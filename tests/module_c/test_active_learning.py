from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from modules.module_c_mlips.active_learning.dft import (
    CP2KLabeler,
    HARTREE_PER_BOHR_TO_EV_PER_A,
    HARTREE_TO_EV,
    build_local_cp2k_command,
    cutoff_converged,
    parse_cp2k_output,
    render_cp2k_input,
    render_slurm_script,
)
from modules.module_c_mlips.active_learning.models import ActiveLearningState
from modules.module_c_mlips.active_learning.training import (
    build_mace_finetune_commands,
    classify_validation_metrics,
    write_mace_datasets,
)
from modules.module_c_mlips.active_learning.workflow import run_active_learning


class ActiveLearningTests(unittest.TestCase):
    def test_cutoff_gate_uses_energy_per_atom_and_maximum_force_delta(self) -> None:
        results = {
            400: {"energy_ev": -100.0, "forces_ev_per_A": [[0.0, 0.0, 0.0]]},
            600: {"energy_ev": -100.0005, "forces_ev_per_A": [[0.01, 0.0, 0.0]]},
            800: {"energy_ev": -100.0006, "forces_ev_per_A": [[0.011, 0.0, 0.0]]},
        }

        report = cutoff_converged(results, atom_count=1)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["selected_cutoff_Ry"], 600)

    def test_cp2k_output_parser_converts_atomic_units(self) -> None:
        content = """ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.): -2.000000
 ATOMIC FORCES in [a.u.]
    1    1 C    0.100000 0.000000 0.000000
 SUM OF ATOMIC FORCES
 PROGRAM ENDED AT 2026-01-01
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cp2k.out"
            path.write_text(content, encoding="utf-8")

            parsed = parse_cp2k_output(path)

        self.assertAlmostEqual(parsed["energy_ev"], -2.0 * HARTREE_TO_EV)
        self.assertAlmostEqual(
            parsed["forces_ev_per_A"][0][0],
            0.1 * HARTREE_PER_BOHR_TO_EV_PER_A,
        )

    def test_slurm_script_records_reproducible_resources(self) -> None:
        script = render_slurm_script(
            "candidate 1",
            ["cp2k.psmp", "-i", "cp2k.inp", "-o", "cp2k.out"],
            {
                "slurm": {
                    "nodes": 2,
                    "ntasks": 32,
                    "time": "06:00:00",
                    "partition": "cpu",
                }
            },
        )

        self.assertIn("#SBATCH --nodes=2", script)
        self.assertIn("#SBATCH --partition=cpu", script)
        self.assertIn("srun cp2k.psmp -i cp2k.inp -o cp2k.out", script)

    def test_local_cp2k_command_uses_configured_mpi_processes(self) -> None:
        command = build_local_cp2k_command(
            ["cp2k", "-i", "cp2k.inp", "-o", "cp2k.out"],
            {
                "local": {
                    "launcher": "mpirun",
                    "mpi_processes_per_job": 16,
                }
            },
        )

        self.assertEqual(
            command,
            [
                "mpirun",
                "-np",
                "16",
                "cp2k",
                "-i",
                "cp2k.inp",
                "-o",
                "cp2k.out",
            ],
        )

    def test_local_cp2k_execution_records_resources_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = {
                "candidate_id": "candidate-1",
                "job_directory": str(root),
                "command": ["cp2k", "-i", "cp2k.inp", "-o", "cp2k.out"],
            }
            labeler = CP2KLabeler(
                {
                    "scheduler": "local",
                    "executable": "cp2k",
                    "local": {
                        "launcher": "mpirun",
                        "mpi_processes_per_job": 8,
                        "omp_threads_per_process": 2,
                        "max_parallel_jobs": 1,
                    },
                }
            )
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="launcher output",
                stderr="",
            )
            with patch(
                "modules.module_c_mlips.active_learning.dft._record_cp2k_version"
            ), patch(
                "modules.module_c_mlips.active_learning.dft.subprocess.run",
                return_value=completed,
            ) as run:
                results = labeler.submit([job])

            call = run.call_args
            self.assertEqual(call.args[0][:4], ["mpirun", "-np", "8", "cp2k"])
            self.assertEqual(call.kwargs["env"]["OMP_NUM_THREADS"], "2")
            self.assertEqual(results[0]["status"], "completed")
            self.assertEqual(results[0]["execution_resources"]["max_parallel_jobs"], 1)
            self.assertEqual(
                (root / "launcher.stdout.log").read_text(encoding="utf-8"),
                "launcher output",
            )

    def test_mbd_requires_an_explicit_project_template(self) -> None:
        atoms = _MinimalAtoms()
        with self.assertRaisesRegex(ValueError, "project-validated"):
            render_cp2k_input(
                atoms,
                project="mg_mof_74",
                settings={"profile": "pbe_mbd"},
            )

    def test_test_split_cannot_leak_into_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            write_mace_datasets(
                [],
                training_ids={"same"},
                validation_ids=set(),
                test_ids={"same"},
                output_directory="unused",
            )

    def test_mace_uses_validation_file_not_held_out_test_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            commands = build_mace_finetune_commands(
                {
                    "train": "train.extxyz",
                    "valid": "valid.extxyz",
                    "test": "test.extxyz",
                },
                temporary,
                committee_size=1,
                base_model="mace_mp_0a_small",
                device="cpu",
                seeds=[12345],
            )

        command = commands[0]["command"]
        valid_index = command.index("--valid_file")
        self.assertEqual(command[valid_index + 1], "valid.extxyz")
        self.assertNotIn("test.extxyz", command)

    def test_validation_thresholds_have_pass_warning_and_failure(self) -> None:
        settings = _ThresholdSettings()
        passed = classify_validation_metrics(
            {
                "energy_mae_ev": 0.02,
                "force_mae_ev_per_A": 0.05,
                "maximum_committee_energy_std_ev": 0.02,
                "maximum_committee_force_std_ev_per_A": 0.05,
            },
            settings,
        )
        warning = classify_validation_metrics(
            {
                "energy_mae_ev": 0.08,
                "force_mae_ev_per_A": 0.15,
                "maximum_committee_energy_std_ev": 0.08,
                "maximum_committee_force_std_ev_per_A": 0.15,
            },
            settings,
        )

        self.assertEqual(passed["status"], "passed")
        self.assertEqual(warning["status"], "warning")

    def test_state_machine_marks_second_stable_round_dft_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, state_path = self._prepared_models_state(
                Path(temporary),
                stable_rounds=1,
                iteration=1,
                candidate_count=75,
            )
            metrics = {
                "energy_mae_ev": 0.02,
                "force_mae_ev_per_A": 0.05,
                "maximum_committee_energy_std_ev": 0.02,
                "maximum_committee_force_std_ev_per_A": 0.05,
                "best_model_index": 0,
                "committee_members": [],
            }
            with patch(
                "modules.module_c_mlips.active_learning.workflow.load_labels",
                return_value=[],
            ), patch(
                "modules.module_c_mlips.active_learning.workflow.evaluate_mace_committee",
                return_value=metrics,
            ):
                report = run_active_learning(plan, resume=True)

            self.assertEqual(report["status"], "dft_validated")
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["stable_rounds"], 2)

    def test_state_machine_stops_when_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, _ = self._prepared_models_state(
                Path(temporary),
                stable_rounds=0,
                iteration=4,
                candidate_count=150,
            )
            metrics = {
                "energy_mae_ev": 0.5,
                "force_mae_ev_per_A": 0.5,
                "maximum_committee_energy_std_ev": 0.5,
                "maximum_committee_force_std_ev_per_A": 0.5,
                "best_model_index": 0,
                "committee_members": [],
            }
            with patch(
                "modules.module_c_mlips.active_learning.workflow.load_labels",
                return_value=[],
            ), patch(
                "modules.module_c_mlips.active_learning.workflow.evaluate_mace_committee",
                return_value=metrics,
            ):
                report = run_active_learning(plan, resume=True)

            self.assertEqual(report["status"], "adaptation_not_converged")

    def test_submit_does_not_start_mace_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, state_path = self._prepared_models_state(
                Path(temporary),
                stable_rounds=0,
                iteration=0,
                candidate_count=50,
            )
            state = ActiveLearningState.load(state_path)
            state.status = "ready_to_train"
            state.model_paths = []
            state.save(state_path)
            with patch(
                "modules.module_c_mlips.active_learning.workflow.load_labels",
                return_value=[],
            ), patch(
                "modules.module_c_mlips.active_learning.workflow.write_mace_datasets",
                return_value={
                    "train": "train.extxyz",
                    "valid": "valid.extxyz",
                    "test": "test.extxyz",
                },
            ), patch(
                "modules.module_c_mlips.active_learning.workflow.build_mace_finetune_commands",
                return_value=[],
            ), patch(
                "modules.module_c_mlips.active_learning.workflow._execute_training"
            ) as execute_training:
                report = run_active_learning(
                    plan,
                    submit=True,
                    train=False,
                    resume=True,
                )

        self.assertEqual(report["status"], "training_prepared")
        execute_training.assert_not_called()

    @staticmethod
    def _prepared_models_state(
        root: Path,
        *,
        stable_rounds: int,
        iteration: int,
        candidate_count: int,
    ) -> tuple[dict, Path]:
        run_id = "campaign/task"
        active_root = root / "runs" / run_id / "active_learning"
        active_root.mkdir(parents=True)
        state_path = active_root / "state.json"
        state = ActiveLearningState(
            campaign_id="campaign",
            material_id="custom",
            status="models_ready",
            iteration=iteration,
            stable_rounds=stable_rounds,
            candidate_count=candidate_count,
            test_ids=["test"],
            training_ids=["train"],
            validation_ids=["valid"],
            model_paths=["model-0.model", "model-1.model", "model-2.model"],
        )
        state.save(state_path)
        plan = {
            "benchmark": {
                "applicability": {},
                "potential_benchmark": {
                    "validation": {"mode": "active_learning"},
                    "active_learning": {
                        "rounds": 4,
                        "maximum_configurations": 150,
                    },
                },
            },
            "material": {"material_id": "custom"},
            "outputs": {"directory": str(root), "run_id": run_id},
        }
        return plan, state_path


class _ThresholdSettings:
    energy_pass_mae_ev = 0.043
    force_pass_mae_ev_per_A = 0.10
    energy_warning_mae_ev = 0.10
    force_warning_mae_ev_per_A = 0.20


class _MinimalAtoms:
    class _Cell:
        array = (
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 10.0),
        )

    cell = _Cell()
    positions = [(0.0, 0.0, 0.0)]

    @staticmethod
    def get_chemical_symbols() -> list[str]:
        return ["C"]


if __name__ == "__main__":
    unittest.main()
