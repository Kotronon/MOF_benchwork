from __future__ import annotations

import json
import os
import zipfile
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
    prepare_cp2k_scf_retries,
    render_cp2k_input,
    render_slurm_script,
)
from modules.module_c_mlips.active_learning.models import ActiveLearningSettings, ActiveLearningState
from modules.module_c_mlips.active_learning.training import (
    build_mace_finetune_commands,
    classify_validation_metrics,
    execute_mace_commands,
    prune_mace_checkpoints,
    split_multisystem_labels,
    write_mace_datasets,
)
from modules.module_c_mlips.active_learning.workflow import _extend_cutoff_test, run_active_learning


class ActiveLearningTests(unittest.TestCase):
    def test_multisystem_split_holds_out_frameworks_and_mixture_ratios(self) -> None:
        labels = [
            {"candidate_id": "train", "material": "MOF-A", "gas_composition": "CO2"},
            {"candidate_id": "validation", "material": "MOF-A", "gas_composition": "N2"},
            {"candidate_id": "unseen_mof", "material": "MOF-B", "gas_composition": "CO2"},
            {
                "candidate_id": "unseen_ratio",
                "material": "MOF-A",
                "gas_composition": {"CO2": 0.25, "N2": 0.75},
            },
        ]

        splits = split_multisystem_labels(
            labels,
            unseen_materials={"MOF-B"},
            unseen_gas_compositions={'{"CO2":0.25,"N2":0.75}'},
            validation_gas_compositions={'{"N2":1.0}'},
        )

        self.assertEqual(splits["train"], ["train"])
        self.assertEqual(splits["validation"], ["validation"])
        self.assertEqual(splits["test"], ["unseen_mof", "unseen_ratio"])
        self.assertEqual(splits["test_unseen_materials"], ["unseen_mof"])
        self.assertEqual(
            splits["test_unseen_gas_compositions"], ["unseen_ratio"]
        )
        self.assertFalse(set(splits["train"]) & set(splits["test"]))

    def test_multisystem_split_rejects_holdout_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "holdouts must be disjoint"):
            split_multisystem_labels(
                [],
                unseen_materials={"MOF-A"},
                unseen_gas_compositions=set(),
                validation_materials={"MOF-A"},
            )

    def test_cutoff_gate_uses_energy_per_atom_and_maximum_force_delta(self) -> None:
        results = {
            400: {"energy_ev": -100.0, "forces_ev_per_A": [[0.0, 0.0, 0.0]]},
            600: {"energy_ev": -100.0005, "forces_ev_per_A": [[0.01, 0.0, 0.0]]},
            800: {"energy_ev": -100.0006, "forces_ev_per_A": [[0.011, 0.0, 0.0]]},
        }

        report = cutoff_converged(results, atom_count=1)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["selected_cutoff_Ry"], 600)

    def test_cp2k_forces_print_section_is_under_force_eval(self) -> None:
        from ase import Atoms

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                      cell=[10, 10, 10], pbc=True)
        text = render_cp2k_input(atoms, project="forces_check", settings={})
        sections = []
        force_paths = []
        for line in text.splitlines():
            fields = line.strip().split()
            if not fields:
                continue
            if fields[0] == "&END":
                sections.pop()
            elif fields[0].startswith("&"):
                sections.append(fields[0][1:])
                if fields[0] == "&FORCES":
                    force_paths.append(tuple(sections))
        self.assertEqual(force_paths, [("FORCE_EVAL", "PRINT", "FORCES")])

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

    def test_cp2k_output_parser_reads_modern_force_table(self) -> None:
        content = """ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -2.000000
 FORCES| Atomic forces [hartree/bohr]
 FORCES|   Atom     x               y               z               |f|
 FORCES|      1  1.00000000E-01 -2.00000000E-01 3.00000000E-01 3.74165739E-01
 FORCES|      2 -1.00000000E-01  2.00000000E-01 -3.00000000E-01 3.74165739E-01

 PROGRAM ENDED AT 2026-01-01
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cp2k.out"
            path.write_text(content, encoding="utf-8")
            parsed = parse_cp2k_output(path)
            statuses = CP2KLabeler({}).status([
                {"candidate_id": "modern", "output_path": str(path)}
            ])
        self.assertEqual(statuses[0]["status"], "completed")
        self.assertEqual(len(parsed["forces_ev_per_A"]), 2)
        for actual, expected in zip(parsed["forces_ev_per_A"][0], [0.1, -0.2, 0.3]):
            self.assertAlmostEqual(actual, expected * HARTREE_PER_BOHR_TO_EV_PER_A)

    def test_resume_extends_failed_cutoff_test_without_replacing_outputs(self) -> None:
        from ase import Atoms
        from ase.io import write

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, state_path = self._prepared_models_state(
                root, stable_rounds=0, iteration=0, candidate_count=50,
            )
            active_root = state_path.parent
            structure = root / "candidate.extxyz"
            write(structure, Atoms("C", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True))
            original = {"cutoff_test_Ry": [400, 600], "relative_cutoff_Ry": 60}
            jobs = CP2KLabeler(original).prepare_cutoff_test(
                {"candidate_id": "candidate", "structure_path": str(structure)},
                active_root / "cutoff_test",
            )
            snapshots = {}
            for job in jobs:
                output = Path(job["output_path"])
                output.write_text("existing output", encoding="utf-8")
                snapshots[output] = output.read_bytes()
                input_path = Path(job["input_path"])
                snapshots[input_path] = input_path.read_bytes()
            state = ActiveLearningState.load(state_path)
            state.status = "cutoff_not_converged"
            state.failure_reason = "failed gate"
            state.save(state_path)
            configured = {**original, "cutoff_test_Ry": [400, 600, 800, 1000]}
            plan["benchmark"]["potential_benchmark"]["active_learning"]["cp2k"] = configured
            with patch(
                "modules.module_c_mlips.active_learning.workflow._advance_cutoff_test",
                return_value=True,
            ):
                run_active_learning(plan, resume=True)
            resumed = ActiveLearningState.load(state_path)
            self.assertEqual(resumed.status, "awaiting_cutoff_test")
            self.assertIsNone(resumed.failure_reason)
            self.assertEqual(resumed.training_ids, state.training_ids)
            self.assertEqual(resumed.test_ids, state.test_ids)
            self.assertEqual(resumed.history[-1]["added_cutoffs_Ry"], [800, 1000])
            expanded = json.loads((active_root / "cutoff_test/cp2k_jobs.json").read_text())["jobs"]
            self.assertEqual([job["cutoff_Ry"] for job in expanded], [400, 600, 800, 1000])
            for path, content in snapshots.items():
                self.assertEqual(path.read_bytes(), content)
            settings = ActiveLearningSettings(cp2k=configured)
            self.assertFalse(_extend_cutoff_test(settings, resumed, active_root))
            changed = ActiveLearningSettings(cp2k={**configured, "cutoff_test_Ry": [400, 600, 1200], "relative_cutoff_Ry": 80})
            with self.assertRaisesRegex(ValueError, "unchanged CP2K"):
                _extend_cutoff_test(changed, resumed, active_root)

    def test_scf_retry_archives_failures_and_preserves_completed_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = []
            for name in ("failed", "completed", "running"): 
                workdir = root / name
                workdir.mkdir()
                (workdir / "cp2k.inp").write_text("&SCF\n  EPS_SCF 1.000e-06\n  MAX_SCF 200\n&END SCF\n")
                (workdir / "launcher.stderr.log").write_text("old log")
                (workdir / f"{name}-RESTART.wfn").write_bytes(b"wavefunction")
                output = (
                    "[ABORT] SCF run NOT converged."
                    if name == "failed" else
                    "ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -2.0\n"
                    " FORCES| Atomic forces [hartree/bohr]\n"
                    " FORCES| 1 0.1 0.2 0.3 0.4\n\n PROGRAM ENDED AT today\n"
                )
                if name == "running":
                    output = "SCF iteration 61 still in progress\n"
                (workdir / "cp2k.out").write_text(output)
                jobs.append({"candidate_id": name, "job_directory": str(workdir),
                             "input_path": str(workdir / "cp2k.inp"),
                             "output_path": str(workdir / "cp2k.out"),
                             "status": name, "returncode": 1 if name == "failed" else 0})
            manifest = root / "cp2k_jobs.json"
            manifest.write_text(json.dumps({"jobs": jobs}))
            successful = {path: path.read_bytes() for name in ("completed", "running")
                          for path in (root / name).iterdir()}
            retried = prepare_cp2k_scf_retries(manifest, max_scf=1000)
            self.assertEqual(retried, ["failed"])
            failed = root / "failed"
            self.assertFalse((failed / "cp2k.out").exists())
            self.assertEqual((failed / "scf_retry_01/cp2k.out").read_text(), "[ABORT] SCF run NOT converged.")
            self.assertEqual((failed / "scf_retry_01/failed-RESTART.wfn").read_bytes(), b"wavefunction")
            self.assertEqual((failed / "failed-RESTART.wfn").read_bytes(), b"wavefunction")
            self.assertIn("MAX_SCF 1000", (failed / "cp2k.inp").read_text())
            self.assertIn("EPS_SCF 1.000e-06", (failed / "cp2k.inp").read_text())
            self.assertIn("SCF_GUESS RESTART", (failed / "cp2k.inp").read_text())
            repaired = json.loads(manifest.read_text())["jobs"][0]
            self.assertEqual(repaired["status"], "prepared")
            self.assertNotIn("returncode", repaired)
            for path, content in successful.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertEqual(prepare_cp2k_scf_retries(manifest, max_scf=1000), [])

    def test_scf_retry_rejects_other_failures_before_modifying_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = []
            for index, content in enumerate(("[ABORT] SCF run NOT converged.", "[ABORT] out of memory")):
                workdir = root / str(index)
                workdir.mkdir()
                (workdir / "cp2k.out").write_text(content)
                (workdir / "cp2k.inp").write_text("MAX_SCF 200\n")
                jobs.append({"candidate_id": str(index), "job_directory": str(workdir),
                             "input_path": str(workdir / "cp2k.inp"), "output_path": str(workdir / "cp2k.out")})
            manifest = root / "cp2k_jobs.json"
            manifest.write_text(json.dumps({"jobs": jobs}))
            with self.assertRaisesRegex(ValueError, "unrelated to SCF"):
                prepare_cp2k_scf_retries(manifest, max_scf=1000)
            self.assertTrue((root / "0/cp2k.out").exists())
            self.assertEqual((root / "0/cp2k.inp").read_text(), "MAX_SCF 200\n")

    def test_parser_rejects_unconverged_scf_even_with_normal_termination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cp2k.out"
            path.write_text("SCF run NOT converged\nPROGRAM ENDED AT today\n")
            with self.assertRaisesRegex(ValueError, "SCF did not converge"):
                parse_cp2k_output(path)

    def test_cp2k_input_uses_configured_scf_step_limit(self) -> None:
        text = render_cp2k_input(_MinimalAtoms(), project="scf", settings={"max_scf": 1000})
        self.assertIn("MAX_SCF 1000", text)
        self.assertIn("EPS_SCF 1.000e-06", text)
        with self.assertRaisesRegex(ValueError, "max_scf must be positive"):
            render_cp2k_input(_MinimalAtoms(), project="scf", settings={"max_scf": 0})

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
        self.assertIn("--restart_latest", command)
        self.assertIn("--save_all_checkpoints", command)
        self.assertEqual(command[command.index("--eval_interval") + 1], "1")
        self.assertIn("--log_dir", command)
        self.assertNotIn("--logs_dir", command)
        mode_index = command.index("--multiheads_finetuning")
        self.assertEqual(command[mode_index + 1], "False")
        self.assertEqual(command[command.index("--E0s") + 1], "average")
        self.assertEqual(command[command.index("--batch_size") + 1], "1")
        self.assertEqual(command[command.index("--valid_batch_size") + 1], "1")

    def test_training_resumes_legacy_commands_without_mutating_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            models.mkdir()
            (models / "trained.model").write_text("model")
            for command in (["mace_run_train"], ["mace_run_train", "--restart_latest",
                                                     "--save_all_checkpoints", "--eval_interval", "10"]):
                entry = {"command": command, "working_directory": str(root),
                         "model_directory": str(models)}
                original = list(command)
                with patch(
                    "modules.module_c_mlips.active_learning.training.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, "finished", ""),
                ) as run:
                    result = execute_mace_commands([entry])
                executed = run.call_args.args[0]
                self.assertEqual(executed.count("--restart_latest"), 1)
                self.assertEqual(executed.count("--save_all_checkpoints"), 1)
                self.assertEqual(executed.count("--eval_interval"), 1)
                self.assertEqual(executed[executed.index("--eval_interval") + 1], "1")
                self.assertEqual(entry["command"], original)
                self.assertEqual(result[0]["status"], "completed")

    def test_cleanup_preserves_recent_best_and_in_progress_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            (root / "logs").mkdir()
            (root / "logs/run.log").write_text("Epoch 1: loss=0.1\nEpoch 2: loss=1.0\n")
            for epoch in range(7):
                path = checkpoints / f"run_epoch-{epoch}.pt"
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("state", "checkpoint")
                os.utime(path, (1, 1))
            pending = checkpoints / "run_epoch-7.pt"
            pending.write_text("still writing")
            unrelated = checkpoints / "notes.pt"
            unrelated.write_text("leave alone")
            self.assertEqual(prune_mace_checkpoints(checkpoints), 3)
            self.assertEqual({p.name for p in checkpoints.iterdir()}, {
                "run_epoch-1.pt", "run_epoch-4.pt", "run_epoch-5.pt",
                "run_epoch-6.pt", "run_epoch-7.pt", "notes.pt",
            })
            # An incomplete settled save prevents pruning fallback checkpoints.
            os.utime(pending, (1, 1))
            self.assertEqual(prune_mace_checkpoints(checkpoints), 0)
            self.assertTrue((checkpoints / "run_epoch-4.pt").exists())

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
