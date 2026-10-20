from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from modules.module_c_mlips.models import InteractionConfiguration
from modules.module_c_mlips.potential_backends.base import (
    PotentialBackend,
    PotentialResult,
)
from modules.module_c_mlips.strategy_comparison import (
    run_strategy_comparison,
    _evaluate_backends,
)
from modules.module_c_mlips.workflow import _build_backends


try:
    from ase import Atoms
except ImportError:  # pragma: no cover
    Atoms = None


class DummyBackend(PotentialBackend):
    def __init__(self, name: str, energy: float, force: float) -> None:
        self._name = name
        self.energy = energy
        self.force = force

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, configuration: InteractionConfiguration) -> PotentialResult:
        return PotentialResult(
            energy_ev=self.energy,
            forces_ev_per_angstrom=[
                [self.force, 0.0, 0.0]
                for _ in range(len(configuration.atoms))
            ],
            runtime_seconds=0.01,
        )


@unittest.skipIf(Atoms is None, "ASE is required for strategy comparison tests.")
class StrategyComparisonTests(unittest.TestCase):
    def test_umof_backend_requires_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "uMOF checkpoint required"):
                _build_backends(
                    [
                        {
                            "backend": "mace-torch",
                            "name": "umof_mh",
                            "model": str(Path(temporary) / "missing.model"),
                        }
                    ],
                    _minimal_run_plan(Path(temporary)),
                    Path(temporary) / "backends",
                )

    def test_strategy_report_marks_umof_as_cp2k_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_plan = _minimal_run_plan(root)
            run_plan["benchmark"]["potential_benchmark"]["comparison"] = {
                "strategies": ["umof_pretrained"],
                "umof_backends": [
                    {
                        "backend": "mace-torch",
                        "name": "umof_mh",
                        "model": str(root / "umof.model"),
                        "model_family": "umof",
                    }
                ],
            }
            (root / "umof.model").write_text("placeholder", encoding="utf-8")
            configuration = _configuration(reference=False)

            with patch(
                "modules.module_c_mlips.strategy_comparison._load_comparison_configurations",
                return_value=([configuration], False),
            ), patch(
                "modules.module_c_mlips.workflow._build_backends",
                return_value=[DummyBackend("umof_mh", energy=5.0, force=0.0)],
            ):
                report = run_strategy_comparison(run_plan)

        row = report["rows"][0]
        self.assertFalse(report["cp2k_jobs_started"])
        self.assertFalse(row["uses_local_cp2k_labels"])
        self.assertFalse(row["fine_tuned_locally"])
        self.assertIsNone(row["energy_mae_vs_cp2k"])
        self.assertIsNone(row["force_mae_vs_cp2k"])

    def test_comparison_excludes_configured_repulsive_probes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _minimal_run_plan(root)
            potential = plan["benchmark"]["potential_benchmark"]
            potential["exclude_regions"] = ["technical_repulsive_probe"]
            potential["comparison"] = {"strategies": ["umof_pretrained"],
                "umof_backends": [{"backend": "mace-torch", "name": "umof_mh", "model": str(root / "umof.model")}]}
            (root / "umof.model").write_text("placeholder")
            good = _configuration(reference=False)
            probe = replace(good, configuration_id="overlap_probe", region="technical_repulsive_probe")
            with patch("modules.module_c_mlips.strategy_comparison._load_comparison_configurations", return_value=([good, probe], False)), patch(
                "modules.module_c_mlips.workflow._build_backends", return_value=[DummyBackend("umof_mh", 1.0, 0.0)]
            ):
                report = run_strategy_comparison(plan)
            self.assertEqual(report["configuration_count"], 1)
            self.assertEqual([row["configuration_id"] for row in report["evaluations"]], [good.configuration_id])

    def test_completed_evaluations_survive_a_later_backend_failure(self) -> None:
        class FailingBackend(DummyBackend):
            def evaluate(self, configuration):
                raise ValueError("nonfinite test result")

        with tempfile.TemporaryDirectory() as temporary:
            progress = Path(temporary) / "progress.json"
            with self.assertRaisesRegex(RuntimeError, "failing"):
                _evaluate_backends([_configuration(reference=False)],
                    [DummyBackend("good", 1.0, 0.0), FailingBackend("failing", 0.0, 0.0)],
                    progress_path=progress)
            data = json.loads(progress.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertEqual(data["failed_backend"], "failing")
            self.assertEqual(len(data["evaluations"]), 1)
            self.assertEqual(data["evaluations"][0]["backend"], "good")

    def test_strategy_report_computes_errors_when_cp2k_labels_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_plan = _minimal_run_plan(root)
            run_plan["benchmark"]["potential_benchmark"]["comparison"] = {
                "strategies": ["umof_pretrained"],
                "umof_backends": [
                    {
                        "backend": "mace-torch",
                        "name": "umof_mh",
                        "model": str(root / "umof.model"),
                        "model_family": "umof",
                    }
                ],
            }
            (root / "umof.model").write_text("placeholder", encoding="utf-8")
            configuration = _configuration(reference=True)

            with patch(
                "modules.module_c_mlips.strategy_comparison._load_comparison_configurations",
                return_value=([configuration], True),
            ), patch(
                "modules.module_c_mlips.workflow._build_backends",
                return_value=[DummyBackend("umof_mh", energy=4.5, force=0.2)],
            ):
                report = run_strategy_comparison(run_plan)
                self.assertTrue(Path(report["outputs"]["json"]).exists())
                self.assertTrue(Path(report["outputs"]["csv"]).exists())

            row = report["rows"][0]
        self.assertAlmostEqual(row["energy_mae_vs_cp2k"], 0.5)
        self.assertGreater(row["force_mae_vs_cp2k"], 0.0)


def _configuration(*, reference: bool) -> InteractionConfiguration:
    atoms = Atoms(
        symbols=["Zn", "O", "C", "O"],
        positions=np.zeros((4, 3)),
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    return InteractionConfiguration(
        configuration_id="test",
        material="MOF",
        adsorbate="CO2",
        atoms=atoms,
        framework_indices=[0],
        adsorbate_indices=[1, 2, 3],
        reference_interaction_energy_ev=4.0 if reference else None,
        reference_forces_ev_per_angstrom=[[0.0, 0.0, 0.0]] * 4 if reference else None,
    )


def _minimal_run_plan(root: Path) -> dict:
    return {
        "module": {"id": "C"},
        "material": {"material_id": "MOF"},
        "adsorbates": {"components": ["CO2"]},
        "simulation": {
            "cell_representation": "source",
            "unit_cells": [1, 1, 1],
            "cutoff_A": 6.0,
            "minimum_image_policy": "ignore",
        },
        "benchmark": {
            "potential_benchmark": {
                "configuration_set": "smoke",
                "comparison": {},
            }
        },
        "resources": {
            "cif_path": "unused.cif",
            "forcefield": {
                "adsorbates": {"CO2": "unused.def"},
                "files": {"pseudo_atoms": "unused.def"},
            },
        },
        "outputs": {
            "directory": str(root / "outputs"),
            "run_id": "strategy",
            "overwrite": True,
            "save_plots": False,
            "save_csv": True,
        },
    }


if __name__ == "__main__":
    unittest.main()
