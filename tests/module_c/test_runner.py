from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from modules.module_c_mlips.models import InteractionConfiguration
from modules.module_c_mlips.potential_backends.base import (
    PotentialBackend,
    PotentialResult,
)
from modules.module_c_mlips.runner import run_potential_comparison


HAS_ASE = importlib.util.find_spec("ase") is not None


class DeterministicBackend(PotentialBackend):
    def __init__(self, name: str, scale: float) -> None:
        self._name = name
        self.scale = scale
        self.evaluation_count = 0

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
        self,
        configuration: InteractionConfiguration,
    ) -> PotentialResult:
        self.evaluation_count += 1
        if configuration.configuration_id.endswith("_framework"):
            energy = self.scale
            force_value = self.scale
        elif configuration.configuration_id.endswith("_adsorbate"):
            energy = 2.0 * self.scale
            force_value = self.scale
        else:
            energy = 4.0 * self.scale
            force_value = 3.0 * self.scale

        return PotentialResult(
            energy_ev=energy,
            forces_ev_per_angstrom=[
                [force_value, 0.0, 0.0]
                for _ in configuration.atoms
            ],
            runtime_seconds=0.1,
        )


@unittest.skipUnless(HAS_ASE, "ASE is required for runner tests.")
class PotentialComparisonRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        from ase import Atoms

        atoms = Atoms(
            "CO",
            positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        self.configuration = InteractionConfiguration(
            configuration_id="test_configuration",
            material="test_mof",
            adsorbate="CO",
            atoms=atoms,
            framework_indices=[0],
            adsorbate_indices=[1],
        )
        self.baseline = DeterministicBackend("baseline", 1.0)
        self.candidate = DeterministicBackend("candidate", 2.0)

    def test_runs_every_backend_and_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "nested" / "comparison.json"

            report = run_potential_comparison(
                [self.configuration],
                [self.baseline, self.candidate],
                baseline_backend="baseline",
                output_path=output,
            )

            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["configuration_count"], 1)
        self.assertEqual(report["output_path"], str(output))
        self.assertEqual(saved, report)
        self.assertEqual(self.baseline.evaluation_count, 3)
        self.assertEqual(self.candidate.evaluation_count, 3)
        comparison = report["configurations"][0]["comparisons"][0]
        self.assertEqual(comparison["energy_difference_ev"], 1.0)

    def test_reuses_identical_framework_evaluation(self) -> None:
        second = InteractionConfiguration(
            configuration_id="second_configuration",
            material=self.configuration.material,
            adsorbate=self.configuration.adsorbate,
            atoms=self.configuration.atoms.copy(),
            framework_indices=self.configuration.framework_indices.copy(),
            adsorbate_indices=self.configuration.adsorbate_indices.copy(),
        )

        report = run_potential_comparison(
            [self.configuration, second],
            [self.baseline, self.candidate],
            baseline_backend="baseline",
        )

        self.assertTrue(report["shared_framework_cache"]["enabled"])
        self.assertEqual(self.baseline.evaluation_count, 5)
        self.assertEqual(self.candidate.evaluation_count, 5)
        for configuration in report["configurations"]:
            framework = configuration["results"]["baseline"]["framework"]
            self.assertEqual(framework["runtime_seconds"], 0.0)

    def test_requires_at_least_one_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            run_potential_comparison(
                [],
                [self.baseline, self.candidate],
                baseline_backend="baseline",
            )

    def test_requires_two_backends(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least two"):
            run_potential_comparison(
                [self.configuration],
                [self.baseline],
                baseline_backend="baseline",
            )

    def test_rejects_duplicate_backend_names(self) -> None:
        duplicate = DeterministicBackend("baseline", 2.0)

        with self.assertRaisesRegex(ValueError, "unique"):
            run_potential_comparison(
                [self.configuration],
                [self.baseline, duplicate],
                baseline_backend="baseline",
            )

    def test_reports_backend_and_configuration_when_evaluation_fails(self) -> None:
        class FailingBackend(DeterministicBackend):
            def evaluate(self, configuration):
                raise ValueError("intentional failure")

        failing = FailingBackend("failing", 1.0)

        with self.assertRaisesRegex(
            RuntimeError,
            "Backend 'failing' failed for 'test_configuration'",
        ):
            run_potential_comparison(
                [self.configuration],
                [self.baseline, failing],
                baseline_backend="baseline",
            )


if __name__ == "__main__":
    unittest.main()
