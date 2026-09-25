from __future__ import annotations

import json
from math import sqrt
import unittest

from modules.module_c_mlips.comparison import compare_interaction_results
from modules.module_c_mlips.interaction import InteractionResult
from modules.module_c_mlips.potential_backends.base import PotentialResult


def _potential_result(
    forces: list[list[float]],
    runtime_seconds: float,
) -> PotentialResult:
    return PotentialResult(
        energy_ev=0.0,
        forces_ev_per_angstrom=forces,
        runtime_seconds=runtime_seconds,
    )


def _interaction_result(
    backend: str,
    energy_ev: float,
    forces: list[list[float]],
    *,
    configuration_id: str = "mof_co2_configuration",
    runtime_seconds: float = 6.0,
) -> InteractionResult:
    component_runtime = runtime_seconds / 3.0
    return InteractionResult(
        configuration_id=configuration_id,
        backend=backend,
        interaction_energy_ev=energy_ev,
        interaction_forces_ev_per_angstrom=forces,
        combined=_potential_result(forces, component_runtime),
        framework=_potential_result([], component_runtime),
        adsorbate=_potential_result([], component_runtime),
    )


class InteractionComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = _interaction_result(
            "classical",
            1.0,
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            runtime_seconds=6.0,
        )
        self.candidate = _interaction_result(
            "mace",
            1.5,
            [[1.0, -1.0, 0.0], [2.0, 0.0, -2.0]],
            runtime_seconds=12.0,
        )

    def test_calculates_energy_force_and_runtime_metrics(self) -> None:
        report = compare_interaction_results(
            [self.baseline, self.candidate],
            baseline_backend="classical",
        )

        comparison = report["comparisons"][0]
        self.assertEqual(report["baseline_backend"], "classical")
        self.assertEqual(comparison["candidate_backend"], "mace")
        self.assertEqual(comparison["energy_difference_ev"], 0.5)
        self.assertEqual(comparison["absolute_energy_difference_ev"], 0.5)
        self.assertEqual(comparison["force_mae_ev_per_angstrom"], 1.0)
        self.assertAlmostEqual(
            comparison["force_rmse_ev_per_angstrom"],
            sqrt(10.0 / 6.0),
        )
        self.assertEqual(
            comparison["maximum_force_difference_ev_per_angstrom"],
            2.0,
        )
        self.assertEqual(comparison["runtime_difference_seconds"], 6.0)
        self.assertEqual(
            comparison["runtime_ratio_candidate_to_baseline"],
            2.0,
        )
        json.dumps(report)

    def test_can_select_baseline_independently_of_result_order(self) -> None:
        report = compare_interaction_results(
            [self.candidate, self.baseline],
            baseline_backend="classical",
        )

        self.assertEqual(report["baseline_backend"], "classical")
        self.assertEqual(
            report["comparisons"][0]["candidate_backend"],
            "mace",
        )

    def test_requires_two_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least two"):
            compare_interaction_results([self.baseline])

    def test_rejects_different_configurations(self) -> None:
        candidate = _interaction_result(
            "mace",
            1.5,
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            configuration_id="different_configuration",
        )

        with self.assertRaisesRegex(ValueError, "same configuration_id"):
            compare_interaction_results([self.baseline, candidate])

    def test_rejects_different_force_counts(self) -> None:
        candidate = _interaction_result(
            "mace",
            1.5,
            [[0.0, 0.0, 0.0]],
        )

        with self.assertRaisesRegex(ValueError, "same number of forces"):
            compare_interaction_results([self.baseline, candidate])

    def test_rejects_duplicate_backend_names(self) -> None:
        duplicate = _interaction_result(
            "classical",
            1.5,
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )

        with self.assertRaisesRegex(ValueError, "unique name"):
            compare_interaction_results([self.baseline, duplicate])

    def test_rejects_unknown_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "is not present"):
            compare_interaction_results(
                [self.baseline, self.candidate],
                baseline_backend="missing",
            )


if __name__ == "__main__":
    unittest.main()
