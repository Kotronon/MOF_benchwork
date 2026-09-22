from __future__ import annotations

import json
import math
import unittest

from modules.module_c_mlips.potential_backends.base import PotentialResult


class PotentialResultTests(unittest.TestCase):
    def test_valid_result_is_json_serializable(self) -> None:
        result = PotentialResult(
            energy_ev=-12.5,
            forces_ev_per_angstrom=[
                [0.1, 0.0, -0.1],
                [-0.1, 0.0, 0.1],
            ],
            runtime_seconds=0.25,
            stress_ev_per_angstrom_cubed=[0.0] * 6,
            metadata={"backend": "test", "version": "1"},
        )

        serialized = result.to_dict()

        self.assertEqual(serialized["energy_ev"], -12.5)
        self.assertEqual(len(serialized["forces_ev_per_angstrom"]), 2)
        self.assertEqual(serialized["runtime_seconds"], 0.25)
        self.assertEqual(serialized["metadata"]["backend"], "test")
        json.dumps(serialized)

    def test_negative_runtime_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            PotentialResult(
                energy_ev=0.0,
                forces_ev_per_angstrom=[],
                runtime_seconds=-0.1,
            )

    def test_non_finite_energy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "energy_ev must be finite"):
            PotentialResult(
                energy_ev=math.nan,
                forces_ev_per_angstrom=[],
                runtime_seconds=0.1,
            )

    def test_force_must_have_three_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "3 components"):
            PotentialResult(
                energy_ev=0.0,
                forces_ev_per_angstrom=[[1.0, 2.0]],
                runtime_seconds=0.1,
            )

    def test_non_finite_force_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "force components must be finite"):
            PotentialResult(
                energy_ev=0.0,
                forces_ev_per_angstrom=[[0.0, math.inf, 0.0]],
                runtime_seconds=0.1,
            )

    def test_stress_must_use_six_component_voigt_form(self) -> None:
        with self.assertRaisesRegex(ValueError, "six-component Voigt"):
            PotentialResult(
                energy_ev=0.0,
                forces_ev_per_angstrom=[],
                runtime_seconds=0.1,
                stress_ev_per_angstrom_cubed=[0.0, 0.0, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
