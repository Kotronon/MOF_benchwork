from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from modules.module_c_mlips.adsorption_engines import (
    FLAMESAdsorptionEngine,
    compare_engine_results,
)


class AdsorptionEngineContractTests(unittest.TestCase):
    def test_widom_cross_check_accepts_overlapping_confidence_intervals(self) -> None:
        primary = self._widom("mlip_mc", 1.0, 0.10, 25.0, 0.5)
        secondary = self._widom("flames", 1.15, 0.10, 25.8, 0.5)

        report = compare_engine_results(primary, secondary)

        self.assertEqual(report["status"], "passed")
        self.assertTrue(
            report["comparisons"]["henry_coefficient_mol_kg_pa"][
                "confidence_intervals_overlap"
            ]
        )

    def test_gcmc_cross_check_uses_common_mol_per_kg_basis(self) -> None:
        primary = {
            "engine": "mlip_mc",
            "method": "gcmc",
            "pressure_points": [
                {
                    "pressure_bar": 1.0,
                    "loading_mol_per_kg": 2.0,
                    "loading_std_mol_per_kg": 0.1,
                }
            ],
        }
        secondary = {
            "engine": "flames",
            "method": "gcmc",
            "pressure_points": [
                {
                    "pressure_bar": 1.0,
                    "loading_mol_per_kg": 2.08,
                    "loading_std_mol_per_kg": 0.1,
                }
            ],
        }

        report = compare_engine_results(primary, secondary)

        self.assertEqual(report["status"], "passed")

    @unittest.skipUnless(
        importlib.util.find_spec("ase") is not None,
        "ASE is required for the FLAMES adapter contract.",
    )
    def test_flames_gcmc_adapter_uses_pinned_api_and_normalizes_output(self) -> None:
        from ase import Atoms

        class FakeGCMC:
            def __init__(self, **kwargs):
                self.output = Path(kwargs["output_folder"])
                self.assert_adsorbate = kwargs["adsorbate_atoms"]

            def run(self, steps):
                self.steps = steps

            def save_results(self):
                (self.output / "results.json").write_text(
                    json.dumps(
                        {
                            "equilibration": {"equilibrated": True},
                            "enthalpy": {
                                "kJ_mol": {"mean": -20.0, "sd": 1.0}
                            },
                            "absolute_uptake": {
                                "mol/kg": {"mean": 2.0, "sd": 0.2}
                            },
                        }
                    ),
                    encoding="utf-8",
                )

        with tempfile.TemporaryDirectory() as temporary:
            engine = FLAMESAdsorptionEngine(gcmc_class=FakeGCMC)
            result = engine.run_gcmc(
                {
                    "framework": Atoms(
                        "C",
                        positions=[[0.0, 0.0, 0.0]],
                        cell=[10.0, 10.0, 10.0],
                        pbc=True,
                    ),
                    "adsorbate": Atoms(
                        "CO2",
                        positions=[
                            [0.0, 0.0, 0.0],
                            [1.16, 0.0, 0.0],
                            [-1.16, 0.0, 0.0],
                        ],
                    ),
                },
                object(),
                {
                    "temperature_K": 298.15,
                    "pressures_bar": [1.0],
                    "equilibration_steps": 10,
                    "production_steps": 20,
                    "seed": 1,
                    "device": "cpu",
                    "output_directory": temporary,
                    "component": "CO2",
                },
            )

        self.assertEqual(
            result["pressure_points"][0]["loading_mol_per_kg"],
            2.0,
        )
        self.assertTrue(result["pressure_points"][0]["equilibrated"])

    @staticmethod
    def _widom(
        engine: str,
        henry: float,
        henry_uncertainty: float,
        heat: float,
        heat_uncertainty: float,
    ) -> dict:
        return {
            "engine": engine,
            "method": "widom",
            "metrics": {
                "henry_coefficient_mol_kg_pa": {
                    "value": henry,
                    "uncertainty": henry_uncertainty,
                },
                "isosteric_heat_zero_loading_kj_mol": {
                    "value": heat,
                    "uncertainty": heat_uncertainty,
                },
            },
        }


if __name__ == "__main__":
    unittest.main()
