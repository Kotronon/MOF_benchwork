from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from analysis.mlip_widom_replicates import aggregate_widom_replicates


class WidomReplicateTests(unittest.TestCase):
    def test_aggregate_reports_seed_uncertainty_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runs = []
            for index, (seed, henry, qst) in enumerate(
                (
                    (12345, 0.100, 12.0),
                    (23456, 0.101, 12.1),
                    (34567, 0.099, 11.9),
                )
            ):
                run = root / f"run_{index}"
                self._write_run(run, seed=seed, henry=henry, qst=qst)
                runs.append(run)

            report = aggregate_widom_replicates(runs, root / "summary")

            self.assertEqual(report["replicate_count"], 3)
            self.assertEqual(report["seeds"], [12345, 23456, 34567])
            self.assertTrue(report["converged"])
            self.assertAlmostEqual(
                report["metrics"]["henry_coefficient_mmol_g_bar"]["mean"],
                0.1,
            )
            for path in report["outputs"].values():
                self.assertTrue(Path(path).is_file())

    def test_duplicate_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first"
            second = root / "second"
            self._write_run(first, seed=12345, henry=0.1, qst=12.0)
            self._write_run(second, seed=12345, henry=0.1, qst=12.0)

            with self.assertRaisesRegex(ValueError, "seeds must be unique"):
                aggregate_widom_replicates([first, second], root / "summary")

    @staticmethod
    def _write_run(run: Path, *, seed: int, henry: float, qst: float) -> None:
        (run / "results").mkdir(parents=True)
        result = {
            "status": "completed",
            "method": "widom",
            "material": "ZIF-8",
            "adsorbate": "CO2",
            "temperature_K": 298.15,
            "seed": seed,
            "attempts": 10000,
            "model": {
                "backend": "mace-torch",
                "name": "mace_mp_small",
                "model": "small",
                "dispersion": False,
                "default_dtype": "float64",
            },
            "corrected_statistics": {
                "henry_coefficient_mmol_g_bar": henry,
                "isosteric_heat_zero_loading_kj_mol": qst,
            },
        }
        plan = {
            "convergence": {
                "minimum_replicates": 3,
                "relative_ci95_target": 0.05,
            }
        }
        (run / "results" / "mlip_mc_benchmark.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        (run / "run_plan.json").write_text(json.dumps(plan), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
