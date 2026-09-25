from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from analysis.potential_report import create_potential_report


class PotentialReportTests(unittest.TestCase):
    def test_writes_aggregate_json_and_flat_csv(self) -> None:
        report = {
            "configurations": [
                self.configuration("sample_0", 2.0, 2.5, 1.8),
                self.configuration("sample_1", -1.0, -0.75, 2.7),
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = create_potential_report(
                report,
                tmpdir,
                save_csv=True,
                save_plots=False,
            )
            summary = json.loads(
                Path(artifacts["summary_json"]).read_text(encoding="utf-8")
            )
            with Path(artifacts["table_csv"]).open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))

        candidate = summary["candidates"]["candidate"]
        self.assertEqual(summary["row_count"], 2)
        self.assertAlmostEqual(candidate["energy_mae_ev"], 0.375)
        self.assertAlmostEqual(
            candidate["median_absolute_energy_difference_ev"],
            0.375,
        )
        self.assertAlmostEqual(candidate["mean_force_mae_ev_per_angstrom"], 0.15)
        self.assertIn("near_contact", candidate["by_region"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["minimum_host_guest_distance_A"], "1.8")

    @staticmethod
    def configuration(
        configuration_id: str,
        baseline_energy: float,
        candidate_energy: float,
        distance: float,
    ) -> dict:
        difference = candidate_energy - baseline_energy
        return {
            "configuration_id": configuration_id,
            "baseline_backend": "baseline",
            "region": "near_contact",
            "metadata": {"minimum_host_guest_distance_A": distance},
            "results": {
                "baseline": {
                    "interaction_energy_ev": baseline_energy,
                },
                "candidate": {
                    "interaction_energy_ev": candidate_energy,
                },
            },
            "comparisons": [
                {
                    "candidate_backend": "candidate",
                    "energy_difference_ev": difference,
                    "absolute_energy_difference_ev": abs(difference),
                    "force_mae_ev_per_angstrom": 0.1 if distance < 2.0 else 0.2,
                    "force_rmse_ev_per_angstrom": 0.2 if distance < 2.0 else 0.3,
                    "maximum_force_difference_ev_per_angstrom": 0.4,
                    "baseline_runtime_seconds": 0.1,
                    "candidate_runtime_seconds": 0.2,
                    "runtime_ratio_candidate_to_baseline": 2.0,
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
