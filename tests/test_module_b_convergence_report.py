from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from analysis.module_b_convergence_report import (
    _batch_statistics,
    analyze_module_b_convergence,
)


class ModuleBConvergenceReportTests(unittest.TestCase):
    def test_batch_statistics_account_for_temporal_autocorrelation(self) -> None:
        values = [float((index // 10) % 4) for index in range(400)]

        result = _batch_statistics(values)

        self.assertGreater(result["integrated_autocorrelation_time_samples"], 1.0)
        self.assertLess(result["effective_sample_size"], result["sample_count"])
        self.assertGreater(result["batch_count"], 2)
        self.assertIsNotNone(result["standard_error_autocorrelation"])
        self.assertEqual(result["ci95_status"], "bounded")

    def test_report_reads_component_logs_and_writes_summary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "run"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "inputs").mkdir()
            (run_dir / "prepare_summary.json").write_text(
                json.dumps({"production_steps": 50_000, "equilibration_steps": 5_000}),
                encoding="utf-8",
            )
            for seed in (12345, 23456, 34567):
                stem = f"gcmc_1p0bar_seed{seed}"
                (run_dir / "inputs" / f"{stem}.in").write_text(
                    "# pressure_bar metadata: 1.0\n"
                    "# CO2 mole_fraction 0.15 partial_pressure_bar 0.15\n"
                    "# N2 mole_fraction 0.85 partial_pressure_bar 0.85\n"
                    f"fix gcmc_co2 adsorbate_co2 gcmc 1 10 10 0 {seed} 298.15\n",
                    encoding="utf-8",
                )
                lines = ["Step Atoms v_nco2 v_nn2"]
                for step in range(0, 55_001, 100):
                    phase = (step // 1_000) % 4
                    co2 = 1.0 + phase * 0.1 + (seed - 12345) / 100_000
                    n2 = 2.0 + phase * 0.2 + (seed - 12345) / 100_000
                    lines.append(f"{step} 433 {co2} {n2}")
                (run_dir / "logs" / f"{stem}.log").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )

            report = analyze_module_b_convergence(run_dir)

            self.assertEqual(report["windows"]["early_start_step"], 5_000)
            self.assertEqual(report["windows"]["early_end_step_exclusive"], 25_000)
            self.assertEqual(report["windows"]["late_start_step"], 25_000)
            self.assertEqual(len(report["results"]), 3)
            self.assertEqual(report["results"][0]["seed_count"], 3)
            self.assertEqual(
                report["results"][2]["ci95_method"],
                "fieller_paired_seeds_plus_autocorrelation_batch_error",
            )
            self.assertTrue(Path(report["outputs"]["json"]).is_file())
            self.assertTrue(Path(report["outputs"]["csv"]).is_file())
            self.assertTrue(Path(report["outputs"]["markdown"]).is_file())


if __name__ == "__main__":
    unittest.main()