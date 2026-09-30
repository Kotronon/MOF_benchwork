from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from analysis.mlip_widom_energy_diagnostics import (
    _summarize_rows,
    read_widom_structure_trace,
)


class WidomEnergyDiagnosticTests(unittest.TestCase):
    def test_reads_full_binary_structure_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "log_widom.bin"
            numbers = (6, 8)
            positions = (0.0, 0.0, 0.0, 1.2, 0.0, 0.0)
            cell = (10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0)
            with path.open("wb") as handle:
                handle.write(struct.pack("iddi", 7, -0.15, -10.0, 2))
                handle.write(struct.pack("2i", *numbers))
                handle.write(struct.pack("6d", *positions))
                handle.write(struct.pack("9d", *cell))

            records = read_widom_structure_trace(path)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].trial, 7)
            self.assertAlmostEqual(records[0].interaction_energy_ev, -0.15)
            np.testing.assert_array_equal(records[0].atomic_numbers, numbers)
            self.assertEqual(records[0].positions_A.shape, (2, 3))
            self.assertEqual(records[0].cell_A.shape, (3, 3))

    def test_summary_quantifies_d3_weight_amplification(self) -> None:
        rows = [
            {
                "d3_correction_ev": -0.1,
                "minimum_host_guest_distance_A": 2.0,
                "boltzmann_weight_no_d3": 1.0,
                "boltzmann_weight_with_d3": 10.0,
            },
            {
                "d3_correction_ev": -0.2,
                "minimum_host_guest_distance_A": 3.0,
                "boltzmann_weight_no_d3": 1.0,
                "boltzmann_weight_with_d3": 20.0,
            },
        ]

        summary = _summarize_rows(rows, attempts=4, label="test")

        self.assertAlmostEqual(summary["d3_correction_ev"]["mean"], -0.15)
        self.assertAlmostEqual(
            summary["mean_boltzmann_weight_ratio_d3_to_no_d3"], 15.0
        )
        self.assertAlmostEqual(
            summary["with_d3"]["average_over_all_attempts"], 7.5
        )


if __name__ == "__main__":
    unittest.main()
