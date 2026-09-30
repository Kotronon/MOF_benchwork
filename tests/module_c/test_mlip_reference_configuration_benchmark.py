from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.mlip_reference_configuration_benchmark import (
    benchmark_reference_configurations,
)


HAS_ASE = importlib.util.find_spec("ase") is not None

if HAS_ASE:
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.io import write


@unittest.skipUnless(HAS_ASE, "ASE is required for reference benchmark tests.")
class ReferenceConfigurationBenchmarkTests(unittest.TestCase):
    def test_writes_metrics_and_supercell_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            structures = root / "top.extxyz"
            frames = []
            for rank, offset in enumerate((0.0, 0.2), start=1):
                atoms = Atoms(
                    "ZnHCOO",
                    positions=[
                        [0.0, 0.0, 0.0],
                        [1.5, 0.0, 0.0],
                        [3.0 + offset, 0.0, 0.0],
                        [4.1 + offset, 0.0, 0.0],
                        [1.9 + offset, 0.0, 0.0],
                    ],
                    cell=[10.0, 10.0, 10.0],
                    pbc=True,
                )
                atoms.info.update(
                    {
                        "weight_rank": rank,
                        "seed": 12345,
                        "trial": rank,
                        "minimum_host_guest_distance_A": 0.4 + offset,
                        "interaction_energy_no_d3_ev": -0.05,
                        "interaction_energy_with_d3_ev": -0.20,
                    }
                )
                frames.append(atoms)
            write(structures, frames, format="extxyz")

            report = benchmark_reference_configurations(
                structures,
                root / "unused-model.pth",
                root / "result",
                framework_atom_count=2,
                calculator=PairCalculator(),
                supercell=(2, 1, 1),
                supercell_count=1,
            )

            self.assertEqual(report["configuration_count"], 2)
            self.assertIn(report["preferred_candidate"], {"mace_mp_no_d3", "mace_mp_d3"})
            self.assertEqual(
                report["supercell_invariance"]["configuration_count"], 1
            )
            for path in report["outputs"].values():
                self.assertTrue(Path(path).is_file())


if HAS_ASE:
    class PairCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces"),
            system_changes=all_changes,
        ) -> None:
            super().calculate(atoms, properties, system_changes)
            positions = atoms.get_positions()
            energy = 0.0
            for first in range(len(atoms)):
                for second in range(first + 1, len(atoms)):
                    distance = np.linalg.norm(positions[first] - positions[second])
                    energy += 0.01 / max(distance, 0.1)
            self.results = {
                "energy": energy,
                "forces": np.zeros((len(atoms), 3)),
            }


if __name__ == "__main__":
    unittest.main()
