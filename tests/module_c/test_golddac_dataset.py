from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from modules.module_c_mlips.datasets.golddac import (
    load_golddac_configurations,
)


HAS_ASE = importlib.util.find_spec("ase") is not None


@unittest.skipUnless(HAS_ASE, "ASE is required for GoldDAC dataset tests.")
class GoldDACDatasetTests(unittest.TestCase):
    def test_loads_dft_reference_and_component_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.xyz"
            self._write_frame(path)

            configurations = load_golddac_configurations(
                path,
                adsorbates=["CO2"],
            )

        configuration = configurations[0]
        self.assertEqual(configuration.material, "Test-MOF")
        self.assertEqual(configuration.adsorbate, "CO2")
        self.assertEqual(configuration.region, "equilibrium")
        self.assertEqual(configuration.framework_indices, [0, 1])
        self.assertEqual(configuration.adsorbate_indices, [2, 3, 4])
        self.assertAlmostEqual(configuration.reference_interaction_energy_ev, -0.5)
        self.assertEqual(
            configuration.metadata["reference_force_mode"],
            "combined_total",
        )
        self.assertFalse(configuration.metadata["training_data"])

    def test_rejects_train_split_to_prevent_benchmark_leakage(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved for model development"):
            load_golddac_configurations("unused", split="train")

    def test_rejects_inconsistent_interaction_energy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.xyz"
            self._write_frame(path, interaction_energy=-0.4)

            with self.assertRaisesRegex(ValueError, "consistency check"):
                load_golddac_configurations(path)

    @staticmethod
    def _write_frame(path: Path, interaction_energy: float = -0.5) -> None:
        import numpy as np
        from ase import Atoms
        from ase.io import write

        atoms = Atoms(
            "ZnOCOO",
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.2, 0.0, 0.0],
                [6.4, 0.0, 0.0],
            ],
            cell=[12.0, 12.0, 12.0],
            pbc=True,
            tags=[0, 0, 1, 1, 1],
        )
        atoms.info.update(
            {
                "name": "Test-MOF_CO2_E_A_2.0",
                "group": "Test-MOF",
                "metal": "Zn",
                "DFT_E_total": -10.5,
                "DFT_E_mof": -8.0,
                "DFT_E_gas": -2.0,
                "DFT_E_int": interaction_energy,
            }
        )
        atoms.arrays["REF_forces"] = np.full((len(atoms), 3), 0.25)
        write(path, atoms, format="extxyz")


if __name__ == "__main__":
    unittest.main()
