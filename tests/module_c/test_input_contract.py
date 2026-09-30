from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from modules.module_c_mlips.input_contract import validate_model_input_contract


HAS_ASE = importlib.util.find_spec("ase") is not None


@unittest.skipUnless(HAS_ASE, "ASE is required for structure contract tests.")
class ModelInputContractTests(unittest.TestCase):
    def setUp(self) -> None:
        from ase import Atoms

        self.framework = Atoms("Zn2", cell=[16.991, 16.991, 16.991], pbc=True)
        self.adsorbate = Atoms("OsCoOs")
        self.adsorbate.new_array(
            "physical_species",
            np.asarray(["O", "C", "O"], dtype="U8"),
        )
        self.contract = {
            "framework_atom_count": 2,
            "cell_lengths_A": [16.991, 16.991, 16.991],
            "cell_tolerance_A": 0.001,
            "cell_angles_deg": [90.0, 90.0, 90.0],
            "cell_angle_tolerance_deg": 0.001,
            "adsorbate_physical_symbols": ["O", "C", "O"],
            "adsorbate_model_symbols": ["Os", "Co", "Os"],
        }

    def test_matching_contract_returns_manifest(self) -> None:
        result = validate_model_input_contract(
            self.framework,
            self.adsorbate,
            self.contract,
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["adsorbate_model_symbols"], ["Os", "Co", "Os"])

    def test_wrong_framework_is_rejected(self) -> None:
        contract = {**self.contract, "framework_atom_count": 276}

        with self.assertRaisesRegex(ValueError, "expected 276 framework atoms"):
            validate_model_input_contract(
                self.framework,
                self.adsorbate,
                contract,
            )

    def test_wrong_guest_labels_are_rejected(self) -> None:
        self.adsorbate.set_chemical_symbols(["O", "C", "O"])

        with self.assertRaisesRegex(ValueError, "adsorbate model symbols"):
            validate_model_input_contract(
                self.framework,
                self.adsorbate,
                self.contract,
            )


if __name__ == "__main__":
    unittest.main()
