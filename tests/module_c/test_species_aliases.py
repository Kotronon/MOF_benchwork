from __future__ import annotations

import importlib.util
import unittest

from modules.module_c_mlips.species_aliases import (
    apply_adsorbate_species_aliases,
    validate_model_species_mapping,
)


HAS_ASE = importlib.util.find_spec("ase") is not None

if HAS_ASE:
    from ase import Atoms


@unittest.skipUnless(HAS_ASE, "ASE is required for species alias tests.")
class SpeciesAliasTests(unittest.TestCase):
    def test_goeminne_aliases_change_only_guest_labels_and_preserve_masses(self) -> None:
        adsorbate = Atoms(
            "OCO",
            positions=[[0, 0, 1.16], [0, 0, 0], [0, 0, -1.16]],
        )
        physical_masses = adsorbate.get_masses().copy()

        aliased, manifest = apply_adsorbate_species_aliases(
            adsorbate,
            {"C": "Co", "O": "Os"},
        )

        self.assertEqual(adsorbate.get_chemical_symbols(), ["O", "C", "O"])
        self.assertEqual(aliased.get_chemical_symbols(), ["Os", "Co", "Os"])
        self.assertEqual(
            aliased.arrays["physical_species"].tolist(),
            ["O", "C", "O"],
        )
        self.assertTrue((aliased.get_masses() == physical_masses).all())
        self.assertEqual(
            manifest["replacement_counts"],
            {"C->Co": 1, "O->Os": 2},
        )

    def test_explicit_species_mapping_must_cover_aliases(self) -> None:
        with self.assertRaisesRegex(ValueError, "Co, Os"):
            validate_model_species_mapping(
                ["Zn", "C", "Co", "Os"],
                {"Zn": "Zn", "C": "C"},
            )


if __name__ == "__main__":
    unittest.main()
