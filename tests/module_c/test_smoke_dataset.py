from __future__ import annotations

import importlib.util
import unittest

from modules.module_c_mlips.datasets.smoke import build_smoke_configurations


HAS_ASE = importlib.util.find_spec("ase") is not None


@unittest.skipUnless(HAS_ASE, "ASE is required for smoke-dataset tests.")
class ModuleCSmokeDatasetTests(unittest.TestCase):
    def test_builds_deterministic_framework_co2_configurations(self) -> None:
        configurations = build_smoke_configurations(
            "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif",
            "CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def",
        )

        self.assertEqual(len(configurations), 3)
        self.assertEqual(
            [configuration.configuration_id for configuration in configurations],
            [
                "IRMOF-1_CO2_cell_center",
                "IRMOF-1_CO2_eighth_cell",
                "IRMOF-1_CO2_overlap_probe",
            ],
        )

        for configuration in configurations:
            self.assertEqual(len(configuration.atoms), 109)
            self.assertEqual(configuration.framework_indices, list(range(106)))
            self.assertEqual(configuration.adsorbate_indices, [106, 107, 108])
            self.assertTrue(
                set(configuration.framework_indices).isdisjoint(
                    configuration.adsorbate_indices
                )
            )
            self.assertEqual(configuration.source, "generated_smoke")
            self.assertEqual(
                configuration.atoms.get_chemical_symbols()[-3:],
                ["O", "C", "O"],
            )
            self.assertEqual(
                configuration.atoms.get_initial_charges()[-3:].tolist(),
                [-0.35, 0.7, -0.35],
            )
            self.assertEqual(
                [
                    (bond.atom1_index, bond.atom2_index, bond.forcefield_type)
                    for bond in configuration.bonds
                ],
                [
                    (106, 107, "RIGID_BOND"),
                    (107, 108, "RIGID_BOND"),
                ],
            )

    def test_generation_is_reproducible(self) -> None:
        arguments = (
            "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif",
            "CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def",
        )

        first = build_smoke_configurations(*arguments)
        second = build_smoke_configurations(*arguments)

        for first_configuration, second_configuration in zip(first, second):
            self.assertEqual(
                first_configuration.atoms.get_positions().tolist(),
                second_configuration.atoms.get_positions().tolist(),
            )

    def test_can_build_cutoff_valid_conventional_cell(self) -> None:
        configurations = build_smoke_configurations(
            "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif",
            "CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def",
            cell_representation="conventional",
            cutoff_A=12.8,
            minimum_image_policy="error",
        )

        self.assertEqual(len(configurations[0].atoms), 427)
        self.assertEqual(len(configurations[0].framework_indices), 424)
        self.assertEqual(configurations[0].adsorbate_indices, [424, 425, 426])


if __name__ == "__main__":
    unittest.main()
