from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from modules.module_c_mlips.datasets.widom import build_widom_configurations


HAS_ASE = importlib.util.find_spec("ase") is not None
CIF_PATH = "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif"
CO2_PATH = "CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def"
PSEUDO_ATOMS_PATH = "CRAFTED-2.0.0/FORCEFIELDS/UFF/pseudo_atoms.def"


@unittest.skipUnless(HAS_ASE, "ASE is required for Widom-dataset tests.")
class ModuleCWidomDatasetTests(unittest.TestCase):
    def build(self, **overrides):
        arguments = {
            "sample_count": 4,
            "seed": 1729,
            "minimum_distance_A": 1.5,
            "cell_representation": "conventional",
            "cutoff_A": 12.8,
            "minimum_image_policy": "error",
        }
        arguments.update(overrides)
        return build_widom_configurations(
            CIF_PATH,
            CO2_PATH,
            PSEUDO_ATOMS_PATH,
            **arguments,
        )

    def test_builds_reproducible_filtered_random_insertions(self) -> None:
        first = self.build()
        second = self.build()

        self.assertEqual(len(first), 4)
        self.assertEqual(
            [configuration.configuration_id for configuration in first],
            [f"IRMOF-1_CO2_widom_{index:05d}" for index in range(4)],
        )
        for left, right in zip(first, second, strict=True):
            np.testing.assert_allclose(left.atoms.positions, right.atoms.positions)
            self.assertGreaterEqual(
                left.metadata["minimum_host_guest_distance_A"],
                1.5,
            )
            self.assertFalse(left.metadata["thermodynamically_unbiased"])
            self.assertEqual(left.source, "generated_widom")
            self.assertEqual(len(left.framework_indices), 424)
            self.assertEqual(left.adsorbate_indices, [424, 425, 426])

    def test_different_seeds_change_insertions(self) -> None:
        first = self.build(seed=1729, sample_count=1)
        second = self.build(seed=1730, sample_count=1)

        self.assertFalse(
            np.allclose(first[0].atoms.positions[-3:], second[0].atoms.positions[-3:])
        )

    def test_rotation_matrix_is_orthonormal(self) -> None:
        configuration = self.build(sample_count=1)[0]
        rotation = np.asarray(configuration.metadata["rotation_matrix"])

        np.testing.assert_allclose(rotation @ rotation.T, np.identity(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_rejects_invalid_sample_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_count"):
            self.build(sample_count=0)


if __name__ == "__main__":
    unittest.main()
