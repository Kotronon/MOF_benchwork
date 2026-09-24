from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from modules.module_c_mlips.datasets import (
    load_lammps_dump_configuration,
    parse_lammps_forcefield_type_map,
)


HAS_ASE = importlib.util.find_spec("ase") is not None
HAS_CRAFTED = Path("CRAFTED-2.0.0").is_dir()
CO2_PATH = Path("CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def")
PSEUDO_ATOMS_PATH = Path(
    "CRAFTED-2.0.0/FORCEFIELDS/UFF/pseudo_atoms.def"
)


class LammpsForcefieldTypeMapTests(unittest.TestCase):
    def test_parses_type_map_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "forcefield.in"
            path.write_text(
                "# Atom type map\n"
                "# 1: Zn -> Zn_ (framework)\n"
                "# 5: O_co2 -> O_co2 (adsorbate)\n",
                encoding="utf-8",
            )

            result = parse_lammps_forcefield_type_map(path)

        self.assertEqual(result, {1: "Zn", 5: "O_co2"})


@unittest.skipUnless(
    HAS_ASE and HAS_CRAFTED,
    "ASE and CRAFTED are required for dump dataset tests.",
)
class LammpsDumpDatasetTests(unittest.TestCase):
    def test_loads_framework_and_co2_from_custom_dump(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            forcefield = root / "forcefield.in"
            forcefield.write_text(
                "# 1: Zn -> Zn_ (framework)\n"
                "# 2: O_co2 -> O_co2 (adsorbate)\n"
                "# 3: C_co2 -> C_co2 (adsorbate)\n",
                encoding="utf-8",
            )
            dump = root / "snapshot.lammpstrj"
            dump.write_text(
                "ITEM: TIMESTEP\n100\n"
                "ITEM: NUMBER OF ATOMS\n4\n"
                "ITEM: BOX BOUNDS pp pp pp\n"
                "2.0 12.0\n3.0 13.0\n4.0 14.0\n"
                "ITEM: ATOMS id mol type q x y z\n"
                "4 2 2 -0.35 7.0 8.0 7.84\n"
                "1 1 1 1.0 2.0 3.0 4.0\n"
                "3 2 3 0.7 7.0 8.0 9.0\n"
                "2 2 2 -0.35 7.0 8.0 10.16\n",
                encoding="utf-8",
            )

            configuration = load_lammps_dump_configuration(
                dump,
                forcefield,
                CO2_PATH,
                PSEUDO_ATOMS_PATH,
            )

        self.assertEqual(configuration.configuration_id, "IRMOF-1_CO2_dump_step_100")
        self.assertEqual(configuration.framework_indices, [0])
        self.assertEqual(configuration.adsorbate_indices, [1, 2, 3])
        self.assertEqual(configuration.atoms.cell.lengths().tolist(), [10.0, 10.0, 10.0])
        self.assertEqual(configuration.atoms.positions[0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(
            [(bond.atom1_index, bond.atom2_index) for bond in configuration.bonds],
            [(1, 2), (2, 3)],
        )


if __name__ == "__main__":
    unittest.main()
