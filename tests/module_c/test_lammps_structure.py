from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from converter.forcefield_to_lammps import (
    build_lammps_forcefield,
    load_forcefield_parameters,
)
from converter.molecule_to_lammps_template import parse_crafted_molecule_def
from modules.module_c_mlips.datasets import build_smoke_configurations
from modules.module_c_mlips.potential_backends.classical_lammps import (
    ClassicalLAMMPSBackend,
)
from modules.module_c_mlips.potential_backends.lammps_structure import (
    build_lammps_structure,
    write_lammps_data,
)


HAS_ASE = importlib.util.find_spec("ase") is not None
HAS_LAMMPS = shutil.which("lmp") is not None
CIF_PATH = "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif"
CO2_PATH = "CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def"
PSEUDO_ATOMS_PATH = "CRAFTED-2.0.0/FORCEFIELDS/UFF/pseudo_atoms.def"
MIXING_RULES_PATH = (
    "CRAFTED-2.0.0/FORCEFIELDS/UFF/force_field_mixing_rules.def"
)


@unittest.skipUnless(HAS_ASE, "ASE is required for LAMMPS-structure tests.")
class LammpsStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = build_smoke_configurations(
            CIF_PATH,
            CO2_PATH,
            PSEUDO_ATOMS_PATH,
        )[0]
        molecule = parse_crafted_molecule_def(CO2_PATH)
        parameters = load_forcefield_parameters(
            {
                "files": {
                    "pseudo_atoms": PSEUDO_ATOMS_PATH,
                    "mixing_rules": MIXING_RULES_PATH,
                }
            }
        )
        forcefield = build_lammps_forcefield(
            self.configuration.atoms.get_chemical_symbols()[:106],
            [atom.atom_type for atom in molecule.atoms],
            parameters,
        )
        self.type_ids = {
            atom_type.label: atom_type.type_id
            for atom_type in forcefield.atom_types
        }

    def build_structure(self):
        return build_lammps_structure(
            self.configuration.atoms,
            self.configuration.framework_indices,
            self.configuration.adsorbate_indices,
            self.type_ids,
            self.configuration.bonds,
        )

    def test_structure_preserves_atoms_cell_components_and_charges(self) -> None:
        configuration = self.configuration
        structure = self.build_structure()

        self.assertEqual(len(structure["atoms"]), 109)
        self.assertEqual(structure["framework_indices"], list(range(106)))
        self.assertEqual(structure["adsorbate_indices"], [106, 107, 108])
        self.assertEqual(
            structure["cell_vectors"],
            configuration.atoms.cell.array.tolist(),
        )
        self.assertEqual(structure["pbc"], [True, True, True])
        self.assertAlmostEqual(
            structure["atoms"][0]["charge"],
            float(configuration.atoms.get_initial_charges()[0]),
        )
        self.assertEqual(structure["atoms"][0]["component"], "framework")
        self.assertEqual(structure["atoms"][-1]["component"], "adsorbate")

    def test_framework_and_co2_types_receive_shared_distinct_type_ids(self) -> None:
        configuration = self.configuration
        structure = self.build_structure()

        self.assertIn("O", structure["type_ids"])
        self.assertIn("O_co2", structure["type_ids"])
        self.assertIn("C_co2", structure["type_ids"])
        self.assertNotEqual(
            structure["type_ids"]["O"],
            structure["type_ids"]["O_co2"],
        )
        self.assertEqual(
            [atom["forcefield_type"] for atom in structure["atoms"][-3:]],
            ["O_co2", "C_co2", "O_co2"],
        )
        self.assertEqual(
            structure["atoms"][-3]["lammps_type_id"],
            structure["atoms"][-1]["lammps_type_id"],
        )

    def test_missing_forcefield_types_are_rejected(self) -> None:
        atoms = self.configuration.atoms.copy()
        del atoms.arrays["forcefield_type"]

        with self.assertRaisesRegex(ValueError, "forcefield_type"):
            build_lammps_structure(
                atoms,
                self.configuration.framework_indices,
                self.configuration.adsorbate_indices,
                self.type_ids,
            )

    def test_incomplete_component_indices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cover every atom"):
            build_lammps_structure(
                self.configuration.atoms,
                self.configuration.framework_indices,
                self.configuration.adsorbate_indices[:-1],
                self.type_ids,
            )

    def test_structure_contains_co2_bonds(self) -> None:
        structure = self.build_structure()

        self.assertEqual(
            structure["bonds"],
            [
                {
                    "atom1_index": 106,
                    "atom2_index": 107,
                    "forcefield_type": "RIGID_BOND",
                },
                {
                    "atom1_index": 107,
                    "atom2_index": 108,
                    "forcefield_type": "RIGID_BOND",
                },
            ],
        )

    def test_writer_creates_combined_lammps_data_file(self) -> None:
        structure = self.build_structure()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = write_lammps_data(
                structure,
                Path(tmpdir) / "mof5_co2.data",
            )
            content = output.read_text(encoding="utf-8")

        self.assertIn("109 atoms", content)
        self.assertIn("2 bonds", content)
        self.assertIn("6 atom types", content)
        self.assertIn("1 bond types", content)
        self.assertIn("Masses", content)
        self.assertIn("Atoms # full", content)
        self.assertIn("Bonds", content)
        self.assertIn("# O_co2", content)
        self.assertIn("# C_co2", content)
        self.assertIn("107 2", content)
        self.assertIn("-0.35000000", content)
        self.assertIn("0.70000000", content)
        self.assertIn("1 1 107 108", content)
        self.assertIn("2 1 108 109", content)

    @unittest.skipUnless(HAS_LAMMPS, "LAMMPS executable is required.")
    def test_lammps_reads_combined_data_and_runs_zero_steps(self) -> None:
        structure = self.build_structure()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_lammps_data(structure, root / "combined.data")
            input_path = root / "in.read_combined"
            input_path.write_text(
                "units real\n"
                "atom_style full\n"
                "boundary p p p\n"
                "bond_style zero\n"
                "read_data combined.data\n"
                "pair_style zero 12.8\n"
                "pair_coeff * *\n"
                "bond_coeff *\n"
                "run 0\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                ["lmp", "-in", str(input_path)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("109 atoms", result.stdout)
        self.assertIn("2 bonds", result.stdout)


class ClassicalLammpsBackendContractTests(unittest.TestCase):
    def test_backend_has_stable_name_and_explicit_unimplemented_error(self) -> None:
        backend = ClassicalLAMMPSBackend(
            lammps_command="lmp",
            forcefield_file=Path("forcefield.inc"),
            working_directory=Path("work"),
        )

        self.assertEqual(backend.name, "uff_ddec_lammps")
        with self.assertRaisesRegex(NotImplementedError, "run-0"):
            backend.evaluate(object())


if __name__ == "__main__":
    unittest.main()
