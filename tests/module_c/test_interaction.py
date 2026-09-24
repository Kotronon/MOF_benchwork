from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from modules.module_c_mlips.datasets import build_smoke_configurations
from modules.module_c_mlips.interaction import (
    build_adsorbate_configuration,
    build_framework_configuration,
    evaluate_interaction,
)
from modules.module_c_mlips.models import InteractionBond, InteractionConfiguration
from modules.module_c_mlips.potential_backends.base import (
    PotentialBackend,
    PotentialResult,
)
from modules.module_c_mlips.potential_backends.classical_lammps import (
    ClassicalLAMMPSBackend,
)


HAS_ASE = importlib.util.find_spec("ase") is not None
HAS_LAMMPS = shutil.which("lmp") is not None
CIF_PATH = "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif"
CO2_PATH = "CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def"
PSEUDO_ATOMS_PATH = "CRAFTED-2.0.0/FORCEFIELDS/UFF/pseudo_atoms.def"
MIXING_RULES_PATH = (
    "CRAFTED-2.0.0/FORCEFIELDS/UFF/force_field_mixing_rules.def"
)


class DeterministicBackend(PotentialBackend):
    @property
    def name(self) -> str:
        return "deterministic"

    def evaluate(
        self,
        configuration: InteractionConfiguration,
    ) -> PotentialResult:
        if configuration.configuration_id.endswith("_framework"):
            energy = 3.0
            force = [1.0, 2.0, 3.0]
        elif configuration.configuration_id.endswith("_adsorbate"):
            energy = 2.0
            force = [4.0, 5.0, 6.0]
        else:
            energy = 10.0
            force = [10.0, 20.0, 30.0]
        return PotentialResult(
            energy_ev=energy,
            forces_ev_per_angstrom=[force.copy() for _ in configuration.atoms],
            runtime_seconds=0.1,
            metadata={"atom_count": len(configuration.atoms)},
        )


@unittest.skipUnless(HAS_ASE, "ASE is required for interaction tests.")
class InteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = build_smoke_configurations(
            CIF_PATH,
            CO2_PATH,
            PSEUDO_ATOMS_PATH,
        )[0]

    def test_component_configurations_preserve_cell_and_remap_bonds(self) -> None:
        framework = build_framework_configuration(self.configuration)
        adsorbate = build_adsorbate_configuration(self.configuration)

        self.assertEqual(len(framework.atoms), 106)
        self.assertEqual(framework.framework_indices, list(range(106)))
        self.assertEqual(framework.adsorbate_indices, [])
        self.assertEqual(framework.bonds, [])
        self.assertEqual(len(adsorbate.atoms), 3)
        self.assertEqual(adsorbate.framework_indices, [])
        self.assertEqual(adsorbate.adsorbate_indices, [0, 1, 2])
        self.assertEqual(
            [
                (bond.atom1_index, bond.atom2_index, bond.forcefield_type)
                for bond in adsorbate.bonds
            ],
            [(0, 1, "RIGID_BOND"), (1, 2, "RIGID_BOND")],
        )
        self.assertEqual(
            framework.atoms.cell.array.tolist(),
            self.configuration.atoms.cell.array.tolist(),
        )
        self.assertEqual(
            adsorbate.atoms.cell.array.tolist(),
            self.configuration.atoms.cell.array.tolist(),
        )

    def test_cross_component_bond_is_rejected(self) -> None:
        configuration = InteractionConfiguration(
            configuration_id="cross_bond",
            material=self.configuration.material,
            adsorbate=self.configuration.adsorbate,
            atoms=self.configuration.atoms.copy(),
            framework_indices=self.configuration.framework_indices.copy(),
            adsorbate_indices=self.configuration.adsorbate_indices.copy(),
            bonds=[InteractionBond(0, 106, "CROSS")],
        )

        with self.assertRaisesRegex(ValueError, "between framework and adsorbate"):
            build_framework_configuration(configuration)

    def test_fake_backend_subtracts_energy_and_forces(self) -> None:
        result = evaluate_interaction(
            self.configuration,
            DeterministicBackend(),
        )

        self.assertEqual(result.interaction_energy_ev, 5.0)
        self.assertAlmostEqual(result.runtime_seconds, 0.3)
        self.assertEqual(
            result.interaction_forces_ev_per_angstrom[0],
            [9.0, 18.0, 27.0],
        )
        self.assertEqual(
            result.interaction_forces_ev_per_angstrom[-1],
            [6.0, 15.0, 24.0],
        )
        json.dumps(result.to_dict())

    @unittest.skipUnless(HAS_LAMMPS, "LAMMPS executable is required.")
    def test_real_lammps_interaction_uses_three_separate_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            backend = ClassicalLAMMPSBackend(
                lammps_command="lmp",
                pseudo_atoms_file=Path(PSEUDO_ATOMS_PATH),
                mixing_rules_file=Path(MIXING_RULES_PATH),
                working_directory=root,
            )

            result = evaluate_interaction(self.configuration, backend)

            expected_directories = [
                root / self.configuration.configuration_id,
                root / f"{self.configuration.configuration_id}_framework",
                root / f"{self.configuration.configuration_id}_adsorbate",
            ]
            for directory in expected_directories:
                self.assertTrue((directory / "combined.data").exists())
                self.assertTrue((directory / "log.lammps").exists())
                self.assertTrue((directory / "forces.dump").exists())

        self.assertTrue(math.isfinite(result.interaction_energy_ev))
        self.assertEqual(len(result.interaction_forces_ev_per_angstrom), 109)
        self.assertEqual(result.combined.metadata["atom_count"], 109)
        self.assertEqual(result.framework.metadata["atom_count"], 106)
        self.assertEqual(result.adsorbate.metadata["atom_count"], 3)


if __name__ == "__main__":
    unittest.main()
