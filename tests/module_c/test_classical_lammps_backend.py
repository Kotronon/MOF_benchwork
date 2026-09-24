from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from modules.module_c_mlips.datasets import build_smoke_configurations
from modules.module_c_mlips.potential_backends.classical_lammps import (
    KCAL_PER_MOL_TO_EV,
    ClassicalLAMMPSBackend,
    parse_force_dump,
    parse_run_zero_energy,
    parse_run_zero_thermo,
    render_run_zero_input,
)


HAS_ASE = importlib.util.find_spec("ase") is not None
HAS_LAMMPS = shutil.which("lmp") is not None
CIF_PATH = "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif"
CO2_PATH = "CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def"
PSEUDO_ATOMS_PATH = "CRAFTED-2.0.0/FORCEFIELDS/UFF/pseudo_atoms.def"
MIXING_RULES_PATH = (
    "CRAFTED-2.0.0/FORCEFIELDS/UFF/force_field_mixing_rules.def"
)


class RunZeroInputTests(unittest.TestCase):
    def test_renderer_contains_energy_and_force_output(self) -> None:
        content = render_run_zero_input()

        self.assertIn("units real", content)
        self.assertIn("atom_style full", content)
        self.assertIn("bond_style zero", content)
        self.assertIn("read_data combined.data", content)
        self.assertIn("include forcefield.inc", content)
        self.assertIn(
            "thermo_style custom step atoms pe evdwl ecoul elong",
            content,
        )
        self.assertIn("id type fx fy fz", content)
        self.assertIn("run 0", content)

    def test_renderer_omits_bond_commands_without_bonds(self) -> None:
        content = render_run_zero_input(has_bonds=False)

        self.assertNotIn("bond_style", content)
        self.assertNotIn("bond_coeff", content)
        self.assertIn("run 0", content)


class RunZeroParserTests(unittest.TestCase):
    def test_energy_parser_returns_last_potential_energy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.lammps"
            log_path.write_text(
                "Step Atoms PotEng E_vdwl E_coul E_long\n"
                "0 109 -12.5 -10.0 -1.0 -1.5\n",
                encoding="utf-8",
            )

            energy = parse_run_zero_energy(log_path)

        self.assertEqual(energy, -12.5)

    def test_thermo_parser_returns_energy_decomposition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.lammps"
            log_path.write_text(
                "Step Atoms PotEng E_vdwl E_coul E_long\n"
                "0 109 -12.5 -10.0 -1.0 -1.5\n",
                encoding="utf-8",
            )

            thermo = parse_run_zero_thermo(log_path)

        self.assertEqual(
            thermo,
            {
                "potential_energy": -12.5,
                "vdw": -10.0,
                "coulomb_short_range": -1.0,
                "coulomb_long_range": -1.5,
            },
        )

    def test_energy_parser_rejects_log_without_potential_energy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.lammps"
            log_path.write_text("LAMMPS log without thermo data\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "No PotEng"):
                parse_run_zero_energy(log_path)

    def test_force_parser_uses_last_frame_and_sorts_by_atom_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_path = Path(tmpdir) / "forces.dump"
            dump_path.write_text(
                "ITEM: TIMESTEP\n0\n"
                "ITEM: NUMBER OF ATOMS\n2\n"
                "ITEM: ATOMS id type fx fy fz\n"
                "2 1 2.0 0.0 0.0\n"
                "1 1 1.0 0.0 0.0\n"
                "ITEM: TIMESTEP\n1\n"
                "ITEM: NUMBER OF ATOMS\n2\n"
                "ITEM: ATOMS id type fx fy fz\n"
                "2 1 4.0 5.0 6.0\n"
                "1 1 1.0 2.0 3.0\n",
                encoding="utf-8",
            )

            forces = parse_force_dump(dump_path)

        self.assertEqual(forces, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@unittest.skipUnless(HAS_ASE, "ASE is required for backend tests.")
class ClassicalLammpsBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = build_smoke_configurations(
            CIF_PATH,
            CO2_PATH,
            PSEUDO_ATOMS_PATH,
        )[0]

    def build_backend(
        self,
        *,
        command: str = "lmp",
        working_directory: Path | None = None,
    ) -> ClassicalLAMMPSBackend:
        return ClassicalLAMMPSBackend(
            lammps_command=command,
            pseudo_atoms_file=Path(PSEUDO_ATOMS_PATH),
            mixing_rules_file=Path(MIXING_RULES_PATH),
            working_directory=working_directory,
        )

    def test_backend_reports_missing_lammps_executable(self) -> None:
        backend = self.build_backend(command="missing-lammps-command")

        with self.assertRaisesRegex(RuntimeError, "was not found"):
            backend.evaluate(self.configuration)

    @unittest.skipUnless(HAS_LAMMPS, "LAMMPS executable is required.")
    def test_real_forcefield_run_zero_returns_energy_and_forces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            result = self.build_backend(
                working_directory=workdir,
            ).evaluate(self.configuration)
            run_directory = workdir / self.configuration.configuration_id

            self.assertTrue((run_directory / "combined.data").exists())
            self.assertTrue((run_directory / "forcefield.inc").exists())
            self.assertTrue((run_directory / "in.run_zero").exists())
            self.assertTrue((run_directory / "log.lammps").exists())
            self.assertTrue((run_directory / "forces.dump").exists())

        self.assertTrue(math.isfinite(result.energy_ev))
        self.assertEqual(len(result.forces_ev_per_angstrom), 109)
        self.assertTrue(
            all(
                math.isfinite(component)
                for force in result.forces_ev_per_angstrom
                for component in force
            )
        )
        self.assertGreaterEqual(result.runtime_seconds, 0.0)
        self.assertEqual(result.metadata["backend"], "uff_ddec_lammps")
        self.assertEqual(result.metadata["atom_count"], 109)
        self.assertEqual(
            set(result.metadata["energy_components_kcal_per_mol"]),
            {
                "potential_energy",
                "vdw",
                "coulomb_short_range",
                "coulomb_long_range",
            },
        )
        self.assertAlmostEqual(
            result.energy_ev,
            result.metadata["energy_kcal_per_mol"] * KCAL_PER_MOL_TO_EV,
        )


if __name__ == "__main__":
    unittest.main()
