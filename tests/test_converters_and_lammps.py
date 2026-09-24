from __future__ import annotations

import csv
import json
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import benchmark
from pipeline.applicability import assess_applicability, load_applicability_rules
from analysis.block_convergence import analyze_block_convergence
from analysis.cell_comparison import compare_cell_runs
from analysis.kspace_sensitivity import compare_kspace_runs
from analysis.reproducibility_manifest import create_reproducibility_manifest
from converter.cif_to_lammps_data import (
    cell_perpendicular_widths,
    convert_cif_to_lammps_data,
    load_framework_structure,
    parse_cif_atom_site_charges,
)
from converter.forcefield_to_lammps import (
    build_lammps_forcefield,
    load_forcefield_parameters,
    parse_mixing_rules,
    parse_pseudo_atoms,
    plan_forcefield_to_lammps,
    render_lammps_forcefield_include,
    write_lammps_forcefield_include,
)
from converter.molecule_to_lammps_template import build_molecule_template, parse_crafted_molecule_def

from parsers.lammps_log_parser import summarize_adsorption, get_log, log_to_json, parse_lammps_log
from pipeline.comparision import compare_variant_results, get_directory
from pipeline.evaluate import evaluate_all_references, evaluate_isotherm, framework_mass_from_lammps_data, sim_results_to_csv
from pipeline.adsorbate_registry import infer_adsorbate_properties
from pipeline.nist_isodb_parser import find_nist_isotherm_candidates, load_nist_isotherm
from pipeline.registry_generation import (
    generate_adsorbate_registry,
    generate_capability_database,
    generate_crafted_material_registry,
    generate_forcefield_registry,
    generate_reference_registry,
    write_generated_registries,
)
from pipeline.runners import (
    _aggregate_replicates,
    _convergence_report,
    _existing_pressure_point_result,
    _lammps_log_completed,
    _latest_restart_file,
    _restart_step,
)
from pipeline.variants import build_variant_configs
from resolvers.forcefield_resolver import ForcefieldResolver
from resolvers.material_resolver import MaterialResolver


HAS_ASE = importlib.util.find_spec("ase") is not None
HAS_LAMMPS = shutil.which("lmp") is not None
HAS_NIST_ISODB = Path("isodb-library/Library").exists()

class ConvertersAndLammpsTests(unittest.TestCase):
    def test_parse_co2_crafted_molecule_definition(self) -> None:
        molecule = parse_crafted_molecule_def("CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def")

        self.assertEqual(molecule.name, "CO2")
        self.assertEqual(len(molecule.atoms), 3)
        self.assertEqual(len(molecule.bonds), 2)
        self.assertAlmostEqual(molecule.critical_temperature_K, 304.1282)
        self.assertTrue(molecule.rigid)

    def test_parse_n2_crafted_molecule_definition_includes_center_atom(self) -> None:
        molecule = parse_crafted_molecule_def("CRAFTED-2.0.0/FORCEFIELDS/UFF/N2.def")

        self.assertEqual(molecule.name, "N2")
        self.assertEqual(len(molecule.atoms), 3)
        self.assertIn("N_com", [atom.atom_type for atom in molecule.atoms])
        self.assertEqual(len(molecule.bonds), 2)

    def test_parse_raspa_single_site_methane_definition(self) -> None:
        molecule = parse_crafted_molecule_def("external/RASPA2/molecules/ExampleDefinitions/methane.def")

        self.assertEqual(molecule.name, "methane")
        self.assertEqual(len(molecule.atoms), 1)
        self.assertEqual(molecule.atoms[0].atom_type, "CH4")
        self.assertEqual((molecule.atoms[0].x, molecule.atoms[0].y, molecule.atoms[0].z), (0.0, 0.0, 0.0))
        self.assertEqual(len(molecule.bonds), 0)

    def test_build_molecule_template_writes_minimal_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "co2.template"

            result = build_molecule_template("CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def", output_path)

            text = output_path.read_text(encoding="utf-8")
        self.assertEqual(result, output_path)
        self.assertIn("3 atoms", text)
        self.assertIn("2 bonds", text)
        self.assertIn("Coords", text)
        self.assertIn("Bonds", text)

    def test_parse_irmof1_cif_charges(self) -> None:
        records = parse_cif_atom_site_charges("CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif")

        self.assertEqual(len(records), 106)
        self.assertGreater(max(abs(record.charge) for record in records), 0.1)
        self.assertEqual(records[0].label, "Zn")

    @unittest.skipUnless(HAS_ASE, "ASE is required for framework geometry loading.")
    def test_load_framework_structure_reads_geometry_and_explicit_charges(self) -> None:
        structure = load_framework_structure("CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif")

        self.assertEqual(structure.atom_count, 106)
        self.assertEqual(structure.symbols[0], "Zn")
        self.assertAlmostEqual(structure.charges[0], 1.103828)
        self.assertGreater(max(abs(charge) for charge in structure.charges), 0.1)

    @unittest.skipUnless(HAS_ASE, "ASE is required for framework cell construction.")
    def test_conventional_irmof1_cell_has_424_atoms_and_preserves_charges(self) -> None:
        source = load_framework_structure("CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif")
        structure = load_framework_structure(
            "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif",
            cell_representation="conventional",
            unit_cells=[1, 1, 1],
            cutoff_A=12.8,
        )

        self.assertEqual(structure.atom_count, 424)
        self.assertTrue(all(abs(length - 25.832) < 1e-5 for length in structure.cell_lengths))
        self.assertTrue(all(abs(angle - 90.0) < 1e-8 for angle in structure.cell_angles))
        self.assertAlmostEqual(sum(structure.charges), 4 * sum(source.charges), places=8)
        self.assertGreaterEqual(min(cell_perpendicular_widths(structure.cell_vectors)), 25.6)

    @unittest.skipUnless(HAS_ASE, "ASE is required for framework cell construction.")
    def test_crafted_primitive_2x2x2_cell_has_848_atoms(self) -> None:
        structure = load_framework_structure(
            "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif",
            cell_representation="primitive",
            unit_cells=[2, 2, 2],
            cutoff_A=12.8,
        )

        self.assertEqual(structure.atom_count, 848)
        self.assertGreaterEqual(min(cell_perpendicular_widths(structure.cell_vectors)), 25.6)

    @unittest.skipUnless(HAS_ASE, "ASE is required for framework cell validation.")
    def test_primitive_1x1x1_cell_is_rejected_for_12p8_angstrom_cutoff(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than twice the real-space cutoff"):
            load_framework_structure(
                "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif",
                cell_representation="primitive",
                unit_cells=[1, 1, 1],
                cutoff_A=12.8,
            )

    @unittest.skipUnless(HAS_ASE, "ASE is required for framework data conversion.")
    def test_framework_converter_writes_conventional_irmof1_data_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "IRMOF-1-conventional.data"
            convert_cif_to_lammps_data(
                "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif",
                output_path,
                cell_representation="conventional",
                unit_cells=[1, 1, 1],
                cutoff_A=12.8,
            )
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("424 atoms", text)
        self.assertIn(
            "0.0000000000 0.0000000000 0.0000000000 xy xz yz",
            text,
        )

    @unittest.skipUnless(HAS_ASE, "ASE is required for framework data conversion.")
    def test_framework_converter_writes_triclinic_irmof1_data_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "IRMOF-1.data"

            result = convert_cif_to_lammps_data("CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif", output_path)

            text = output_path.read_text(encoding="utf-8")
        self.assertEqual(result, output_path)
        self.assertIn("106 atoms", text)
        self.assertIn("xy xz yz", text)
        self.assertIn("Atoms # full", text)
        self.assertIn("1 1 1 1.10382800", text)

    @unittest.skipUnless(HAS_ASE and HAS_LAMMPS, "ASE and lmp are required for the LAMMPS read_data smoke test.")
    def test_lammps_can_read_generated_data_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            data_file = tmpdir_path / "IRMOF-1.data"
            input_file = tmpdir_path / "read_data.in"

            convert_cif_to_lammps_data(
                "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif",
                data_file,
            )

            input_file.write_text(
                f"""units real
                atom_style full
                boundary p p p
                read_data {data_file}
                run 0
                """,
                encoding="utf-8",
            )

            result = subprocess.run(
                ["lmp", "-in", str(input_file), "-log", str(tmpdir_path / "log.lammps")],
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("106 atoms", result.stdout)

    @unittest.skipUnless(HAS_ASE and HAS_LAMMPS, "ASE and lmp are required for the conventional-cell smoke test.")
    def test_lammps_can_read_conventional_irmof1_data_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_file = root / "IRMOF-1-conventional.data"
            input_file = root / "read_conventional.in"
            convert_cif_to_lammps_data(
                "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif",
                data_file,
                cell_representation="conventional",
                unit_cells=[1, 1, 1],
                cutoff_A=12.8,
            )
            input_file.write_text(
                f"units real\natom_style full\nboundary p p p\nread_data {data_file}\nrun 0\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                ["lmp", "-in", str(input_file), "-log", str(root / "log.lammps")],
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("424 atoms", result.stdout)

    @unittest.skipUnless(HAS_ASE and HAS_LAMMPS, "ASE and lmp are required for the LAMMPS forcefield smoke test.")
    def test_lammps_run_zero_reads_data_pair_coeffs_and_co2_template(self) -> None:
        run_plan = benchmark.build_run_plan(benchmark.load_benchmark_data("benchmark.json"))
        parameters = load_forcefield_parameters(run_plan["resources"]["forcefield"])
        forcefield = build_lammps_forcefield(
            framework_symbols=["Zn", "H", "C", "O"],
            adsorbate_atom_types=["O_co2", "C_co2"],
            parameters=parameters,
        )
        extra_masses = {
            atom_type.label: atom_type.mass
            for atom_type in forcefield.atom_types
            if atom_type.source == "adsorbate" and atom_type.mass is not None
        }
        type_ids = {atom_type.label: atom_type.type_id for atom_type in forcefield.atom_types}
        charges = {
            atom_type.label: atom_type.charge
            for atom_type in forcefield.atom_types
            if atom_type.source == "adsorbate" and atom_type.charge is not None
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            data_file = tmpdir_path / "IRMOF-1.data"
            template_file = tmpdir_path / "CO2.template"
            forcefield_file = tmpdir_path / "forcefield.in"
            input_file = tmpdir_path / "run_zero.in"

            convert_cif_to_lammps_data(
                run_plan["material"]["cif_path"],
                data_file,
                extra_atom_types=extra_masses,
                extra_bond_types=1,
            )
            build_molecule_template(
                run_plan["resources"]["forcefield"]["adsorbates"]["CO2"],
                template_file,
                atom_type_ids=type_ids,
                atom_charges=charges,
            )
            write_lammps_forcefield_include(forcefield, forcefield_file)
            input_file.write_text(
                f"""units real
                atom_style full
                boundary p p p
                read_data {data_file} extra/special/per/atom 2
                molecule co2 {template_file}
                include {forcefield_file}
                run 0
                """,
                encoding="utf-8",
            )

            result = subprocess.run(
                ["lmp", "-in", str(input_file), "-log", str(tmpdir_path / "log.lammps")],
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("Loop time", result.stdout)

    @unittest.skipUnless(HAS_ASE and HAS_LAMMPS, "ASE and lmp are required for the GCMC smoke test.")
    def test_lammps_can_run_materialized_gcmc_smoke_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = benchmark.load_benchmark_data("benchmark.json")
            config["output"]["directory"] = str(Path(tmpdir) / "module_a")
            run_plan = benchmark.build_run_plan(config)
            prepare_plan = benchmark.prepare_benchmark(run_plan)
            result = benchmark.materialize_benchmark(prepare_plan)
            gcmc_input = Path(result["files"]["gcmc_test_input"])
            log_file = Path(result["working_directory"]) / "logs" / "gcmc_test.log"

            completed = subprocess.run(
                ["lmp", "-in", str(gcmc_input), "-log", str(log_file)],
                capture_output=True,
                text=True,
            )
            parsed_data = parse_lammps_log(completed.stdout)

        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertIn("Loop time", completed.stdout)
        summary = summarize_adsorption(
            parsed_data["rows"],
            framework_atoms=106,
            adsorbate_atoms_per_molecule=3,
        )
        self.assertTrue(summary["inserted"])
        self.assertGreaterEqual(summary["max_adsorbates"], 1)


if __name__ == "__main__":
    unittest.main()
