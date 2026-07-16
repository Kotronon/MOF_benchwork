from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import benchmark
from converter.cif_to_lammps_data import (
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


HAS_ASE = importlib.util.find_spec("ase") is not None
HAS_LAMMPS = shutil.which("lmp") is not None


class BenchmarkPipelineTests(unittest.TestCase):
    def test_load_benchmark_data_reads_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.json"
            path.write_text(json.dumps({"material": {"name": "MOF-5"}}), encoding="utf-8")

            data = benchmark.load_benchmark_data(path)

        self.assertEqual(data["material"]["name"], "MOF-5")

    def test_normalize_config_applies_v1_defaults(self) -> None:
        config = benchmark.normalize_config({"material": {"name": "MOF-5"}})

        self.assertEqual(config["adsorbates"]["components"], ["CO2"])
        self.assertEqual(config["benchmark"]["task"], "auto")
        self.assertIsNone(config["output"]["directory"])
        self.assertEqual(config["simulation"]["method"], "GCMC")

    def test_mof5_alias_resolves_to_irmof1(self) -> None:
        config = benchmark.normalize_config({"material": {"name": "MOF-5", "charge_scheme": "DDEC"}})

        resolved = benchmark.resolve_benchmark(config)

        self.assertEqual(resolved["material"]["material_id"], "IRMOF-1")
        self.assertTrue(resolved["material"]["cif_path"].endswith("CIF_FILES/DDEC/IRMOF-1.cif"))

    def test_selects_module_a_for_single_component_gcmc(self) -> None:
        config = benchmark.normalize_config({"material": {"name": "MOF-5"}})

        module = benchmark.select_module(config)

        self.assertEqual(module["id"], "A")

    def test_selects_module_b_for_mixture(self) -> None:
        config = benchmark.normalize_config(
            {
                "material": {"name": "MOF-5"},
                "adsorbates": {"components": ["CO2", "N2"], "mixture": {"CO2": 0.2, "N2": 0.8}},
            }
        )

        module = benchmark.select_module(config)

        self.assertEqual(module["id"], "B")

    def test_unknown_material_has_clear_error(self) -> None:
        config = benchmark.normalize_config({"material": {"name": "NOT_A_REAL_MOF"}})

        with self.assertRaisesRegex(LookupError, "Could not resolve material"):
            benchmark.resolve_benchmark(config)

    def test_dry_run_cli_outputs_run_plan_without_running_lammps(self) -> None:
        result = subprocess.run(
            [sys.executable, "benchmark.py", "benchmark.json", "--dry-run"],
            check=True,
            capture_output=True,
            text=True,
        )
        run_plan = json.loads(result.stdout)

        self.assertEqual(run_plan["status"], "planned")
        self.assertEqual(run_plan["module"]["id"], "A")
        self.assertEqual(run_plan["material"]["material_id"], "IRMOF-1")
        self.assertIn("forcefield", run_plan["resources"])

    def test_gitignore_excludes_external_data_and_outputs(self) -> None:
        gitignore = Path(".gitignore").read_text(encoding="utf-8")

        for pattern in [".DS_Store", "CRAFTED-2.0.0/", "CRAFTED-2.0.0.tar.xz", "outputs/", "reports/"]:
            self.assertIn(pattern, gitignore)

    def test_module_a_prepare_returns_json_serializable_plan(self) -> None:
        run_plan = benchmark.build_run_plan(benchmark.load_benchmark_data("benchmark.json"))

        prepare_plan = benchmark.prepare_benchmark(run_plan)

        self.assertEqual(prepare_plan["status"], "prepared_plan")
        self.assertEqual(prepare_plan["side_effects"], "none")
        self.assertIn("framework_data", prepare_plan["planned_files"])
        self.assertIn("forcefield_include", prepare_plan["planned_files"])
        self.assertIn("run0_input", prepare_plan["planned_files"])
        self.assertEqual(len(prepare_plan["planned_files"]["input_scripts"]), len(run_plan["conditions"]["pressures_bar"]))
        json.dumps(prepare_plan)

    @unittest.skipUnless(HAS_ASE, "ASE is required for materializing framework data.")
    def test_materialize_benchmark_writes_prepare_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = benchmark.load_benchmark_data("benchmark.json")
            config["output"]["directory"] = str(Path(tmpdir) / "module_a")
            run_plan = benchmark.build_run_plan(config)
            prepare_plan = benchmark.prepare_benchmark(run_plan)

            result = benchmark.materialize_benchmark(prepare_plan)

            framework_data = Path(result["files"]["framework_data"])
            molecule_template = Path(result["files"]["molecule_templates"]["CO2"])
            forcefield_include = Path(result["files"]["forcefield_include"])
            run0_input = Path(result["files"]["run0_input"])
            gcmc_test_input = Path(result["files"]["gcmc_test_input"])
            summary = Path(result["files"]["summary"])

            self.assertTrue(framework_data.exists())
            self.assertTrue(molecule_template.exists())
            self.assertTrue(forcefield_include.exists())
            self.assertTrue(run0_input.exists())
            self.assertTrue(gcmc_test_input.exists())
            self.assertTrue(summary.exists())
            self.assertIn("extra/special/per/atom 2", run0_input.read_text(encoding="utf-8"))
            self.assertIn("extra/bond/per/atom 2 extra/special/per/atom 2", gcmc_test_input.read_text(encoding="utf-8"))
            self.assertIn("fix gcmc_co2 adsorbate gcmc", gcmc_test_input.read_text(encoding="utf-8"))
            self.assertIn("pair_coeff", forcefield_include.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "materialized")
        json.dumps(result)

    def test_gcmc_input_builder_uses_valid_lammps_gcmc_shape(self) -> None:
        script = benchmark.gcmc_input_builder(
            {
                "framework_data": "IRMOF-1.data",
                "molecule_templates": {"CO2": "CO2.template"},
                "forcefield_include": "forcefield.in",
                "framework_atom_types": [1, 2, 3, 4],
                "adsorbate_atom_types": [5, 6],
                "component": "CO2",
                "temperature_K": 298.15,
                "pressure_bar": 0.01,
                "chemical_potential_kcal_mol": -10.0,
                "displacement_A": 1.0,
                "run_steps": 1,
                "gcmc_every_steps": 1,
                "exchange_attempts": 1,
                "move_attempts": 1,
                "seed": 12345,
                "extra_bond_per_atom": 2,
                "extra_special_per_atom": 2,
            }
        )

        self.assertIn("read_data IRMOF-1.data extra/bond/per/atom 2 extra/special/per/atom 2", script)
        self.assertIn("group framework type 1 2 3 4", script)
        self.assertIn("group adsorbate type 5 6", script)
        self.assertIn("fix gcmc_co2 adsorbate gcmc 1 1 1 0 12345 298.15 -10 1 mol co2 group adsorbate full_energy", script)
        self.assertIn("run 1", script)

    def test_dry_run_does_not_create_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = benchmark.load_benchmark_data("benchmark.json")
            output_dir = Path(tmpdir) / "planned_output"
            config["output"]["directory"] = str(output_dir)
            config_path = Path(tmpdir) / "benchmark.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            subprocess.run(
                [sys.executable, "benchmark.py", str(config_path), "--dry-run"],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertFalse(output_dir.exists())

    def test_missing_adsorbate_definition_has_clear_error(self) -> None:
        config = benchmark.normalize_config(
            {
                "material": {"name": "MOF-5"},
                "adsorbates": {"components": ["AR"]},
            }
        )

        with self.assertRaisesRegex(LookupError, "Adsorbate definition"):
            benchmark.resolve_benchmark(config)

    def test_parse_pseudo_atoms_reads_adsorbate_masses_and_charges(self) -> None:
        pseudo_atoms = parse_pseudo_atoms("CRAFTED-2.0.0/FORCEFIELDS/UFF/pseudo_atoms.def")

        self.assertEqual(set(pseudo_atoms), {"C_co2", "O_co2", "N_n2", "N_com"})
        self.assertEqual(pseudo_atoms["C_co2"].element, "C")
        self.assertAlmostEqual(pseudo_atoms["C_co2"].mass, 12.0107)
        self.assertAlmostEqual(pseudo_atoms["C_co2"].charge, 0.7)
        self.assertAlmostEqual(pseudo_atoms["O_co2"].charge, -0.35)
        self.assertAlmostEqual(pseudo_atoms["N_com"].mass, 0.0)
        self.assertAlmostEqual(pseudo_atoms["N_com"].charge, 0.964)

    def test_parse_mixing_rules_reads_uff_lj_parameters(self) -> None:
        lj_parameters, mixing_rule = parse_mixing_rules(
            "CRAFTED-2.0.0/FORCEFIELDS/UFF/force_field_mixing_rules.def"
        )

        self.assertEqual(mixing_rule, "Lorentz-Berthelot")
        self.assertAlmostEqual(lj_parameters["Zn_"].epsilon_K, 62.35)
        self.assertAlmostEqual(lj_parameters["Zn_"].sigma_A, 2.462)
        self.assertAlmostEqual(lj_parameters["C_co2"].epsilon_K, 27.0)
        self.assertAlmostEqual(lj_parameters["O_co2"].sigma_A, 3.05)
        self.assertNotIn("N_com", lj_parameters)

    def test_load_forcefield_parameters_accepts_resolved_forcefield_config(self) -> None:
        run_plan = benchmark.build_run_plan(benchmark.load_benchmark_data("benchmark.json"))

        parameters = load_forcefield_parameters(run_plan["resources"]["forcefield"])

        self.assertEqual(parameters.mixing_rule, "Lorentz-Berthelot")
        self.assertIn("Zn_", parameters.lj_parameters)
        self.assertIn("C_co2", parameters.pseudo_atoms)
        json.dumps(parameters.to_dict())

    def test_plan_forcefield_to_lammps_is_side_effect_free_and_serializable(self) -> None:
        run_plan = benchmark.build_run_plan(benchmark.load_benchmark_data("benchmark.json"))
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "forcefield.lammps"

            plan = plan_forcefield_to_lammps(
                run_plan["resources"]["forcefield"],
                run_plan["adsorbates"]["components"],
                output_path,
            )

            self.assertFalse(output_path.exists())
        self.assertEqual(plan["converter"], "forcefield_to_lammps")
        self.assertEqual(plan["status"], "planned")
        self.assertEqual(plan["components"], ["CO2"])
        json.dumps(plan)

    def test_build_lammps_forcefield_assigns_framework_and_co2_types(self) -> None:
        run_plan = benchmark.build_run_plan(benchmark.load_benchmark_data("benchmark.json"))
        parameters = load_forcefield_parameters(run_plan["resources"]["forcefield"])
        molecule = parse_crafted_molecule_def(run_plan["resources"]["forcefield"]["adsorbates"]["CO2"])

        forcefield = build_lammps_forcefield(
            framework_symbols=["Zn", "H", "C", "O"],
            adsorbate_atom_types=[atom.atom_type for atom in molecule.atoms],
            parameters=parameters,
        )

        self.assertEqual([atom_type.label for atom_type in forcefield.atom_types], ["Zn", "H", "C", "O", "O_co2", "C_co2"])
        self.assertEqual([atom_type.type_id for atom_type in forcefield.atom_types], [1, 2, 3, 4, 5, 6])
        self.assertEqual(len(forcefield.pair_coefficients), 21)
        self.assertEqual(forcefield.atom_types[4].mass, 15.9994)
        self.assertEqual(forcefield.atom_types[5].charge, 0.7)
        json.dumps(forcefield.to_dict())

    def test_lammps_forcefield_include_renders_mixed_pair_coefficients(self) -> None:
        run_plan = benchmark.build_run_plan(benchmark.load_benchmark_data("benchmark.json"))
        parameters = load_forcefield_parameters(run_plan["resources"]["forcefield"])
        forcefield = build_lammps_forcefield(
            framework_symbols=["Zn", "H", "C", "O"],
            adsorbate_atom_types=["O_co2", "C_co2"],
            parameters=parameters,
        )

        include_text = render_lammps_forcefield_include(forcefield)
        zn_o_co2 = next(
            coefficient
            for coefficient in forcefield.pair_coefficients
            if coefficient.label_i == "Zn" and coefficient.label_j == "O_co2"
        )

        self.assertIn("pair_style lj/cut/coul/long 12.8", include_text)
        self.assertIn("kspace_style pppm 1.0e-4", include_text)
        self.assertIn("pair_coeff 1 5", include_text)
        self.assertAlmostEqual(zn_o_co2.sigma_A, (2.462 + 3.05) / 2.0)
        self.assertAlmostEqual(zn_o_co2.epsilon_kcal_mol, (62.35 * 79.0) ** 0.5 * 0.0019872041)

    def test_write_lammps_forcefield_include_writes_expected_file(self) -> None:
        run_plan = benchmark.build_run_plan(benchmark.load_benchmark_data("benchmark.json"))
        parameters = load_forcefield_parameters(run_plan["resources"]["forcefield"])
        forcefield = build_lammps_forcefield(
            framework_symbols=["Zn", "H", "C", "O"],
            adsorbate_atom_types=["O_co2", "C_co2"],
            parameters=parameters,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "forcefield.in"

            result = write_lammps_forcefield_include(forcefield, output_path)

            text = output_path.read_text(encoding="utf-8")
        self.assertEqual(result, output_path)
        self.assertIn("# 5: O_co2 -> O_co2 (adsorbate)", text)
        self.assertIn("pair_coeff 5 6", text)

    @unittest.skipUnless(HAS_ASE, "ASE is required for framework data conversion.")
    def test_framework_data_can_include_extra_co2_atom_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "IRMOF-1.data"

            convert_cif_to_lammps_data(
                "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif",
                output_path,
                extra_atom_types={"O_co2": 15.9994, "C_co2": 12.0107},
                extra_bond_types=1,
            )

            text = output_path.read_text(encoding="utf-8")
        self.assertIn("6 atom types", text)
        self.assertIn("0 bonds", text)
        self.assertIn("1 bond types", text)
        self.assertIn("5 15.99940000 # O_co2", text)
        self.assertIn("6 12.01070000 # C_co2", text)

    def test_molecule_template_can_use_global_co2_atom_type_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "CO2.template"

            build_molecule_template(
                "CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def",
                output_path,
                atom_type_ids={"O_co2": 5, "C_co2": 6},
                atom_charges={"O_co2": -0.35, "C_co2": 0.7},
            )

            text = output_path.read_text(encoding="utf-8")
        self.assertIn("1 5", text)
        self.assertIn("2 6", text)
        self.assertIn("3 5", text)
        self.assertIn("Charges", text)
        self.assertIn("1 -0.35000000", text)
        self.assertIn("2 0.70000000", text)
        self.assertIn("3 -0.35000000", text)

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
        
    def test_lammps_log_parser_summarizes_adsorption_from_thermo_rows(self) -> None:
        log_data = """LAMMPS
Step Atoms Temp PotEng TotEng Press
0 106 0.0 10.0 10.0 1.0
1 109 2.0 9.0 9.5 1.1
2 112 3.0 8.0 8.5 1.2
Loop time of 0.1 on 1 procs
"""
        parsed_data = parse_lammps_log(log_data)
        summary = summarize_adsorption(parsed_data["rows"], framework_atoms=106, adsorbate_atoms_per_molecule=3)

        self.assertEqual(len(parsed_data["rows"]), 3)
        self.assertEqual(parsed_data["rows"][1]["Atoms"], 109)
        self.assertEqual(summary["sample_count"], 3)
        self.assertTrue(summary["inserted"])
        self.assertEqual(summary["max_adsorbates"], 2)
        self.assertAlmostEqual(summary["mean_adsorbates"], 1.0)
        self.assertEqual(summary["max_atoms"], 112)

    def test_log_to_json_writes_adsorption_summary(self) -> None:
        log_data = """LAMMPS
Step Atoms Temp PotEng TotEng Press
0 106 0.0 10.0 10.0 1.0
1 109 2.0 9.0 9.5 1.1
2 112 3.0 8.0 8.5 1.2
Loop time of 0.1 on 1 procs
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "gcmc_test.log"
            output_path = Path(tmpdir) / "summary" / "gcmc_test_summary.json"
            log_path.write_text(log_data, encoding="utf-8")

            summary = log_to_json(
                log_path,
                output_path,
                framework_atoms=106,
                adsorbate_atoms_per_molecule=3,
            )

            written_summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["max_adsorbates"], 2)
        self.assertEqual(written_summary["max_adsorbates"], 2)
        self.assertTrue(written_summary["inserted"])

if __name__ == "__main__":
    unittest.main()
