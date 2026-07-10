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
from modules.module_a_adsorption import prepare as prepare_module_a


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

        prepare_plan = prepare_module_a(run_plan)

        self.assertEqual(prepare_plan["status"], "prepared_plan")
        self.assertEqual(prepare_plan["side_effects"], "none")
        self.assertIn("framework_data", prepare_plan["planned_files"])
        self.assertEqual(len(prepare_plan["planned_files"]["input_scripts"]), len(run_plan["conditions"]["pressures_bar"]))
        json.dumps(prepare_plan)

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
            )

            text = output_path.read_text(encoding="utf-8")
        self.assertIn("6 atom types", text)
        self.assertIn("5 15.99940000 # O_co2", text)
        self.assertIn("6 12.01070000 # C_co2", text)

    def test_molecule_template_can_use_global_co2_atom_type_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "CO2.template"

            build_molecule_template(
                "CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def",
                output_path,
                atom_type_ids={"O_co2": 5, "C_co2": 6},
            )

            text = output_path.read_text(encoding="utf-8")
        self.assertIn("1 5", text)
        self.assertIn("2 6", text)
        self.assertIn("3 5", text)

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

if __name__ == "__main__":
    unittest.main()
