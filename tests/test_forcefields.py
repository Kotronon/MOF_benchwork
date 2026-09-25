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

class ForcefieldsTests(unittest.TestCase):
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
        self.assertAlmostEqual(lj_parameters["N_com"].epsilon_K, 0.0)
        self.assertAlmostEqual(lj_parameters["N_com"].sigma_A, 1.0)

    def test_parse_raspa_mixing_rules_handles_inline_comments(self) -> None:
        lj_parameters, mixing_rule = parse_mixing_rules(
            "external/RASPA2/forcefield/ExampleMoleculeForceField/force_field_mixing_rules.def"
        )

        self.assertEqual(mixing_rule, "Lorentz-Berthelot")
        self.assertAlmostEqual(lj_parameters["CH4"].epsilon_K, 158.5)
        self.assertAlmostEqual(lj_parameters["CH4"].sigma_A, 3.72)
        self.assertAlmostEqual(lj_parameters["Ar"].sigma_A, 3.38)

    def test_load_forcefield_parameters_accepts_resolved_forcefield_config(self) -> None:
        run_plan = benchmark.build_run_plan(benchmark.load_benchmark_data("benchmark.json"))

        parameters = load_forcefield_parameters(run_plan["resources"]["forcefield"])

        self.assertEqual(parameters.mixing_rule, "Lorentz-Berthelot")
        self.assertIn("Zn_", parameters.lj_parameters)
        self.assertIn("C_co2", parameters.pseudo_atoms)

    def test_forcefield_resolver_uses_raspa2_adsorbate_fallback(self) -> None:
        forcefield = ForcefieldResolver().resolve(
            {"framework": "UFF", "adsorbate": "auto", "cross_interactions": "auto"},
            ["CH4"],
        )
        parameters = load_forcefield_parameters(forcefield)

        self.assertTrue(forcefield["adsorbates"]["CH4"].endswith("methane.def"))
        self.assertEqual(forcefield["adsorbate_sources"]["CH4"], "raspa2_example_molecule_forcefield")
        self.assertIn("CH4", parameters.pseudo_atoms)
        self.assertIn("CH4", parameters.lj_parameters)
        self.assertIn("Zn_", parameters.lj_parameters)
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
        self.assertNotIn("pair_modify shift yes", include_text)
        self.assertIn("kspace_style pppm 1.0e-4", include_text)
        self.assertIn("pair_coeff 1 5", include_text)
        self.assertAlmostEqual(zn_o_co2.sigma_A, (2.462 + 3.05) / 2.0)
        self.assertAlmostEqual(zn_o_co2.epsilon_kcal_mol, (62.35 * 79.0) ** 0.5 * 0.0019872041)

    def test_lammps_forcefield_include_can_enable_shifted_lj(self) -> None:
        run_plan = benchmark.build_run_plan(benchmark.load_benchmark_data("benchmark.json"))
        parameters = load_forcefield_parameters(run_plan["resources"]["forcefield"])
        forcefield = build_lammps_forcefield(
            framework_symbols=["Zn", "H", "C", "O"],
            adsorbate_atom_types=["O_co2", "C_co2"],
            parameters=parameters,
            pair_modify_shift=True,
        )

        include_text = render_lammps_forcefield_include(forcefield)

        self.assertIn("pair_style lj/cut/coul/long 12.8", include_text)
        self.assertIn("pair_modify shift yes", include_text)
        self.assertIn("kspace_style pppm 1.0e-4", include_text)
        self.assertLess(
            include_text.index("pair_modify shift yes"),
            include_text.index("kspace_style pppm 1.0e-4"),
        )

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


if __name__ == "__main__":
    unittest.main()
