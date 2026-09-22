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

class ResolversAndRegistryTests(unittest.TestCase):
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

    def test_zif7_module_a_case_is_automatically_marked_outside_validated_scope(self) -> None:
        rules = load_applicability_rules()
        run_plan = benchmark.build_run_plan(
            benchmark.load_benchmark_data("input_json_files/benchmark_zif7_co2_module_a_warning_test.json")
        )
        applicability = run_plan["benchmark"]["applicability"]

        self.assertIn("ZIF-7", rules["module_A"]["warning_materials"])
        self.assertEqual(run_plan["module"]["id"], "A")
        self.assertEqual(run_plan["material"]["material_id"], "ZIF-7")
        self.assertEqual(applicability["status"], "warning")
        self.assertEqual(applicability["matched_rule"], "module_A.warning_materials.ZIF-7")
        self.assertTrue(applicability["can_attempt_simulation"])
        self.assertTrue(applicability["requires_user_confirmation"])
        self.assertEqual(applicability["recommended_module"], "D")
        self.assertIn("gate-opening", applicability["reason"])
        self.assertEqual(applicability["geometry"]["pore_volume_cm3_g"], 0.0)
        self.assertTrue(any(check["name"] == "pore_volume_cm3_g" for check in applicability["checks"]))

    def test_applicability_uses_crafted_geometry_for_unknown_warning_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            geometry = root / "geometry.csv"
            rules = root / "rules.json"
            geometry.write_text(
                "FrameworkName,D_is,D_fs,D_isfs,ASA_m^2/cm^3,ASA_m^2/g,NASA_m^2/cm^3,"
                "NASA_m^2/g,Unitcell_volume,Density,AV_Volume_fraction,AV_cm^3/g,"
                "NAV_Volume_fraction,NAV_cm^3/g,n_pockets\n"
                "NARROW-MOF,5.0,2.0,5.0,0,0,0,0,1000,1.0,0.0,0.0,0,0,0\n",
                encoding="utf-8",
            )
            rules.write_text(
                json.dumps(
                    {
                        "module_A": {
                            "required": {
                                "engine": "LAMMPS",
                                "method": "GCMC",
                                "framework": "rigid",
                                "single_component": True,
                                "min_pore_volume_cm3_g": 0.01,
                            },
                            "adsorbate_diameter_policy": {
                                "source": "generated_from_lj_sigma",
                                "sigma_scale": 1.1,
                                "fallback_A": 3.5,
                            },
                            "geometry_source": str(geometry),
                            "warning_materials": {},
                        }
                    }
                ),
                encoding="utf-8",
            )

            applicability = assess_applicability(
                benchmark.normalize_config({"material": {"name": "NARROW-MOF"}}),
                {"id": "A", "name": "Module A", "slug": "module_A"},
                {
                    "material": {"material_id": "NARROW-MOF"},
                    "forcefield": ForcefieldResolver().resolve(
                        {"framework": "UFF", "adsorbate": "auto", "cross_interactions": "auto"},
                        ["CO2"],
                    ),
                },
                rules_path=rules,
            )

        self.assertEqual(applicability["status"], "warning")
        self.assertTrue(applicability["requires_user_confirmation"])
        self.assertEqual(applicability["geometry"]["pore_volume_cm3_g"], 0.0)
        self.assertGreater(applicability["adsorbate_properties"]["access_diameter_A"], 3.3)
        self.assertTrue(any("pore-limiting diameter" in reason for reason in applicability["reasons"]))

    def test_material_resolver_merges_safe_nist_aliases_without_editing_alias_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cif_dir = root / "crafted" / "CIF_FILES" / "DDEC"
            nist_dir = root / "isodb" / "Library" / "10.test"
            cif_dir.mkdir(parents=True)
            nist_dir.mkdir(parents=True)
            (cif_dir / "HKUST-1.cif").write_text("data_HKUST-1\n", encoding="utf-8")
            (root / "aliases.json").write_text("{}", encoding="utf-8")
            (nist_dir / "safe.json").write_text(
                json.dumps({"adsorbent": {"name": "CuBTC (HKUST-1)"}}),
                encoding="utf-8",
            )
            (nist_dir / "composite.json").write_text(
                json.dumps({"adsorbent": {"name": "CNT@HKUST-1"}}),
                encoding="utf-8",
            )

            resolver = MaterialResolver(
                crafted_root=root / "crafted",
                alias_file=root / "aliases.json",
                nist_root=root / "isodb",
            )
            match = resolver.resolve("CuBTC (HKUST-1)", charge_scheme="DDEC")

        self.assertEqual(match.material_id, "HKUST-1")
        self.assertEqual(match.matched_by, "alias")
        self.assertIn("CuBTC (HKUST-1)", match.aliases)
        self.assertNotIn("CNT@HKUST-1", match.aliases)

    def test_adsorbate_properties_are_generated_from_crafted_forcefield(self) -> None:
        forcefield = ForcefieldResolver().resolve(
            {"framework": "UFF", "adsorbate": "auto", "cross_interactions": "auto"},
            ["CO2"],
        )

        properties = infer_adsorbate_properties("CO2", forcefield)

        self.assertEqual(properties["source"], "crafted")
        self.assertEqual(properties["component"], "CO2")
        self.assertEqual(properties["atom_count"], 3)
        self.assertAlmostEqual(properties["molar_mass_g_mol"], 44.0095, places=4)
        self.assertAlmostEqual(properties["net_charge_e"], 0.0, places=6)
        self.assertGreater(properties["access_diameter_A"], 3.3)

    def test_generated_registries_include_material_and_adsorbate_metadata(self) -> None:
        materials = generate_crafted_material_registry()
        adsorbates = generate_adsorbate_registry()
        forcefields = generate_forcefield_registry()
        references = generate_reference_registry()

        self.assertIn("IRMOF-1", materials)
        self.assertEqual(materials["IRMOF-1"]["geometry"]["pore_volume_cm3_g"], 0.845545)
        self.assertIn("CO2", adsorbates)
        self.assertIn("UFF", adsorbates["CO2"])
        self.assertGreater(adsorbates["CO2"]["UFF"]["access_diameter_A"], 3.3)
        self.assertIn("CH4", adsorbates)
        self.assertIn("RASPA2_ExampleMoleculeForceField", adsorbates["CH4"])
        self.assertGreater(adsorbates["CH4"]["RASPA2_ExampleMoleculeForceField"]["access_diameter_A"], 3.7)
        self.assertIn("UFF", forcefields)
        self.assertIn("CO2", forcefields["UFF"]["adsorbates"])
        self.assertTrue(references["isotherm"])
        self.assertTrue(any(reference["material_id"] == "IRMOF-1" for reference in references["isotherm"]))

    def test_capability_database_and_registry_writer_include_resource_sections(self) -> None:
        database = generate_capability_database()

        self.assertEqual(database["schema_version"], 1)
        self.assertIn("materials", database)
        self.assertIn("adsorbates", database)
        self.assertIn("molecule_definitions", database)
        self.assertIn("forcefields", database)
        self.assertIn("references", database)
        self.assertGreater(database["summary"]["material_count"], 0)
        self.assertGreater(database["summary"]["adsorbate_count"], 0)
        self.assertGreater(database["summary"]["molecule_definition_count"], 0)
        self.assertGreater(database["summary"]["forcefield_count"], 0)
        self.assertGreater(database["summary"]["isotherm_reference_count"], 0)
        self.assertIn("PENTANE", database["molecule_definitions"])
        self.assertFalse(database["molecule_definitions"]["PENTANE"][0]["usable_by_current_lammps_pipeline"])

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_generated_registries(output_dir=tmpdir)

            self.assertTrue(Path(paths["capability_database"]).exists())
            self.assertTrue(Path(paths["molecule_definition_registry"]).exists())
            self.assertTrue(Path(paths["forcefield_registry"]).exists())
            self.assertTrue(Path(paths["reference_registry"]).exists())

    def test_forcefield_resolver_accepts_additional_raspa2_molecules(self) -> None:
        forcefield = ForcefieldResolver().resolve(
            {"framework": "UFF", "adsorbate": "auto", "cross_interactions": "auto"},
            ["PENTANE"],
        )

        self.assertTrue(forcefield["adsorbates"]["PENTANE"].endswith("pentane.def"))
        self.assertEqual(forcefield["adsorbate_sources"]["PENTANE"], "raspa2_example_molecule_forcefield")

    def test_run_confirmation_rejects_not_implemented_module_before_prepare(self) -> None:
        config = benchmark.normalize_config(
            {
                "material": {"name": "MOF-5"},
                "adsorbates": {
                    "components": ["CO2", "N2"],
                    "mixture": {"CO2": 0.2, "N2": 0.8},
                },
            }
        )
        run_plan = benchmark.build_run_plan(config)

        with self.assertRaisesRegex(RuntimeError, "only materializes and runs Module A"):
            benchmark._confirm_supported_capability(run_plan)

    def test_unknown_material_has_clear_error(self) -> None:
        config = benchmark.normalize_config({"material": {"name": "NOT_A_REAL_MOF"}})

        with self.assertRaisesRegex(LookupError, "Could not resolve material"):
            benchmark.resolve_benchmark(config)

    @unittest.skipUnless(HAS_NIST_ISODB, "NIST ISODB library is required for NIST reference tests.")
    def test_nist_isodb_candidates_rank_irmof1_co2_at_298k(self) -> None:
        candidates = find_nist_isotherm_candidates(
            "isodb-library",
            material_names=["IRMOF-1", "MOF-5"],
            components=["CO2"],
            temperature_K=298.15,
            pressures_bar=[0.1, 1.0, 10.0],
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["source"], "nist_isodb")
        self.assertEqual(candidates[0]["format"], "nist_json")
        self.assertEqual(candidates[0]["component"], "CO2")
        self.assertLessEqual(candidates[0]["temperature_delta_K"], 2.0)
        dois = [candidate["doi"] for candidate in candidates]
        self.assertEqual(len(dois), len(set(dois)))

    @unittest.skipUnless(HAS_NIST_ISODB, "NIST ISODB library is required for NIST reference tests.")
    def test_nist_outlier_filter_marks_candidates_far_from_crafted(self) -> None:
        run_plan = benchmark.build_run_plan(benchmark.load_benchmark_data("benchmark_tao2022_mof5_co2.json"))
        excluded = [
            reference
            for reference in run_plan["resources"]["references"]
            if reference.get("doi") == "10.1007/s00894-010-0720-x"
        ]

        self.assertTrue(excluded)
        self.assertTrue(excluded[0]["excluded"])
        self.assertFalse(excluded[0]["use_in_evaluation"])
        self.assertIn("loading_ratio_gt_3_vs_primary_reference", excluded[0]["exclusion_reasons"])

    @unittest.skipUnless(HAS_NIST_ISODB, "NIST ISODB library is required for NIST reference tests.")
    def test_reference_resolver_auto_keeps_crafted_primary_and_adds_nist_candidates(self) -> None:
        config = benchmark.load_benchmark_data("benchmark.json")
        config["benchmark"]["reference"]["source"] = "auto"

        run_plan = benchmark.build_run_plan(config)
        references = run_plan["resources"]["references"]
        selected = [reference for reference in references if reference["selected"]]

        self.assertTrue(any(reference["source"] == "crafted" for reference in references))
        self.assertTrue(any(reference["source"] == "nist_isodb" for reference in references))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["source"], "crafted")
        self.assertIn("excess", config["benchmark"]["reference"]["allowed_reference_basis"])
        self.assertTrue(config["benchmark"]["reference"]["allow_excess_reference_basis"])

    def test_load_nist_isotherm_converts_json_to_reference_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nist.json"
            path.write_text(
                json.dumps(
                    {
                        "DOI": "10.test/example",
                        "adsorbates": [{"name": "Carbon Dioxide"}],
                        "adsorbent": {"name": "IRMOF-1"},
                        "adsorptionUnits": "mmol/g",
                        "pressureUnits": "bar",
                        "temperature": 298,
                        "isotherm_data": [
                            {
                                "pressure": 1.0,
                                "species_data": [{"adsorption": 2.5, "composition": 1}],
                                "total_adsorption": 2.5,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            points = load_nist_isotherm(path, component="CO2")

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["source"], "nist_isodb")
        self.assertEqual(points[0]["doi"], "10.test/example")
        self.assertAlmostEqual(points[0]["pressure_Pa"], 100000.0)
        self.assertAlmostEqual(points[0]["loading_mol_per_kg"], 2.5)

    def test_load_nist_isotherm_accepts_null_total_adsorption(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nist.json"
            path.write_text(
                json.dumps(
                    {
                        "DOI": "10.test/example",
                        "adsorbates": [{"name": "Carbon Dioxide"}],
                        "adsorbent": {"name": "IRMOF-1"},
                        "adsorptionUnits": "mmol/g",
                        "pressureUnits": "bar",
                        "temperature": 298,
                        "isotherm_data": [
                            {
                                "pressure": 1.0,
                                "species_data": [{"adsorption": 2.5, "composition": 1}],
                                "total_adsorption": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            points = load_nist_isotherm(path, component="CO2")

        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0]["loading_mol_per_kg"], 2.5)


if __name__ == "__main__":
    unittest.main()
