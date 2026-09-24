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

class PlanningTests(unittest.TestCase):
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
        self.assertEqual(config["simulation"]["seeds"], [12345])
        self.assertEqual(config["convergence"]["minimum_replicates"], 3)
        self.assertTrue(config["output"]["save_dumps"])
        self.assertEqual(config["output"]["dump_every_steps"], 1000)
        self.assertTrue(config["output"]["save_restarts"])
        self.assertEqual(config["output"]["restart_every_steps"], 50000)

    def test_normalize_config_rejects_duplicate_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            benchmark.normalize_config(
                {"material": {"name": "MOF-5"}, "simulation": {"seeds": [12345, 12345]}}
            )

    def test_prepare_builds_one_run_per_pressure_and_seed(self) -> None:
        config = benchmark.load_benchmark_data("benchmark.json")
        config["conditions"]["pressures_bar"] = [0.1, 1.0]
        config["simulation"]["seeds"] = [101, 202, 303]

        prepare_plan = benchmark.prepare_benchmark(benchmark.build_run_plan(config))
        scripts = prepare_plan["planned_files"]["input_scripts"]

        self.assertEqual(len(scripts), 6)
        self.assertEqual({script["seed"] for script in scripts}, {101, 202, 303})
        self.assertTrue(all(f"seed{script['seed']}" in script["path"] for script in scripts))
        self.assertTrue(all("/restarts/" in script["restart"] for script in scripts))
        self.assertTrue(all(script["restart"].endswith(".restart.*") for script in scripts))
        self.assertEqual(prepare_plan["parameters"]["restart_every_steps"], 50000)

    def test_build_variant_configs_applies_pppm_overrides(self) -> None:
        config = benchmark.load_benchmark_data("benchmark_mof5_co2_pppm_variants_smoke_test.json")

        variant_configs = build_variant_configs(config)

        self.assertEqual(len(variant_configs), 3)
        self.assertEqual({variant["benchmark"]["module"] for variant in variant_configs}, {"C"})
        self.assertEqual(
            [variant["benchmark"]["variant"]["name"] for variant in variant_configs],
            ["UFF-pppm_1e-4", "UFF-pppm_1e-5", "UFF-pppm_1e-6"],
        )
        self.assertEqual(
            [float(variant["simulation"]["kspace_accuracy"]) for variant in variant_configs],
            [1e-4, 1e-5, 1e-6],
        )
        self.assertEqual(
            [variant["benchmark"]["variant"]["overrides"]["kspace_accuracy"] for variant in variant_configs],
            [1e-4, 1e-5, 1e-6],
        )
        self.assertTrue(
            all(
                variant["output"]["directory"].endswith(
                    f"runs/smoke_MOF5_CO2_pppm_variants/{variant['benchmark']['variant']['name']}"
                )
                for variant in variant_configs
            )
        )

    def test_compare_variant_results_reads_isotherm_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "material": {"name": "MOF-5"},
                "benchmark": {"task": "potential_benchmark"},
                "output": {
                    "directory": str(Path(tmpdir) / "module_C"),
                    "run_id": "variant_smoke",
                },
            }
            variant_root = get_directory(config)
            for name, loading in [("pppm_1e-4", 1.0), ("pppm_1e-5", 1.2)]:
                variant_dir = variant_root / name / "work"
                variant_dir.mkdir(parents=True)
                benchmark.save_benchmark_data(
                    variant_dir / "isotherm_summary.json",
                    {
                        "status": "completed",
                        "results": [
                            {
                                "pressure_bar": 1.0,
                                "replicate_count": 1,
                                "seeds": [12345],
                                "converged": True,
                                "mean_wall_time_seconds": 2.0,
                                "total_wall_time_seconds": 2.0,
                                "summary": {
                                    "mean_adsorbates": loading,
                                    "standard_error_adsorbates": None,
                                },
                            }
                        ],
                    },
                )

            output_path = variant_root / "variant_comparison.json"
            comparison = compare_variant_results(config, output_path)

            self.assertEqual(comparison["variant_count"], 2)
            self.assertEqual(comparison["point_count"], 2)
            self.assertTrue(output_path.exists())
            self.assertEqual(
                [(row["variant"], row["mean_adsorbates"]) for row in comparison["rows"]],
                [("pppm_1e-4", 1.0), ("pppm_1e-5", 1.2)],
            )

    def test_prepare_can_disable_dump_files(self) -> None:
        config = benchmark.load_benchmark_data("benchmark.json")
        config["conditions"]["pressures_bar"] = [1.0]
        config["output"]["save_dumps"] = False
        config["output"]["dump_every_steps"] = 100000

        prepare_plan = benchmark.prepare_benchmark(benchmark.build_run_plan(config))
        scripts = prepare_plan["planned_files"]["input_scripts"]

        self.assertIsNone(scripts[0]["dump"])
        self.assertEqual(prepare_plan["parameters"]["dump_every_steps"], 100000)

    def test_aggregate_replicates_reports_seed_uncertainty_and_convergence(self) -> None:
        replicate_results = [
            {
                "pressure_bar": 1.0,
                "replicate_index": index,
                "seed": seed,
                "wall_time_seconds": float(index),
                "summary": {
                    "sample_count": 10,
                    "inserted": True,
                    "max_atoms": 112,
                    "max_adsorbates": 2.0,
                    "mean_adsorbates": value,
                },
            }
            for index, (seed, value) in enumerate(
                [(101, 0.99), (202, 1.00), (303, 1.01), (404, 1.00), (505, 1.00)],
                start=1,
            )
        ]

        aggregated = _aggregate_replicates(
            replicate_results,
            {"minimum_replicates": 3, "relative_ci95_target": 0.05},
        )
        report = _convergence_report(
            aggregated,
            {"minimum_replicates": 3, "relative_ci95_target": 0.05},
        )

        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["replicate_count"], 5)
        self.assertAlmostEqual(aggregated[0]["summary"]["mean_adsorbates"], 1.0)
        self.assertAlmostEqual(aggregated[0]["mean_wall_time_seconds"], 3.0)
        self.assertAlmostEqual(aggregated[0]["total_wall_time_seconds"], 15.0)
        self.assertIsNotNone(aggregated[0]["summary"]["standard_error_adsorbates"])
        self.assertTrue(aggregated[0]["converged"])
        self.assertTrue(report["all_pressure_points_converged"])

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
            gcmc_input = Path(result["files"]["gcmc_inputs"][0])
            summary = Path(result["files"]["summary"])

            self.assertTrue(framework_data.exists())
            self.assertTrue(molecule_template.exists())
            self.assertTrue(forcefield_include.exists())
            self.assertTrue(run0_input.exists())
            self.assertTrue(gcmc_test_input.exists())
            self.assertTrue(summary.exists())
            self.assertEqual(result["parameters"]["framework_atom_count"], 424)
            self.assertEqual(result["parameters"]["adsorbate_atoms_per_molecule"], 3)
            self.assertIn("424 atoms", framework_data.read_text(encoding="utf-8"))
            run0_content = run0_input.read_text(encoding="utf-8")
            self.assertIn("extra/special/per/atom 2", run0_content)
            self.assertIn("bond_style zero", run0_content)
            self.assertIn("special_bonds lj/coul 0.0 0.0 0.0", run0_content)
            self.assertIn(
                "thermo_style custom step atoms pe evdwl ecoul elong",
                run0_content,
            )
            self.assertIn("extra/bond/per/atom 2 extra/special/per/atom 2", gcmc_test_input.read_text(encoding="utf-8"))
            self.assertIn("fix gcmc_co2 adsorbate gcmc", gcmc_test_input.read_text(encoding="utf-8"))
            self.assertIn("restart 50000", gcmc_input.read_text(encoding="utf-8"))
            self.assertTrue(result["files"]["restart_files"][0].endswith(".restart.*"))
            self.assertIn("molecule_templates", result["files"]["gcmc_runs"][0])
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
                "chemical_potential_kcal_mol": 0.0,
                "displacement_A": 1.0,
                "run_steps": 1,
                "gcmc_every_steps": 1,
                "exchange_attempts": 1,
                "move_attempts": 1,
                "seed": 12345,
                "extra_bond_per_atom": 2,
                "extra_special_per_atom": 2,
                "restart_file": "restarts/gcmc_001bar.restart.*",
                "restart_every_steps": 50,
            }
        )

        self.assertIn("read_data IRMOF-1.data extra/bond/per/atom 2 extra/special/per/atom 2", script)
        self.assertIn("group framework type 1 2 3 4", script)
        self.assertIn("group adsorbate type 5 6", script)
        self.assertIn(
            "fix gcmc_co2 adsorbate gcmc 1 1 1 0 12345 298.15 0 1 "
            "mol co2 group adsorbate full_energy pressure 0.00986923 fugacity_coeff 1",
            script,
        )
        self.assertIn("restart 50 restarts/gcmc_001bar.restart.*", script)
        self.assertIn("run 1", script)

    def test_gcmc_input_builder_can_resume_from_lammps_restart(self) -> None:
        script = benchmark.gcmc_input_builder(
            {
                "framework_data": "IRMOF-1.data",
                "molecule_templates": {"CO2": "CO2.template"},
                "forcefield_include": "forcefield.in",
                "framework_atom_types": [1, 2, 3, 4],
                "adsorbate_atom_types": [5, 6],
                "component": "CO2",
                "temperature_K": 298.15,
                "pressure_bar": 1.0,
                "run_steps": 1000,
                "gcmc_every_steps": 1,
                "exchange_attempts": 10,
                "move_attempts": 10,
                "seed": 104729,
                "restart_file": "restarts/gcmc_1bar_seed104729.restart.*",
                "restart_every_steps": 50,
                "read_restart_file": "restarts/gcmc_1bar_seed104729.restart.500",
            }
        )

        self.assertIn("read_restart restarts/gcmc_1bar_seed104729.restart.500", script)
        self.assertNotIn("read_data IRMOF-1.data", script)
        self.assertIn("fix gcmc_co2 adsorbate gcmc", script)
        self.assertIn("restart 50 restarts/gcmc_1bar_seed104729.restart.*", script)
        self.assertIn("run 1000 upto", script)

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

    def test_cli_overrides_kspace_settings_in_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = benchmark.load_benchmark_data("benchmark.json")
            config["output"]["directory"] = str(Path(tmpdir) / "planned_output")
            config_path = Path(tmpdir) / "benchmark.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "benchmark.py",
                    str(config_path),
                    "--dry-run",
                    "--kspace-style",
                    "ewald",
                    "--kspace-accuracy",
                    "1e-6",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            run_plan = json.loads(result.stdout)

        self.assertEqual(run_plan["simulation"]["kspace_style"], "ewald")
        self.assertEqual(float(run_plan["simulation"]["kspace_accuracy"]), 1e-6)

    def test_normalize_config_accepts_pair_modify_shift(self) -> None:
        config = benchmark.normalize_config(
            {"material": {"name": "MOF-5"}, "simulation": {"pair_modify_shift": True}}
        )

        self.assertTrue(config["simulation"]["pair_modify_shift"])

    def test_normalize_config_rejects_invalid_pair_modify_shift(self) -> None:
        with self.assertRaisesRegex(TypeError, "simulation.pair_modify_shift"):
            benchmark.normalize_config(
                {"material": {"name": "MOF-5"}, "simulation": {"pair_modify_shift": "yes"}}
            )

    def test_normalize_config_rejects_invalid_kspace_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "simulation.kspace_style"):
            benchmark.normalize_config({"material": {"name": "MOF-5"}, "simulation": {"kspace_style": "ppm"}})
        with self.assertRaisesRegex(ValueError, "simulation.kspace_accuracy"):
            benchmark.normalize_config({"material": {"name": "MOF-5"}, "simulation": {"kspace_accuracy": 0}})

    def test_missing_adsorbate_definition_has_clear_error(self) -> None:
        config = benchmark.normalize_config(
            {
                "material": {"name": "MOF-5"},
                "adsorbates": {"components": ["XE"]},
            }
        )

        with self.assertRaisesRegex(LookupError, "Adsorbate definition"):
            benchmark.resolve_benchmark(config)


if __name__ == "__main__":
    unittest.main()
