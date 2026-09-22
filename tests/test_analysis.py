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

class AnalysisTests(unittest.TestCase):
    def test_block_convergence_analyzes_completed_production_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = root / "run"
            (run / "data").mkdir(parents=True)
            (run / "inputs").mkdir()
            (run / "logs").mkdir()
            (run / "data" / "IRMOF-1.data").write_text(
                "LAMMPS data\n\n106 atoms\n",
                encoding="utf-8",
            )
            (run / "prepare_summary.json").write_text(
                json.dumps(
                    {
                        "prepare_plan": {
                            "parameters": {
                                "production_steps": 500,
                                "equilibration_steps": 500,
                                "adsorbate_atoms_per_molecule": 3,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run / "inputs" / "gcmc_1bar.in").write_text(
                "# pressure_bar metadata: 1.0\n"
                "fix gcmc_co2 adsorbate gcmc 1 10 10 0 12345 298.15 0 1\n"
                "run 1000\n",
                encoding="utf-8",
            )
            thermo_rows = "\n".join(
                f"{step} {112 if step >= 500 else 106} 0 0"
                for step in range(0, 1001, 100)
            )
            (run / "logs" / "gcmc_1bar.log").write_text(
                f"Step Atoms Temp Press\n{thermo_rows}\n",
                encoding="utf-8",
            )

            report = analyze_block_convergence(
                run,
                root / "analysis",
                block_size=250,
            )

            self.assertFalse(report["provisional"])
            self.assertTrue(report["all_completed_runs_converged"])
            self.assertEqual(report["runs"][0]["complete_block_count"], 2)
            self.assertEqual(report["runs"][0]["latest_step"], 1000)
            self.assertTrue(Path(report["outputs"]["csv"]).exists())
            self.assertTrue(Path(report["outputs"]["json"]).exists())
            if report["outputs"]["plot"] is not None:
                self.assertTrue(Path(report["outputs"]["plot"]).exists())

    def test_reproducibility_manifest_hashes_immutable_run_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = root / "run"
            (run / "data").mkdir(parents=True)
            (run / "inputs").mkdir()
            (run / "data" / "framework.data").write_text(
                "framework\n",
                encoding="utf-8",
            )
            (run / "inputs" / "gcmc.in").write_text(
                "run 10\n",
                encoding="utf-8",
            )
            (run / "prepare_summary.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            config = root / "benchmark.json"
            config.write_text('{"material": "MOF-5"}\n', encoding="utf-8")

            manifest = create_reproducibility_manifest(
                run,
                root / "manifest",
                project_root=root,
                config_file=config,
                launch_command="python benchmark.py benchmark.json",
            )

            self.assertEqual(
                manifest["launch_command"],
                "python benchmark.py benchmark.json",
            )
            self.assertEqual(len(manifest["checksums_sha256"]), 4)
            self.assertTrue(Path(manifest["outputs"]["json"]).exists())
            self.assertTrue(Path(manifest["outputs"]["checksums"]).exists())

    def test_cell_comparison_reports_expected_ratios_and_slowest_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primitive = root / "primitive"
            conventional = root / "conventional"
            for run, atom_count, molecule_counts, runtimes in [
                (primitive, 106, [2.0, 4.0], [10.0, 20.0]),
                (conventional, 424, [8.0, 16.0], [30.0, 80.0]),
            ]:
                (run / "data").mkdir(parents=True)
                (run / "data" / "IRMOF-1.data").write_text(
                    f"LAMMPS data\n\n{atom_count} atoms\n",
                    encoding="utf-8",
                )
                (run / "evaluated_isotherm.csv").write_text(
                    "pressure_bar,loading_mol_per_kg\n1.0,1.0\n5.0,2.0\n",
                    encoding="utf-8",
                )
                (run / "isotherm_summary.json").write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "results": [
                                {
                                    "pressure_bar": pressure,
                                    "wall_time_seconds": runtime,
                                    "summary": {"mean_adsorbates": count},
                                }
                                for pressure, runtime, count in zip(
                                    [1.0, 5.0], runtimes, molecule_counts, strict=True
                                )
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            report = compare_cell_runs(primitive, conventional, root / "comparison")

            self.assertEqual(report["expected_r_n"], 4.0)
            self.assertTrue(report["runtime_ratio_available"])
            self.assertTrue(all(row["r_n"] == 4.0 for row in report["rows"]))
            self.assertTrue(all(row["r_q"] == 1.0 for row in report["rows"]))
            self.assertEqual(
                report["slowest_conventional_pressure"]["pressure_bar"], 5.0
            )
            self.assertEqual(
                report["slowest_runtime_ratio_pressure"][
                    "runtime_ratio_conventional_primitive"
                ],
                4.0,
            )
            self.assertTrue(Path(report["outputs"]["csv"]).exists())
            self.assertTrue(Path(report["outputs"]["replicate_csv"]).exists())
            self.assertEqual(report["replicate_row_count"], 4)
            self.assertTrue(Path(report["outputs"]["json"]).exists())
            if report["outputs"]["plot"] is not None:
                self.assertTrue(Path(report["outputs"]["plot"]).exists())

    def test_kspace_sensitivity_compares_completed_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = []
            for label, accuracy, loading in [
                ("pppm_1e-4", 1e-4, 10.0),
                ("pppm_1e-5", 1e-5, 10.5),
            ]:
                run = root / label
                run.mkdir()
                (run / "evaluated_isotherm.csv").write_text(
                    "pressure_bar,loading_absolute_mol_per_kg,loading_excess_mol_per_kg\n"
                    f"1.0,{loading},{loading - 0.1}\n",
                    encoding="utf-8",
                )
                (run / "prepare_summary.json").write_text(
                    json.dumps(
                        {
                            "result": {
                                "parameters": {
                                    "kspace_style": "pppm",
                                    "kspace_accuracy": accuracy,
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                runs.append((label, run))

            result = compare_kspace_runs(runs, root / "comparison")
            csv_path = Path(result["csv"])
            csv_exists = csv_path.exists()
            with csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(result["row_count"], 2)
        self.assertTrue(csv_exists)
        changed = next(row for row in rows if row["run_label"] == "pppm_1e-5")
        self.assertAlmostEqual(float(changed["absolute_delta_percent"]), 5.0)


if __name__ == "__main__":
    unittest.main()
