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

class EvaluationTests(unittest.TestCase):
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

    def test_sim_results_to_csv_writes_flat_isotherm_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "isotherm_summary.json"
            output_path = Path(tmpdir) / "isotherm_summary.csv"
            data_path = Path(tmpdir) / "framework.data"
            reference_path = Path(tmpdir) / "reference.csv"
            data_path.write_text(
                """LAMMPS data

2 atoms
1 atom types

Masses

1 10.0 # X

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
2 1 1 0.0 1.0 0.0 0.0
""",
                encoding="utf-8",
            )
            reference_path.write_text(
                "# pressure[Pa],mean_volume[mol/kg],mean_error[mol/kg]\n"
                "1.000000000000000000e+05,8.000000000000000000e+01,1.000000000000000000e+00\n",
                encoding="utf-8",
            )
            input_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "results": [
                            {
                                "pressure_bar": 1.0,
                                "log_file": "gcmc_1bar.log",
                                "summary_file": "gcmc_1bar_summary.json",
                                "summary": {
                                    "sample_count": 10,
                                    "inserted": True,
                                    "max_atoms": 124,
                                    "max_adsorbates": 6.0,
                                    "mean_adsorbates": 2.0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = sim_results_to_csv(input_path, output_path, framework_data=data_path, reference_csv=reference_path)
            text = output_path.read_text(encoding="utf-8")
            with output_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            framework_mass_amu = framework_mass_from_lammps_data(data_path)

        self.assertEqual(result, output_path)
        self.assertAlmostEqual(framework_mass_amu, 20.0)
        self.assertIn("pressure_bar,pressure_Pa,sample_count,inserted,max_adsorbates,mean_adsorbates_per_cell", text)
        self.assertEqual(rows[0]["reference_source"], "crafted")
        self.assertEqual(rows[0]["reference_match"], "exact")
        self.assertEqual(float(rows[0]["absolute_error_mol_per_kg"]), 20.0)
        self.assertEqual(float(rows[0]["relative_error_percent"]), 25.0)

    def test_sim_results_to_csv_accepts_nist_json_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "isotherm_summary.json"
            output_path = root / "isotherm_summary.csv"
            data_path = root / "framework.data"
            reference_path = root / "nist.json"
            data_path.write_text(
                """LAMMPS data

1 atoms
1 atom types

Masses

1 10.0 # X

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
""",
                encoding="utf-8",
            )
            reference_path.write_text(
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
                                "species_data": [{"adsorption": 100.0, "composition": 1}],
                                "total_adsorption": 100.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            input_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "results": [
                            {
                                "pressure_bar": 1.0,
                                "summary": {
                                    "sample_count": 10,
                                    "inserted": True,
                                    "max_atoms": 13,
                                    "max_adsorbates": 4.0,
                                    "mean_adsorbates": 1.0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            sim_results_to_csv(input_path, output_path, framework_data=data_path, reference_csv=reference_path)
            text = output_path.read_text(encoding="utf-8")
            with output_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertIn("nist_isodb,10.test/example,unknown", text)
        self.assertEqual(rows[0]["reference_match"], "exact")
        self.assertEqual(rows[0]["reference_source"], "nist_isodb")
        self.assertEqual(rows[0]["reference_doi"], "10.test/example")
        self.assertEqual(rows[0]["reference_basis"], "unknown")
        self.assertEqual(rows[0]["reference_compared_against"], "absolute_assumed_for_unknown_reference")
        self.assertEqual(float(rows[0]["comparison_error_mol_per_kg"]), 0.0)

    def test_excess_reference_compares_against_excess_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "isotherm_summary.json"
            output_path = root / "isotherm_summary.csv"
            data_path = root / "framework.data"
            reference_path = root / "nist_excess.json"
            data_path.write_text(
                """LAMMPS data

1 atoms
1 atom types

Masses

1 10.0 # X

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
""",
                encoding="utf-8",
            )
            absolute_loading = 100.0
            gas_density = 100000.0 / (8.31446261815324 * 300.0)
            excess_loading = absolute_loading - gas_density * 0.001
            reference_path.write_text(
                json.dumps(
                    {
                        "DOI": "10.test/excess",
                        "adsorbates": [{"name": "Carbon Dioxide"}],
                        "adsorbent": {"name": "IRMOF-1"},
                        "adsorptionUnits": "mmol/g",
                        "pressureUnits": "bar",
                        "temperature": 300,
                        "category": "excess adsorption",
                        "isotherm_data": [
                            {
                                "pressure": 1.0,
                                "species_data": [{"adsorption": excess_loading, "composition": 1}],
                                "total_adsorption": excess_loading,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            input_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "results": [
                            {
                                "pressure_bar": 1.0,
                                "summary": {
                                    "sample_count": 10,
                                    "inserted": True,
                                    "max_atoms": 4,
                                    "max_adsorbates": 1.0,
                                    "mean_adsorbates": 1.0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            sim_results_to_csv(
                input_path,
                output_path,
                framework_data=data_path,
                reference_csv=reference_path,
                evaluation_config={
                    "report_excess": True,
                    "pore_volume_cm3_g": 1.0,
                    "gas_density_backend": "ideal",
                    "temperature_K": 300.0,
                },
            )
            with output_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["reference_basis"], "excess")
        self.assertEqual(rows[0]["reference_compared_against"], "excess")
        self.assertAlmostEqual(float(rows[0]["loading_absolute_mol_per_kg"]), absolute_loading)
        self.assertAlmostEqual(float(rows[0]["loading_excess_mol_per_kg"]), excess_loading)
        self.assertAlmostEqual(float(rows[0]["comparison_error_mol_per_kg"]), 0.0)

    def test_evaluate_isotherm_writes_readable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data").mkdir()
            (root / "source" / "references").mkdir(parents=True)
            sim_path = root / "isotherm_summary.json"
            data_path = root / "data" / "framework.data"
            reference_path = root / "source" / "references" / "reference.csv"
            data_path.write_text(
                """LAMMPS data

1 atoms
1 atom types

Masses

1 10.0 # X

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
""",
                encoding="utf-8",
            )
            reference_path.write_text(
                "# pressure[Pa],mean_volume[mol/kg],mean_error[mol/kg]\n"
                "1.0e+05,100.0,2.0\n",
                encoding="utf-8",
            )
            sim_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "results": [
                            {
                                "pressure_bar": 1.0,
                                "summary": {
                                    "sample_count": 10,
                                    "inserted": True,
                                    "max_atoms": 13,
                                    "max_adsorbates": 4.0,
                                    "mean_adsorbates": 1.0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = evaluate_isotherm(sim_path)
            csv_exists = Path(result["evaluated_csv"]).exists()
            report_exists = Path(result["evaluation_report"]).exists()

        self.assertEqual(result["point_count"], 1)
        self.assertEqual(result["mean_absolute_error_mol_per_kg"], 0.0)
        self.assertTrue(csv_exists)
        self.assertTrue(report_exists)

    def test_evaluate_all_references_writes_combined_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "data").mkdir()
            (run_dir / "source" / "references").mkdir(parents=True)
            data_path = run_dir / "data" / "framework.data"
            crafted_path = run_dir / "source" / "references" / "crafted.csv"
            nist_path = run_dir / "source" / "references" / "nist.json"
            sim_path = run_dir / "isotherm_summary.json"

            data_path.write_text(
                """LAMMPS data

1 atoms
1 atom types

Masses

1 10.0 # X

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
""",
                encoding="utf-8",
            )
            crafted_path.write_text(
                "# pressure[Pa],mean_volume[mol/kg],mean_error[mol/kg]\n"
                "1.0e+05,90.0,1.0\n",
                encoding="utf-8",
            )
            gas_density = 100000.0 / (8.31446261815324 * 300.0)
            excess_loading = 100.0 - gas_density * 0.001
            nist_path.write_text(
                json.dumps(
                    {
                        "DOI": "10.test/nist",
                        "adsorbates": [{"name": "Carbon Dioxide"}],
                        "adsorbent": {"name": "IRMOF-1"},
                        "adsorptionUnits": "mmol/g",
                        "pressureUnits": "bar",
                        "temperature": 298,
                        "category": "excess adsorption",
                        "isotherm_data": [
                            {
                                "pressure": 1.0,
                                "species_data": [{"adsorption": excess_loading, "composition": 1}],
                                "total_adsorption": excess_loading,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            sim_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "results": [
                            {
                                "pressure_bar": 1.0,
                                "summary": {
                                    "sample_count": 10,
                                    "inserted": True,
                                    "max_atoms": 13,
                                    "max_adsorbates": 4.0,
                                    "mean_adsorbates": 1.0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "prepare_summary.json").write_text(
                json.dumps(
                    {
                        "prepare_plan": {"parameters": {"components": ["CO2"]}},
                        "result": {
                            "files": {
                                "source_files": {
                                    "reference_files": [
                                        {
                                            "path": str(crafted_path),
                                            "copied_path": str(crafted_path),
                                            "source": "crafted",
                                            "format": "crafted_csv",
                                            "selected": True,
                                        },
                                        {
                                            "path": str(nist_path),
                                            "copied_path": str(nist_path),
                                            "source": "nist_isodb",
                                            "format": "nist_json",
                                            "doi": "10.test/nist",
                                            "selected": False,
                                        },
                                    ]
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = evaluate_all_references(
                run_dir,
                evaluation_config={
                    "report_excess": True,
                    "pore_volume_cm3_g": 1.0,
                    "gas_density_backend": "ideal",
                    "temperature_K": 300.0,
                },
            )
            combined_csv = Path(result["combined_csv"])
            basis_csv = Path(result["basis_comparison_csv"])
            basis_report = Path(result["basis_comparison_report"])
            basis_absolute_plot = Path(result["basis_absolute_plot"])
            basis_excess_plot = Path(result["basis_excess_plot"])
            candidates_json = Path(result["reference_candidates"])
            combined_csv_exists = combined_csv.exists()
            basis_csv_exists = basis_csv.exists()
            basis_report_exists = basis_report.exists()
            basis_absolute_plot_exists = basis_absolute_plot.exists()
            basis_excess_plot_exists = basis_excess_plot.exists()
            candidates_json_exists = candidates_json.exists()
            combined_text = combined_csv.read_text(encoding="utf-8")
            basis_text = basis_csv.read_text(encoding="utf-8")
            with basis_csv.open("r", encoding="utf-8") as handle:
                basis_rows = list(csv.DictReader(handle))

        self.assertEqual(result["reference_count"], 2)
        self.assertTrue(combined_csv_exists)
        self.assertTrue(basis_csv_exists)
        self.assertTrue(basis_report_exists)
        self.assertTrue(basis_absolute_plot_exists)
        self.assertTrue(basis_excess_plot_exists)
        self.assertTrue(candidates_json_exists)
        self.assertIn("simulation_mol_per_kg", combined_text)
        self.assertIn("crafted_crafted_reference_mol_per_kg", combined_text)
        self.assertIn("nist_isodb_10_test_nist_reference_mol_per_kg", combined_text)
        self.assertIn("selected_simulation_mol_per_kg", basis_text)
        self.assertEqual(result["basis_comparison_row_count"], 2)
        crafted_row = next(row for row in basis_rows if row["reference_source"] == "crafted")
        nist_row = next(row for row in basis_rows if row["reference_source"] == "nist_isodb")
        self.assertEqual(crafted_row["basis_interpretation"], "matched_absolute_reference")
        self.assertEqual(nist_row["reference_basis"], "excess")
        self.assertEqual(nist_row["reference_compared_against"], "excess")
        self.assertEqual(nist_row["basis_interpretation"], "matched_excess_reference")
        self.assertAlmostEqual(float(nist_row["comparison_error_mol_per_kg"]), 0.0)


if __name__ == "__main__":
    unittest.main()
