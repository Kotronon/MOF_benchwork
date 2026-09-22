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

class RunnersAndResumeTests(unittest.TestCase):
    def test_resume_detects_completed_and_incomplete_lammps_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            completed_log = root / "completed.log"
            incomplete_log = root / "incomplete.log"
            completed_log.write_text(
                "Step Atoms Temp\n"
                "0 10 0\n"
                "100 13 0\n"
                "Loop time of 1.0 on 1 procs for 100 steps with 13 atoms\n",
                encoding="utf-8",
            )
            incomplete_log.write_text(
                "Step Atoms Temp\n"
                "0 10 0\n"
                "80 13 0\n",
                encoding="utf-8",
            )

            self.assertTrue(_lammps_log_completed(completed_log, expected_steps=100))
            self.assertFalse(_lammps_log_completed(incomplete_log, expected_steps=100))

    def test_resume_selects_latest_lammps_restart_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            restart_dir = root / "restarts"
            restart_dir.mkdir()
            old_restart = restart_dir / "gcmc_1bar_seed101.restart.50000"
            new_restart = restart_dir / "gcmc_1bar_seed101.restart.150000"
            other_restart = restart_dir / "gcmc_2bar_seed101.restart.999999"
            old_restart.write_text("old", encoding="utf-8")
            new_restart.write_text("new", encoding="utf-8")
            other_restart.write_text("other", encoding="utf-8")

            latest = _latest_restart_file(
                {"restart": str(restart_dir / "gcmc_1bar_seed101.restart.*")}
            )

        self.assertEqual(_restart_step(old_restart), 50000)
        self.assertEqual(latest, new_restart)

    def test_resume_reuses_summary_when_combined_restart_log_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_log = root / "gcmc_1bar_seed101.log"
            combined_log = root / "gcmc_1bar_seed101_combined.log"
            summary_file = root / "gcmc_1bar_seed101_summary.json"
            base_log.write_text(
                "Step Atoms Temp\n"
                "0 10 0\n"
                "80 13 0\n",
                encoding="utf-8",
            )
            combined_log.write_text(
                "Step Atoms Temp\n"
                "0 10 0\n"
                "80 13 0\n"
                "Step Atoms Temp\n"
                "100 16 0\n"
                "Loop time of 1.0 on 1 procs for 20 steps with 16 atoms\n",
                encoding="utf-8",
            )
            summary_file.write_text(json.dumps({"mean_adsorbates": 2.0}), encoding="utf-8")

            reused = _existing_pressure_point_result(
                {
                    "path": str(root / "gcmc_1bar_seed101.in"),
                    "log": str(base_log),
                    "pressure_bar": 1.0,
                    "replicate_index": 1,
                    "seed": 101,
                },
                index=1,
                total_runs=1,
                discard_fraction=0.0,
                framework_atoms=10,
                adsorbate_atoms_per_molecule=3,
                expected_steps=100,
            )

        self.assertIsNotNone(reused)
        self.assertEqual(reused["status"], "reused_summary")
        self.assertEqual(reused["log_file"], str(combined_log))

    def test_resume_parses_completed_log_and_ignores_incomplete_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            completed_log = root / "gcmc_1bar_seed101.log"
            incomplete_log = root / "gcmc_2bar_seed101.log"
            completed_log.write_text(
                "Step Atoms Temp\n"
                "0 10 0\n"
                "100 13 0\n"
                "Loop time of 1.0 on 1 procs for 100 steps with 13 atoms\n",
                encoding="utf-8",
            )
            incomplete_log.write_text(
                "Step Atoms Temp\n"
                "0 10 0\n"
                "80 13 0\n",
                encoding="utf-8",
            )
            incomplete_log.with_name("gcmc_2bar_seed101_summary.json").write_text(
                json.dumps({"mean_adsorbates": 999.0}),
                encoding="utf-8",
            )

            reused = _existing_pressure_point_result(
                {
                    "path": str(root / "gcmc_1bar_seed101.in"),
                    "log": str(completed_log),
                    "pressure_bar": 1.0,
                    "replicate_index": 1,
                    "seed": 101,
                },
                index=1,
                total_runs=2,
                discard_fraction=0.0,
                framework_atoms=10,
                adsorbate_atoms_per_molecule=3,
                expected_steps=100,
            )
            ignored = _existing_pressure_point_result(
                {
                    "path": str(root / "gcmc_2bar_seed101.in"),
                    "log": str(incomplete_log),
                    "pressure_bar": 2.0,
                    "replicate_index": 1,
                    "seed": 101,
                },
                index=2,
                total_runs=2,
                discard_fraction=0.0,
                framework_atoms=10,
                adsorbate_atoms_per_molecule=3,
                expected_steps=100,
            )

            self.assertIsNotNone(reused)
            self.assertEqual(reused["status"], "reused_log")
            self.assertTrue(Path(reused["summary_file"]).exists())
            self.assertAlmostEqual(reused["summary"]["mean_adsorbates"], 0.5)
            self.assertIsNone(ignored)

    def test_run_isotherm_rejects_invalid_parallel_jobs(self) -> None:
        with self.assertRaisesRegex(ValueError, "jobs must be at least 1"):
            benchmark.run_isotherm({"files": {"gcmc_runs": []}}, jobs=0)


if __name__ == "__main__":
    unittest.main()
