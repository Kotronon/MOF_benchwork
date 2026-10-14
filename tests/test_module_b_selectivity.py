from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import benchmark
from modules.module_b_selectivity import (
    build_candidate_pool,
    evaluate_selectivity,
    initialize_campaign_state,
    select_next_candidate,
)
from parsers.lammps_log_parser import (
    parse_lammps_log,
    summarize_component_adsorption,
)
from pipeline.lammps_inputs import gcmc_input_builder
from pipeline.runners import _write_mixture_results_csv, _write_selectivity_surface_csv


class ModuleBSelectivityTests(unittest.TestCase):
    def test_mixture_config_selects_module_b(self) -> None:
        config = benchmark.normalize_config(
            {
                "material": {"name": "MOF-5"},
                "adsorbates": {
                    "components": ["CO2", "N2"],
                    "mixture": {"CO2": 0.15, "N2": 0.85},
                },
            }
        )

        self.assertEqual(benchmark.select_module(config)["id"], "B")
        self.assertEqual(config["adsorbates"]["mixture"], {"CO2": 0.15, "N2": 0.85})

    def test_single_component_mixture_is_normalized_to_module_a(self) -> None:
        config = benchmark.normalize_config(
            {
                "material": {"name": "MOF-5"},
                "adsorbates": {
                    "components": ["CO2"],
                    "mixture": {"CO2": 1.0},
                },
            }
        )

        self.assertIsNone(config["adsorbates"]["mixture"])
        self.assertEqual(benchmark.select_module(config)["id"], "A")

    def test_mixture_config_rejects_invalid_mole_fraction_sum(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            benchmark.normalize_config(
                {
                    "material": {"name": "MOF-5"},
                    "adsorbates": {
                        "components": ["CO2", "N2"],
                        "mixture": {"CO2": 0.2, "N2": 0.9},
                    },
                }
            )

    def test_mixture_config_rejects_mismatched_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match adsorbates.components"):
            benchmark.normalize_config(
                {
                    "material": {"name": "MOF-5"},
                    "adsorbates": {
                        "components": ["CO2", "N2"],
                        "mixture": {"CO2": 0.15, "CH4": 0.85},
                    },
                }
            )

    def test_module_b_prepare_plan_is_json_serializable(self) -> None:
        run_plan = benchmark.build_run_plan(
            benchmark.load_benchmark_data("benchmark_mof5_co2_n2_mixture_smoke_test.json")
        )

        prepare_plan = benchmark.prepare_benchmark(run_plan)

        self.assertEqual(run_plan["module"]["id"], "B")
        self.assertEqual(prepare_plan["parameters"]["mixture"], {"CO2": 0.15, "N2": 0.85})
        self.assertIn("CO2", prepare_plan["planned_files"]["molecule_templates"])
        self.assertIn("N2", prepare_plan["planned_files"]["molecule_templates"])
        json.dumps(prepare_plan)

    def test_mixture_gcmc_input_builder_writes_two_component_fixes(self) -> None:
        script = gcmc_input_builder(
            {
                "framework_data": "IRMOF-1.data",
                "molecule_templates": {"CO2": "CO2.template", "N2": "N2.template"},
                "forcefield_include": "forcefield.in",
                "framework_atom_types": [1, 2, 3, 4],
                "adsorbate_atom_types": [5, 6, 7, 8],
                "component_adsorbate_atom_types": {"CO2": [5, 6], "N2": [7, 8]},
                "component_atoms_per_molecule": {"CO2": 3, "N2": 3},
                "components": ["CO2", "N2"],
                "mixture": {"CO2": 0.15, "N2": 0.85},
                "temperature_K": 298.15,
                "pressure_bar": 10.0,
                "run_steps": 1,
                "gcmc_every_steps": 1,
                "exchange_attempts": 1,
                "move_attempts": 1,
                "seed": 12345,
                "fugacity_coefficients": {"CO2": 0.99, "N2": 1.01},
            }
        )

        self.assertIn("molecule co2 CO2.template", script)
        self.assertIn("molecule n2 N2.template", script)
        self.assertIn("group adsorbate_co2 type 5 6", script)
        self.assertIn("group adsorbate_n2 type 7 8", script)
        self.assertIn("variable nco2 equal count(adsorbate_co2)/3", script)
        self.assertIn("variable nn2 equal count(adsorbate_n2)/3", script)
        self.assertIn("# CO2 mole_fraction 0.15 partial_pressure_bar 1.5", script)
        self.assertIn("# N2 mole_fraction 0.85 partial_pressure_bar 8.5", script)
        self.assertIn("fix gcmc_co2 adsorbate_co2 gcmc", script)
        self.assertIn("fix gcmc_n2 adsorbate_n2 gcmc", script)
        self.assertIn("pressure 1.48038 fugacity_coeff 0.99", script)
        self.assertIn("pressure 8.38885 fugacity_coeff 1.01", script)
        self.assertIn("thermo_style custom step atoms temp pe etotal press v_nco2 v_nn2", script)

    def test_fake_mixture_log_yields_component_means_and_selectivity(self) -> None:
        log_data = """LAMMPS
Step Atoms Temp PotEng TotEng Press v_nco2 v_nn2
0 106 0.0 10.0 10.0 1.0 1 2
1 115 2.0 9.0 9.5 1.1 2 1
2 118 3.0 8.0 8.5 1.2 3 1
Loop time of 0.1 on 1 procs
"""
        parsed = parse_lammps_log(log_data)

        summary = summarize_component_adsorption(
            parsed["rows"],
            component_atom_counts={"CO2": 3, "N2": 3},
            mixture={"CO2": 0.15, "N2": 0.85},
            framework_mass_amu=1000.0,
        )

        self.assertAlmostEqual(summary["components"]["CO2"]["mean_molecules"], 2.0)
        self.assertAlmostEqual(summary["components"]["N2"]["mean_molecules"], 4.0 / 3.0)
        self.assertAlmostEqual(summary["components"]["CO2"]["loading_mol_per_kg"], 2.0)
        self.assertIsNotNone(summary["selectivity"]["CO2/N2"]["value"])

    def test_selectivity_handles_zero_denominator(self) -> None:
        selectivity = evaluate_selectivity(
            {"CO2": 2.0, "N2": 0.0},
            {"CO2": 0.15, "N2": 0.85},
        )

        self.assertIsNone(selectivity["CO2/N2"]["value"])
        self.assertEqual(selectivity["CO2/N2"]["reason"], "undefined_zero_adsorbed_or_gas_fraction")

    def test_active_learning_state_is_json_serializable_and_selects_pending_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "module_b_campaign_state.json"

            state = initialize_campaign_state(
                path,
                pressures_bar=[0.1, 1.0, 10.0],
                co2_mole_fractions=[0.05, 0.5, 0.85],
            )
            next_candidate = select_next_candidate(state)

            self.assertTrue(path.exists())
            self.assertEqual(len(build_candidate_pool([0.1, 1.0, 10.0], [0.05, 0.5, 0.85])), 9)
            self.assertIsNotNone(next_candidate)
            json.dumps(state)

    def test_mixture_result_csv_outputs_component_and_selectivity_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results = [
                {
                    "pressure_bar": 1.0,
                    "replicate_count": 1,
                    "seeds": [12345],
                    "summary": {
                        "components": {
                            "CO2": {
                                "gas_phase_mole_fraction": 0.15,
                                "mean_molecules": 0.5,
                                "max_molecules": 1.0,
                                "standard_error_molecules": None,
                            },
                            "N2": {
                                "gas_phase_mole_fraction": 0.85,
                                "mean_molecules": 0.5,
                                "max_molecules": 1.0,
                                "standard_error_molecules": None,
                            },
                        },
                        "selectivity": {
                            "CO2/N2": {"value": 5.666666666666666},
                            "N2/CO2": {"value": 0.1764705882352941},
                        },
                    },
                }
            ]

            mixture_csv = _write_mixture_results_csv(root / "mixture_results.csv", results)
            selectivity_csv = _write_selectivity_surface_csv(root / "selectivity_surface.csv", results)

            mixture_text = Path(mixture_csv).read_text(encoding="utf-8")
            selectivity_text = Path(selectivity_csv).read_text(encoding="utf-8")

        self.assertIn("mean_CO2_molecules_per_cell", mixture_text)
        self.assertIn("selectivity_CO2_N2", mixture_text)
        self.assertIn("CO2/N2", selectivity_text)


if __name__ == "__main__":
    unittest.main()
