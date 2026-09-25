from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

from analysis.mlip_widom import (
    ANGSTROM3_TO_M3,
    ATOMIC_MASS_KG,
    AVOGADRO_PER_MOL,
    BOLTZMANN_EV_PER_K,
    BOLTZMANN_J_PER_K,
    EV_TO_KJ_PER_MOL,
    analyze_widom_attempts,
)
from engines.mlip_mc_runner import (
    run_mlip_mc_gcmc,
    run_mlip_mc_widom,
    summarize_widom_samples,
)


HAS_ASE = importlib.util.find_spec("ase") is not None

if HAS_ASE:
    from ase import Atoms
    from ase.units import bar


class WidomStatisticsTests(unittest.TestCase):
    @unittest.skipUnless(HAS_ASE, "ASE is required for Widom statistics.")
    def test_overlap_rejections_contribute_zero_to_all_attempt_average(self) -> None:
        result = summarize_widom_samples(
            [0.0, 0.0],
            attempts=4,
            temperature_K=300.0,
        )

        self.assertAlmostEqual(result["average_boltzmann_factor"], 0.5)
        self.assertAlmostEqual(result["weighted_adsorption_energy_ev"], 0.0)
        self.assertEqual(result["valid_energy_count"], 2)

    def test_thermodynamics_and_block_uncertainty_include_overlaps(self) -> None:
        result = analyze_widom_attempts(
            [0.0, None, 0.0, None],
            temperature_K=300.0,
            framework_volume_A3=1000.0,
            framework_mass_amu=12.0,
            block_size=2,
            convergence_checkpoints=[2, 4],
        )

        overall = result["overall"]
        expected_qst = BOLTZMANN_EV_PER_K * 300.0 * EV_TO_KJ_PER_MOL
        expected_henry = (
            1000.0
            * ANGSTROM3_TO_M3
            * 0.5
            / (
                BOLTZMANN_J_PER_K
                * 300.0
                * AVOGADRO_PER_MOL
                * 12.0
                * ATOMIC_MASS_KG
            )
        )
        self.assertAlmostEqual(overall["average_boltzmann_factor"], 0.5)
        self.assertAlmostEqual(
            overall["henry_coefficient_mol_kg_pa"], expected_henry
        )
        self.assertAlmostEqual(
            overall["isosteric_heat_zero_loading_kj_mol"],
            expected_qst,
        )
        uncertainty = result["uncertainty"]
        self.assertEqual(uncertainty["complete_block_count"], 2)
        self.assertAlmostEqual(
            uncertainty["metrics"]["henry_coefficient_mol_kg_pa"][
                "standard_error"
            ],
            0.0,
        )
        self.assertEqual([row["attempts"] for row in result["convergence"]], [2, 4])


@unittest.skipUnless(HAS_ASE, "ASE is required for MLIP-MC runner tests.")
class MlipMcRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.framework = Atoms(
            "C",
            positions=[[0.0, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        self.adsorbate = Atoms(
            "CO2",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.16], [0.0, 0.0, -1.16]],
        )

    def test_widom_runner_writes_normalized_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_mlip_mc_widom(
                object(),
                self.framework,
                self.adsorbate,
                temperature_K=300.0,
                trial_count=4,
                seed=123,
                device="cpu",
                output_directory=temporary_directory,
                block_size=2,
                convergence_checkpoints=[2, 4],
                engine_class=FakeWidom,
            )

            self.assertEqual(result["attempts"], 4)
            self.assertEqual(result["valid_insertions"], 2)
            self.assertEqual(result["overlap_rejections"], 2)
            self.assertAlmostEqual(
                result["corrected_statistics"]["average_boltzmann_factor"],
                0.5,
            )
            self.assertEqual(result["attempt_trace_status"], "ordered_binary_log")
            self.assertEqual(result["uncertainty"]["complete_block_count"], 2)
            self.assertEqual(len(result["convergence"]), 2)
            self.assertTrue(
                Path(result["analysis_files"]["convergence_csv"]).is_file()
            )
            self.assertTrue(
                Path(result["analysis_files"]["convergence_plot"]).is_file()
            )
            self.assertTrue(Path(result["output_path"]).is_file())

    def test_gcmc_runner_summarizes_only_production_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_mlip_mc_gcmc(
                object(),
                self.framework,
                self.adsorbate,
                component="CO2",
                temperature_K=300.0,
                pressures_bar=[1.0],
                equilibration_steps=2,
                production_steps=3,
                seed=123,
                device="cpu",
                output_directory=temporary_directory,
                engine_class=FakeGcmc,
                eos_factory=lambda _component: FakeEos(),
            )

            point = result["pressure_points"][0]
            self.assertEqual(point["production_sample_count"], 3)
            self.assertAlmostEqual(
                point["mean_uptake_molecules_per_unit_cell"],
                2.0,
            )
            self.assertAlmostEqual(point["fugacity_bar"], 0.9)
            self.assertTrue(Path(result["output_path"]).is_file())


class FakeWidom:
    def __init__(self, **arguments) -> None:
        self.output_directory = Path(arguments["output_dir"])
        self.stats = {"valid_insertions": 2, "vdw_overlaps": 2}

    def run(self, trial_count: int) -> None:
        data = {
            "temperature": 300.0,
            "attempts": trial_count,
            "valid_insertions": 2,
            "raw_adsorption_energies": [0.0, 0.0],
        }
        (self.output_directory / "widom_results.json").write_text(
            json.dumps(data),
            encoding="utf-8",
        )
        log_path = self.output_directory / "log_widom.bin"
        with log_path.open("wb") as handle:
            for trial in (1, 3):
                handle.write(struct.pack("iddi", trial, 0.0, 0.0, 1))
                handle.write(struct.pack("i", 6))
                handle.write(struct.pack("3d", 0.0, 0.0, 0.0))
                handle.write(
                    struct.pack(
                        "9d",
                        10.0,
                        0.0,
                        0.0,
                        0.0,
                        10.0,
                        0.0,
                        0.0,
                        0.0,
                        10.0,
                    )
                )


class FakeGcmc:
    def __init__(self, **arguments) -> None:
        self.output_directory = Path(arguments["output_dir"])
        self.pressure = float(arguments["P"] / bar)
        self.moves = {"insertion": {"attempted": 2, "accepted": 1}}

    def run(self, total_steps: int) -> None:
        data = {
            "uptake": [9, 9, 1, 2, 3],
            "interaction_energy": [0, 0, -1, -2, -3],
            "total_energy": [0] * total_steps,
        }
        path = self.output_directory / f"results_{self.pressure:.4f}bar.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        transition_path = (
            self.output_directory / f"log_{self.pressure:.4f}bar.bin"
        )
        with transition_path.open("wb") as handle:
            for step, uptake, energy in (
                (1, 9, 0.0),
                (2, 9, 0.0),
                (3, 1, -1.0),
                (4, 2, -2.0),
                (5, 3, -3.0),
            ):
                handle.write(
                    struct.pack("iiddi", step, uptake, energy, energy, 4)
                )


class FakeEos:
    def calculate_fugacity(self, _temperature: float, pressure: float) -> float:
        return 0.9 * pressure


if __name__ == "__main__":
    unittest.main()
