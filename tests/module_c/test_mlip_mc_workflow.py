from __future__ import annotations

import unittest

from pipeline.config import load_benchmark_data
from pipeline.planning import build_run_plan


WIDOM_CONFIG = "input_json_files/benchmark_zif8_co2_mlip_mc_widom_smoke.json"
GCMC_CONFIG = "input_json_files/benchmark_zif8_co2_mlip_mc_gcmc_smoke.json"
NEQUIP_CONFIG = (
    "input_json_files/benchmark_zif8_co2_nequip_finetuned_widom_smoke.json"
)


class MlipMcWorkflowPlanningTests(unittest.TestCase):
    def test_widom_config_selects_executable_module_c_workflow(self) -> None:
        plan = build_run_plan(load_benchmark_data(WIDOM_CONFIG))

        self.assertEqual(plan["module"]["id"], "C")
        self.assertEqual(plan["simulation"]["engine"], "MLIP_MC")
        self.assertEqual(plan["simulation"]["method"], "WIDOM")
        applicability = plan["benchmark"]["applicability"]
        self.assertEqual(applicability["status"], "supported")
        self.assertEqual(
            applicability["current_module_capability"],
            "mlip_mc_widom",
        )
        checks = {item["name"]: item for item in applicability["checks"]}
        self.assertEqual(checks["widom_analysis"]["status"], "passed")

    def test_invalid_widom_checkpoint_is_rejected_by_applicability(self) -> None:
        config = load_benchmark_data(WIDOM_CONFIG)
        settings = config["benchmark"]["potential_benchmark"]["mlip_mc"]
        settings["convergence_checkpoints"] = [10, 21]

        plan = build_run_plan(config)

        self.assertEqual(plan["benchmark"]["applicability"]["status"], "unsupported")
        checks = {
            item["name"]: item
            for item in plan["benchmark"]["applicability"]["checks"]
        }
        self.assertEqual(checks["widom_analysis"]["status"], "failed")

    def test_gcmc_config_selects_executable_module_c_workflow(self) -> None:
        plan = build_run_plan(load_benchmark_data(GCMC_CONFIG))

        self.assertEqual(plan["module"]["id"], "C")
        self.assertEqual(plan["simulation"]["method"], "GCMC")
        applicability = plan["benchmark"]["applicability"]
        self.assertEqual(applicability["status"], "supported")
        checks = {item["name"]: item for item in applicability["checks"]}
        self.assertEqual(checks["gcmc_steps"]["status"], "passed")

    def test_nequip_reference_config_is_supported_without_loading_model(self) -> None:
        plan = build_run_plan(load_benchmark_data(NEQUIP_CONFIG))

        model = plan["benchmark"]["potential_benchmark"]["mlip_mc"]["model"]
        self.assertEqual(model["backend"], "nequip")
        self.assertEqual(model["loader"], "legacy")
        self.assertEqual(model["energy_mode"], "interaction_direct")
        self.assertEqual(
            model["adsorbate_species_aliases"],
            {"C": "Co", "O": "Os"},
        )
        self.assertEqual(
            model["vdw_radius_aliases"],
            {"Co": "C", "Os": "O"},
        )
        self.assertFalse(model["dispersion"])
        self.assertEqual(plan["conditions"]["temperature_K"], 273.0)
        checks = {
            item["name"]: item
            for item in plan["benchmark"]["applicability"]["checks"]
        }
        self.assertEqual(checks["mlip_backend"]["status"], "passed")
        self.assertEqual(checks["energy_mode"]["status"], "passed")
        self.assertEqual(checks["nequip_dispersion"]["status"], "passed")
        self.assertEqual(
            checks["goeminne_adsorbate_species"]["status"],
            "passed",
        )
        self.assertEqual(checks["goeminne_vdw_radii"]["status"], "passed")
        self.assertEqual(
            checks["goeminne_species_mapping"]["status"],
            "passed",
        )
        self.assertEqual(checks["goeminne_input_contract"]["status"], "passed")

    def test_nequip_reference_rejects_double_counted_dispersion(self) -> None:
        config = load_benchmark_data(NEQUIP_CONFIG)
        model = config["benchmark"]["potential_benchmark"]["mlip_mc"]["model"]
        model["dispersion"] = True

        plan = build_run_plan(config)

        applicability = plan["benchmark"]["applicability"]
        self.assertEqual(applicability["status"], "unsupported")
        checks = {item["name"]: item for item in applicability["checks"]}
        self.assertEqual(checks["nequip_dispersion"]["status"], "failed")

    def test_goeminne_reference_rejects_missing_guest_aliases(self) -> None:
        config = load_benchmark_data(NEQUIP_CONFIG)
        model = config["benchmark"]["potential_benchmark"]["mlip_mc"]["model"]
        model.pop("adsorbate_species_aliases")

        plan = build_run_plan(config)

        checks = {
            item["name"]: item
            for item in plan["benchmark"]["applicability"]["checks"]
        }
        self.assertEqual(
            checks["goeminne_adsorbate_species"]["status"],
            "failed",
        )

    def test_goeminne_reference_rejects_missing_input_contract(self) -> None:
        config = load_benchmark_data(NEQUIP_CONFIG)
        model = config["benchmark"]["potential_benchmark"]["mlip_mc"]["model"]
        model.pop("input_contract")

        plan = build_run_plan(config)

        checks = {
            item["name"]: item
            for item in plan["benchmark"]["applicability"]["checks"]
        }
        self.assertEqual(checks["goeminne_input_contract"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
