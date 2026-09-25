from __future__ import annotations

import unittest

from pipeline.config import load_benchmark_data
from pipeline.planning import build_run_plan


WIDOM_CONFIG = "input_json_files/benchmark_zif8_co2_mlip_mc_widom_smoke.json"
GCMC_CONFIG = "input_json_files/benchmark_zif8_co2_mlip_mc_gcmc_smoke.json"


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


if __name__ == "__main__":
    unittest.main()
