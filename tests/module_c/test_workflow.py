from __future__ import annotations

from pathlib import Path
import unittest

from modules.module_c_mlips.potential_backends.classical_lammps import (
    ClassicalLAMMPSBackend,
)
from modules.module_c_mlips.potential_backends.mace import MaceBackend
from modules.module_c_mlips.workflow import (
    _build_backends,
    _build_configurations,
    _working_directory,
)
from pipeline.config import load_benchmark_data
from pipeline.planning import build_run_plan


CONFIG_PATH = (
    "input_json_files/benchmark_mof5_co2_potentials_smoke_test.json"
)
WIDOM_CONFIG_PATH = (
    "input_json_files/benchmark_mof5_co2_potentials_widom_pilot.json"
)


class PotentialWorkflowTests(unittest.TestCase):
    def test_smoke_config_builds_supported_module_c_plan(self) -> None:
        run_plan = build_run_plan(load_benchmark_data(CONFIG_PATH))

        self.assertEqual(run_plan["module"]["id"], "C")
        self.assertEqual(
            run_plan["benchmark"]["applicability"]["status"],
            "supported",
        )
        self.assertTrue(
            run_plan["benchmark"]["applicability"][
                "can_attempt_simulation"
            ]
        )
        settings = run_plan["benchmark"]["potential_benchmark"]
        self.assertEqual(settings["baseline_backend"], "uff_ddec_lammps")
        self.assertEqual(len(settings["backends"]), 2)

    def test_widom_config_builds_supported_module_c_plan(self) -> None:
        run_plan = build_run_plan(load_benchmark_data(WIDOM_CONFIG_PATH))

        applicability = run_plan["benchmark"]["applicability"]
        self.assertEqual(applicability["status"], "supported")
        checks = {check["name"]: check for check in applicability["checks"]}
        self.assertEqual(checks["configuration_set"]["actual"], "widom")
        self.assertEqual(checks["widom_dataset"]["status"], "passed")

    def test_backend_factory_builds_named_classical_and_mace_backends(self) -> None:
        run_plan = build_run_plan(load_benchmark_data(CONFIG_PATH))
        settings = run_plan["benchmark"]["potential_benchmark"]
        backends = _build_backends(
            settings["backends"],
            run_plan,
            Path("test_work") / "backends",
        )

        self.assertEqual(
            [backend.name for backend in backends],
            ["uff_ddec_lammps", "mace_mp_small"],
        )
        self.assertIsInstance(backends[0], ClassicalLAMMPSBackend)
        self.assertIsInstance(backends[1], MaceBackend)
        self.assertEqual(backends[1].model, "small")
        self.assertEqual(backends[1].default_dtype, "float32")

    def test_all_artifacts_share_one_run_directory(self) -> None:
        run_plan = build_run_plan(load_benchmark_data(CONFIG_PATH))

        working_directory = _working_directory(run_plan)

        self.assertEqual(
            working_directory,
            Path("outputs/module_C_potential_benchmark/runs/")
            / "mof5_co2_uff_mace_smoke_001",
        )

    def test_backend_factory_rejects_unknown_type(self) -> None:
        run_plan = build_run_plan(load_benchmark_data(CONFIG_PATH))

        with self.assertRaisesRegex(ValueError, "Unsupported potential backend"):
            _build_backends(
                [
                    {"type": "classical_lammps", "name": "classical"},
                    {"type": "unknown", "name": "unknown"},
                ],
                run_plan,
                Path("test_work") / "backends",
            )

    def test_widom_configuration_set_is_selected_from_settings(self) -> None:
        run_plan = build_run_plan(load_benchmark_data(CONFIG_PATH))
        settings = {
            "configuration_set": "widom",
            "dataset": {
                "sample_count": 2,
                "seed": 2468,
                "minimum_distance_A": 1.5,
                "random_orientations": True,
            },
        }
        component = run_plan["adsorbates"]["components"][0]

        configurations = _build_configurations(
            settings,
            run_plan,
            component,
            run_plan["resources"]["forcefield"],
        )

        self.assertEqual(len(configurations), 2)
        self.assertTrue(
            all(item.source == "generated_widom" for item in configurations)
        )

    def test_unknown_configuration_set_is_rejected(self) -> None:
        run_plan = build_run_plan(load_benchmark_data(CONFIG_PATH))
        component = run_plan["adsorbates"]["components"][0]

        with self.assertRaisesRegex(ValueError, "configuration_set"):
            _build_configurations(
                {"configuration_set": "unknown"},
                run_plan,
                component,
                run_plan["resources"]["forcefield"],
            )


if __name__ == "__main__":
    unittest.main()
