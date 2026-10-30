from __future__ import annotations

import unittest

import benchmark
from modules.registry import module_by_id, registered_modules
from resolvers.task_resolver import default_metrics_for_module


class ModuleRegistryTests(unittest.TestCase):
    def test_registered_modules_have_unique_ids_and_aliases(self) -> None:
        modules = registered_modules()
        ids = [module.id for module in modules]
        aliases = [
            alias
            for module in modules
            for alias in module.task_aliases
        ]

        self.assertEqual(ids, ["A", "B", "C", "D"])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(aliases), len(set(aliases)))

    def test_registered_modules_implement_required_interface(self) -> None:
        required_methods = [
            "metadata",
            "matches",
            "assess",
            "prepare",
            "materialize",
            "run_test",
            "run",
            "evaluate",
        ]

        for module in registered_modules():
            with self.subTest(module=module.id):
                self.assertTrue(module.name)
                self.assertTrue(module.slug)
                self.assertTrue(module.default_metrics)
                for method_name in required_methods:
                    self.assertTrue(callable(getattr(module, method_name)))

    def test_default_metrics_are_served_from_registry(self) -> None:
        self.assertEqual(
            default_metrics_for_module("B"),
            list(module_by_id("B").default_metrics),
        )

    def test_single_component_mixture_with_fraction_one_selects_module_a(self) -> None:
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

    def test_single_component_mixture_rejects_non_unit_fraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            benchmark.normalize_config(
                {
                    "material": {"name": "MOF-5"},
                    "adsorbates": {
                        "components": ["CO2"],
                        "mixture": {"CO2": 0.8},
                    },
                }
            )

    def test_true_mixture_selects_module_b(self) -> None:
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

    def test_potential_benchmark_settings_select_module_c(self) -> None:
        config = benchmark.normalize_config(
            {
                "material": {"name": "MOF-5"},
                "benchmark": {
                    "potential_benchmark": {
                        "workflow": "static",
                    },
                },
            }
        )

        self.assertEqual(benchmark.select_module(config)["id"], "C")

    def test_module_d_simulation_remains_unavailable(self) -> None:
        module = module_by_id("D")

        self.assertEqual(module.metadata()["id"], "D")
        with self.assertRaisesRegex(NotImplementedError, "simulation execution is unavailable"):
            module.run({"module": {"id": "D"}})


if __name__ == "__main__":
    unittest.main()
