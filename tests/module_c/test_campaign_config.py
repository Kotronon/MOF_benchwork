from __future__ import annotations

import unittest

from pipeline.config import normalize_config


class ModuleCCampaignConfigTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "material": {
                "cif_path": "structures/custom.cif",
            },
            "benchmark": {
                "task": "potential_benchmark",
                "potential_benchmark": {
                    "workflow": "campaign",
                    "validation": {"mode": "screening"},
                },
            },
        }

    def test_direct_cif_infers_material_name_and_campaign_defaults(self) -> None:
        normalized = normalize_config(self._config())

        self.assertEqual(normalized["material"]["name"], "custom")
        settings = normalized["benchmark"]["potential_benchmark"]
        self.assertEqual(settings["adsorption_engine"], "mlip_mc")
        self.assertFalse(settings["active_learning"]["enabled"])
        self.assertEqual(
            settings["active_learning"]["validation_configurations"],
            6,
        )
        self.assertEqual(
            settings["campaign"]["stages"]["gcmc_production"][
                "production_steps"
            ],
            1_000_000,
        )

    def test_active_learning_budget_must_cover_all_allocations(self) -> None:
        config = self._config()
        active = config["benchmark"]["potential_benchmark"][
            "active_learning"
        ] = {
            "initial_training_configurations": 30,
            "test_configurations": 20,
            "rounds": 4,
            "configurations_per_round": 25,
            "maximum_configurations": 149,
        }
        self.assertEqual(active["maximum_configurations"], 149)

        with self.assertRaisesRegex(ValueError, "exceed"):
            normalize_config(config)

    def test_non_co2_adsorbate_is_retained_for_applicability_rejection(self) -> None:
        config = self._config()
        config["adsorbates"] = {"components": ["N2"]}

        normalized = normalize_config(config)

        self.assertEqual(normalized["adsorbates"]["components"], ["N2"])


if __name__ == "__main__":
    unittest.main()
