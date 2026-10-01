from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from modules.module_c_mlips.applicability import assess_campaign_structure
from modules.module_c_mlips.campaign import (
    _validated_campaign_model,
    build_campaign_tasks,
)
from modules.module_c_mlips.active_learning.workflow import _stratified_split
from pipeline.config import normalize_config


class ModuleCCampaignTests(unittest.TestCase):
    def _normalized(self) -> dict:
        return normalize_config(
            {
                "material": {
                    "name": "custom_mof",
                    "cif_path": "structures/custom_mof.cif",
                },
                "benchmark": {
                    "task": "potential_benchmark",
                    "potential_benchmark": {
                        "workflow": "campaign",
                        "validation": {"mode": "screening"},
                        "campaign": {
                            "id": "test_campaign",
                            "systems": [
                                {
                                    "name": "unknown_one",
                                    "cif_path": "structures/one.cif",
                                },
                                {
                                    "name": "unknown_two",
                                    "cif_path": "structures/two.cif",
                                },
                            ],
                            "models": [
                                {
                                    "backend": "mace-torch",
                                    "name": "model_a",
                                    "model": "small",
                                },
                                {
                                    "backend": "mace-torch",
                                    "name": "model_b",
                                    "model": "small",
                                },
                            ],
                        },
                    },
                },
                "simulation": {"seeds": [11, 22]},
            }
        )

    def test_task_expansion_does_not_require_a_material_whitelist(self) -> None:
        tasks = build_campaign_tasks(
            self._normalized(),
            stage="widom",
            selected_systems=["unknown_two"],
            selected_models=["model_b"],
        )

        self.assertEqual(len(tasks), 2)
        self.assertEqual({task["seed"] for task in tasks}, {11, 22})
        self.assertTrue(
            all(task["system"] == "unknown_two" for task in tasks)
        )

    def test_holdout_split_preserves_candidate_type_fractions(self) -> None:
        candidates = [
            {
                "candidate_id": f"{category}_{index}",
                "candidate_type": category,
            }
            for category, count in (
                ("framework", 10),
                ("single_adsorbate", 20),
                ("multiple_adsorbates", 20),
            )
            for index in range(count)
        ]

        test, training = _stratified_split(
            candidates,
            test_count=20,
            fractions={
                "framework": 0.20,
                "single_adsorbate": 0.40,
                "multiple_adsorbates": 0.40,
            },
        )

        counts = {
            category: sum(
                item["candidate_type"] == category for item in test
            )
            for category in {
                "framework",
                "single_adsorbate",
                "multiple_adsorbates",
            }
        }
        self.assertEqual(
            counts,
            {
                "framework": 4,
                "single_adsorbate": 8,
                "multiple_adsorbates": 8,
            },
        )
        self.assertFalse(
            {item["candidate_id"] for item in test}
            & {item["candidate_id"] for item in training}
        )

    def test_validated_model_is_selected_from_active_learning_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            state_path = (
                output
                / "runs"
                / "campaign"
                / "custom__mace__active_learning__seed1"
                / "active_learning"
                / "state.json"
            )
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "material_id": "custom",
                        "status": "dft_validated",
                        "model_paths": ["first.model", "best.model"],
                        "validation": {
                            "metrics": {"best_model_index": 1}
                        },
                    }
                ),
                encoding="utf-8",
            )
            plan = {
                "material": {"material_id": "custom"},
                "outputs": {
                    "directory": str(output),
                    "run_id": "campaign/custom__model__widom__seed1",
                },
                "benchmark": {
                    "potential_benchmark": {
                        "active_learning": {"device": "cpu"}
                    }
                },
            }

            model, state = _validated_campaign_model(plan)

        self.assertEqual(model["model"], "best.model")
        self.assertEqual(state["status"], "dft_validated")

    @unittest.skipUnless(
        importlib.util.find_spec("ase") is not None,
        "ASE is required for structure applicability.",
    )
    def test_unknown_valid_cif_is_screening_only_not_rejected(self) -> None:
        cif = Path("CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif")
        if not cif.is_file():
            self.skipTest("CRAFTED fixture is unavailable.")
        config = self._normalized()
        config["material"] = {
            "name": "unregistered_mof",
            "cif_path": str(cif),
            "charge_scheme": "DDEC",
            "cif_source": str(cif),
            "source": "custom",
        }
        config["simulation"]["unit_cells"] = [1, 1, 1]
        config["simulation"]["cutoff_A"] = 6.0
        resolved = {
            "material": {
                "material_id": "unregistered_mof",
                "cif_path": str(cif),
            },
            "unit_cells": [1, 1, 1],
            "forcefield": {
                "adsorbates": {
                    "CO2": "CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def"
                }
            },
        }

        report = assess_campaign_structure(config, resolved)

        self.assertTrue(report["can_attempt_simulation"])
        self.assertEqual(report["potential_status"], "screening_only")
        self.assertEqual(
            report["reference_status"],
            "no_external_reference",
        )


if __name__ == "__main__":
    unittest.main()
