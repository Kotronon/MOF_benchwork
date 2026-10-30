"""Check follow-up scope and protection of the completed pilot."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modules.module_c_mlips import umof_adsorption_refinement as refinement
from modules.module_c_mlips.umof_workflow import atomic_json, read_json


class RefinementTests(unittest.TestCase):
    def test_only_two_models_two_pressures_and_no_widom(self):
        config = refinement.load_config(refinement.DEFAULT_CONFIG)
        jobs = list(refinement.jobs(config, Path("output")))
        self.assertEqual(len(jobs), 4)
        self.assertEqual({j[0] for j in jobs}, {"baseline", "stage1"})
        self.assertEqual({j[1] for j in jobs}, {12345})
        self.assertEqual({j[2] for j in jobs}, {.4655, .9905})
        self.assertTrue(all("gcmc-" in str(j[-1]) for j in jobs))
        self.assertEqual(config["gcmc"]["equilibration_steps"], 5000)
        self.assertEqual(config["gcmc"]["production_steps"], 10000)
        self.assertNotEqual(config["output_directory"], "outputs/umof_polar_adsorption_comparison")

    def test_rejects_changed_original_input_before_loading_models(self):
        recipe = read_json(refinement.DEFAULT_CONFIG)
        recipe["source_config"]["sha256"] = "0"*64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"recipe.json"
            atomic_json(path, recipe)
            with patch.object(refinement.original, "load_config") as loader:
                with self.assertRaisesRegex(ValueError, "Original configuration changed"):
                    refinement.load_config(path)
                loader.assert_not_called()

    def test_rejects_duplicate_pressures_or_extra_seed(self):
        recipe = read_json(refinement.DEFAULT_CONFIG)
        for key, value in (("pressures_bar", [.4655, .4655]), ("seeds", [12345, 23456])):
            current = copy.deepcopy(recipe); current[key] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"recipe.json"; atomic_json(path, current)
                with self.assertRaises(ValueError):
                    refinement.load_config(path)


if __name__ == "__main__":
    unittest.main()
