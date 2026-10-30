"""Verify the isolated follow-up uses Stage 1 and the original data splits."""
from pathlib import Path
import unittest

import torch

from modules.module_c_mlips import umof_stage1_followup as followup
from modules.module_c_mlips.umof_finetuning import sha256_file
from modules.module_c_mlips.umof_workflow import read_json


class Stage1FollowupTests(unittest.TestCase):
    def setUp(self):
        self.root = followup.DEFAULT_OUTPUT
        self.config = read_json(self.root / "followup.json")

    def test_command_uses_stage1_without_stage_two_or_new_matrix(self):
        command = followup.command(self.root, self.config)
        self.assertEqual(command[command.index("--foundation_head")+1], "Default")
        self.assertEqual(command[command.index("--foundation_model")+1], str(self.root / self.config["foundation"]))
        self.assertEqual(command[command.index("--lr")+1], "3e-05")
        self.assertEqual(command[command.index("--max_num_epochs")+1], "5")
        self.assertNotIn("--swa", command)
        self.assertNotIn("--start_swa", command)

    def test_original_data_splits_are_unchanged_and_no_test_data_are_used(self):
        parent = followup.SOURCE / "025pct/seed-12345"
        originals = {"train": parent / "datasets/train.extxyz", "valid": parent / "datasets/valid.extxyz",
            "replay_train": followup.SOURCE / "assets/replay-12345/train.extxyz",
            "replay_valid": followup.SOURCE / "assets/replay-12345/valid.extxyz"}
        for name, original in originals.items():
            self.assertEqual(sha256_file(original), sha256_file(self.root / f"assets/{name}.extxyz"))
        self.assertFalse((self.root / "assets/test.extxyz").exists())
        self.assertEqual(self.config["counts"], {"train": 211, "valid": 46, "replay_train": 200, "replay_valid": 100})

    def test_head_extraction_preserves_shared_electrostatic_buffers(self):
        parent = torch.load(self.root / "assets/stage1.model", map_location="cpu", weights_only=False)
        single = torch.load(self.root / self.config["foundation"], map_location="cpu", weights_only=False)
        self.assertEqual(single.heads, ["Default"])
        shared = [k for k in parent.state_dict() if k.endswith(".cl_scale")]
        self.assertTrue(shared)
        for key in shared:
            self.assertTrue(torch.equal(parent.state_dict()[key], single.state_dict()[key]), key)
        probe = read_json(self.root / "head_extraction_check.json")
        self.assertEqual(probe["status"], "passed")
        self.assertLessEqual(probe["energy_difference_eV"], .002)


if __name__ == "__main__":
    unittest.main()
