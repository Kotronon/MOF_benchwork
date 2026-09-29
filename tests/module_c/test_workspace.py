from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from modules.module_c_mlips.workspace import initialize_workspace


class WorkspaceTests(unittest.TestCase):
    def test_overwrite_replaces_existing_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "run"
            stale_restart = run_directory / "engine" / "restart.json"
            stale_restart.parent.mkdir(parents=True)
            stale_restart.write_text("stale", encoding="utf-8")

            run_plan = {"outputs": {"overwrite": True}}
            initialize_workspace(run_directory, run_plan)

            self.assertFalse(stale_restart.exists())
            self.assertTrue((run_directory / "engine").is_dir())
            self.assertEqual(
                json.loads((run_directory / "run_plan.json").read_text()),
                run_plan,
            )

    def test_no_overwrite_preserves_existing_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "run"
            marker = run_directory / "marker.txt"
            marker.parent.mkdir(parents=True)
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                initialize_workspace(
                    run_directory,
                    {"outputs": {"overwrite": False}},
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
