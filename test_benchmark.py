from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import benchmark


class BenchmarkPipelineTests(unittest.TestCase):
    def test_load_benchmark_data_reads_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.json"
            path.write_text(json.dumps({"material": {"name": "MOF-5"}}), encoding="utf-8")

            data = benchmark.load_benchmark_data(path)

        self.assertEqual(data["material"]["name"], "MOF-5")

    def test_normalize_config_applies_v1_defaults(self) -> None:
        config = benchmark.normalize_config({"material": {"name": "MOF-5"}})

        self.assertEqual(config["adsorbates"]["components"], ["CO2"])
        self.assertEqual(config["benchmark"]["task"], "auto")
        self.assertIsNone(config["output"]["directory"])
        self.assertEqual(config["simulation"]["method"], "GCMC")

    def test_mof5_alias_resolves_to_irmof1(self) -> None:
        config = benchmark.normalize_config({"material": {"name": "MOF-5", "charge_scheme": "DDEC"}})

        resolved = benchmark.resolve_benchmark(config)

        self.assertEqual(resolved["material"]["material_id"], "IRMOF-1")
        self.assertTrue(resolved["material"]["cif_path"].endswith("CIF_FILES/DDEC/IRMOF-1.cif"))

    def test_selects_module_a_for_single_component_gcmc(self) -> None:
        config = benchmark.normalize_config({"material": {"name": "MOF-5"}})

        module = benchmark.select_module(config)

        self.assertEqual(module["id"], "A")

    def test_selects_module_b_for_mixture(self) -> None:
        config = benchmark.normalize_config(
            {
                "material": {"name": "MOF-5"},
                "adsorbates": {"components": ["CO2", "N2"], "mixture": {"CO2": 0.2, "N2": 0.8}},
            }
        )

        module = benchmark.select_module(config)

        self.assertEqual(module["id"], "B")

    def test_unknown_material_has_clear_error(self) -> None:
        config = benchmark.normalize_config({"material": {"name": "NOT_A_REAL_MOF"}})

        with self.assertRaisesRegex(LookupError, "Could not resolve material"):
            benchmark.resolve_benchmark(config)

    def test_dry_run_cli_outputs_run_plan_without_running_lammps(self) -> None:
        result = subprocess.run(
            [sys.executable, "benchmark.py", "benchmark.json", "--dry-run"],
            check=True,
            capture_output=True,
            text=True,
        )
        run_plan = json.loads(result.stdout)

        self.assertEqual(run_plan["status"], "planned")
        self.assertEqual(run_plan["module"]["id"], "A")
        self.assertEqual(run_plan["material"]["material_id"], "IRMOF-1")
        self.assertIn("forcefield", run_plan["resources"])


if __name__ == "__main__":
    unittest.main()
