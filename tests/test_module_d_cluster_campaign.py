"""Independent cluster cases and collection, using invented numerical data."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from benchmark import build_run_plan
from modules.module_d.flexibility import CYCLE_COLUMNS, prepare_flexibility, materialize_flexibility
from modules.module_d.general import materialize_general, render_input, verify_bundle
from tests.test_module_d_general import fixture
from tests.test_module_d_flexibility import measured_rows
from analysis.module_d_flexibility import summarize_cycles

spec = importlib.util.spec_from_file_location("mil53_campaign_runner", Path(__file__).parents[1] / "scripts/run_module_d_mil53.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ClusterCampaignTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("MODULE_D_LAMMPS"), "Set MODULE_D_LAMMPS for a short native fixture run")
    def test_native_independent_cases_and_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp, molecule=True)
            p = config["benchmark"]["flexibility"]
            p["md"].update(ensemble="NPT", cell_mode="tri", external_pressure_bar=100., barostat_damping_ps=1.)
            p["trajectory"] = dict(interval_steps=5, precision_digits=16)
            plan = build_run_plan(config)
            root = Path(materialize_flexibility(prepare_flexibility(plan))["working_directory"])
            command = [os.environ["MODULE_D_LAMMPS"]]
            runner.run_all_cases(plan, root, command)
            completed_times = {}
            for case in plan["cases"]:
                case_id = case["case_id"]
                job = root / "jobs" / case_id
                completed_times[case_id] = (job / "cycle_results.csv").stat().st_mtime_ns
                for link in (job / "assets", job / "smoke" / case_id / "assets", job / "results" / case_id / "assets"):
                    self.assertTrue(link.is_symlink())
                    self.assertEqual(link.resolve(), (root / "assets").resolve())
            runner.run_all_cases(plan, root, command)
            for case_id, timestamp in completed_times.items():
                self.assertEqual((root / "jobs" / case_id / "cycle_results.csv").stat().st_mtime_ns, timestamp)
            report = runner.collect_cases(plan, root)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(report["comparisons"]), 2)

    def test_single_flexible_case_uses_transferred_assets_and_can_be_analyzed(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_run_plan(fixture(tmp))
            root = Path(materialize_flexibility(prepare_flexibility(plan))["working_directory"])
            Path(tmp, "dry.data").unlink()
            case = plan["cases"][1]
            selected, output = runner.job_plan(plan, root, case)
            self.assertEqual(selected["cases"], [case])
            self.assertTrue(all(e["status"] == "present" for e in selected["benchmark"]["applicability"]["assets"]))
            materialize_general(dict(run_plan=selected, working_directory=output,
                                     shared_assets_directory=root / "assets"))
            self.assertTrue((output / "assets").is_symlink())
            report = summarize_cycles(selected, measured_rows(selected))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["comparisons"], [])
            (root / "assets/initial_structure").write_text("changed fixture")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_bundle(output)

    def test_collection_requires_all_jobs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_run_plan(fixture(tmp))
            root = Path(materialize_flexibility(prepare_flexibility(plan))["working_directory"])
            for case in plan["cases"]:
                selected, output = runner.job_plan(plan, root, case)
                materialize_general(dict(run_plan=selected, working_directory=output))
                (output / "results").mkdir()
                (output / "results/status.json").write_text(json.dumps(dict(status="complete", command=["fixture-lmp"], cases=[dict(case_id=case["case_id"])])))
                with (output / "cycle_results.csv").open("x", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CYCLE_COLUMNS)
                    writer.writeheader()
                    writer.writerows(measured_rows(selected))
                if case == plan["cases"][0]:
                    with self.assertRaises(FileNotFoundError):
                        runner.collect_cases(plan, root)
                    self.assertFalse((root / "cycle_results.csv").exists())
            report = runner.collect_cases(plan, root)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(report["comparisons"]), 1)
            with self.assertRaises(FileExistsError):
                runner.collect_cases(plan, root)

    def test_configured_precise_output_and_invalid_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            config["benchmark"]["flexibility"]["trajectory"] = dict(interval_steps=5, precision_digits=16)
            plan = build_run_plan(config)
            deck = render_input(plan, plan["cases"][1])
            self.assertIn("custom 5 trajectory.lammpstrj", deck)
            self.assertIn("format float %.16g", deck)
            config["benchmark"]["flexibility"]["trajectory"]["precision_digits"] = 18
            with self.assertRaises(ValueError):
                build_run_plan(config)

    def test_isotherm_json_covers_all_branches_seeds_and_pressures(self):
        config = json.loads(Path("input_json_files/benchmark_module_d_mil53_co2_isotherm.json").read_text())
        plan = build_run_plan(config)
        self.assertFalse(plan["benchmark"]["applicability"]["blockers"])
        self.assertEqual(len(plan["cases"]), 90)
        self.assertEqual({c["variant"] for c in plan["cases"]}, {"rigid", "hybrid_nvt", "hybrid_npt"})
        self.assertEqual(len({(c["variant"], c["point_id"], c["seed"]) for c in plan["cases"]}), 90)
        for case in plan["cases"]:
            deck = render_input(plan, case)
            self.assertIn("run 200000\n", deck)
            self.assertIn("run 400000\n", deck)
            self.assertIn("format float %.16g", deck)


if __name__ == "__main__":
    unittest.main()
