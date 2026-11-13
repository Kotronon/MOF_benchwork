"""Exercise scheduling, finite budgets and failure handling without simulations."""
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from modules import remaining_runs as runner


class RemainingRunsTests(unittest.TestCase):
    def test_b_step_waits_for_real_existing_writer_without_executing(self):
        from modules import module_b_comparison as compare
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "source" / "campaign.json"
            target.parent.mkdir()
            (root / "comparison.json").write_text(json.dumps({"source": str(target)}))
            with (target.parent / ".writer.lock").open("a") as owner, \
                 patch.object(compare.al, "execute") as execute:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(runner.b_step(root, "al", 3)["status"], "waiting_for_existing_writer")
                execute.assert_not_called()

    def test_b_step_refuses_execution_after_ceiling(self):
        from modules import module_b_comparison as compare
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "campaign.json"
            target.write_text(json.dumps({"status": "ready"}))
            (root / "comparison.json").write_text(json.dumps({"source": str(target)}))
            with patch.object(compare.al, "propose", return_value={"status": "request"}), \
                 patch.object(compare.al, "execute") as execute:
                with self.assertRaisesRegex(RuntimeError, "ceiling reached"):
                    runner.b_step(root, "al", 3, execute_allowed=False)
                execute.assert_not_called()

    def test_writer_probe_distinguishes_active_and_stale_locks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".writer.lock"
            self.assertFalse(runner.lock_busy(p))
            with p.open("a") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(runner.lock_busy(p))
                fcntl.flock(owner, fcntl.LOCK_UN)
            self.assertFalse(runner.lock_busy(p))

    def test_finite_ceiling_includes_retries_and_requires_explicit_budget(self):
        state = {"options": {"max_training_points": 24, "max_precision_retries": 1}, "holdout": list(range(20))}
        self.assertEqual(runner.request_limit(state), 48)
        self.assertEqual(runner.request_limit(state, True), 40)
        state["options"]["max_training_points"] = None
        with self.assertRaises(ValueError):
            runner.request_limit(state)

    def run_b(self, decisions):
        state = {"options": {"max_training_points": 1, "max_precision_retries": 0}}
        with patch("modules.module_b_comparison.read", side_effect=[{"source": "campaign.json"}, state]), \
             patch("modules.module_b_cluster.check_dependencies"), \
             patch.object(runner, "b_step", side_effect=decisions) as step, \
             patch.object(runner, "save_status"), patch.object(runner, "log"), \
             patch.object(runner.time, "sleep") as sleep:
            runner.drive_b(Path("comparison"), "al", 3, 30)
        return step, sleep

    def test_waits_for_existing_writer_then_finishes_within_ceiling(self):
        step, sleep = self.run_b([
            {"status": "waiting_for_existing_writer"},
            {"status": "execution_budget_exhausted"},
            {"status": "budget_exhausted"},
        ])
        sleep.assert_called_once_with(30)
        self.assertEqual([c.kwargs["execute_allowed"] for c in step.call_args_list], [True, True, False])

    def test_waits_for_shared_initial_design(self):
        _, sleep = self.run_b([{"status": "waiting_for_shared_initial_design"}, {"status": "budget_exhausted"}])
        sleep.assert_called_once()

    def test_precision_failure_is_a_terminal_scientific_result(self):
        step, _ = self.run_b([{"status": "holdout_precision_unmet"}])
        self.assertEqual(step.call_count, 1)

    def test_unknown_status_or_execution_error_is_not_retried(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected status"):
            self.run_b([{"status": "technical_error"}])
        with self.assertRaisesRegex(RuntimeError, "failed simulation"):
            self.run_b([RuntimeError("failed simulation")])

    def test_gpu_check_ignores_other_gpus(self):
        with patch.object(runner.subprocess, "check_output", return_value="GPU-other, 123\nGPU-selected, 456\n"):
            self.assertEqual(runner.gpu_pids("GPU-selected"), [456])

    def test_umof_waits_for_gpu_and_finishes_polar_before_mh(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for system in ("polar", "mh"):
                (root / f"{system}.json").write_text(json.dumps({"pressures_bar": [1], "seeds": [7], "models": [{"label": "pretrained"}]}))
            calls = []
            def command(bundle, system, action, env, *extra):
                calls.append((system, action))
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-test")
                if action == "run":
                    p = root / "results" / system / "pretrained_p1_s7" / "completed.json"
                    p.parent.mkdir(parents=True)
                    p.write_text("{}")
            with patch.object(runner, "supervisor_lock", return_value=nullcontext()), \
                 patch.object(runner, "gpu_uuid", return_value="GPU-test"), \
                 patch.object(runner, "gpu_pids", side_effect=[[123], [], []]), \
                 patch.object(runner, "umof_command", side_effect=command), \
                 patch.object(runner, "save_status"), patch.object(runner, "log"), \
                 patch.object(runner.time, "sleep") as sleep:
                runner.drive_umof(root, 0, 30)
            sleep.assert_called_once_with(30)
            self.assertEqual(calls, [("polar", "status"), ("polar", "run"), ("polar", "report"),
                                     ("mh", "status"), ("mh", "run"), ("mh", "report")])

    def test_umof_reuses_completed_jobs_without_waiting_for_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for system in ("polar", "mh"):
                (root / f"{system}.json").write_text(json.dumps({"pressures_bar": [1], "seeds": [7], "models": [{"label": "pretrained"}]}))
                marker = root / "results" / system / "pretrained_p1_s7" / "completed.json"
                marker.parent.mkdir(parents=True)
                marker.write_text("{}")
            with patch.object(runner, "supervisor_lock", return_value=nullcontext()), \
                 patch.object(runner, "gpu_uuid", return_value="GPU-test"), \
                 patch.object(runner, "gpu_pids") as occupancy, \
                 patch.object(runner, "umof_command") as command, \
                 patch.object(runner, "save_status"):
                runner.drive_umof(root, 0, 30)
            occupancy.assert_not_called()
            self.assertEqual([c.args[2] for c in command.call_args_list], ["status", "report", "status", "report"])

    def test_supervisor_stops_waiting_random_branch_if_al_fails(self):
        class Process:
            def __init__(self, pid, code):
                self.pid, self.code = pid, code
            def poll(self):
                return self.code
            def wait(self, timeout=None):
                if self.code is None:
                    self.code = -15
                return self.code
        args = SimpleNamespace(scope="module-b", comparison_root=Path("comparison"), bundle=Path("bundle"),
                               jobs=3, poll_seconds=30, gpu=None)
        children = [Process(1001, 1), Process(1002, 0), Process(1003, None)]
        with tempfile.TemporaryDirectory() as d, \
             patch.object(runner, "OUTPUT", Path(d)), \
             patch.object(runner.subprocess, "Popen", side_effect=children), \
             patch.object(runner.os, "killpg") as kill, \
             patch.object(runner, "save_status") as status, patch.object(runner, "log"):
            with self.assertRaisesRegex(RuntimeError, "Failed branches: al, random"):
                runner.supervise(args)
            kill.assert_called_once_with(1003, runner.signal.SIGTERM)
            status.assert_called_with("random", "blocked_by_al_error")

    def test_interrupt_terminates_owned_process_groups(self):
        class Process:
            pid, code = 1001, None
            def poll(self):
                return self.code
            def wait(self, timeout=None):
                self.code = -15
                return self.code
        args = SimpleNamespace(scope="umof", comparison_root=Path("comparison"), bundle=Path("bundle"),
                               jobs=3, poll_seconds=30, gpu=0)
        with tempfile.TemporaryDirectory() as d, \
             patch.object(runner, "OUTPUT", Path(d)), \
             patch.object(runner.subprocess, "Popen", return_value=Process()) as popen, \
             patch.object(runner.os, "killpg") as kill, \
             patch.object(runner.time, "sleep", side_effect=KeyboardInterrupt), \
             patch.object(runner, "log"):
            with self.assertRaises(KeyboardInterrupt):
                runner.supervise(args)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            kill.assert_called_once_with(1001, runner.signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
