"""Finish the prepared Module B comparison and frozen uMOF confirmation budgets.

This supervisor changes no scientific settings. It waits for existing writers,
executes one bounded request/job at a time, and does not retry failed children.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
COMPARISON = Path("outputs/module_B_mof5_al_comparison")
BUNDLE = Path("reports/module_c_closure/cluster_bundle")
OUTPUT = Path("outputs/remaining_runs")
B_TERMINAL = {
    "budget_exhausted", "candidate_pool_exhausted", "surrogate_uncertainty_target_met",
    "pac_target_met", "simulation_precision_unmet", "holdout_completed", "holdout_precision_unmet",
}


def read(path):
    return json.loads(Path(path).read_text())


def log(message):
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def save_status(stage, status, **details):
    target = OUTPUT / f"{stage}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(stage=stage, status=status,
        updated_at=datetime.now(timezone.utc).isoformat(), **details), indent=2) + "\n")
    os.replace(temporary, target)


@contextmanager
def supervisor_lock(name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / f".{name}.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another remaining-runs supervisor owns {name}.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def lock_busy(path):
    """Probe an existing advisory lock; a stale filename is not a busy writer."""
    if not path.exists():
        return False
    with path.open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def request_limit(state, holdout=False):
    """Finite ceiling including every allowed retry, even after interruption."""
    budget = len(state["holdout"]) if holdout else state["options"]["max_training_points"]
    retries = state["options"]["max_precision_retries"]
    if type(budget) is not int or budget < 1 or type(retries) is not int or retries < 0:
        raise ValueError("A finite positive point budget and nonnegative retry limit are required.")
    return budget * (retries + 1)


def b_step(root, role, jobs, execute_allowed=True):
    from modules.module_b import comparison as compare
    from modules.module_b.runtime import campaign_lock

    manifest = compare.read(root / "comparison.json")
    target = Path(manifest["source"]) if role == "al" else compare.campaign(root, role)
    try:
        with campaign_lock(target):
            state = compare.read(target)
            if state["status"] == "technical_error":
                raise RuntimeError(f"Resolve the saved technical error in {target}: {state.get('last_error')}")
            if not execute_allowed:
                decision = compare.al.propose(target, holdout=role == "holdouts")
                if decision["status"] in {"request", "awaiting_simulation"}:
                    raise RuntimeError(f"{role}: request ceiling reached; refusing further execution.")
                return decision
            if role == "al":
                compare.checked_source(manifest)
                if state.get("campaign_kind") != "production" or state.get("material_key") != "mof5":
                    raise ValueError("Expected the prepared MOF-5 production campaign.")
                compare.capture(root, "al", state)
                result = compare.al.execute(target, 1, jobs=jobs)
                compare.capture(root, "al", compare.read(target))
                return result
            return compare.run(root, role, 1, jobs=jobs)
    except RuntimeError as exc:
        if isinstance(exc.__cause__, BlockingIOError):
            return {"status": "waiting_for_existing_writer"}
        raise


def drive_b(root, role, jobs, poll):
    from modules.module_b import comparison as compare
    from modules.module_b.cluster import check_dependencies

    check_dependencies()
    manifest = compare.read(root / "comparison.json")
    target = Path(manifest["source"]) if role == "al" else compare.campaign(root, role)
    ceiling = request_limit(compare.read(target), holdout=role == "holdouts")
    executed = 0
    previous = None
    while True:
        result = b_step(root, role, jobs, execute_allowed=executed < ceiling)
        status = result["status"]
        if status == "execution_budget_exhausted":
            executed += 1
        save_status(role, status, requests_this_invocation=executed, campaign=str(target))
        if status != previous or status == "execution_budget_exhausted":
            log(f"{role}: {json.dumps(result)}")
        previous = status
        if status in {"waiting_for_existing_writer", "waiting_for_shared_initial_design"}:
            time.sleep(poll)
            continue
        if status == "execution_budget_exhausted":
            continue
        if status not in B_TERMINAL:
            raise RuntimeError(f"{role}: unexpected status {status}; refusing an automatic retry.")
        return


def gpu_uuid(index):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    devices = dict(line.strip().split(", ", 1) for line in output.splitlines() if line.strip())
    if str(index) not in devices:
        raise ValueError(f"GPU {index} does not exist: {list(devices)}")
    return devices[str(index)]


def gpu_pids(uuid):
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    return [int(parts[1]) for line in output.splitlines()
            if len(parts := [p.strip() for p in line.split(",")]) == 2 and parts[0] == uuid]


def umof_jobs(bundle, system):
    config = read(bundle / f"{system}.json")
    return [f"{m['label']}_p{p:g}_s{s}" for p in config["pressures_bar"]
            for s in config["seeds"] for m in config["models"]]


def umof_command(bundle, system, action, env, *extra):
    # Always run the unchanged bundled code, preserving the frozen identities.
    subprocess.run([sys.executable, "-u", "-m", "modules.module_c_mlips.closure_campaign",
                    action, "--directory", ".", "--system", system, *extra],
                   cwd=bundle, env=env, check=True)


def drive_umof(bundle, gpu, poll):
    uuid = gpu_uuid(gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
    with supervisor_lock(f"gpu-{uuid}"):
        # POLAR finishes first; MH therefore cannot compete with it for memory.
        for system in ("polar", "mh"):
            umof_command(bundle, system, "status", env)  # includes asset/result verification
            for index, job in enumerate(umof_jobs(bundle, system)):
                directory = bundle / "results" / system / job
                previous = None
                while not (directory / "completed.json").exists():
                    # Full-campaign launchers lock one job at a time. Probe all
                    # locks before claiming GPU memory, including initialization.
                    busy = any(lock_busy(p) for p in (bundle / "results").glob("*/*/.writer.lock"))
                    pids = gpu_pids(uuid)
                    if busy or pids:
                        message = f"GPU {gpu}: waiting for existing jobs/processes {pids}"
                        if message != previous:
                            log(message)
                            save_status("umof", "waiting_for_gpu_or_writer", system=system, job=job, gpu=gpu)
                        previous = message
                        time.sleep(poll)
                        continue
                    log(f"{system}: run remaining job {index + 1}: {job} on GPU {gpu}")
                    save_status("umof", "running", system=system, job=job, gpu=gpu)
                    umof_command(bundle, system, "run", env,
                                 "--job-index", str(index), "--max-jobs", "1")
                    if not (directory / "completed.json").exists():
                        raise RuntimeError(f"uMOF job returned without completion: {job}")
            umof_command(bundle, system, "report", env)
        save_status("umof", "finished_within_configured_budgets", gpu=gpu)


def plan(scope, root, bundle, gpu):
    if scope in {"all", "module-b"}:
        manifest = read(root / "comparison.json")
        for role, path in (("al", Path(manifest["source"])),
                           ("holdouts", root / "holdouts/production/mof5/campaign.json"),
                           ("random", root / "random/production/mof5/campaign.json")):
            state = read(path)
            log(f"{role}: {len(state['results'])} training, {len(state['holdout_results'])}/{len(state['holdout'])} holdouts; "
                f"point budget {state['options']['max_training_points']}, retries {state['options']['max_precision_retries']}")
    if scope in {"all", "umof"}:
        for system in ("polar", "mh"):
            jobs = umof_jobs(bundle, system)
            done = sum((bundle / "results" / system / job / "completed.json").exists() for job in jobs)
            log(f"{system}: {done}/{len(jobs)} completion markers; GPU {gpu}; POLAR then MH")
    log("Plan only: no simulation, GPU allocation or campaign write.")


def supervise(args):
    stages = (["al", "holdouts", "random"] if args.scope in {"all", "module-b"} else [])
    if args.scope in {"all", "umof"}:
        stages.append("umof")
    children, handles = {}, []
    def stop(signum, frame):
        raise KeyboardInterrupt
    old_term = signal.signal(signal.SIGTERM, stop)
    try:
        # Acquire every selected branch lock before launching anything.
        from contextlib import ExitStack
        with ExitStack() as stack:
            for stage in stages:
                stack.enter_context(supervisor_lock(stage))
            for stage in stages:
                handle = (OUTPUT / f"{stage}.log").open("a", buffering=1)
                handles.append(handle)
                cmd = [sys.executable, "-u", "-m", "modules.remaining_runs", args.scope,
                       "--worker", stage, "--comparison-root", str(args.comparison_root),
                       "--bundle", str(args.bundle), "--jobs", str(args.jobs), "--poll-seconds", str(args.poll_seconds)]
                if args.gpu is not None:
                    cmd += ["--gpu", str(args.gpu)]
                children[stage] = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                log(f"Started {stage}; log: {OUTPUT / (stage + '.log')}")
            remaining = dict(children)
            failed = []
            b_reported = False
            while remaining:
                for stage, process in list(remaining.items()):
                    code = process.poll()
                    if code is not None:
                        log(f"{stage}: {'finished' if code == 0 else 'FAILED'} (exit {code})")
                        if code:
                            failed.append(stage)
                        del remaining[stage]
                if args.scope in {"all", "module-b"} and not b_reported and not (set(remaining) & {"al", "holdouts", "random"}):
                    if not (set(failed) & {"al", "holdouts", "random"}):
                        from modules.module_b.comparison import report
                        report(args.comparison_root)
                        log(f"Module B report refreshed: {args.comparison_root / 'report/report.md'}")
                    b_reported = True
                if remaining:
                    # Random needs the starts. A failed AL branch must not leave
                    # a random worker waiting forever for data that cannot arrive.
                    if "al" in failed and "random" in remaining:
                        process = remaining.pop("random")
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait()
                        failed.append("random")
                        save_status("random", "blocked_by_al_error")
                        log("random: stopped because AL failed")
                    if remaining:
                        time.sleep(1)
            if failed:
                raise RuntimeError(f"Failed branches: {', '.join(failed)}. Inspect {OUTPUT}/*.log; no error retries were started.")
            log("Configured runs finished. Check reports for remaining precision/sampling limitations.")
    finally:
        for process in children.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in children.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()
        signal.signal(signal.SIGTERM, old_term)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=["module-b", "umof", "all"])
    parser.add_argument("--jobs", type=int, choices=[1, 2, 3], default=3,
                        help="parallel seeds per Module B branch (up to 3 branches)")
    parser.add_argument("--gpu", type=int, help="physical nvidia-smi GPU index; waits until unused")
    parser.add_argument("--comparison-root", type=Path, default=COMPARISON)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--plan", action="store_true", help="read-only inventory; start nothing")
    parser.add_argument("--worker", choices=["al", "holdouts", "random", "umof"], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.poll_seconds < 1 or not args.poll_seconds < float("inf"):
        parser.error("--poll-seconds must be finite and >= 1")
    if args.scope in {"all", "umof"} and (args.gpu is None or args.gpu < 0):
        parser.error("Specify --gpu with the physical nvidia-smi index for uMOF.")
    if args.comparison_root.is_absolute():
        parser.error("--comparison-root must be repository-relative.")
    os.chdir(PROJECT)
    args.bundle = args.bundle.resolve()
    if args.plan:
        plan(args.scope, args.comparison_root, args.bundle, args.gpu)
        return 0
    if args.worker:
        try:
            if args.worker == "umof":
                drive_umof(args.bundle, args.gpu, args.poll_seconds)
            else:
                drive_b(args.comparison_root, args.worker, args.jobs, args.poll_seconds)
        except BaseException as exc:
            save_status(args.worker, "error", error=str(exc))
            raise
    else:
        supervise(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
