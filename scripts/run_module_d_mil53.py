"""MIL-53/CO2 campaigns from a repository checkout or a legacy portable archive."""
from copy import deepcopy
import csv
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys


def job_plan(plan, campaign, case):
    """Create an independent one-case plan using only transferred assets."""
    from modules.module_d.flexibility import asset
    result = deepcopy(plan)
    directory = Path(campaign).resolve() / "jobs" / case["case_id"]
    result["cases"] = [deepcopy(case)]
    result["state_points"] = [p for p in plan["state_points"] if p["point_id"] == case["point_id"]]
    result["simulation"]["seeds"] = [case["seed"]]
    result["outputs"].update(directory=str(directory), resume=False, overwrite=False)
    result["benchmark"]["applicability"]["assets"] = [
        asset(Path(campaign).resolve() / "assets" / entry["role"], entry["role"])
        for entry in plan["benchmark"]["applicability"]["assets"]]
    if any(entry["status"] != "present" for entry in result["benchmark"]["applicability"]["assets"]):
        raise ValueError("Transferred job assets are incomplete.")
    return result, directory


def run_case(plan, campaign, index, command):
    from modules.module_d.general import materialize_general, execute_bundle
    from analysis.module_d_flexibility import evaluate_flexibility
    if index is None or not 1 <= index <= len(plan["cases"]):
        raise ValueError(f"--case-index must be between 1 and {len(plan['cases'])}.")
    case = plan["cases"][index-1]
    selected, directory = job_plan(plan, campaign, case)
    print(f"Case {index}/{len(plan['cases'])}: {case['case_id']}", flush=True)
    materialize_general(dict(run_plan=selected, working_directory=str(directory),
                             shared_assets_directory=Path(campaign).resolve() / "assets"))
    execute_bundle(directory, smoke=True, command=command)
    execute_bundle(directory, command=command)
    return evaluate_flexibility(directory)


def collect_cases(plan, campaign):
    """Require every planned job, validate records, then form comparisons."""
    from modules.module_d.general import verify_bundle
    from modules.module_d.flexibility import CYCLE_COLUMNS
    from analysis.module_d_flexibility import summarize_cycles, evaluate_flexibility
    campaign = Path(campaign)
    rows = []
    commands = []
    for case in plan["cases"]:
        expected, directory = job_plan(plan, campaign, case)
        if verify_bundle(directory) != expected:
            raise ValueError(f"Changed job plan: {case['case_id']}")
        status = json.loads((directory / "results/status.json").read_text())
        if status["status"] != "complete" or [c["case_id"] for c in status["cases"]] != [case["case_id"]]:
            raise ValueError(f"Incomplete job: {case['case_id']}")
        commands.append(status["command"])
        with (directory / "cycle_results.csv").open(newline="") as handle:
            job_rows = list(csv.DictReader(handle))
        if any(row["case_id"] != case["case_id"] for row in job_rows):
            raise ValueError(f"Unexpected measurements: {case['case_id']}")
        rows.extend(job_rows)
    if any(command != commands[0] for command in commands):
        raise ValueError("All cases must use the same LAMMPS command.")
    if summarize_cycles(plan, rows)["status"] != "complete":
        raise ValueError("Incomplete campaign measurements.")
    with (campaign / "cycle_results.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CYCLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return evaluate_flexibility(campaign)


def run_all_cases(plan, campaign, command):
    """Run directly and sequentially, reusing only verified complete cases."""
    from modules.module_d.general import verify_bundle
    from analysis.module_d_flexibility import summarize_cycles
    for index, case in enumerate(plan["cases"], 1):
        expected, directory = job_plan(plan, campaign, case)
        if not directory.exists():
            run_case(plan, campaign, index, command)
            continue
        # Incomplete attempts are preserved and require a fresh campaign.
        if verify_bundle(directory) != expected:
            raise ValueError(f"Changed case plan: {case['case_id']}")
        status_path = directory / "results/status.json"
        if not status_path.is_file():
            raise ValueError(f"Incomplete prior attempt: {directory}")
        status = json.loads(status_path.read_text())
        if (status["status"] != "complete" or status["command"] != command or
                [row["case_id"] for row in status["cases"]] != [case["case_id"]]):
            raise ValueError(f"Incomplete case or changed LAMMPS command: {directory}")
        with (directory / "cycle_results.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if summarize_cycles(expected, rows)["status"] != "complete":
            raise ValueError(f"Incomplete measurements: {directory}")
        print(f"Already complete {index}/{len(plan['cases'])}: {case['case_id']}", flush=True)


def check_capabilities(help_text):
    required = {
        "Atom": {"full"}, "Pair": {"hybrid/overlay", "coul/long", "table"},
        "Bond": {"hybrid", "mm3", "table", "zero"},
        "Angle": {"hybrid", "mm3", "cosine/periodic", "cross"},
        "Dihedral": {"fourier"}, "Improper": {"distharm"},
        "KSpace": {"ewald"}, "Fix": {"gcmc", "rigid/nvt/small", "npt", "nvt"},
    }
    sections = re.split(r"\* ([A-Za-z]+) styles:?[ \t]*\n", help_text)
    available = {sections[i].lower(): set(sections[i+1].split())
                 for i in range(1, len(sections)-1, 2)}
    missing = [f"{category}:{style}" for category, styles in required.items()
               for style in sorted(styles - available.get(category.lower(), set()))]
    if missing:
        raise ValueError("LAMMPS lacks required styles: " + ", ".join(missing) +
                         ". Build with MOLECULE KSPACE RIGID MC YAFF EXTRA-MOLECULE; see README.md.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "smoke", "pilot", "run", "all", "case", "collect"], default="check", nargs="?")
    parser.add_argument("--case-index", type=int, help="1-based campaign case index")
    parser.add_argument("--directory", type=Path, help="Prepared campaign directory in the existing repository checkout")
    parser.add_argument("--command", nargs="+", default=["lmp"])
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    if args.directory is not None:
        campaign = args.directory.resolve()
        sys.path.insert(0, str(root.parent))
    else:
        campaign = root / "campaign"
        os.chdir(root)
        for name, expected in json.loads((root / "package_manifest.json").read_text()).items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid manifest path")
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("Package checksum mismatch: " + name)
    from modules.module_d.general import verify_bundle, execute_bundle
    from analysis.module_d_flexibility import evaluate_flexibility
    plan = verify_bundle(campaign)
    if plan["benchmark"]["applicability"]["blockers"]:
        raise ValueError(plan["benchmark"]["applicability"]["blockers"])
    if args.stage == "collect":
        report = collect_cases(plan, campaign)
        print(f"Collected {len(report['cases'])} cases: {campaign / 'evaluation/module_d_report.json'}", flush=True)
        return
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    result = subprocess.run([*args.command, "-help"], capture_output=True,
                            text=True, check=True, timeout=60)
    check_capabilities(result.stdout + result.stderr)
    print("PASS: inputs and required LAMMPS styles verified.", flush=True)
    if args.stage == "check":
        return
    if args.stage == "all":
        run_all_cases(plan, campaign, args.command)
        if not (campaign / "cycle_results.csv").exists():
            collect_cases(plan, campaign)
        print(f"Complete: {campaign / 'evaluation/module_d_report.json'}", flush=True)
        return
    if args.stage == "case":
        run_case(plan, campaign, args.case_index, args.command)
        print("PASS: case completed. Sampling and potential remain unvalidated.", flush=True)
        return
    execute_bundle(campaign, smoke=args.stage == "smoke", command=args.command)
    if args.stage in {"pilot", "run"}:
        report = evaluate_flexibility(campaign)
        if report["status"] != "complete":
            raise RuntimeError("Incomplete pilot evaluation")
        print("Results: campaign/evaluation/module_d_report.json", flush=True)
    print(f"PASS: {args.stage}. Technical execution only; potential and sampling remain unvalidated.", flush=True)


if __name__ == "__main__":
    main()
