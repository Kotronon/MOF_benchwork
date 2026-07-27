from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from time import perf_counter
from typing import Any

from pipeline.config import save_benchmark_data
from pipeline.evaluate import evaluate_isotherm
from pipeline.runners import (
    _aggregate_replicates,
    _convergence_report,
    _run_pressure_point,
    _write_convergence_csv,
)


def resume_isotherm(
    run_dir: str | Path,
    *,
    jobs: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resume a materialized run by rerunning only pressure points without summaries."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1.")
    run_path = Path(run_dir).resolve()
    prepare_path = run_path / "prepare_summary.json"
    if not prepare_path.exists():
        raise FileNotFoundError(f"Missing prepare summary: {prepare_path}")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    materialized = prepare.get("result")
    if not isinstance(materialized, dict):
        raise ValueError(f"Missing materialized run metadata in {prepare_path}")
    materialized = _absolute_materialized_plan(materialized, run_path)
    runs = materialized.get("files", {}).get("gcmc_runs", [])
    if not runs:
        raise ValueError(f"No materialized GCMC runs found in {prepare_path}")

    completed: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, run in enumerate(runs, start=1):
        summary_path = _summary_path(run)
        if summary_path.exists():
            completed[index] = _completed_result(run, summary_path, index)
        else:
            pending.append((index, run))

    plan = {
        "status": "dry_run" if dry_run else "planned",
        "run_directory": str(run_path),
        "jobs": jobs,
        "completed": [_run_label(runs[index - 1]) for index in sorted(completed)],
        "pending": [_run_label(run) for _index, run in pending],
        "completed_count": len(completed),
        "pending_count": len(pending),
        "total_count": len(runs),
    }
    if dry_run:
        return plan
    if not pending:
        return _finalize(materialized, completed, jobs, elapsed_seconds=0.0)

    archive_dir = _archive_interrupted_logs(run_path, [run for _index, run in pending])
    started_at = perf_counter()
    resumed: dict[int, dict[str, Any]] = {}
    discard_fraction = _discard_fraction(materialized)
    framework_atoms = int(materialized["parameters"]["framework_atom_count"])
    adsorbate_atoms = int(materialized["parameters"]["adsorbate_atoms_per_molecule"])

    if jobs == 1:
        for index, run in pending:
            resumed[index] = _run_pressure_point(
                run,
                index,
                len(runs),
                discard_fraction,
                framework_atoms,
                adsorbate_atoms,
            )
    else:
        print(
            f"Resuming {len(pending)} missing pressure points with {jobs} parallel jobs",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    _run_pressure_point,
                    run,
                    index,
                    len(runs),
                    discard_fraction,
                    framework_atoms,
                    adsorbate_atoms,
                ): index
                for index, run in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                resumed[index] = future.result()

    results = {**completed, **resumed}
    report = _finalize(
        materialized,
        results,
        jobs,
        elapsed_seconds=perf_counter() - started_at,
    )
    report["resume"] = {
        **plan,
        "status": "completed",
        "archive_directory": str(archive_dir) if archive_dir else None,
    }
    resume_path = run_path / "resume_report.json"
    save_benchmark_data(resume_path, report["resume"])
    report["resume_report"] = str(resume_path)
    return report


def _absolute_materialized_plan(plan: dict[str, Any], run_path: Path) -> dict[str, Any]:
    plan = dict(plan)
    files = dict(plan.get("files", {}))
    converted_runs = []
    for run in files.get("gcmc_runs", []):
        converted = dict(run)
        input_name = Path(str(run["path"])).name
        log_name = Path(str(run["log"])).name
        converted["path"] = str(run_path / "inputs" / input_name)
        converted["log"] = str(run_path / "logs" / log_name)
        if run.get("dump"):
            converted["dump"] = str(run_path / "dumps" / Path(str(run["dump"])).name)
        converted_runs.append(converted)
    files["gcmc_runs"] = converted_runs
    plan["files"] = files
    plan["working_directory"] = str(run_path)
    return plan


def _summary_path(run: dict[str, Any]) -> Path:
    log_path = Path(str(run["log"]))
    return log_path.with_name(log_path.stem + "_summary.json")


def _completed_result(
    run: dict[str, Any], summary_path: Path, index: int
) -> dict[str, Any]:
    log_path = Path(str(run["log"]))
    return {
        "pressure_bar": float(run["pressure_bar"]),
        "replicate_index": int(run.get("replicate_index", index)),
        "seed": int(run.get("seed", 12345)),
        "input_script": str(run["path"]),
        "command": run.get("command"),
        "log_file": str(log_path),
        "summary_file": str(summary_path),
        "discard_fraction": None,
        "wall_time_seconds": _log_wall_time_seconds(log_path),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def _archive_interrupted_logs(
    run_path: Path, pending_runs: list[dict[str, Any]]
) -> Path | None:
    existing = [Path(str(run["log"])) for run in pending_runs if Path(str(run["log"])).exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = run_path / "logs" / f"interrupted_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    for log_path in existing:
        shutil.move(str(log_path), archive / log_path.name)
    return archive


def _finalize(
    materialized: dict[str, Any],
    results_by_index: dict[int, dict[str, Any]],
    jobs: int,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    runs = materialized["files"]["gcmc_runs"]
    missing = [index for index in range(1, len(runs) + 1) if index not in results_by_index]
    if missing:
        raise ValueError(f"Cannot finalize; missing run indices: {missing}")
    results = [results_by_index[index] for index in range(1, len(runs) + 1)]
    convergence_config = materialized.get("convergence", {})
    aggregated = _aggregate_replicates(results, convergence_config)
    convergence = _convergence_report(aggregated, convergence_config)
    summary = {
        "status": "completed",
        "jobs": jobs,
        "wall_time_seconds": elapsed_seconds,
        "results": aggregated,
        "replicate_results": results,
        "convergence": convergence,
    }
    run_path = Path(str(materialized["working_directory"]))
    summary_path = run_path / "isotherm_summary.json"
    convergence_json = run_path / "convergence_report.json"
    convergence_csv = run_path / "convergence_report.csv"
    save_benchmark_data(summary_path, summary)
    save_benchmark_data(convergence_json, convergence)
    _write_convergence_csv(convergence_csv, aggregated)
    evaluation = evaluate_isotherm(
        summary_path,
        evaluation_config=materialized.get("evaluation"),
    )
    return {
        **summary,
        "summary_file": str(summary_path),
        "convergence_report": str(convergence_json),
        "convergence_csv": str(convergence_csv),
        "evaluation": evaluation,
    }


def _discard_fraction(materialized: dict[str, Any]) -> float:
    parameters = materialized["parameters"]
    equilibration = int(parameters["equilibration_steps"])
    production = int(parameters["production_steps"])
    return equilibration / (equilibration + production)


def _log_wall_time_seconds(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Total wall time:\s*(?:(\d+)-)?(\d+):(\d+):(\d+)", text)
    if match is None:
        return None
    days, hours, minutes, seconds = (int(value or 0) for value in match.groups())
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


def _run_label(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "pressure_bar": float(run["pressure_bar"]),
        "seed": int(run.get("seed", 12345)),
        "input": str(run["path"]),
        "log": str(run["log"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rerun only missing pressure points in an interrupted materialized isotherm."
    )
    parser.add_argument("run_dir")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show completed and pending pressure points without changing files or starting LAMMPS.",
    )
    args = parser.parse_args(argv)
    report = resume_isotherm(args.run_dir, jobs=args.jobs, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
