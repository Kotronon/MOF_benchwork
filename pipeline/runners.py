from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
from typing import Any

from engines.lammps_runner import build_lammps_command
from parsers.lammps_log_parser import log_to_json
from pipeline.config import save_benchmark_data
from pipeline.evaluate import evaluate_isotherm


def run_benchmark(materialized_plan: dict[str, Any]) -> dict[str, Any]:
    gcmc_input = materialized_plan["files"]["gcmc_test_input"]
    log_file = Path(materialized_plan["working_directory"]) / "logs" / "gcmc_test.log"
    summary_file = Path(materialized_plan["working_directory"]) / "gcmc_test_summary.json"

    command = build_lammps_command(gcmc_input, log_file=log_file)
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr + result.stdout)

    summary = log_to_json(
        log_file,
        summary_file,
        framework_atoms=106,
        adsorbate_atoms_per_molecule=3,
        discard_fraction=_discard_fraction(materialized_plan),
    )

    return {
        "status": "completed",
        "command": command,
        "log_file": str(log_file),
        "summary_file": str(summary_file),
        "discard_fraction": _discard_fraction(materialized_plan),
        "summary": summary,
    }


def run_isotherm(materialized_plan: dict[str, Any], jobs: int = 1) -> dict[str, Any]:
    """Run all materialized pressure-point GCMC inputs."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1.")

    runs = materialized_plan["files"]["gcmc_runs"]
    total_runs = len(runs)
    discard_fraction = _discard_fraction(materialized_plan)
    if jobs == 1:
        results = [
            _run_pressure_point(run, index, total_runs, discard_fraction)
            for index, run in enumerate(runs, start=1)
        ]
    else:
        print(f"Running {total_runs} pressure points with {jobs} parallel jobs", flush=True)
        indexed_runs = list(enumerate(runs, start=1))
        results_by_index: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_run_pressure_point, run, index, total_runs, discard_fraction): index
                for index, run in indexed_runs
            }
            for future in as_completed(futures):
                index = futures[future]
                results_by_index[index] = future.result()
        results = [results_by_index[index] for index, _run in indexed_runs]

    isotherm_summary = {
        "status": "completed",
        "jobs": jobs,
        "results": results,
    }

    output_path = Path(materialized_plan["working_directory"]) / "isotherm_summary.json"
    save_benchmark_data(output_path, isotherm_summary)

    evaluation = evaluate_isotherm(output_path, evaluation_config=materialized_plan.get("evaluation"))

    return {
        **isotherm_summary,
        "summary_file": str(output_path),
        "evaluation": evaluation,
    }


def _run_pressure_point(
    run: dict[str, Any],
    index: int,
    total_runs: int,
    discard_fraction: float,
) -> dict[str, Any]:
    input_script = run["path"]
    pressure_bar = run["pressure_bar"]
    log_file = Path(run["log"])
    summary_file = log_file.with_name(log_file.stem + "_summary.json")

    print(f"[{index}/{total_runs}] Running {pressure_bar} bar", flush=True)

    command = build_lammps_command(input_script, log_file=log_file)
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr + result.stdout)

    summary = log_to_json(
        log_file,
        summary_file,
        framework_atoms=106,
        adsorbate_atoms_per_molecule=3,
        discard_fraction=discard_fraction,
    )

    print(f"[{index}/{total_runs}] Finished {pressure_bar} bar", flush=True)
    return {
        "pressure_bar": pressure_bar,
        "input_script": input_script,
        "command": command,
        "log_file": str(log_file),
        "summary_file": str(summary_file),
        "discard_fraction": discard_fraction,
        "summary": summary,
    }


def _discard_fraction(materialized_plan: dict[str, Any]) -> float:
    parameters = materialized_plan.get("parameters", {})
    run_steps = int(parameters.get("run_steps", 0))
    equilibration_steps = int(parameters.get("equilibration_steps", 0))
    if run_steps <= 0 or equilibration_steps <= 0:
        return 0.0
    return min(equilibration_steps / run_steps, 0.999999)
