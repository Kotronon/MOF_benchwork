from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import math
from pathlib import Path
from statistics import mean, stdev
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
        framework_atoms=_framework_atom_count(materialized_plan),
        adsorbate_atoms_per_molecule=_adsorbate_atom_count(materialized_plan),
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
    framework_atoms = _framework_atom_count(materialized_plan)
    adsorbate_atoms_per_molecule = _adsorbate_atom_count(materialized_plan)
    if jobs == 1:
        results = [
            _run_pressure_point(
                run,
                index,
                total_runs,
                discard_fraction,
                framework_atoms,
                adsorbate_atoms_per_molecule,
            )
            for index, run in enumerate(runs, start=1)
        ]
    else:
        print(f"Running {total_runs} pressure points with {jobs} parallel jobs", flush=True)
        indexed_runs = list(enumerate(runs, start=1))
        results_by_index: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    _run_pressure_point,
                    run,
                    index,
                    total_runs,
                    discard_fraction,
                    framework_atoms,
                    adsorbate_atoms_per_molecule,
                ): index
                for index, run in indexed_runs
            }
            for future in as_completed(futures):
                index = futures[future]
                results_by_index[index] = future.result()
        results = [results_by_index[index] for index, _run in indexed_runs]

    convergence_config = materialized_plan.get("convergence", {})
    aggregated_results = _aggregate_replicates(results, convergence_config)
    convergence_report = _convergence_report(aggregated_results, convergence_config)
    isotherm_summary = {
        "status": "completed",
        "jobs": jobs,
        "results": aggregated_results,
        "replicate_results": results,
        "convergence": convergence_report,
    }

    output_path = Path(materialized_plan["working_directory"]) / "isotherm_summary.json"
    save_benchmark_data(output_path, isotherm_summary)
    convergence_json = Path(materialized_plan["working_directory"]) / "convergence_report.json"
    convergence_csv = Path(materialized_plan["working_directory"]) / "convergence_report.csv"
    save_benchmark_data(convergence_json, convergence_report)
    _write_convergence_csv(convergence_csv, aggregated_results)

    evaluation = evaluate_isotherm(output_path, evaluation_config=materialized_plan.get("evaluation"))

    return {
        **isotherm_summary,
        "summary_file": str(output_path),
        "convergence_report": str(convergence_json),
        "convergence_csv": str(convergence_csv),
        "evaluation": evaluation,
    }


def _run_pressure_point(
    run: dict[str, Any],
    index: int,
    total_runs: int,
    discard_fraction: float,
    framework_atoms: int,
    adsorbate_atoms_per_molecule: int,
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
        framework_atoms=framework_atoms,
        adsorbate_atoms_per_molecule=adsorbate_atoms_per_molecule,
        discard_fraction=discard_fraction,
    )

    print(f"[{index}/{total_runs}] Finished {pressure_bar} bar", flush=True)
    return {
        "pressure_bar": pressure_bar,
        "replicate_index": run.get("replicate_index", 1),
        "seed": run.get("seed", 12345),
        "input_script": input_script,
        "command": command,
        "log_file": str(log_file),
        "summary_file": str(summary_file),
        "discard_fraction": discard_fraction,
        "summary": summary,
    }


def _aggregate_replicates(
    results: list[dict[str, Any]],
    convergence_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate independent seed replicates into one result per pressure."""
    config = convergence_config or {}
    minimum_replicates = int(config.get("minimum_replicates", 3))
    relative_target = float(config.get("relative_ci95_target", 0.05))
    grouped: dict[float, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(float(result["pressure_bar"]), []).append(result)

    aggregated = []
    for pressure_bar, replicates in grouped.items():
        replicates.sort(key=lambda item: (int(item.get("replicate_index", 1)), int(item.get("seed", 0))))
        values = [float(item["summary"]["mean_adsorbates"]) for item in replicates]
        replicate_count = len(values)
        average = mean(values)
        standard_deviation = stdev(values) if replicate_count >= 2 else None
        standard_error = standard_deviation / math.sqrt(replicate_count) if standard_deviation is not None else None
        ci95_half_width = (
            _t_critical_975(replicate_count - 1) * standard_error
            if standard_error is not None
            else None
        )
        relative_ci95 = (
            ci95_half_width / abs(average)
            if ci95_half_width is not None and average != 0
            else None
        )
        converged = (
            replicate_count >= minimum_replicates
            and relative_ci95 is not None
            and relative_ci95 <= relative_target
        )
        sample_summary = replicates[0]["summary"]
        aggregated.append(
            {
                "pressure_bar": pressure_bar,
                "replicate_count": replicate_count,
                "seeds": [int(item.get("seed", 12345)) for item in replicates],
                "converged": converged,
                "summary": {
                    **sample_summary,
                    "sample_count": sum(int(item["summary"].get("sample_count", 0)) for item in replicates),
                    "inserted": any(bool(item["summary"].get("inserted")) for item in replicates),
                    "max_atoms": max(int(item["summary"].get("max_atoms", 0)) for item in replicates),
                    "max_adsorbates": max(float(item["summary"].get("max_adsorbates", 0)) for item in replicates),
                    "mean_adsorbates": average,
                    "replicate_mean_adsorbates": values,
                    "standard_deviation_adsorbates": standard_deviation,
                    "standard_error_adsorbates": standard_error,
                    "ci95_half_width_adsorbates": ci95_half_width,
                    "ci95_lower_adsorbates": average - ci95_half_width if ci95_half_width is not None else None,
                    "ci95_upper_adsorbates": average + ci95_half_width if ci95_half_width is not None else None,
                    "relative_ci95_half_width": relative_ci95,
                },
            }
        )
    return sorted(aggregated, key=lambda item: item["pressure_bar"])


def _convergence_report(
    results: list[dict[str, Any]],
    convergence_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = convergence_config or {}
    converged_count = sum(bool(result["converged"]) for result in results)
    return {
        "criterion": "two-sided_95_percent_t_interval_relative_half_width",
        "minimum_replicates": int(config.get("minimum_replicates", 3)),
        "relative_ci95_target": float(config.get("relative_ci95_target", 0.05)),
        "pressure_point_count": len(results),
        "converged_pressure_point_count": converged_count,
        "all_pressure_points_converged": bool(results) and converged_count == len(results),
        "pressure_points": [
            {
                "pressure_bar": result["pressure_bar"],
                "replicate_count": result["replicate_count"],
                "seeds": result["seeds"],
                "mean_adsorbates": result["summary"]["mean_adsorbates"],
                "standard_deviation_adsorbates": result["summary"]["standard_deviation_adsorbates"],
                "standard_error_adsorbates": result["summary"]["standard_error_adsorbates"],
                "ci95_half_width_adsorbates": result["summary"]["ci95_half_width_adsorbates"],
                "relative_ci95_half_width": result["summary"]["relative_ci95_half_width"],
                "converged": result["converged"],
            }
            for result in results
        ],
    }


def _write_convergence_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pressure_bar", "replicate_count", "seeds", "mean_adsorbates",
        "standard_deviation_adsorbates", "standard_error_adsorbates",
        "ci95_half_width_adsorbates", "relative_ci95_half_width", "converged",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            summary = result["summary"]
            writer.writerow(
                {
                    "pressure_bar": result["pressure_bar"],
                    "replicate_count": result["replicate_count"],
                    "seeds": ";".join(str(seed) for seed in result["seeds"]),
                    "mean_adsorbates": summary["mean_adsorbates"],
                    "standard_deviation_adsorbates": summary["standard_deviation_adsorbates"],
                    "standard_error_adsorbates": summary["standard_error_adsorbates"],
                    "ci95_half_width_adsorbates": summary["ci95_half_width_adsorbates"],
                    "relative_ci95_half_width": summary["relative_ci95_half_width"],
                    "converged": result["converged"],
                }
            )


def _t_critical_975(degrees_of_freedom: int) -> float:
    """Return the two-sided 95% Student-t critical value without SciPy."""
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive.")
    return table.get(degrees_of_freedom, 1.96)


def _discard_fraction(materialized_plan: dict[str, Any]) -> float:
    parameters = materialized_plan.get("parameters", {})
    run_steps = int(parameters.get("run_steps", 0))
    equilibration_steps = int(parameters.get("equilibration_steps", 0))
    if run_steps <= 0 or equilibration_steps <= 0:
        return 0.0
    return min(equilibration_steps / run_steps, 0.999999)


def _framework_atom_count(materialized_plan: dict[str, Any]) -> int:
    count = int(materialized_plan.get("parameters", {}).get("framework_atom_count", 0))
    if count <= 0:
        raise ValueError("Materialized plan is missing a positive framework_atom_count.")
    return count


def _adsorbate_atom_count(materialized_plan: dict[str, Any]) -> int:
    count = int(
        materialized_plan.get("parameters", {}).get("adsorbate_atoms_per_molecule", 0)
    )
    if count <= 0:
        raise ValueError("Materialized plan is missing a positive adsorbate_atoms_per_molecule.")
    return count
