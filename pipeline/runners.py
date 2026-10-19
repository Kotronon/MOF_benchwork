from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import math
import re
from pathlib import Path
from statistics import mean, stdev
import subprocess
from time import perf_counter
from typing import Any

from engines.lammps_runner import build_lammps_command
from parsers.lammps_log_parser import get_log, log_to_json, parse_lammps_log
from pipeline.config import load_benchmark_data, save_benchmark_data
from pipeline.evaluate import evaluate_isotherm
from pipeline.lammps_inputs import gcmc_input_builder
from pipeline.mixture_statistics import fieller_interval, mean_interval, precision_met


def run_benchmark(materialized_plan: dict[str, Any]) -> dict[str, Any]:
    runs = materialized_plan["files"].get("gcmc_runs", [])
    if runs and runs[0].get("sampler") == "cbmc":
        from engines.flexible_guest_runner import run_pressure_point
        return run_pressure_point(runs[0], 1, 1, _discard_fraction(materialized_plan))
    gcmc_input = materialized_plan["files"]["gcmc_test_input"]
    log_file = Path(materialized_plan["working_directory"]) / "logs" / "gcmc_test.log"
    summary_file = Path(materialized_plan["working_directory"]) / "gcmc_test_summary.json"

    command = build_lammps_command(gcmc_input, log_file=log_file)
    started_at = perf_counter()
    result = subprocess.run(command, capture_output=True, text=True)
    wall_time_seconds = perf_counter() - started_at

    if result.returncode != 0:
        raise RuntimeError(result.stderr + result.stdout)

    summary = log_to_json(
        log_file,
        summary_file,
        framework_atoms=_framework_atom_count(materialized_plan),
        adsorbate_atoms_per_molecule=_adsorbate_atom_count(materialized_plan),
        discard_fraction=_discard_fraction(materialized_plan),
        component_atom_counts=_component_atom_counts(materialized_plan),
        mixture=_mixture(materialized_plan),
    )

    return {
        "status": "completed",
        "command": command,
        "log_file": str(log_file),
        "summary_file": str(summary_file),
        "discard_fraction": _discard_fraction(materialized_plan),
        "wall_time_seconds": wall_time_seconds,
        "summary": summary,
    }


def run_isotherm(materialized_plan: dict[str, Any], jobs: int = 1) -> dict[str, Any]:
    """Run all materialized pressure-point GCMC inputs."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1.")

    isotherm_started_at = perf_counter()
    runs = materialized_plan["files"]["gcmc_runs"]
    if jobs > 1 and any(run.get("sampler") == "cbmc" for run in runs):
        raise ValueError("The LAMMPS-library CBMC backend runs serially (jobs=1); use separate processes for independent campaigns.")
    total_runs = len(runs)
    discard_fraction = _discard_fraction(materialized_plan)
    framework_atoms = _framework_atom_count(materialized_plan)
    adsorbate_atoms_per_molecule = _adsorbate_atom_count(materialized_plan)
    component_atom_counts = _component_atom_counts(materialized_plan)
    mixture = _mixture(materialized_plan)
    resume = bool(materialized_plan.get("resume", False))
    expected_steps = int(materialized_plan.get("parameters", {}).get("run_steps", 0) or 0)
    indexed_runs = list(enumerate(runs, start=1))
    results_by_index: dict[int, dict[str, Any]] = {}
    pending_runs: list[tuple[int, dict[str, Any]]] = []

    for index, run in indexed_runs:
        existing = None
        if resume:
            existing = _existing_pressure_point_result(
                run,
                index,
                total_runs,
                discard_fraction,
                framework_atoms,
                adsorbate_atoms_per_molecule,
                expected_steps,
                component_atom_counts=component_atom_counts,
                mixture=mixture,
            )
        if existing is None:
            run_to_submit = dict(run)
            if resume:
                restart_file = _latest_restart_file(run)
                if restart_file is not None:
                    run_to_submit["read_restart"] = str(restart_file)
            pending_runs.append((index, run_to_submit))
        else:
            results_by_index[index] = existing

    if resume:
        reused_count = total_runs - len(pending_runs)
        print(
            f"Resuming isotherm: reusing {reused_count} completed pressure points; "
            f"running {len(pending_runs)} remaining points",
            flush=True,
        )

    if jobs == 1:
        for index, run in pending_runs:
            results_by_index[index] = _run_pressure_point(
                run,
                index,
                total_runs,
                discard_fraction,
                framework_atoms,
                adsorbate_atoms_per_molecule,
                component_atom_counts,
                mixture,
            )
    elif pending_runs:
        print(f"Running {len(pending_runs)} of {total_runs} pressure points with {jobs} parallel jobs", flush=True)
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
                    component_atom_counts,
                    mixture,
                ): index
                for index, run in pending_runs
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
        "wall_time_seconds": perf_counter() - isotherm_started_at,
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
    mixture_csv = _write_mixture_results_csv(
        Path(materialized_plan["working_directory"]) / "mixture_results.csv",
        aggregated_results,
    )
    selectivity_csv = _write_selectivity_surface_csv(
        Path(materialized_plan["working_directory"]) / "selectivity_surface.csv",
        aggregated_results,
    )

    evaluation = evaluate_isotherm(output_path, evaluation_config=materialized_plan.get("evaluation"))

    return {
        **isotherm_summary,
        "summary_file": str(output_path),
        "convergence_report": str(convergence_json),
        "convergence_csv": str(convergence_csv),
        "mixture_results_csv": str(mixture_csv) if mixture_csv else None,
        "selectivity_surface_csv": str(selectivity_csv) if selectivity_csv else None,
        "evaluation": evaluation,
    }


def _run_pressure_point(
    run: dict[str, Any],
    index: int,
    total_runs: int,
    discard_fraction: float,
    framework_atoms: int,
    adsorbate_atoms_per_molecule: int,
    component_atom_counts: dict[str, int] | None = None,
    mixture: dict[str, float] | None = None,
) -> dict[str, Any]:
    if run.get("sampler") == "cbmc":
        from engines.flexible_guest_runner import run_pressure_point
        return run_pressure_point(run, index, total_runs, discard_fraction)
    input_script = run["path"]
    pressure_bar = run["pressure_bar"]
    base_log_file = Path(run["log"])
    summary_file = base_log_file.with_name(base_log_file.stem + "_summary.json")
    read_restart = run.get("read_restart")
    actual_log_file = base_log_file
    if read_restart:
        restart_step = _restart_step(Path(read_restart)) or "latest"
        input_script = _write_resume_input(run, Path(read_restart))
        actual_log_file = base_log_file.with_name(f"{base_log_file.stem}_resume_{restart_step}.log")
        print(
            f"[{index}/{total_runs}] Resuming {pressure_bar} bar from {read_restart}",
            flush=True,
        )
    else:
        print(f"[{index}/{total_runs}] Running {pressure_bar} bar", flush=True)

    command = build_lammps_command(input_script, log_file=actual_log_file)
    started_at = perf_counter()
    result = subprocess.run(command, capture_output=True, text=True)
    wall_time_seconds = perf_counter() - started_at

    if result.returncode != 0:
        raise RuntimeError(result.stderr + result.stdout)

    summary_log_file = _combine_pressure_point_logs(base_log_file) if read_restart else base_log_file
    summary = log_to_json(
        summary_log_file,
        summary_file,
        framework_atoms=framework_atoms,
        adsorbate_atoms_per_molecule=adsorbate_atoms_per_molecule,
        discard_fraction=discard_fraction,
        component_atom_counts=component_atom_counts,
        mixture=mixture,
    )

    print(f"[{index}/{total_runs}] Finished {pressure_bar} bar", flush=True)
    return _pressure_point_result(
        run,
        command,
        summary_log_file,
        summary_file,
        discard_fraction,
        wall_time_seconds=wall_time_seconds,
        summary=summary,
        status="resumed_from_restart" if read_restart else "completed",
    )


def _write_resume_input(run: dict[str, Any], restart_file: Path) -> str:
    resume_input = Path(run.get("resume_input") or Path(run["path"]).with_suffix(".resume.in"))
    resume_input.parent.mkdir(parents=True, exist_ok=True)
    resume_input.write_text(
        gcmc_input_builder(
            {
                "framework_data": run.get("framework_data", "unused_for_restart"),
                "molecule_templates": run["molecule_templates"],
                "forcefield_include": run["forcefield_include"],
                "framework_atom_types": run["framework_atom_types"],
                "adsorbate_atom_types": run["adsorbate_atom_types"],
                "component_adsorbate_atom_types": run.get("component_adsorbate_atom_types", {}),
                "component_atoms_per_molecule": run.get("component_atoms_per_molecule", {}),
                "components": run.get("components", [run["component"]]),
                "mixture": run.get("mixture"),
                "component": run["component"],
                "temperature_K": run["temperature_K"],
                "pressure_bar": run["pressure_bar"],
                "chemical_potential_kcal_mol": run.get("chemical_potential_kcal_mol", 0.0),
                "displacement_A": run.get("displacement_A", 1.0),
                "run_steps": run["run_steps"],
                "gcmc_every_steps": run.get("gcmc_every_steps", 1),
                "exchange_attempts": run.get("exchange_attempts", 10),
                "move_attempts": run.get("move_attempts", 10),
                "seed": run.get("seed", 12345),
                "extra_bond_per_atom": run.get("extra_bond_per_atom", 0),
                "extra_special_per_atom": run.get("extra_special_per_atom", 0),
                "fugacity_coeff": run.get("fugacity_coeff", 1.0),
                "fugacity_coefficients": run.get("fugacity_coefficients", {}),
                "dump_file": run.get("dump"),
                "dump_every_steps": run.get("dump_every_steps", 1000),
                "restart_file": run.get("restart"),
                "restart_every_steps": run.get("restart_every_steps", 0),
                "read_restart_file": str(restart_file),
            }
        ),
        encoding="utf-8",
    )
    return str(resume_input)


def _existing_pressure_point_result(
    run: dict[str, Any],
    index: int,
    total_runs: int,
    discard_fraction: float,
    framework_atoms: int,
    adsorbate_atoms_per_molecule: int,
    expected_steps: int,
    component_atom_counts: dict[str, int] | None = None,
    mixture: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    if run.get("sampler") == "cbmc":
        from engines.flexible_guest_runner import run_pressure_point
        return run_pressure_point(run, index, total_runs, discard_fraction)
    input_script = run["path"]
    pressure_bar = run["pressure_bar"]
    log_file = Path(run["log"])
    summary_file = log_file.with_name(log_file.stem + "_summary.json")
    completed_log = _completed_pressure_point_log(log_file, expected_steps)
    command = build_lammps_command(input_script, log_file=log_file)

    if summary_file.exists():
        if log_file.exists() and completed_log is None:
            print(
                f"[{index}/{total_runs}] Existing summary for {pressure_bar} bar ignored "
                "because the log is incomplete",
                flush=True,
            )
            return None
        print(f"[{index}/{total_runs}] Reusing completed {pressure_bar} bar", flush=True)
        return _pressure_point_result(
            run,
            command,
            completed_log or log_file,
            summary_file,
            discard_fraction,
            wall_time_seconds=None,
            summary=load_benchmark_data(summary_file),
            status="reused_summary",
        )

    if completed_log is not None:
        summary = log_to_json(
            completed_log,
            summary_file,
            framework_atoms=framework_atoms,
            adsorbate_atoms_per_molecule=adsorbate_atoms_per_molecule,
            discard_fraction=discard_fraction,
            component_atom_counts=component_atom_counts,
            mixture=mixture,
        )
        print(f"[{index}/{total_runs}] Parsed completed {pressure_bar} bar log", flush=True)
        return _pressure_point_result(
            run,
            command,
            completed_log,
            summary_file,
            discard_fraction,
            wall_time_seconds=None,
            summary=summary,
            status="reused_log",
        )

    return None


def _completed_pressure_point_log(base_log_file: Path, expected_steps: int) -> Path | None:
    combined_log = base_log_file.with_name(f"{base_log_file.stem}_combined.log")
    if combined_log.exists() and _lammps_log_completed(combined_log, expected_steps):
        return combined_log
    if _lammps_log_completed(base_log_file, expected_steps):
        return base_log_file
    return None


def _latest_restart_file(run: dict[str, Any]) -> Path | None:
    restart_pattern = run.get("restart")
    if not restart_pattern:
        return None
    pattern_path = Path(restart_pattern)
    candidates = [path for path in pattern_path.parent.glob(pattern_path.name) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (_restart_step(path) or -1, path.stat().st_mtime))


def _restart_step(path: Path) -> int | None:
    matches = re.findall(r"(\d+)", path.name)
    return int(matches[-1]) if matches else None


def _pressure_point_log_segments(base_log_file: Path) -> list[Path]:
    resume_logs = sorted(
        base_log_file.parent.glob(f"{base_log_file.stem}_resume_*.log"),
        key=lambda path: (_restart_step(path) or -1, path.stat().st_mtime),
    )
    return [path for path in [base_log_file, *resume_logs] if path.exists()]


def _combine_pressure_point_logs(base_log_file: Path) -> Path:
    combined_log = base_log_file.with_name(f"{base_log_file.stem}_combined.log")
    with combined_log.open("w", encoding="utf-8") as output:
        for segment in _pressure_point_log_segments(base_log_file):
            output.write(get_log(segment))
            output.write("\n")
    return combined_log


def _lammps_log_completed(log_file: Path, expected_steps: int) -> bool:
    try:
        log_data = get_log(log_file)
    except FileNotFoundError:
        return False
    if "Loop time" not in log_data:
        return False
    rows = parse_lammps_log(log_data)["rows"]
    if not rows:
        return False
    if expected_steps <= 0:
        return True
    steps = [int(row["Step"]) for row in rows if "Step" in row]
    return bool(steps) and max(steps) >= expected_steps


def _pressure_point_result(
    run: dict[str, Any],
    command: list[str],
    log_file: Path,
    summary_file: Path,
    discard_fraction: float,
    wall_time_seconds: float | None,
    summary: dict[str, Any],
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "status": status,
        "pressure_bar": run["pressure_bar"],
        "replicate_index": run.get("replicate_index", 1),
        "seed": run.get("seed", 12345),
        "input_script": run["path"],
        "command": command,
        "log_file": str(log_file),
        "summary_file": str(summary_file),
        "discard_fraction": discard_fraction,
        "wall_time_seconds": wall_time_seconds,
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
        values = [
            sum(float(c["mean_molecules"]) for c in item["summary"]["components"].values())
            if len(item["summary"].get("components", {})) > 1
            else float(item["summary"]["mean_adsorbates"])
            for item in replicates
        ]
        wall_times = [
            float(item["wall_time_seconds"])
            for item in replicates
            if item.get("wall_time_seconds") is not None
        ]
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
        component_summary = _aggregate_component_summaries(replicates)
        if component_summary:
            seeds = [item.get("seed") for item in replicates]
            if None in seeds or len(set(seeds)) != len(seeds):
                raise ValueError("Mixture uncertainty requires distinct, identified seed replicates.")
            statistics = [v for v in component_summary["components"].values() if v["gas_phase_mole_fraction"] > 0]
            requested_pairs = config.get("selectivity_pairs")
            if requested_pairs is not None and (not requested_pairs or set(requested_pairs) - set(component_summary["selectivity"])):
                raise ValueError("selectivity_pairs must name existing component ratios.")
            mixture_config = {"relative_ci95_target": 0.2, **config}
            for statistic in [*component_summary["components"].values(), *component_summary["selectivity"].values()]:
                statistic["precision_met"] = precision_met(statistic, replicate_count, mixture_config)
            statistics += [v for pair,v in component_summary["selectivity"].items()
                           if (pair in requested_pairs if requested_pairs is not None else pair.split("/") == sorted(pair.split("/")))
                           and all(component_summary["components"][c]["gas_phase_mole_fraction"] > 0 for c in pair.split("/"))]
            converged = all(statistic["precision_met"] for statistic in statistics)
        aggregated.append(
            {
                "pressure_bar": pressure_bar,
                "replicate_count": replicate_count,
                "seeds": [int(item.get("seed", 12345)) for item in replicates],
                "converged": converged,
                **({"precision_met": converged, "convergence_scope": "between_seed_precision_only"} if component_summary else {}),
                "mean_wall_time_seconds": mean(wall_times) if wall_times else None,
                "total_wall_time_seconds": sum(wall_times) if wall_times else None,
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
                    **component_summary,
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
        "criterion": ("component_t_and_selectivity_fieller_relative_half_width"
                      if any("components" in r["summary"] for r in results)
                      else "two-sided_95_percent_t_interval_relative_half_width"),
        "interpretation": "Between-seed precision; not a proof of equilibrium or time-series convergence.",
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
                **({"components": result["summary"]["components"],
                    "selectivity": result["summary"]["selectivity"],
                    "precision_met": result["converged"]} if "components" in result["summary"] else {}),
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


def _write_mixture_results_csv(path: Path, results: list[dict[str, Any]]) -> Path | None:
    if not any("components" in result.get("summary", {}) for result in results):
        return None
    component_names = sorted({
        component
        for result in results
        for component in result.get("summary", {}).get("components", {})
    })
    fieldnames = ["pressure_bar", "replicate_count", "seeds"]
    for component in component_names:
        fieldnames.extend(
            [
                f"y_{component}",
                f"mean_{component}_molecules_per_cell",
                f"max_{component}_molecules_per_cell",
                f"standard_error_{component}_molecules",
            ]
        )
    for left in component_names:
        for right in component_names:
            if left != right:
                fieldnames.append(f"selectivity_{left}_{right}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            summary = result.get("summary", {})
            components = summary.get("components", {})
            row: dict[str, Any] = {
                "pressure_bar": result["pressure_bar"],
                "replicate_count": result["replicate_count"],
                "seeds": ";".join(str(seed) for seed in result["seeds"]),
            }
            for component in component_names:
                component_summary = components.get(component, {})
                row[f"y_{component}"] = component_summary.get("gas_phase_mole_fraction")
                row[f"mean_{component}_molecules_per_cell"] = component_summary.get("mean_molecules")
                row[f"max_{component}_molecules_per_cell"] = component_summary.get("max_molecules")
                row[f"standard_error_{component}_molecules"] = component_summary.get("standard_error_molecules")
            for key, selectivity in summary.get("selectivity", {}).items():
                row[f"selectivity_{key.replace('/', '_')}"] = selectivity.get("value")
            writer.writerow(row)
    return path


def _write_selectivity_surface_csv(path: Path, results: list[dict[str, Any]]) -> Path | None:
    if not any("selectivity" in result.get("summary", {}) for result in results):
        return None
    fieldnames = ["pressure_bar", "selectivity", "value", "reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for key, selectivity in result.get("summary", {}).get("selectivity", {}).items():
                writer.writerow(
                    {
                        "pressure_bar": result["pressure_bar"],
                        "selectivity": key,
                        "value": selectivity.get("value"),
                        "reason": selectivity.get("reason", ""),
                    }
                )
    return path


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


def _component_atom_counts(materialized_plan: dict[str, Any]) -> dict[str, int] | None:
    counts = materialized_plan.get("parameters", {}).get("component_atoms_per_molecule")
    if not counts:
        return None
    return {str(component).upper(): int(count) for component, count in counts.items()}


def _mixture(materialized_plan: dict[str, Any]) -> dict[str, float] | None:
    mixture = materialized_plan.get("parameters", {}).get("mixture")
    if not mixture:
        return None
    return {str(component).upper(): float(fraction) for component, fraction in mixture.items()}


def _aggregate_component_summaries(replicates: list[dict[str, Any]]) -> dict[str, Any]:
    if not replicates or "components" not in replicates[0].get("summary", {}):
        return {}

    component_names = list(replicates[0]["summary"]["components"])
    if any(set(item["summary"].get("components", {})) != set(component_names) for item in replicates):
        raise ValueError("Mixture replicates must contain the same components.")
    t_critical = _t_critical_975(len(replicates) - 1) if len(replicates) > 1 else None
    components: dict[str, dict[str, Any]] = {}
    for component in component_names:
        values = [
            float(replicate["summary"]["components"][component]["mean_molecules"])
            for replicate in replicates
        ]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Mixture component means must be finite and nonnegative.")
        component_sample = replicates[0]["summary"]["components"][component]
        fractions = [item["summary"]["components"][component].get("gas_phase_mole_fraction") for item in replicates]
        if any(value != fractions[0] for value in fractions):
            raise ValueError("Cannot aggregate mixture seeds with different gas compositions.")
        component_mean = mean(values)
        components[component] = {
            **component_sample,
            "mean_molecules": component_mean,
            "replicate_mean_molecules": values,
            **mean_interval(values, t_critical),
            "standard_deviation_molecules": stdev(values) if len(values) >= 2 else None,
            "standard_error_molecules": (
                stdev(values) / math.sqrt(len(values)) if len(values) >= 2 else None
            ),
            "max_molecules": max(
                float(replicate["summary"]["components"][component].get("max_molecules", 0.0))
                for replicate in replicates
            ),
            "sample_count": sum(
                int(replicate["summary"]["components"][component].get("sample_count", 0))
                for replicate in replicates
            ),
        }

    mixture = {
        component: float(summary.get("gas_phase_mole_fraction", 0.0) or 0.0)
        for component, summary in components.items()
    }
    if any(not math.isfinite(y) or y < 0 for y in mixture.values()) or not math.isclose(sum(mixture.values()), 1.0, abs_tol=1e-8):
        raise ValueError("Mixture gas fractions must be positive and sum to one.")
    selectivity = _selectivity_from_components(components, mixture)
    for pair, statistic in selectivity.items():
        left, right = pair.split("/")
        if mixture[left] == 0 or mixture[right] == 0:
            statistic.update(value=None, ci95_lower=None, ci95_upper=None, ci95_half_width=None,
                             relative_ci95_half_width=None, ci95_status="undefined_gas_fraction")
            continue
        scale = mixture[right] / mixture[left]
        statistic.update(fieller_interval(components[left]["replicate_mean_molecules"],
                                         components[right]["replicate_mean_molecules"], scale, t_critical))
    return {
        "components": components,
        "selectivity": selectivity,
    }


def _selectivity_from_components(
    components: dict[str, dict[str, Any]],
    mixture: dict[str, float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = list(components)
    total_adsorbed = sum(float(component["mean_molecules"]) for component in components.values())
    for left in names:
        for right in names:
            if left == right:
                continue
            key = f"{left}/{right}"
            left_adsorbed = float(components[left]["mean_molecules"])
            right_adsorbed = float(components[right]["mean_molecules"])
            left_gas = float(mixture.get(left, 0.0) or 0.0)
            right_gas = float(mixture.get(right, 0.0) or 0.0)
            if total_adsorbed <= 0.0 or right_adsorbed <= 0.0 or left_gas <= 0.0 or right_gas <= 0.0:
                result[key] = {
                    "value": None,
                    "reason": "undefined_zero_adsorbed_or_gas_fraction",
                }
                continue
            x_left = left_adsorbed / total_adsorbed
            x_right = right_adsorbed / total_adsorbed
            result[key] = {
                "value": (x_left / x_right) / (left_gas / right_gas),
                "adsorbed_mole_fraction_left": x_left,
                "adsorbed_mole_fraction_right": x_right,
                "gas_mole_fraction_left": left_gas,
                "gas_mole_fraction_right": right_gas,
            }
    return result
