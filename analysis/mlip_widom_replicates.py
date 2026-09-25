"""Aggregate independent MLIP-MC Widom seeds."""

from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from pipeline.config import save_benchmark_data


METRICS = {
    "henry_coefficient_mmol_g_bar": r"$K_H$ / mmol g$^{-1}$ bar$^{-1}$",
    "isosteric_heat_zero_loading_kj_mol": r"$Q_{st}^{0}$ / kJ mol$^{-1}$",
}


def aggregate_widom_replicates(
    run_directories: list[str | Path],
    output_directory: str | Path,
    *,
    relative_ci95_target: float | None = None,
) -> dict[str, Any]:
    """Combine independent Widom runs and write convergence artifacts."""
    if len(run_directories) < 2:
        raise ValueError("At least two Widom run directories are required.")
    runs = [_load_run(Path(path)) for path in run_directories]
    _validate_compatible_runs(runs)

    first = runs[0]
    target = (
        float(relative_ci95_target)
        if relative_ci95_target is not None
        else float(first["relative_ci95_target"])
    )
    minimum_replicates = int(first["minimum_replicates"])
    summaries = {
        metric: _replicate_statistics(
            [float(run["metrics"][metric]) for run in runs]
        )
        for metric in METRICS
    }
    converged = len(runs) >= minimum_replicates and all(
        summary["relative_ci95_half_width"] is not None
        and summary["relative_ci95_half_width"] <= target
        for summary in summaries.values()
    )
    report = {
        "schema_version": 1,
        "status": "completed",
        "material": first["material"],
        "adsorbate": first["adsorbate"],
        "temperature_K": first["temperature_K"],
        "model": first["model"],
        "replicate_count": len(runs),
        "seeds": [run["seed"] for run in runs],
        "attempts_per_replicate": [run["attempts"] for run in runs],
        "minimum_replicates": minimum_replicates,
        "relative_ci95_target": target,
        "converged": converged,
        "metrics": summaries,
        "runs": [
            {
                "run_directory": run["run_directory"],
                "seed": run["seed"],
                "attempts": run["attempts"],
                **run["metrics"],
            }
            for run in runs
        ],
        "interpretation": (
            "Seed convergence requires the configured minimum number of "
            "independent replicates and relative 95% Student-t half-widths "
            "at or below the configured target for both reported metrics."
        ),
    }

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "widom_replicate_summary.json"
    csv_path = output / "widom_replicates.csv"
    plot_path = output / "widom_replicate_summary.png"
    save_benchmark_data(json_path, report)
    _write_csv(report["runs"], csv_path)
    _write_plot(report, plot_path)
    report["outputs"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    save_benchmark_data(json_path, report)
    return report


def _load_run(run_directory: Path) -> dict[str, Any]:
    result_path = run_directory / "results" / "mlip_mc_benchmark.json"
    plan_path = run_directory / "run_plan.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing MLIP-MC result: {result_path}")
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing run plan: {plan_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if result.get("method") != "widom" or result.get("status") != "completed":
        raise ValueError(f"Run is not a completed Widom result: {run_directory}")
    statistics = result.get("corrected_statistics", {})
    missing = [name for name in METRICS if statistics.get(name) is None]
    if missing:
        raise ValueError(
            f"Widom run {run_directory} lacks metrics: {', '.join(missing)}"
        )
    convergence = plan.get("convergence", {})
    model = result.get("model", {})
    return {
        "run_directory": str(run_directory),
        "material": result.get("material"),
        "adsorbate": result.get("adsorbate"),
        "temperature_K": float(result["temperature_K"]),
        "model": {
            key: model.get(key)
            for key in ("backend", "name", "model", "dispersion", "default_dtype")
        },
        "seed": int(result["seed"]),
        "attempts": int(result["attempts"]),
        "minimum_replicates": int(convergence.get("minimum_replicates", 3)),
        "relative_ci95_target": float(
            convergence.get("relative_ci95_target", 0.05)
        ),
        "metrics": {name: float(statistics[name]) for name in METRICS},
    }


def _validate_compatible_runs(runs: list[dict[str, Any]]) -> None:
    first = runs[0]
    compatibility_fields = (
        "material",
        "adsorbate",
        "temperature_K",
        "model",
        "minimum_replicates",
        "relative_ci95_target",
    )
    for run in runs[1:]:
        mismatched = [name for name in compatibility_fields if run[name] != first[name]]
        if mismatched:
            raise ValueError(
                "Incompatible Widom replicate settings in "
                f"{run['run_directory']}: {', '.join(mismatched)}"
            )
    seeds = [run["seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Widom replicate seeds must be unique.")


def _replicate_statistics(values: list[float]) -> dict[str, float | int | None]:
    count = len(values)
    mean = fmean(values)
    standard_deviation = stdev(values) if count >= 2 else None
    standard_error = (
        None if standard_deviation is None else standard_deviation / sqrt(count)
    )
    ci95 = (
        None
        if standard_error is None
        else _student_t_critical_95(count - 1) * standard_error
    )
    return {
        "replicate_count": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95_half_width": ci95,
        "relative_ci95_half_width": (
            None if ci95 is None or mean == 0 else abs(ci95 / mean)
        ),
    }


def _student_t_critical_95(degrees_of_freedom: int) -> float:
    critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
    }
    if degrees_of_freedom in critical:
        return critical[degrees_of_freedom]
    if degrees_of_freedom < 25:
        return critical[20]
    if degrees_of_freedom < 30:
        return critical[25]
    return 1.96


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "run_directory",
        "seed",
        "attempts",
        *METRICS,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(report: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    seeds = [str(row["seed"]) for row in report["runs"]]
    for axis, (metric, ylabel) in zip(axes, METRICS.items()):
        values = [row[metric] for row in report["runs"]]
        summary = report["metrics"][metric]
        axis.scatter(seeds, values, color="#2878B5", label="Independent seed")
        axis.axhline(summary["mean"], color="#333333", linewidth=1.4, label="Mean")
        if summary["ci95_half_width"] is not None:
            lower = summary["mean"] - summary["ci95_half_width"]
            upper = summary["mean"] + summary["ci95_half_width"]
            axis.axhspan(lower, upper, color="#2878B5", alpha=0.15, label="95% CI")
        axis.set_xlabel("Seed")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize="small")
    status = "converged" if report["converged"] else "not converged"
    fig.suptitle(f"MLIP-MC Widom seed reproducibility ({status})")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate independent MLIP-MC Widom runs."
    )
    parser.add_argument("run_directories", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--relative-ci95-target", type=float)
    args = parser.parse_args(argv)
    report = aggregate_widom_replicates(
        args.run_directories,
        args.output_dir,
        relative_ci95_target=args.relative_ci95_target,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
