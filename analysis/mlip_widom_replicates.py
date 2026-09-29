"""Aggregate independent MLIP-MC Widom seeds."""

from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from analysis.mlip_widom import analyze_widom_attempts, read_widom_attempt_trace
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
    compare_references: bool = False,
    crafted_root: str | Path | None = "CRAFTED-2.0.0",
    isodb_root: str | Path | None = "isodb-library",
    curated_reference_paths: tuple[str | Path, ...] = (),
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
    seed_summaries = {
        metric: _replicate_statistics(
            [float(run["metrics"][metric]) for run in runs]
        )
        for metric in METRICS
    }
    seed_converged = len(runs) >= minimum_replicates and all(
        summary["relative_ci95_half_width"] is not None
        and summary["relative_ci95_half_width"] <= target
        for summary in seed_summaries.values()
    )
    pooled = _pooled_block_analysis(runs, relative_ci95_target=target)
    checkpoint = _checkpoint_analysis(runs, relative_tolerance=target)
    converged = (
        len(runs) >= minimum_replicates
        and pooled["converged"]
        and checkpoint["stable"]
    )
    report = {
        "schema_version": 2,
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
        "seed_reproducibility_converged": seed_converged,
        "pooled_block_sampling_converged": pooled["converged"],
        "checkpoint_stable": checkpoint["stable"],
        "metrics": seed_summaries,
        "seed_reproducibility": {
            "converged": seed_converged,
            "metrics": seed_summaries,
        },
        "pooled_block_analysis": pooled,
        "checkpoint_convergence": checkpoint,
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
            "Independent Widom insertions are assessed primarily through pooled, "
            "non-overlapping blocks across all seeds. Overall convergence requires "
            "the configured seed count, pooled relative 95% Student-t half-widths "
            "within the target, and stable final cumulative checkpoints. Seed-mean "
            "Student-t intervals remain a conservative reproducibility diagnostic."
        ),
    }

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "widom_replicate_summary.json"
    csv_path = output / "widom_replicates.csv"
    plot_path = output / "widom_replicate_summary.png"
    block_csv_path = output / "widom_pooled_blocks.csv"
    block_plot_path = output / "widom_pooled_block_summary.png"
    save_benchmark_data(json_path, report)
    _write_csv(report["runs"], csv_path)
    _write_plot(report, plot_path)
    _write_block_csv(pooled["blocks"], block_csv_path)
    _write_pooled_plot(report, block_plot_path)
    report["outputs"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "pooled_blocks_csv": str(block_csv_path),
        "pooled_blocks_plot": str(block_plot_path),
    }
    if compare_references:
        from analysis.mlip_widom_reference import (
            create_widom_reference_comparison,
        )

        reference_report = create_widom_reference_comparison(
            json_path,
            crafted_root,
            output / "reference_comparison",
            isodb_root=isodb_root,
            curated_reference_paths=curated_reference_paths,
        )
        report["reference_comparison"] = {
            "assessment": reference_report["assessment"],
            "goeminne_reference": reference_report["goeminne_reference"],
            "outputs": reference_report["outputs"],
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
    analysis_settings = (
        plan.get("benchmark", {})
        .get("potential_benchmark", {})
        .get("mlip_mc", {})
    )
    model = result.get("model", {})
    attempts = int(result["attempts"])
    block_size = int(
        result.get("uncertainty", {}).get(
            "block_size_attempts",
            analysis_settings.get("block_size", max(1, attempts // 10)),
        )
    )
    trial_log_path = (
        run_directory / "engine" / "mlip_mc" / "widom" / "log_widom.bin"
    )
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
        "attempts": attempts,
        "block_size": block_size,
        "minimum_replicates": int(convergence.get("minimum_replicates", 3)),
        "relative_ci95_target": float(
            convergence.get("relative_ci95_target", 0.05)
        ),
        "metrics": {name: float(statistics[name]) for name in METRICS},
        "convergence": result.get("convergence", []),
        "attempt_trace": read_widom_attempt_trace(
            trial_log_path,
            attempts=attempts,
        ),
        "framework_volume_A3": float(result["framework_volume_A3"]),
        "framework_mass_amu": float(result["framework_mass_amu"]),
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
        "attempts",
        "block_size",
        "framework_volume_A3",
        "framework_mass_amu",
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


def _pooled_block_analysis(
    runs: list[dict[str, Any]],
    *,
    relative_ci95_target: float,
) -> dict[str, Any]:
    first = runs[0]
    combined_trace = [
        value
        for run in runs
        for value in run["attempt_trace"]
    ]
    analysis = analyze_widom_attempts(
        combined_trace,
        temperature_K=first["temperature_K"],
        framework_volume_A3=first["framework_volume_A3"],
        framework_mass_amu=first["framework_mass_amu"],
        block_size=first["block_size"],
        convergence_checkpoints=[len(combined_trace)],
    )
    uncertainty = analysis["uncertainty"]
    block_count = int(uncertainty["complete_block_count"])
    t_critical = _student_t_critical_95(block_count - 1)
    metrics = {}
    for metric in METRICS:
        estimate = float(analysis["overall"][metric])
        standard_error = uncertainty["metrics"][metric]["standard_error"]
        ci95 = (
            None
            if standard_error is None
            else t_critical * float(standard_error)
        )
        metrics[metric] = {
            "estimate": estimate,
            "standard_error": standard_error,
            "student_t_critical_95": t_critical,
            "degrees_of_freedom": block_count - 1,
            "ci95_half_width": ci95,
            "relative_ci95_half_width": (
                None if ci95 is None or estimate == 0 else abs(ci95 / estimate)
            ),
        }
    converged = all(
        values["relative_ci95_half_width"] is not None
        and values["relative_ci95_half_width"] <= relative_ci95_target
        for values in metrics.values()
    )
    blocks = uncertainty["blocks"]
    blocks_per_run = len(blocks) // len(runs)
    for index, block in enumerate(blocks):
        run_index = min(index // blocks_per_run, len(runs) - 1)
        block["seed"] = runs[run_index]["seed"]
        block["seed_block_index"] = index - run_index * blocks_per_run + 1
    return {
        "method": "pooled_non_overlapping_block_sem",
        "independent_sampling_assumption": (
            "Widom insertion positions and orientations are independently sampled."
        ),
        "total_attempts": len(combined_trace),
        "valid_insertions": int(analysis["overall"]["valid_energy_count"]),
        "valid_fraction": (
            float(analysis["overall"]["valid_energy_count"])
            / len(combined_trace)
        ),
        "block_size_attempts": first["block_size"],
        "complete_block_count": block_count,
        "relative_ci95_target": relative_ci95_target,
        "converged": converged,
        "metrics": metrics,
        "blocks": blocks,
    }


def _checkpoint_analysis(
    runs: list[dict[str, Any]],
    *,
    relative_tolerance: float,
) -> dict[str, Any]:
    checkpoint_maps = {
        run["seed"]: {int(row["attempts"]): row for row in run["convergence"]}
        for run in runs
    }
    checkpoint_sets = [set(values) for values in checkpoint_maps.values()]
    if not checkpoint_sets or any(values != checkpoint_sets[0] for values in checkpoint_sets[1:]):
        raise ValueError("Widom replicates must contain matching convergence checkpoints.")
    checkpoints = sorted(checkpoint_sets[0])
    rows = []
    for checkpoint in checkpoints:
        row: dict[str, Any] = {"attempts_per_seed": checkpoint}
        for metric in METRICS:
            values = [
                float(checkpoint_maps[run["seed"]][checkpoint][metric])
                for run in runs
            ]
            row[metric] = _replicate_statistics(values)
        rows.append(row)

    changes = {}
    if len(rows) >= 2:
        previous, final = rows[-2], rows[-1]
        for metric in METRICS:
            previous_mean = float(previous[metric]["mean"])
            final_mean = float(final[metric]["mean"])
            changes[metric] = abs(final_mean - previous_mean) / abs(final_mean)
    stable = bool(changes) and all(
        value <= relative_tolerance for value in changes.values()
    )
    return {
        "checkpoints": rows,
        "final_two_checkpoint_relative_change": changes,
        "relative_tolerance": relative_tolerance,
        "stable": stable,
    }


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
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
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


def _write_block_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0]) if rows else ["block_index"]
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
    status = (
        "converged"
        if report["seed_reproducibility_converged"]
        else "not converged"
    )
    fig.suptitle(f"MLIP-MC Widom seed reproducibility ({status})")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_pooled_plot(report: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    pooled = report["pooled_block_analysis"]
    blocks = pooled["blocks"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for axis, (metric, ylabel) in zip(axes, METRICS.items()):
        x_values = [row["block_index"] for row in blocks]
        y_values = [row[metric] for row in blocks]
        colors = [row["seed"] for row in blocks]
        for seed in report["seeds"]:
            selected = [
                (x, y)
                for x, y, color in zip(x_values, y_values, colors)
                if color == seed
            ]
            axis.scatter(
                [item[0] for item in selected],
                [item[1] for item in selected],
                s=28,
                label=f"Seed {seed}",
            )
        summary = pooled["metrics"][metric]
        estimate = summary["estimate"]
        ci95 = summary["ci95_half_width"]
        axis.axhline(estimate, color="#222222", linewidth=1.4, label="Pooled estimate")
        if ci95 is not None:
            axis.axhspan(
                estimate - ci95,
                estimate + ci95,
                color="#777777",
                alpha=0.15,
                label="Pooled 95% CI",
            )
        axis.set_xlabel("Independent 1000-attempt block")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize="small", ncol=2)
    status = "converged" if pooled["converged"] else "not converged"
    fig.suptitle(f"Pooled Widom block convergence ({status})")
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
    parser.add_argument(
        "--compare-references",
        action="store_true",
        help="Also create the layered CRAFTED, experimental, and curated comparison.",
    )
    parser.add_argument("--crafted-root", default="CRAFTED-2.0.0")
    parser.add_argument("--isodb-root", default="isodb-library")
    parser.add_argument(
        "--curated-reference",
        action="append",
        default=[],
        help="Curated DFT/finetuned Widom reference JSON; may be repeated.",
    )
    args = parser.parse_args(argv)
    report = aggregate_widom_replicates(
        args.run_directories,
        args.output_dir,
        relative_ci95_target=args.relative_ci95_target,
        compare_references=args.compare_references,
        crafted_root=args.crafted_root,
        isodb_root=args.isodb_root,
        curated_reference_paths=tuple(args.curated_reference),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
