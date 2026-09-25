"""Tabular and graphical reports for Module C potential comparisons."""

from __future__ import annotations

import csv
import json
from math import sqrt
import os
from pathlib import Path
from statistics import mean, median
from typing import Any


def create_potential_report(
    report: dict[str, Any] | str | Path,
    output_directory: str | Path,
    *,
    save_csv: bool = True,
    save_plots: bool = True,
) -> dict[str, Any]:
    """Write aggregate metrics, a flat table, and diagnostic plots."""
    data = _load_report(report)
    rows = _comparison_rows(data)
    if not rows:
        raise ValueError("Potential comparison report contains no candidate results.")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary = _aggregate_rows(rows)
    summary_path = output / "potential_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    artifacts: dict[str, Any] = {
        "summary_json": str(summary_path),
        "summary": summary,
    }
    if save_csv:
        csv_path = output / "potential_metrics.csv"
        _write_csv(rows, csv_path)
        artifacts["table_csv"] = str(csv_path)
    if save_plots:
        plot_path = output / "potential_benchmark_summary.png"
        _write_plots(rows, plot_path)
        artifacts["plot_png"] = str(plot_path)
    return artifacts


def _load_report(report: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    with Path(report).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for configuration in report.get("configurations", []):
        baseline_name = configuration["baseline_backend"]
        baseline = configuration["results"][baseline_name]
        metadata = configuration.get("metadata", {})
        distance = metadata.get("minimum_host_guest_distance_A")
        for comparison in configuration.get("comparisons", []):
            candidate_name = comparison["candidate_backend"]
            candidate = configuration["results"][candidate_name]
            rows.append(
                {
                    "configuration_id": configuration["configuration_id"],
                    "region": configuration.get("region"),
                    "minimum_host_guest_distance_A": distance,
                    "minimum_distance_filter_A": metadata.get(
                        "minimum_distance_filter_A"
                    ),
                    "thermodynamically_unbiased": metadata.get(
                        "thermodynamically_unbiased"
                    ),
                    "baseline_backend": baseline_name,
                    "candidate_backend": candidate_name,
                    "baseline_interaction_energy_ev": baseline[
                        "interaction_energy_ev"
                    ],
                    "candidate_interaction_energy_ev": candidate[
                        "interaction_energy_ev"
                    ],
                    "energy_difference_ev": comparison["energy_difference_ev"],
                    "absolute_energy_difference_ev": comparison[
                        "absolute_energy_difference_ev"
                    ],
                    "force_mae_ev_per_angstrom": comparison[
                        "force_mae_ev_per_angstrom"
                    ],
                    "force_rmse_ev_per_angstrom": comparison[
                        "force_rmse_ev_per_angstrom"
                    ],
                    "maximum_force_difference_ev_per_angstrom": comparison[
                        "maximum_force_difference_ev_per_angstrom"
                    ],
                    "baseline_runtime_seconds": comparison[
                        "baseline_runtime_seconds"
                    ],
                    "candidate_runtime_seconds": comparison[
                        "candidate_runtime_seconds"
                    ],
                    "runtime_ratio_candidate_to_baseline": comparison[
                        "runtime_ratio_candidate_to_baseline"
                    ],
                }
            )
    return rows


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(row["candidate_backend"], []).append(row)

    candidates = {}
    for candidate, candidate_rows in by_candidate.items():
        regions: dict[str, list[dict[str, Any]]] = {}
        for row in candidate_rows:
            regions.setdefault(str(row.get("region") or "unclassified"), []).append(row)
        candidates[candidate] = {
            **_metric_summary(candidate_rows),
            "by_region": {
                region: _metric_summary(region_rows)
                for region, region_rows in sorted(regions.items())
            },
        }
    filters = sorted(
        {
            float(row["minimum_distance_filter_A"])
            for row in rows
            if row["minimum_distance_filter_A"] is not None
        }
    )
    unbiased_values = {
        row["thermodynamically_unbiased"]
        for row in rows
        if row["thermodynamically_unbiased"] is not None
    }
    return {
        "schema_version": 1,
        "baseline_backend": rows[0]["baseline_backend"],
        "candidate_count": len(candidates),
        "row_count": len(rows),
        "sampling": {
            "minimum_distance_filters_A": filters,
            "thermodynamically_unbiased": (
                unbiased_values == {True} if unbiased_values else None
            ),
        },
        "candidates": candidates,
        "interpretation": (
            "Differences quantify agreement with the configured baseline; "
            "they do not establish accuracy without an independent reference."
        ),
    }


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    energy_differences = [row["energy_difference_ev"] for row in rows]
    absolute_energy_differences = [abs(value) for value in energy_differences]
    force_maes = [row["force_mae_ev_per_angstrom"] for row in rows]
    force_rmses = [row["force_rmse_ev_per_angstrom"] for row in rows]
    runtime_ratios = [
        row["runtime_ratio_candidate_to_baseline"]
        for row in rows
        if row["runtime_ratio_candidate_to_baseline"] is not None
    ]
    return {
        "configuration_count": len(rows),
        "energy_mae_ev": mean(absolute_energy_differences),
        "median_absolute_energy_difference_ev": median(
            absolute_energy_differences
        ),
        "energy_rmse_ev": sqrt(mean(value * value for value in energy_differences)),
        "mean_force_mae_ev_per_angstrom": mean(force_maes),
        "median_force_mae_ev_per_angstrom": median(force_maes),
        "mean_force_rmse_ev_per_angstrom": mean(force_rmses),
        "maximum_force_difference_ev_per_angstrom": max(
            row["maximum_force_difference_ev_per_angstrom"] for row in rows
        ),
        "mean_baseline_runtime_seconds": mean(
            row["baseline_runtime_seconds"] for row in rows
        ),
        "mean_candidate_runtime_seconds": mean(
            row["candidate_runtime_seconds"] for row in rows
        ),
        "median_runtime_ratio_candidate_to_baseline": (
            median(runtime_ratios) if runtime_ratios else None
        ),
    }


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(rows: list[dict[str, Any]], output_path: Path) -> None:
    cache_directory = output_path.parent / ".plot_cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory / "matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required when output.save_plots is enabled."
        ) from exc

    candidates = list(dict.fromkeys(row["candidate_backend"] for row in rows))
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    all_energies: list[float] = []
    for index, candidate in enumerate(candidates):
        candidate_rows = [row for row in rows if row["candidate_backend"] == candidate]
        baseline = [row["baseline_interaction_energy_ev"] for row in candidate_rows]
        predicted = [row["candidate_interaction_energy_ev"] for row in candidate_rows]
        all_energies.extend(baseline + predicted)
        axes[0, 0].scatter(baseline, predicted, label=candidate, alpha=0.75, color=colors(index))
    lower, upper = min(all_energies), max(all_energies)
    if lower == upper:
        lower -= 1.0
        upper += 1.0
    axes[0, 0].plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(xlabel="Baseline interaction energy (eV)", ylabel="Candidate interaction energy (eV)", title="Energy parity")
    axes[0, 0].legend(frameon=False)

    for index, candidate in enumerate(candidates):
        candidate_rows = [
            row
            for row in rows
            if row["candidate_backend"] == candidate
            and row["minimum_host_guest_distance_A"] is not None
        ]
        if candidate_rows:
            axes[0, 1].scatter(
                [row["minimum_host_guest_distance_A"] for row in candidate_rows],
                [row["energy_difference_ev"] for row in candidate_rows],
                label=candidate,
                alpha=0.75,
                color=colors(index),
            )
    axes[0, 1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(xlabel="Minimum host-guest distance (A)", ylabel="Candidate - baseline energy (eV)", title="Energy error by separation")
    if axes[0, 1].collections:
        axes[0, 1].legend(frameon=False)

    force_data = [
        [row["force_mae_ev_per_angstrom"] for row in rows if row["candidate_backend"] == candidate]
        for candidate in candidates
    ]
    axes[1, 0].boxplot(force_data, tick_labels=candidates, showmeans=True)
    axes[1, 0].set(ylabel="Force MAE (eV/A)", title="Force-error distribution")

    baseline_name = rows[0]["baseline_backend"]
    runtime_labels = [baseline_name, *candidates]
    runtime_values = [mean(row["baseline_runtime_seconds"] for row in rows)] + [
        mean(
            row["candidate_runtime_seconds"]
            for row in rows
            if row["candidate_backend"] == candidate
        )
        for candidate in candidates
    ]
    axes[1, 1].bar(runtime_labels, runtime_values, color=["#4c4c4c", *[colors(index) for index in range(len(candidates))]])
    axes[1, 1].set(ylabel="Mean runtime per interaction evaluation (s)", title="Observed runtime")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.tick_params(axis="x", labelrotation=15)
    figure.suptitle("Module C potential comparison")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
