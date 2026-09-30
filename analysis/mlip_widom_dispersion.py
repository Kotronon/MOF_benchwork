"""Compare MLIP-MC Widom results with and without dispersion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pipeline.config import save_benchmark_data


METRICS = {
    "henry_coefficient_mmol_g_bar": {
        "label": r"$K_H$ / mmol g$^{-1}$ bar$^{-1}$",
        "title": "Henry coefficient",
        "logarithmic": True,
        "reference_uncertainty": "henry_uncertainty_mmol_g_bar",
    },
    "isosteric_heat_zero_loading_kj_mol": {
        "label": r"$Q_{st}^{0}$ / kJ mol$^{-1}$",
        "title": "Zero-loading adsorption heat",
        "logarithmic": False,
        "reference_uncertainty": "isosteric_heat_uncertainty_kj_mol",
    },
}


def create_dispersion_comparison(
    baseline_summary_path: str | Path,
    dispersion_summary_path: str | Path,
    reference_comparison_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write paired and reference plots for no-D3 and D3 Widom results."""
    baseline = _load_json(baseline_summary_path)
    dispersion = _load_json(dispersion_summary_path)
    reference_report = _load_json(reference_comparison_path)
    _validate_compatible(baseline, dispersion, reference_report)

    variants = [
        _variant_summary("MACE-MP without D3", baseline),
        _variant_summary("MACE-MP + D3", dispersion),
    ]
    paired_runs = _paired_runs(baseline, dispersion)
    references = reference_report.get("references", [])
    ratios = {
        metric: {
            "d3_to_no_d3": (
                variants[1][metric]["estimate"]
                / variants[0][metric]["estimate"]
            ),
            "d3_to_references": {
                reference["source_id"]: (
                    variants[1][metric]["estimate"] / float(reference[metric])
                )
                for reference in references
                if reference.get(metric) not in (None, 0)
            },
        }
        for metric in METRICS
    }
    report = {
        "schema_version": 1,
        "material": baseline["material"],
        "adsorbate": baseline["adsorbate"],
        "temperature_K": float(baseline["temperature_K"]),
        "variants": variants,
        "paired_runs": paired_runs,
        "references": references,
        "ratios": ratios,
        "limitations": (
            f"The D3 aggregate contains {dispersion['replicate_count']} "
            "independent seed(s). Error bars for candidates are pooled-block "
            "95% confidence intervals, not model-form uncertainty."
        ),
    }

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paired_plot = output / "widom_d3_paired_comparison.png"
    reference_plot = output / "widom_d3_reference_comparison.png"
    csv_path = output / "widom_d3_reference_comparison.csv"
    json_path = output / "widom_d3_comparison.json"
    _write_paired_plot(report, paired_plot)
    _write_reference_plot(report, reference_plot)
    _write_csv(report, csv_path)
    report["outputs"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "paired_plot": str(paired_plot),
        "reference_plot": str(reference_plot),
    }
    save_benchmark_data(json_path, report)
    return report


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing comparison input: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _validate_compatible(
    baseline: dict[str, Any],
    dispersion: dict[str, Any],
    reference_report: dict[str, Any],
) -> None:
    fields = ("material", "adsorbate", "temperature_K")
    for field in fields:
        values = (
            baseline.get(field),
            dispersion.get(field),
            reference_report.get(field),
        )
        if values[0] != values[1] or values[0] != values[2]:
            raise ValueError(
                f"Incompatible {field} in dispersion comparison: {values}."
            )
    if baseline.get("model", {}).get("dispersion") is not False:
        raise ValueError("The baseline summary must have dispersion=false.")
    if dispersion.get("model", {}).get("dispersion") is not True:
        raise ValueError("The dispersion summary must have dispersion=true.")


def _variant_summary(label: str, summary: dict[str, Any]) -> dict[str, Any]:
    pooled = summary["pooled_block_analysis"]
    return {
        "label": label,
        "model": summary.get("model", {}),
        "replicate_count": int(summary["replicate_count"]),
        **{
            metric: {
                "estimate": float(pooled["metrics"][metric]["estimate"]),
                "ci95_half_width": pooled["metrics"][metric].get(
                    "ci95_half_width"
                ),
            }
            for metric in METRICS
        },
    }


def _paired_runs(
    baseline: dict[str, Any],
    dispersion: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_by_seed = {int(run["seed"]): run for run in baseline["runs"]}
    dispersion_by_seed = {int(run["seed"]): run for run in dispersion["runs"]}
    common_seeds = sorted(set(baseline_by_seed) & set(dispersion_by_seed))
    if not common_seeds:
        raise ValueError("No common seeds exist for the paired D3 comparison.")
    return [
        {
            "seed": seed,
            "without_d3": {
                metric: float(baseline_by_seed[seed][metric])
                for metric in METRICS
            },
            "with_d3": {
                metric: float(dispersion_by_seed[seed][metric])
                for metric in METRICS
            },
        }
        for seed in common_seeds
    ]


def _write_paired_plot(report: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    seed_colors = {12345: "#2878B5", 23456: "#E07A2D", 34567: "#3A923A"}
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for axis, (metric, specification) in zip(axes, METRICS.items()):
        for run in report["paired_runs"]:
            values = [run["without_d3"][metric], run["with_d3"][metric]]
            color = seed_colors.get(run["seed"], "#666666")
            axis.plot(
                [0, 1],
                values,
                marker="o",
                color=color,
                linewidth=1.7,
                markersize=6,
                label=f"Seed {run['seed']}",
            )
        for position, variant in enumerate(report["variants"]):
            metric_data = variant[metric]
            axis.errorbar(
                position,
                metric_data["estimate"],
                yerr=metric_data["ci95_half_width"],
                fmt="D",
                color="#111111",
                capsize=5,
                markersize=7,
                linewidth=1.6,
                zorder=5,
            )
        axis.set_xticks([0, 1], ["MACE-MP\nwithout D3", "MACE-MP\nwith D3"])
        axis.set_ylabel(specification["label"])
        axis.set_title(specification["title"])
        axis.grid(axis="y", alpha=0.25)
        if specification["logarithmic"]:
            axis.set_yscale("log")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=max(1, len(labels)),
    )
    fig.suptitle(
        f"Effect of D3 dispersion on {report['material']}/{report['adsorbate']} "
        f"Widom results at {report['temperature_K']:.2f} K",
        y=1.08,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_reference_plot(report: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    candidate_colors = ("#2878B5", "#E07A2D")
    reference_colors = {
        "classical_simulation": "#6A4C93",
        "experiment": "#C14953",
        "dft": "#00876C",
        "dft_finetuned": "#00876C",
        "finetuned_mlip": "#00876C",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4))
    for axis, (metric, specification) in zip(axes, METRICS.items()):
        entries = []
        for color, variant in zip(candidate_colors, report["variants"]):
            entries.append(
                {
                    "label": variant["label"],
                    "value": variant[metric]["estimate"],
                    "uncertainty": variant[metric]["ci95_half_width"],
                    "color": color,
                    "marker": "D",
                }
            )
        for reference in report["references"]:
            if reference.get(metric) is None:
                continue
            entries.append(
                {
                    "label": reference["label"],
                    "value": float(reference[metric]),
                    "uncertainty": reference.get(
                        specification["reference_uncertainty"]
                    ),
                    "color": reference_colors.get(
                        reference.get("reference_class"), "#555555"
                    ),
                    "marker": "s",
                }
            )
        for position, entry in enumerate(entries):
            axis.errorbar(
                position,
                entry["value"],
                yerr=entry["uncertainty"],
                fmt=entry["marker"],
                color=entry["color"],
                capsize=5,
                markersize=8,
                linewidth=1.6,
                zorder=4,
            )
        axis.set_xticks(
            range(len(entries)),
            [entry["label"] for entry in entries],
            rotation=20,
            ha="right",
        )
        axis.set_ylabel(specification["label"])
        axis.set_title(specification["title"])
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelsize=8)
        if specification["logarithmic"]:
            axis.set_yscale("log")
    fig.suptitle(
        f"{report['material']}/{report['adsorbate']} at "
        f"{report['temperature_K']:.2f} K: MACE-MP dispersion and references"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "source",
        "source_class",
        "replicate_count",
        "henry_coefficient_mmol_g_bar",
        "henry_ci95_half_width_mmol_g_bar",
        "isosteric_heat_zero_loading_kj_mol",
        "qst_ci95_half_width_kj_mol",
    ]
    rows = [
        {
            "source": variant["label"],
            "source_class": "candidate_mlip",
            "replicate_count": variant["replicate_count"],
            "henry_coefficient_mmol_g_bar": variant[
                "henry_coefficient_mmol_g_bar"
            ]["estimate"],
            "henry_ci95_half_width_mmol_g_bar": variant[
                "henry_coefficient_mmol_g_bar"
            ]["ci95_half_width"],
            "isosteric_heat_zero_loading_kj_mol": variant[
                "isosteric_heat_zero_loading_kj_mol"
            ]["estimate"],
            "qst_ci95_half_width_kj_mol": variant[
                "isosteric_heat_zero_loading_kj_mol"
            ]["ci95_half_width"],
        }
        for variant in report["variants"]
    ]
    rows.extend(
        {
            "source": reference["label"],
            "source_class": reference.get("reference_class"),
            "replicate_count": None,
            "henry_coefficient_mmol_g_bar": reference.get(
                "henry_coefficient_mmol_g_bar"
            ),
            "henry_ci95_half_width_mmol_g_bar": reference.get(
                "henry_uncertainty_mmol_g_bar"
            ),
            "isosteric_heat_zero_loading_kj_mol": reference.get(
                "isosteric_heat_zero_loading_kj_mol"
            ),
            "qst_ci95_half_width_kj_mol": reference.get(
                "isosteric_heat_uncertainty_kj_mol"
            ),
        }
        for reference in report["references"]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare MLIP-MC Widom results with and without D3."
    )
    parser.add_argument("baseline_summary")
    parser.add_argument("dispersion_summary")
    parser.add_argument("reference_comparison")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = create_dispersion_comparison(
        args.baseline_summary,
        args.dispersion_summary,
        args.reference_comparison,
        args.output_dir,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
