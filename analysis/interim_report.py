from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shutil
from typing import Any

from analysis.block_convergence import analyze_block_convergence, _input_metadata
from analysis.cell_comparison import compare_cell_runs
from analysis.plotting import legend_outside_right
from converter.cif_to_lammps_data import convert_cif_to_lammps_data
from pipeline.evaluate import evaluate_all_references, evaluate_isotherm
from pipeline.runners import _aggregate_replicates, _convergence_report


INTERIM_FIELDS = [
    "pressure_bar",
    "replicate_count",
    "seeds",
    "n_co2_primitive",
    "n_co2_conventional",
    "r_n",
    "expected_r_n",
    "r_n_deviation_percent",
    "q_primitive_mol_per_kg",
    "q_conventional_mol_per_kg",
    "r_q",
    "expected_r_q",
    "r_q_deviation_percent",
    "ci95_half_width_adsorbates",
    "relative_ci95_half_width",
    "seed_converged",
]

PROGRESS_FIELDS = [
    "pressure_bar",
    "seed",
    "latest_step",
    "progress_percent",
    "run_complete",
    "production_sample_count",
    "complete_block_count",
    "block_converged",
]


def create_interim_report(
    primitive_run: str | Path,
    conventional_run: str | Path,
    output_dir: str | Path,
    *,
    seed_runs: list[str | Path] | None = None,
    block_size: int = 50_000,
) -> dict[str, Any]:
    """Create a non-destructive report from all currently completed pressure points."""
    primitive_path = Path(primitive_run).resolve()
    conventional_path = Path(conventional_run).resolve()
    additional_paths = [Path(path).resolve() for path in (seed_runs or [])]
    target = Path(output_dir).resolve()

    for run_path in [primitive_path, conventional_path, *additional_paths]:
        if not run_path.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_path}")

    target.mkdir(parents=True, exist_ok=True)
    convergence_config = _run_section(conventional_path, "convergence") or {}
    evaluation_config = _run_section(conventional_path, "evaluation") or {}

    primitive_replicates = _collect_completed_replicates(primitive_path)
    conventional_replicates: list[dict[str, Any]] = []
    for run_path in [conventional_path, *additional_paths]:
        conventional_replicates.extend(_collect_completed_replicates(run_path))
    _reject_duplicate_seed_points(conventional_replicates)

    if not primitive_replicates:
        raise ValueError(f"No completed pressure points found in {primitive_path}")
    if not conventional_replicates:
        raise ValueError(f"No completed pressure points found in {conventional_path}")

    primitive_results = _aggregate_replicates(primitive_replicates, convergence_config)
    conventional_results = _aggregate_replicates(
        conventional_replicates, convergence_config
    )
    primitive_view = _write_run_view(
        target / "interim_data" / "primitive",
        primitive_path,
        primitive_replicates,
        primitive_results,
        convergence_config,
    )
    conventional_view = _write_run_view(
        target / "interim_data" / "conventional",
        conventional_path,
        conventional_replicates,
        conventional_results,
        convergence_config,
    )

    primitive_evaluation = evaluate_isotherm(
        primitive_view / "isotherm_summary.json",
        output_dir=primitive_view,
        evaluation_config=_run_section(primitive_path, "evaluation") or evaluation_config,
    )
    conventional_evaluation = evaluate_isotherm(
        conventional_view / "isotherm_summary.json",
        output_dir=conventional_view,
        evaluation_config=evaluation_config,
    )
    reference_evaluation = evaluate_all_references(
        conventional_view,
        evaluation_config=evaluation_config,
    )
    comparison = compare_cell_runs(
        primitive_view,
        conventional_view,
        target / "cell_comparison",
    )
    block_report = _safe_block_convergence(
        conventional_path,
        target / "block_convergence",
        block_size=block_size,
        planned_log_count=len(conventional_replicates),
    )

    progress_rows = _progress_rows(block_report)
    _write_csv(target / "progress.csv", progress_rows, PROGRESS_FIELDS)
    interim_rows = _interim_rows(comparison, conventional_results)
    _write_csv(target / "interim_results.csv", interim_rows, INTERIM_FIELDS)
    seed_plot = _write_seed_plot(
        target / "seed_reproducibility.png", conventional_results
    )

    copied_plots: dict[str, str | None] = {
        "isotherm": _copy_plot(
            conventional_evaluation.get("isotherm_plot"),
            target / "interim_isotherm.png",
        ),
        "all_references": _copy_plot(
            reference_evaluation.get("combined_plot"),
            target / "interim_isotherm_all_references.png",
        ),
        "cell_comparison": _copy_plot(
            comparison["outputs"].get("plot"),
            target / "cell_comparison.png",
        ),
        "block_convergence": _copy_plot(
            block_report["outputs"].get("plot"),
            target / "block_convergence.png",
        ),
        "seed_reproducibility": str(seed_plot) if seed_plot else None,
    }

    completed_pressures = [
        float(result["pressure_bar"]) for result in conventional_results
    ]
    seed_converged = sum(bool(result["converged"]) for result in conventional_results)
    report = {
        "status": "provisional",
        "primitive_run": str(primitive_path),
        "conventional_runs": [
            str(path) for path in [conventional_path, *additional_paths]
        ],
        "completed_pressure_count": len(completed_pressures),
        "completed_pressures_bar": completed_pressures,
        "planned_pressure_count": block_report["log_count"],
        "seed_converged_pressure_count": seed_converged,
        "seed_convergence_requires": {
            "minimum_replicates": int(convergence_config.get("minimum_replicates", 3)),
            "relative_ci95_target": float(
                convergence_config.get("relative_ci95_target", 0.05)
            ),
        },
        "block_convergence": block_report,
        "cell_comparison": comparison,
        "primitive_evaluation": primitive_evaluation,
        "conventional_evaluation": conventional_evaluation,
        "reference_evaluation": reference_evaluation,
        "outputs": {
            "json": str(target / "interim_report.json"),
            "markdown": str(target / "interim_report.md"),
            "results_csv": str(target / "interim_results.csv"),
            "progress_csv": str(target / "progress.csv"),
            "plots": copied_plots,
        },
    }
    (target / "interim_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(target / "interim_report.md", report, interim_rows, progress_rows)
    return report


def _safe_block_convergence(
    run_path: Path,
    output_dir: Path,
    *,
    block_size: int,
    planned_log_count: int,
) -> dict[str, Any]:
    try:
        return analyze_block_convergence(
            run_path,
            output_dir,
            block_size=block_size,
        )
    except FileNotFoundError as exc:
        if "No pressure-point logs found" not in str(exc):
            raise
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_directory": str(run_path),
            "block_size": block_size,
            "log_count": planned_log_count,
            "completed_log_count": 0,
            "all_completed_runs_converged": False,
            "provisional": True,
            "unavailable_reason": str(exc),
            "runs": [],
            "outputs": {
                "csv": None,
                "json": str(output_dir / "block_convergence_summary.json"),
                "plot": None,
            },
        }
        (output_dir / "block_convergence_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary


def _collect_completed_replicates(run_dir: Path) -> list[dict[str, Any]]:
    full_summary = run_dir / "isotherm_summary.json"
    if full_summary.exists():
        data = json.loads(full_summary.read_text(encoding="utf-8"))
        source = data.get("replicate_results") or data.get("results", [])
        return [_normalize_result(item, run_dir, index) for index, item in enumerate(source, 1)]

    planned = _planned_metadata(run_dir)
    results = []
    for index, summary_path in enumerate(
        sorted((run_dir / "logs").glob("gcmc_*bar*_summary.json")), 1
    ):
        stem = summary_path.stem.removesuffix("_summary")
        input_path = run_dir / "inputs" / f"{stem}.in"
        metadata = planned.get(stem)
        if metadata is None:
            metadata = _input_metadata(input_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        log_path = run_dir / "logs" / f"{stem}.log"
        results.append(
            {
                "pressure_bar": float(metadata["pressure_bar"]),
                "replicate_index": int(metadata.get("replicate_index", index)),
                "seed": int(metadata["seed"]),
                "input_script": str(input_path),
                "log_file": str(log_path),
                "summary_file": str(summary_path),
                "wall_time_seconds": _log_wall_time_seconds(log_path),
                "summary": summary,
            }
        )
    return results


def _normalize_result(item: dict[str, Any], run_dir: Path, index: int) -> dict[str, Any]:
    result = dict(item)
    pressure = float(result["pressure_bar"])
    stem = Path(str(result.get("log_file", ""))).stem
    metadata = _planned_metadata(run_dir).get(stem, {})
    seed = result.get("seed", metadata.get("seed"))
    if seed is None:
        seed = _seed_for_pressure(run_dir, pressure)
    result["pressure_bar"] = pressure
    result["replicate_index"] = int(result.get("replicate_index", index))
    result["seed"] = int(seed) if seed is not None else index
    result.setdefault("wall_time_seconds", None)
    return result


def _planned_metadata(run_dir: Path) -> dict[str, dict[str, Any]]:
    prepare = _load_prepare(run_dir)
    scripts = (
        prepare.get("prepare_plan", {})
        .get("planned_files", {})
        .get("input_scripts", [])
    )
    return {
        Path(str(script.get("log", ""))).stem: script
        for script in scripts
        if script.get("log")
    }


def _seed_for_pressure(run_dir: Path, pressure: float) -> int | None:
    for metadata in _planned_metadata(run_dir).values():
        if float(metadata.get("pressure_bar", -1)) == pressure:
            seed = metadata.get("seed")
            if seed is not None:
                return int(seed)
    parameters = _load_prepare(run_dir).get("prepare_plan", {}).get("parameters", {})
    seeds = parameters.get("seeds")
    if isinstance(seeds, list) and seeds:
        return int(seeds[0])
    seed = parameters.get("seed")
    return int(seed) if seed is not None else 12345


def _load_prepare(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "prepare_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_section(run_dir: Path, name: str) -> dict[str, Any] | None:
    prepare = _load_prepare(run_dir)
    value = prepare.get("prepare_plan", {}).get(name)
    if isinstance(value, dict):
        return value
    value = prepare.get("result", {}).get(name)
    return value if isinstance(value, dict) else None


def _log_wall_time_seconds(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Total wall time:\s*(?:(\d+)-)?(\d+):(\d+):(\d+)", text)
    if match is None:
        return None
    days, hours, minutes, seconds = (int(value or 0) for value in match.groups())
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


def _reject_duplicate_seed_points(results: list[dict[str, Any]]) -> None:
    seen: set[tuple[float, int]] = set()
    for result in results:
        key = (float(result["pressure_bar"]), int(result["seed"]))
        if key in seen:
            raise ValueError(
                f"Duplicate conventional result for {key[0]:g} bar and seed {key[1]}. "
                "Do not pass the same seed run twice."
            )
        seen.add(key)


def _write_run_view(
    target: Path,
    source_run: Path,
    replicates: list[dict[str, Any]],
    results: list[dict[str, Any]],
    convergence_config: dict[str, Any],
) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    data_target = target / "data"
    data_target.mkdir(exist_ok=True)
    _copy_or_generate_framework_data(source_run, data_target)
    prepare_path = source_run / "prepare_summary.json"
    if prepare_path.exists():
        shutil.copy2(prepare_path, target / "prepare_summary.json")
    summary = {
        "status": "provisional",
        "results": results,
        "replicate_results": replicates,
        "convergence": _convergence_report(results, convergence_config),
    }
    (target / "isotherm_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return target


def _copy_or_generate_framework_data(source_run: Path, data_target: Path) -> Path:
    data_files = sorted((source_run / "data").glob("*.data"))
    if len(data_files) == 1:
        target = data_target / data_files[0].name
        shutil.copy2(data_files[0], target)
        return target
    if len(data_files) > 1:
        raise ValueError(f"Expected at most one framework data file in {source_run / 'data'}")

    framework_plan = (
        _load_prepare(source_run)
        .get("prepare_plan", {})
        .get("planned_files", {})
        .get("framework_data", {})
    )
    input_cif = framework_plan.get("input_cif")
    output_name = Path(str(framework_plan.get("output_data", "IRMOF-1.data"))).name
    if not input_cif:
        raise ValueError(
            f"No framework data file found in {source_run / 'data'} and no CIF conversion plan "
            f"found in {source_run / 'prepare_summary.json'}."
        )

    target = data_target / output_name
    convert_cif_to_lammps_data(
        input_cif,
        target,
        atom_style=str(framework_plan.get("atom_style", "full")),
        cell_representation=str(framework_plan.get("cell_representation", "source")),
        unit_cells=framework_plan.get("unit_cells", [1, 1, 1]),
        cutoff_A=framework_plan.get("cutoff_A"),
        minimum_image_policy=str(framework_plan.get("minimum_image_policy", "error")),
    )
    return target


def _progress_rows(block_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "pressure_bar": run["pressure_bar"],
            "seed": run["seed"],
            "latest_step": run["latest_step"],
            "progress_percent": 100.0 * float(run["progress_fraction"]),
            "run_complete": run["run_complete"],
            "production_sample_count": run["production_sample_count"],
            "complete_block_count": run["complete_block_count"],
            "block_converged": run["converged"],
        }
        for run in block_report["runs"]
    ]


def _interim_rows(
    comparison: dict[str, Any], conventional_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_pressure = {
        float(result["pressure_bar"]): result for result in conventional_results
    }
    rows = []
    for comparison_row in comparison["rows"]:
        pressure = float(comparison_row["pressure_bar"])
        result = by_pressure[pressure]
        summary = result["summary"]
        rows.append(
            {
                "pressure_bar": pressure,
                "replicate_count": result["replicate_count"],
                "seeds": ";".join(str(seed) for seed in result["seeds"]),
                **{
                    field: comparison_row.get(field)
                    for field in INTERIM_FIELDS
                    if field in comparison_row
                },
                "ci95_half_width_adsorbates": summary.get("ci95_half_width_adsorbates"),
                "relative_ci95_half_width": summary.get("relative_ci95_half_width"),
                "seed_converged": result["converged"],
            }
        )
    return rows


def _write_seed_plot(path: Path, results: list[dict[str, Any]]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    pressures = [float(result["pressure_bar"]) for result in results]
    means = [float(result["summary"]["mean_adsorbates"]) for result in results]
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 8.4), sharex=True)
    axes[0].plot(
        pressures,
        means,
        "D-",
        color="black",
        linewidth=1.8,
        label="Mean across available seeds",
    )
    seeds: dict[int, list[tuple[float, float]]] = {}
    ci_label_written = False
    for result in results:
        values = result["summary"].get("replicate_mean_adsorbates", [])
        for seed, value in zip(result["seeds"], values, strict=False):
            seeds.setdefault(int(seed), []).append(
                (float(result["pressure_bar"]), float(value))
            )
        uncertainty = result["summary"].get("ci95_half_width_adsorbates")
        if uncertainty is not None:
            axes[0].errorbar(
                [result["pressure_bar"]],
                [result["summary"]["mean_adsorbates"]],
                yerr=[uncertainty],
                fmt="none",
                capsize=3,
                color="black",
                linewidth=1.5,
                label="95% confidence interval" if not ci_label_written else None,
            )
            ci_label_written = True
    show_seed_labels = len(seeds) > 1
    for seed, values in seeds.items():
        values.sort()
        axes[0].plot(
            [value[0] for value in values],
            [value[1] for value in values],
            "o-",
            alpha=0.65,
            linewidth=1.0,
            label=f"seed {seed}" if show_seed_labels else None,
        )
    for result in results:
        axes[0].annotate(
            f"n={result['replicate_count']}",
            (
                float(result["pressure_bar"]),
                float(result["summary"]["mean_adsorbates"]),
            ),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize="small",
        )

    relative = [
        (
            100.0 * float(result["summary"]["relative_ci95_half_width"])
            if result["summary"].get("relative_ci95_half_width") is not None
            else None
        )
        for result in results
    ]
    available = [
        (pressure, value)
        for pressure, value in zip(pressures, relative)
        if value is not None
    ]
    if available:
        axes[1].plot(
            [value[0] for value in available],
            [value[1] for value in available],
            "o-",
            color="tab:green",
            label="Relative 95% CI half-width",
        )
    unavailable = [
        pressure
        for pressure, value in zip(pressures, relative)
        if value is None
    ]
    unavailable_y = 1.0
    if unavailable:
        axes[1].scatter(
            unavailable,
            [unavailable_y] * len(unavailable),
            marker="x",
            color="tab:gray",
            label="CI unavailable (n=1)",
        )
        for pressure in unavailable:
            axes[1].annotate(
                "n=1",
                (pressure, unavailable_y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                color="tab:gray",
                fontsize="small",
            )
    axes[1].axhline(5.0, color="black", linestyle="--", label="target: 5%")
    axes[0].set_title("Seed reproducibility (completed pressure points only)")
    axes[0].set_ylabel(r"Mean $N_{CO_2}$ per conventional cell")
    axes[1].set_ylabel("Relative 95% CI half-width / %")
    axes[1].set_xlabel("Pressure / bar")
    axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(pressures)
        axis.set_xticklabels([f"{pressure:g}" for pressure in pressures])
        axis.minorticks_off()
        axis.grid(alpha=0.3)
        legend_outside_right(axis, fontsize="small")
    figure.text(
        0.5,
        0.01,
        "Converged when n >= 3 and relative 95% CI half-width <= 5%.",
        ha="center",
        fontsize="small",
    )
    figure.tight_layout(rect=(0, 0.035, 0.78, 1))
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path

def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _copy_plot(source: str | None, target: Path) -> str | None:
    if source is None:
        return None
    source_path = Path(source)
    if not source_path.exists():
        return None
    if source_path.resolve() != target.resolve():
        shutil.copy2(source_path, target)
    return str(target)


def _write_markdown(
    path: Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    progress_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Provisional MOF-5 CO2 interim report",
        "",
        "> This report is provisional. Only completed pressure-point summaries are used as isotherm results; live logs are used only for progress and block diagnostics.",
        "",
        f"Completed pressure points: {report['completed_pressure_count']} of {report['planned_pressure_count']}",
        "",
        "## Primitive--conventional comparison",
        "",
        "| Pressure / bar | Seeds | N primitive | N conventional | R_N | R_q | Seed converged |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pressure_bar']:g} | {row['replicate_count']} | "
            f"{row['n_co2_primitive']:.6g} | {row['n_co2_conventional']:.6g} | "
            f"{row['r_n']:.6g} | {row['r_q']:.6g} | {row['seed_converged']} |"
        )
    lines.extend(
        [
            "",
            "## Progress",
            "",
            "| Pressure / bar | Seed | Progress / % | Complete | Block converged |",
            "|---:|---:|---:|:---:|:---:|",
        ]
    )
    for row in progress_rows:
        lines.append(
            f"| {float(row['pressure_bar']):g} | {row['seed']} | "
            f"{float(row['progress_percent']):.1f} | {row['run_complete']} | "
            f"{row['block_converged']} |"
        )
    _append_reference_curation_summary(lines, report)
    lines.extend(
        [
            "",
            "## Block convergence plot interpretation",
            "",
            "`block_convergence.png` contains two panels for each pressure point with available LAMMPS log data:",
            "",
            "- Upper panel: block mean `N_CO2`. Each point is the mean number of CO2 molecules within one production block. This shows local block-to-block fluctuations.",
            "- Lower panel: cumulative mean `N_CO2`. Each point is the running mean over all production blocks up to that point. This shows whether the final adsorption estimate stabilizes over time.",
            "",
            "A stable cumulative mean is the more direct indicator for the final adsorption value. The block mean is useful for detecting noisy or drifting individual blocks. The vertical dashed line marks the end of equilibration and the beginning of the production region used for analysis.",
            "",
            "## Figures",
            "",
            "- `interim_isotherm.png`: completed conventional points and selected reference",
            "- `interim_isotherm_all_references.png`: completed points against CRAFTED/NIST references",
            "- `cell_comparison.png`: primitive--conventional counts, loadings, ratios, and runtimes",
            "- `block_convergence.png`: block and cumulative means, including provisional live logs",
            "- `seed_reproducibility.png`: seed means and relative 95% confidence intervals",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_reference_curation_summary(lines: list[str], report: dict[str, Any]) -> None:
    reference_report = report.get("reference_evaluation", {})
    candidates_path = reference_report.get("reference_candidates")
    candidates = _load_reference_candidates(candidates_path)
    if not candidates:
        return

    active = [candidate for candidate in candidates if candidate.get("use_in_evaluation", True) and not candidate.get("excluded")]
    excluded = [candidate for candidate in candidates if candidate.get("excluded")]
    active_crafted = sum(candidate.get("source") == "crafted" for candidate in active)
    active_nist = sum(candidate.get("source") == "nist_isodb" for candidate in active)
    excluded_nist = sum(candidate.get("source") == "nist_isodb" for candidate in excluded)

    lines.extend(
        [
            "",
            "## Reference curation summary",
            "",
            "The full NIST ISODB search space is not listed in the human-readable report. Instead, the report shows the ranked candidates that passed the resolver policy and archives all candidate metadata in `reference_candidates.json`.",
            "",
            "| Category | Count |",
            "|---|---:|",
            f"| Active CRAFTED references | {active_crafted} |",
            f"| Active NIST ISODB references | {active_nist} |",
            f"| Excluded NIST ISODB candidates | {excluded_nist} |",
            f"| Total ranked candidates stored in metadata | {len(candidates)} |",
            "",
        ]
    )

    reason_counts = Counter(
        reason
        for candidate in excluded
        for reason in candidate.get("exclusion_reasons", [])
    )
    if reason_counts:
        lines.extend(
            [
                "Excluded references are not silently removed. They remain archived with explicit reasons:",
                "",
                "| Exclusion reason | Count |",
                "|---|---:|",
            ]
        )
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"| `{reason}` | {count} |")
        lines.append("")

    active_references = [
        candidate for candidate in active if candidate.get("source") in {"crafted", "nist_isodb"}
    ]
    if active_references:
        lines.extend(
            [
                "Active references used for plotting and quantitative comparison:",
                "",
                "| Source | DOI / key | Basis | Role |",
                "|---|---|---|---|",
            ]
        )
        for candidate in active_references:
            source = "CRAFTED" if candidate.get("source") == "crafted" else "NIST ISODB"
            identifier = candidate.get("doi") or candidate.get("key", "")
            basis = candidate.get("adsorption_basis") or candidate.get("format", "unknown")
            role = "selected primary" if candidate.get("selected") else "active comparison"
            lines.append(f"| {source} | `{identifier}` | `{basis}` | {role} |")
        lines.append("")


def _load_reference_candidates(path_value: str | None) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [candidate for candidate in data if isinstance(candidate, dict)]
    if isinstance(data, dict):
        candidates = data.get("candidates", [])
        if isinstance(candidates, list):
            return [candidate for candidate in candidates if isinstance(candidate, dict)]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create plots and tables from all currently available run results."
    )
    parser.add_argument("--primitive", required=True, help="Primitive-cell run directory.")
    parser.add_argument(
        "--conventional", required=True, help="Primary conventional-cell run directory."
    )
    parser.add_argument(
        "--seed-run",
        action="append",
        default=[],
        help="Additional conventional seed run directory; may be repeated.",
    )
    parser.add_argument("--output-dir", default="reports/interim")
    parser.add_argument("--block-size", type=int, default=50_000)
    args = parser.parse_args(argv)
    report = create_interim_report(
        args.primitive,
        args.conventional,
        args.output_dir,
        seed_runs=args.seed_run,
        block_size=args.block_size,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
