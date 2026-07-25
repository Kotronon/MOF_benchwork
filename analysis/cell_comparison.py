from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


COMPARISON_FIELDS = [
    "pressure_bar",
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
    "primitive_mean_wall_time_seconds",
    "conventional_mean_wall_time_seconds",
    "runtime_ratio_conventional_primitive",
]

REPLICATE_FIELDS = [
    "cell_form",
    "pressure_bar",
    "seed",
    "n_co2",
    "q_mol_per_kg",
    "wall_time_seconds",
]


def compare_cell_runs(
    primitive_run: str | Path,
    conventional_run: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compare molecule counts, mass-normalized loading, and runtimes by pressure."""
    primitive = _load_run(Path(primitive_run))
    conventional = _load_run(Path(conventional_run))
    common_pressures = sorted(set(primitive["points"]) & set(conventional["points"]))
    if not common_pressures:
        raise ValueError("The two runs have no common pressure points.")

    expected_r_n = conventional["framework_atom_count"] / primitive["framework_atom_count"]
    rows = []
    for pressure in common_pressures:
        primitive_point = primitive["points"][pressure]
        conventional_point = conventional["points"][pressure]
        r_n = _ratio(conventional_point["mean_adsorbates"], primitive_point["mean_adsorbates"])
        r_q = _ratio(conventional_point["loading_mol_per_kg"], primitive_point["loading_mol_per_kg"])
        runtime_ratio = _ratio(
            conventional_point["wall_time_seconds"],
            primitive_point["wall_time_seconds"],
        )
        rows.append(
            {
                "pressure_bar": pressure,
                "n_co2_primitive": primitive_point["mean_adsorbates"],
                "n_co2_conventional": conventional_point["mean_adsorbates"],
                "r_n": r_n,
                "expected_r_n": expected_r_n,
                "r_n_deviation_percent": _relative_deviation_percent(r_n, expected_r_n),
                "q_primitive_mol_per_kg": primitive_point["loading_mol_per_kg"],
                "q_conventional_mol_per_kg": conventional_point["loading_mol_per_kg"],
                "r_q": r_q,
                "expected_r_q": 1.0,
                "r_q_deviation_percent": _relative_deviation_percent(r_q, 1.0),
                "primitive_mean_wall_time_seconds": primitive_point["wall_time_seconds"],
                "conventional_mean_wall_time_seconds": conventional_point["wall_time_seconds"],
                "runtime_ratio_conventional_primitive": runtime_ratio,
            }
        )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / "cell_comparison.csv"
    replicate_csv_path = target / "cell_replicates.csv"
    json_path = target / "cell_comparison_summary.json"
    plot_path = target / "cell_comparison.png"
    _write_csv(csv_path, rows)
    replicate_rows = _build_replicate_rows(
        primitive, conventional, common_pressures
    )
    _write_csv(replicate_csv_path, replicate_rows, REPLICATE_FIELDS)
    plot_result = _write_plot(
        plot_path,
        rows,
        expected_r_n,
        primitive,
        conventional,
    )

    report = {
        "primitive_run": str(Path(primitive_run)),
        "conventional_run": str(Path(conventional_run)),
        "primitive_framework_atom_count": primitive["framework_atom_count"],
        "conventional_framework_atom_count": conventional["framework_atom_count"],
        "expected_r_n": expected_r_n,
        "expected_r_q": 1.0,
        "pressure_point_count": len(rows),
        "replicate_row_count": len(replicate_rows),
        "primitive_runtime_available": primitive["runtime_available"],
        "conventional_runtime_available": conventional["runtime_available"],
        "runtime_ratio_available": primitive["runtime_available"] and conventional["runtime_available"],
        "slowest_primitive_pressure": _slowest_pressure(rows, "primitive_mean_wall_time_seconds"),
        "slowest_conventional_pressure": _slowest_pressure(rows, "conventional_mean_wall_time_seconds"),
        "slowest_runtime_ratio_pressure": _slowest_pressure(
            rows, "runtime_ratio_conventional_primitive"
        ),
        "rows": rows,
        "outputs": {
            "csv": str(csv_path),
            "replicate_csv": str(replicate_csv_path),
            "json": str(json_path),
            "plot": str(plot_result) if plot_result else None,
        },
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _load_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "isotherm_summary.json"
    evaluated_path = run_dir / "evaluated_isotherm.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing isotherm summary: {summary_path}")
    if not evaluated_path.exists():
        raise FileNotFoundError(f"Missing evaluated isotherm: {evaluated_path}")

    data_files = sorted((run_dir / "data").glob("*.data"))
    if len(data_files) != 1:
        raise ValueError(f"Expected exactly one framework data file in {run_dir / 'data'}.")
    framework_atom_count = _framework_atom_count(data_files[0])
    loadings = _load_loadings(evaluated_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = {
        float(result["pressure_bar"]): result
        for result in summary.get("results", [])
    }
    molecule_counts = {
        pressure: float(result["summary"]["mean_adsorbates"])
        for pressure, result in results.items()
    }
    runtimes = _mean_runtimes_by_pressure(summary)
    replicates = _replicates_by_pressure(summary)
    pressures = sorted(set(molecule_counts) & set(loadings))
    points = {
        pressure: {
            "mean_adsorbates": molecule_counts[pressure],
            "loading_mol_per_kg": loadings[pressure],
            "wall_time_seconds": runtimes.get(pressure),
            "ci95_half_width_adsorbates": results[pressure]["summary"].get(
                "ci95_half_width_adsorbates"
            ),
            "ci95_half_width_loading_mol_per_kg": _scaled_uncertainty(
                results[pressure]["summary"].get("ci95_half_width_adsorbates"),
                molecule_counts[pressure],
                loadings[pressure],
            ),
            "replicates": _with_replicate_loadings(
                replicates.get(pressure, []),
                molecule_counts[pressure],
                loadings[pressure],
            ),
        }
        for pressure in pressures
    }
    return {
        "framework_atom_count": framework_atom_count,
        "points": points,
        "runtime_available": bool(points) and all(
            point["wall_time_seconds"] is not None for point in points.values()
        ),
    }


def _framework_atom_count(data_path: Path) -> int:
    for line in data_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1] == "atoms":
            return int(parts[0])
    raise ValueError(f"Could not read framework atom count from {data_path}.")


def _load_loadings(path: Path) -> dict[float, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    preferred_columns = (
        "loading_absolute_mol_per_kg",
        "loading_mol_per_kg",
        "loading_mmol_per_g",
    )
    column = next(
        (name for name in preferred_columns if name in rows[0] and rows[0].get(name) not in (None, "")),
        None,
    )
    if column is None:
        raise ValueError(f"No supported loading column found in {path}.")
    return {float(row["pressure_bar"]): float(row[column]) for row in rows}


def _mean_runtimes_by_pressure(summary: dict[str, Any]) -> dict[float, float]:
    source_results = summary.get("replicate_results") or summary.get("results", [])
    grouped: dict[float, list[float]] = {}
    for result in source_results:
        value = result.get("wall_time_seconds")
        if value is None:
            continue
        grouped.setdefault(float(result["pressure_bar"]), []).append(float(value))
    return {pressure: mean(values) for pressure, values in grouped.items()}

def _replicates_by_pressure(
    summary: dict[str, Any],
) -> dict[float, list[dict[str, Any]]]:
    source_results = summary.get("replicate_results")
    grouped: dict[float, list[dict[str, Any]]] = {}
    if source_results:
        for result in source_results:
            grouped.setdefault(float(result["pressure_bar"]), []).append(
                {
                    "seed": result.get("seed"),
                    "mean_adsorbates": float(result["summary"]["mean_adsorbates"]),
                    "wall_time_seconds": result.get("wall_time_seconds"),
                }
            )
        return grouped

    for result in summary.get("results", []):
        pressure = float(result["pressure_bar"])
        values = result["summary"].get("replicate_mean_adsorbates")
        if values:
            seeds = result.get("seeds", [])
            grouped[pressure] = [
                {
                    "seed": seeds[index] if index < len(seeds) else index + 1,
                    "mean_adsorbates": float(value),
                    "wall_time_seconds": None,
                }
                for index, value in enumerate(values)
            ]
        else:
            grouped[pressure] = [
                {
                    "seed": result.get("seed"),
                    "mean_adsorbates": float(result["summary"]["mean_adsorbates"]),
                    "wall_time_seconds": result.get("wall_time_seconds"),
                }
            ]
    return grouped


def _with_replicate_loadings(
    replicates: list[dict[str, Any]],
    mean_adsorbates: float,
    loading_mol_per_kg: float,
) -> list[dict[str, Any]]:
    loading_per_molecule = (
        loading_mol_per_kg / mean_adsorbates if mean_adsorbates != 0 else None
    )
    return [
        {
            **replicate,
            "loading_mol_per_kg": (
                float(replicate["mean_adsorbates"]) * loading_per_molecule
                if loading_per_molecule is not None
                else None
            ),
        }
        for replicate in replicates
    ]

def _scaled_uncertainty(
    uncertainty: float | None,
    mean_adsorbates: float,
    loading_mol_per_kg: float,
) -> float | None:
    if uncertainty is None or mean_adsorbates == 0:
        return None
    return float(uncertainty) * loading_mol_per_kg / mean_adsorbates


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _relative_deviation_percent(value: float | None, expected: float) -> float | None:
    if value is None or expected == 0:
        return None
    return 100.0 * (value - expected) / expected


def _slowest_pressure(rows: list[dict[str, Any]], field: str) -> dict[str, float] | None:
    available = [row for row in rows if row[field] is not None]
    if not available:
        return None
    slowest = max(available, key=lambda row: float(row[field]))
    return {"pressure_bar": slowest["pressure_bar"], field: slowest[field]}


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames or COMPARISON_FIELDS
        )
        writer.writeheader()
        writer.writerows(rows)


def _build_replicate_rows(
    primitive: dict[str, Any],
    conventional: dict[str, Any],
    pressures: list[float],
) -> list[dict[str, Any]]:
    rows = []
    for cell_form, run in (
        ("primitive", primitive),
        ("conventional", conventional),
    ):
        for pressure in pressures:
            for replicate in run["points"][pressure]["replicates"]:
                rows.append(
                    {
                        "cell_form": cell_form,
                        "pressure_bar": pressure,
                        "seed": replicate["seed"],
                        "n_co2": replicate["mean_adsorbates"],
                        "q_mol_per_kg": replicate["loading_mol_per_kg"],
                        "wall_time_seconds": replicate["wall_time_seconds"],
                    }
                )
    return rows


def _write_plot(
    path: Path,
    rows: list[dict[str, Any]],
    expected_r_n: float,
    primitive: dict[str, Any],
    conventional: dict[str, Any],
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    pressures = [row["pressure_bar"] for row in rows]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    run_styles = (
        ("Primitive", primitive, "tab:blue"),
        ("Conventional", conventional, "tab:orange"),
    )
    _plot_absolute_series(
        axes[0, 0],
        pressures,
        run_styles,
        "mean_adsorbates",
        "mean_adsorbates",
        "ci95_half_width_adsorbates",
        r"$N_{\mathrm{CO_2}}$ per cell",
    )
    _plot_absolute_series(
        axes[0, 1],
        pressures,
        run_styles,
        "loading_mol_per_kg",
        "loading_mol_per_kg",
        "ci95_half_width_loading_mol_per_kg",
        r"$q$ / mol kg$^{-1}$",
    )
    _plot_absolute_series(
        axes[0, 2],
        pressures,
        run_styles,
        "wall_time_seconds",
        "wall_time_seconds",
        None,
        "Mean wall time / s",
    )

    axes[1, 0].axhline(
        expected_r_n,
        color="black",
        linestyle="--",
        label=f"expected {expected_r_n:g}",
    )
    axes[1, 0].plot(pressures, [row["r_n"] for row in rows], "o-")
    axes[1, 0].set_ylabel(r"$R_N$")
    axes[1, 0].legend()

    axes[1, 1].axhline(
        1.0, color="black", linestyle="--", label="expected 1"
    )
    axes[1, 1].plot(pressures, [row["r_q"] for row in rows], "o-")
    axes[1, 1].set_ylabel(r"$R_q$")
    axes[1, 1].legend()

    if any(
        row["runtime_ratio_conventional_primitive"] is not None for row in rows
    ):
        axes[1, 2].plot(
            pressures,
            [row["runtime_ratio_conventional_primitive"] for row in rows],
            "o-",
        )
    else:
        axes[1, 2].text(
            0.5,
            0.5,
            "runtime unavailable",
            ha="center",
            va="center",
            transform=axes[1, 2].transAxes,
        )
    axes[1, 2].set_ylabel("runtime conventional / primitive")

    for axis in axes.flat:
        axis.set_xlabel("Pressure / bar")
        axis.set_xscale("log")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _plot_absolute_series(
    axis: Any,
    pressures: list[float],
    run_styles: tuple[tuple[str, dict[str, Any], str], ...],
    point_field: str,
    replicate_field: str,
    uncertainty_field: str | None,
    ylabel: str,
) -> None:
    plotted = False
    for label, run, color in run_styles:
        values = [
            run["points"][pressure].get(point_field) for pressure in pressures
        ]
        if any(value is not None for value in values):
            axis.plot(
                pressures,
                values,
                "o-",
                color=color,
                label=f"{label} mean",
            )
            if uncertainty_field is not None:
                for pressure, value in zip(pressures, values, strict=True):
                    uncertainty = run["points"][pressure].get(
                        uncertainty_field
                    )
                    if value is not None and uncertainty is not None:
                        axis.errorbar(
                            [pressure],
                            [value],
                            yerr=[uncertainty],
                            fmt="none",
                            color=color,
                            capsize=3,
                            alpha=0.8,
                        )
            plotted = True
        _plot_seed_replicates(
            axis,
            pressures,
            run,
            replicate_field,
            color,
        )
    if plotted:
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "data unavailable",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_ylabel(ylabel)


def _plot_seed_replicates(
    axis: Any,
    pressures: list[float],
    run: dict[str, Any],
    field: str,
    color: str,
) -> None:
    by_seed: dict[Any, list[tuple[float, float]]] = {}
    for pressure in pressures:
        for replicate in run["points"][pressure]["replicates"]:
            value = replicate.get(field)
            if value is None:
                continue
            seed = replicate.get("seed")
            by_seed.setdefault(seed, []).append((pressure, float(value)))
    for values in by_seed.values():
        values.sort()
        axis.plot(
            [value[0] for value in values],
            [value[1] for value in values],
            "o-",
            color=color,
            alpha=0.2,
            linewidth=0.8,
            markersize=3,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare primitive and conventional MOF runs.")
    parser.add_argument("primitive_run", help="Completed primitive-cell run directory.")
    parser.add_argument("conventional_run", help="Completed conventional-cell run directory.")
    parser.add_argument("--output-dir", default="reports/cell_comparison")
    args = parser.parse_args(argv)
    report = compare_cell_runs(args.primitive_run, args.conventional_run, args.output_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
