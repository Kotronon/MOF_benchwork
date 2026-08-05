from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "pressure_bar",
    "run_label",
    "run_directory",
    "kspace_style",
    "kspace_accuracy",
    "loading_absolute_mol_per_kg",
    "loading_excess_mol_per_kg",
    "baseline_label",
    "baseline_absolute_mol_per_kg",
    "baseline_excess_mol_per_kg",
    "absolute_delta_mol_per_kg",
    "absolute_delta_percent",
    "excess_delta_mol_per_kg",
    "excess_delta_percent",
]


def compare_kspace_runs(
    runs: list[tuple[str, str | Path]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compare completed evaluated_isotherm.csv files from KSpace sensitivity runs."""
    if len(runs) < 2:
        raise ValueError("At least two runs are required for a KSpace sensitivity comparison.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaded_runs = [_load_run(label, Path(run_dir)) for label, run_dir in runs]
    baseline = loaded_runs[0]
    rows = _comparison_rows(loaded_runs, baseline)

    csv_path = output / "kspace_sensitivity.csv"
    report_path = output / "kspace_sensitivity.md"
    absolute_plot_path = output / "kspace_sensitivity_absolute.png"
    excess_plot_path = output / "kspace_sensitivity_excess.png"

    _write_csv(rows, csv_path)
    _write_report(rows, report_path)
    absolute_plot = _write_plot(rows, absolute_plot_path, "absolute")
    excess_plot = _write_plot(rows, excess_plot_path, "excess")

    return {
        "status": "completed",
        "run_count": len(loaded_runs),
        "row_count": len(rows),
        "csv": str(csv_path),
        "report": str(report_path),
        "absolute_plot": str(absolute_plot) if absolute_plot else None,
        "excess_plot": str(excess_plot) if excess_plot else None,
    }


def _load_run(label: str, run_dir: Path) -> dict[str, Any]:
    evaluated_csv = run_dir / "evaluated_isotherm.csv"
    prepare_summary = run_dir / "prepare_summary.json"
    if not evaluated_csv.exists():
        raise FileNotFoundError(f"Missing evaluated isotherm CSV: {evaluated_csv}")

    rows = []
    with evaluated_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)

    parameters = {}
    if prepare_summary.exists():
        data = json.loads(prepare_summary.read_text(encoding="utf-8"))
        parameters = data.get("result", {}).get("parameters") or data.get("prepare_plan", {}).get("parameters", {})

    return {
        "label": label,
        "run_dir": run_dir,
        "rows": rows,
        "kspace_style": parameters.get("kspace_style", ""),
        "kspace_accuracy": parameters.get("kspace_accuracy", ""),
    }


def _comparison_rows(loaded_runs: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_by_pressure = {
        float(row["pressure_bar"]): row
        for row in baseline["rows"]
        if row.get("pressure_bar") not in ("", None)
    }
    rows: list[dict[str, Any]] = []
    for run in loaded_runs:
        for row in run["rows"]:
            pressure = float(row["pressure_bar"])
            baseline_row = baseline_by_pressure.get(pressure)
            if baseline_row is None:
                continue
            absolute = _as_float(row.get("loading_absolute_mol_per_kg"))
            excess = _as_float(row.get("loading_excess_mol_per_kg"))
            baseline_absolute = _as_float(baseline_row.get("loading_absolute_mol_per_kg"))
            baseline_excess = _as_float(baseline_row.get("loading_excess_mol_per_kg"))
            rows.append(
                {
                    "pressure_bar": pressure,
                    "run_label": run["label"],
                    "run_directory": str(run["run_dir"]),
                    "kspace_style": run["kspace_style"],
                    "kspace_accuracy": run["kspace_accuracy"],
                    "loading_absolute_mol_per_kg": absolute if absolute is not None else "",
                    "loading_excess_mol_per_kg": excess if excess is not None else "",
                    "baseline_label": baseline["label"],
                    "baseline_absolute_mol_per_kg": baseline_absolute if baseline_absolute is not None else "",
                    "baseline_excess_mol_per_kg": baseline_excess if baseline_excess is not None else "",
                    "absolute_delta_mol_per_kg": _delta(absolute, baseline_absolute),
                    "absolute_delta_percent": _relative_delta_percent(absolute, baseline_absolute),
                    "excess_delta_mol_per_kg": _delta(excess, baseline_excess),
                    "excess_delta_percent": _relative_delta_percent(excess, baseline_excess),
                }
            )
    return sorted(rows, key=lambda item: (float(item["pressure_bar"]), str(item["run_label"])))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# KSpace Sensitivity Comparison",
        "",
        "| pressure / bar | run | kspace | absolute / mol kg^-1 | abs. delta / % | excess / mol kg^-1 | excess delta / % |",
        "|---:|:---|:---|---:|---:|---:|---:|",
    ]
    for row in rows:
        kspace = f"{row['kspace_style']} {row['kspace_accuracy']}".strip()
        lines.append(
            "| "
            f"{_format_optional(row['pressure_bar'])} | "
            f"{row['run_label']} | "
            f"{kspace} | "
            f"{_format_optional(row['loading_absolute_mol_per_kg'])} | "
            f"{_format_optional(row['absolute_delta_percent'])} | "
            f"{_format_optional(row['loading_excess_mol_per_kg'])} | "
            f"{_format_optional(row['excess_delta_percent'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The first run passed on the command line is the baseline.",
            "- Percent deltas are relative to the baseline loading at the same pressure.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(rows: list[dict[str, Any]], path: Path, basis: str) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    loading_key = f"loading_{basis}_mol_per_kg"
    series: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        pressure = _as_float(row.get("pressure_bar"))
        loading = _as_float(row.get(loading_key))
        if pressure is None or loading is None:
            continue
        series.setdefault(str(row["run_label"]), []).append((pressure, loading))
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for label, values in sorted(series.items()):
        points = sorted(values)
        ax.plot(
            [pressure for pressure, _loading in points],
            [loading for _pressure, loading in points],
            marker="o",
            label=label,
        )
    ax.set_xlabel("Pressure / bar")
    ax.set_ylabel("Loading / mol kg$^{-1}$")
    ax.set_title(f"KSpace sensitivity ({basis})")
    ax.grid(True, alpha=0.3)
    if all(pressure > 0 for values in series.values() for pressure, _loading in values):
        ax.set_xscale("log")
        ticks = sorted({pressure for values in series.values() for pressure, _loading in values})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:g}" for tick in ticks])
        ax.minorticks_off()
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _delta(value: float | None, baseline: float | None) -> float | str:
    if value is None or baseline is None:
        return ""
    return value - baseline


def _relative_delta_percent(value: float | None, baseline: float | None) -> float | str:
    if value is None or baseline in (None, 0.0):
        return ""
    return (value - baseline) / baseline * 100.0


def _format_optional(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{number:.6g}"


def _parse_run_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.name, value
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Run arguments must be PATH or LABEL=PATH.")
    return label, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare KSpace sensitivity runs.")
    parser.add_argument("runs", nargs="+", type=_parse_run_arg, help="Run directory as PATH or LABEL=PATH.")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison CSV, report, and plots.")
    args = parser.parse_args(argv)
    result = compare_kspace_runs(args.runs, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
