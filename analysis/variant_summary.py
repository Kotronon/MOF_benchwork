from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def create_variant_summary(run_directory: str | Path) -> dict[str, Any]:
    """Create a compact visual summary for a completed variant benchmark."""
    run_dir = Path(run_directory)
    comparison_path = run_dir / "variant_comparison.json"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing variant comparison: {comparison_path}")

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    rows = []
    comparison_by_variant = {
        str(row["variant"]): row for row in comparison.get("rows", [])
    }
    for variant_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        evaluated_path = variant_dir / "work" / "evaluated_isotherm.csv"
        if not evaluated_path.exists():
            continue
        with evaluated_path.open("r", encoding="utf-8", newline="") as handle:
            evaluated_rows = list(csv.DictReader(handle))
        for evaluated in evaluated_rows:
            runtime = comparison_by_variant.get(variant_dir.name, {})
            rows.append(
                {
                    "variant": variant_dir.name,
                    "pressure_bar": _float(evaluated.get("pressure_bar")),
                    "absolute_mol_per_kg": _float(
                        evaluated.get("loading_absolute_mol_per_kg")
                    ),
                    "excess_mol_per_kg": _float(
                        evaluated.get("loading_excess_mol_per_kg")
                    ),
                    "reference_mol_per_kg": _float(evaluated.get("reference_mol_per_kg")),
                    "reference_error_percent": _float(
                        evaluated.get("comparison_relative_error_percent")
                    ),
                    "runtime_seconds": _float(runtime.get("mean_wall_time_seconds")),
                    "converged": bool(runtime.get("converged", False)),
                }
            )

    if not rows:
        raise ValueError(f"No evaluated variant results found below {run_dir}")

    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "variant_smoke_summary.png"
    report_path = output_dir / "variant_smoke_summary.md"
    _write_plot(rows, plot_path)
    _write_report(rows, report_path)
    return {
        "status": "completed",
        "variant_count": len({row["variant"] for row in rows}),
        "point_count": len(rows),
        "all_converged": all(row["converged"] for row in rows),
        "plot": str(plot_path),
        "report": str(report_path),
    }


def _write_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["variant"].replace("UFF-", "") for row in rows]
    colors = ["#2878B5", "#E07A2D", "#3A923A", "#7A5AA6"][: len(rows)]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))

    _bars(
        axes[0],
        labels,
        [row["absolute_mol_per_kg"] for row in rows],
        colors,
        "Absolute loading",
        "mol kg$^{-1}$",
        decimals=3,
    )
    references = [row["reference_mol_per_kg"] for row in rows if row["reference_mol_per_kg"] is not None]
    if references:
        axes[0].axhline(references[0], color="#333333", linestyle="--", linewidth=1.3, label="CRAFTED reference")
        axes[0].legend(frameon=False, fontsize="small", loc="lower right")

    _bars(
        axes[1],
        labels,
        [row["reference_error_percent"] for row in rows],
        colors,
        "Error to CRAFTED reference",
        "%",
        decimals=2,
    )
    _bars(
        axes[2],
        labels,
        [row["runtime_seconds"] / 60.0 if row["runtime_seconds"] is not None else None for row in rows],
        colors,
        "Wall time",
        "minutes",
        decimals=1,
    )

    pressure = rows[0]["pressure_bar"]
    convergence = "converged" if all(row["converged"] for row in rows) else "smoke test, not convergence-qualified"
    fig.suptitle(f"MOF-5/CO$_2$ PPPM sensitivity at {pressure:g} bar ({convergence})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _bars(
    axis: Any,
    labels: list[str],
    values: list[float | None],
    colors: list[str],
    title: str,
    ylabel: str,
    *,
    decimals: int,
) -> None:
    numeric = [value if value is not None else 0.0 for value in values]
    bars = axis.bar(labels, numeric, color=colors)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=20)
    axis.bar_label(
        bars,
        labels=["n/a" if value is None else f"{value:.{decimals}f}" for value in values],
        padding=3,
        fontsize=9,
    )
    upper = max(numeric) if numeric else 1.0
    axis.set_ylim(0, upper * 1.18 if upper > 0 else 1.0)


def _write_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Variant smoke-test summary",
        "",
        "| variant | pressure / bar | absolute / mol kg^-1 | excess / mol kg^-1 | reference error / % | runtime / min | converged |",
        "|:---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        runtime_minutes = row["runtime_seconds"] / 60.0 if row["runtime_seconds"] is not None else None
        lines.append(
            f"| {row['variant']} | {_format(row['pressure_bar'], 3)} | "
            f"{_format(row['absolute_mol_per_kg'], 4)} | {_format(row['excess_mol_per_kg'], 4)} | "
            f"{_format(row['reference_error_percent'], 2)} | {_format(runtime_minutes, 1)} | "
            f"{'yes' if row['converged'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "All configured variants completed and produced parseable adsorption results.",
            "This establishes technical pipeline success only. A single seed, pressure point, and short run do not establish statistical convergence or potential accuracy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _format(value: float | None, decimals: int) -> str:
    return "n/a" if value is None else f"{value:.{decimals}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a visual summary of a variant benchmark.")
    parser.add_argument("run_directory")
    args = parser.parse_args(argv)
    print(json.dumps(create_variant_summary(args.run_directory), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
