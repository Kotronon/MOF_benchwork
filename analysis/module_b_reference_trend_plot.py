from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REFERENCE_FILES = {
    "10/90 NIST simulation; excess": "10.1039b807470d.Isotherm14.json",
    "~50/50 NIST experiment; basis unspecified": "10.1016j.seppur.2018.01.013.Isotherm4.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _simulation_series(run_dir: Path) -> list[dict[str, Any]]:
    summary = _read_json(run_dir / "isotherm_summary.json")
    convergence_path = (
        run_dir / "analysis" / "module_b_convergence" / "convergence_report.json"
    )
    convergence = _read_json(convergence_path) if convergence_path.exists() else None
    convergence_index = {
        (float(row["pressure_bar"]), row["quantity"]): row
        for row in (convergence or {}).get("results", [])
    }
    prepare = _read_json(run_dir / "prepare_summary.json")
    framework_data = Path(
        prepare["prepare_plan"]["planned_files"]["framework_data"]["output_data"]
    )
    if not framework_data.is_absolute():
        framework_data = PROJECT_ROOT / framework_data
    framework_mass = _framework_mass(framework_data)

    rows = []
    for point in summary.get("results", []):
        pressure = float(point["pressure_bar"])
        components = point["summary"]["components"]
        mixture = {
            name: float(values["gas_phase_mole_fraction"])
            for name, values in components.items()
        }
        component_values = {
            name: float(values["mean_molecules"])
            for name, values in components.items()
        }
        row: dict[str, Any] = {
            "source": "Simulation 15/85; absolute",
            "pressure_bar": pressure,
            "co2_mole_fraction": mixture["CO2"],
            "n2_mole_fraction": mixture["N2"],
            "co2_mmol_g": 1000.0 * component_values["CO2"] / framework_mass,
            "n2_mmol_g": 1000.0 * component_values["N2"] / framework_mass,
            "selectivity": (
                component_values["CO2"]
                / component_values["N2"]
                * mixture["N2"]
                / mixture["CO2"]
                if component_values["N2"] > 0
                else None
            ),
        }
        for component in ("CO2", "N2"):
            result = convergence_index.get((pressure, component), {})
            factor = 1000.0 / framework_mass
            row[f"{component.lower()}_ci95_lower_mmol_g"] = (
                float(result["ci95_lower"]) * factor
                if result.get("ci95_lower") is not None
                else None
            )
            row[f"{component.lower()}_ci95_upper_mmol_g"] = (
                float(result["ci95_upper"]) * factor
                if result.get("ci95_upper") is not None
                else None
            )
        selectivity_result = convergence_index.get((pressure, "CO2/N2"), {})
        row["selectivity_ci95_lower"] = selectivity_result.get("ci95_lower")
        row["selectivity_ci95_upper"] = selectivity_result.get("ci95_upper")
        rows.append(row)
    return sorted(rows, key=lambda row: row["pressure_bar"])


def _framework_mass(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Framework data file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1] == "atoms":
            continue
    from pipeline.evaluate import framework_mass_from_lammps_data

    mass = float(framework_mass_from_lammps_data(path))
    if mass <= 0:
        raise ValueError(f"Invalid framework mass in {path}.")
    return mass


def _reference_series(run_dir: Path) -> list[dict[str, Any]]:
    reference_dir = run_dir / "source" / "references"
    rows = []
    for label, filename in REFERENCE_FILES.items():
        data = _read_json(reference_dir / filename)
        for point in data.get("isotherm_data", []):
            pressure = float(point["pressure"])
            species = point.get("species_data", [])
            values = {}
            fractions = {}
            for item in species:
                key = item["InChIKey"]
                component = "CO2" if key.startswith("CURLTUG") else "N2"
                values[component] = float(item["adsorption"])
                fractions[component] = float(item["composition"])
            if set(values) != {"CO2", "N2"}:
                continue
            selectivity = None
            if pressure > 0 and values["N2"] > 0 and fractions["CO2"] > 0:
                selectivity = (
                    values["CO2"]
                    / values["N2"]
                    * fractions["N2"]
                    / fractions["CO2"]
                )
            rows.append(
                {
                    "source": label,
                    "pressure_bar": pressure,
                    "co2_mole_fraction": fractions["CO2"],
                    "n2_mole_fraction": fractions["N2"],
                    "co2_mmol_g": values["CO2"],
                    "n2_mmol_g": values["N2"],
                    "selectivity": selectivity,
                    "co2_ci95_lower_mmol_g": None,
                    "co2_ci95_upper_mmol_g": None,
                    "n2_ci95_lower_mmol_g": None,
                    "n2_ci95_upper_mmol_g": None,
                    "selectivity_ci95_lower": None,
                    "selectivity_ci95_upper": None,
                }
            )
    return rows


def _write_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    colors = {
        "Simulation 15/85; absolute": "#146b62",
        "10/90 NIST simulation; excess": "#cf5b32",
        "~50/50 NIST experiment; basis unspecified": "#4267a5",
    }
    markers = {
        "Simulation 15/85; absolute": "o",
        "10/90 NIST simulation; excess": "s",
        "~50/50 NIST experiment; basis unspecified": "D",
    }
    figure, axes = plt.subplots(
        1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    for label in colors:
        series = sorted(
            (row for row in rows if row["source"] == label),
            key=lambda row: row["pressure_bar"],
        )
        positive = [row for row in series if row["pressure_bar"] > 0]
        if not positive:
            continue
        color = colors[label]
        marker = markers[label]
        pressures = [row["pressure_bar"] for row in positive]
        for axis, component in zip(axes[:1], ("co2", "n2")):
            values = [row[f"{component}_mmol_g"] for row in positive]
            axis.plot(
                pressures,
                values,
                marker=marker,
                markersize=5.5,
                linewidth=1.8,
                color=color,
                label="_nolegend_",
            )
            if label == "Simulation 15/85; absolute":
                lower = [row.get(f"{component}_ci95_lower_mmol_g") for row in positive]
                upper = [row.get(f"{component}_ci95_upper_mmol_g") for row in positive]
                if all(value is not None for value in [*lower, *upper]):
                    axis.fill_between(pressures, lower, upper, color=color, alpha=0.12)
        selectivities = [row["selectivity"] for row in positive]
        if all(value is not None for value in selectivities):
            axes[1].plot(
                pressures,
                selectivities,
                marker=marker,
                markersize=5.5,
                linewidth=1.8,
                color=color,
                label="_nolegend_",
            )

    axes[0].set_title("Component uptake")
    axes[0].set_ylabel("Uptake (mmol/g)")
    for label in colors:
        series = sorted(
            (row for row in rows if row["source"] == label and row["pressure_bar"] > 0),
            key=lambda row: row["pressure_bar"],
        )
        if series:
            axes[0].plot(
                [row["pressure_bar"] for row in series],
                [row["n2_mmol_g"] for row in series],
                marker=markers[label],
                markersize=4.5,
                linewidth=1.5,
                linestyle="--",
                color=colors[label],
                label="_nolegend_",
            )
    source_handles = [
        Line2D(
            [0], [0], color=colors[label], marker=markers[label], linewidth=1.8,
            markersize=5.5, label=label,
        )
        for label in colors
    ]
    component_handles = [
        Line2D([0], [0], color="#444444", linewidth=1.8, linestyle="-", label="CO2"),
        Line2D([0], [0], color="#444444", linewidth=1.8, linestyle="--", label="N2"),
    ]
    axes[0].legend(
        handles=[*source_handles, *component_handles],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    axes[1].set_title("Selectivity calculated from each dataset")
    axes[1].set_ylabel("CO2/N2 selectivity")
    axes[1].legend(handles=source_handles, frameon=False, fontsize=8, loc="upper left")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Total pressure (bar)")
        pressure_ticks = [0.1, 1, 5, 10, 15, 20, 50, 100]
        axis.set_xticks(pressure_ticks)
        axis.set_xticklabels([f"{value:g}" for value in pressure_ticks])
        axis.grid(True, which="major", color="#d7d9d6", linewidth=0.8)
        axis.grid(True, which="minor", color="#eceeeb", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "IRMOF-1 CO2/N2: qualitative trend comparison\n"
        "Different gas compositions and adsorption bases; not a quantitative validation",
        fontsize=13,
        y=1.03,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_reference_trend_plot(
    run_dir: str | Path, output_dir: str | Path | None = None
) -> dict[str, str]:
    run_path = Path(run_dir).resolve()
    rows = _simulation_series(run_path) + _reference_series(run_path)
    target = Path(output_dir).resolve() if output_dir else run_path / "analysis" / "reference_trend"
    target.mkdir(parents=True, exist_ok=True)
    plot_path = target / "reference_trend_comparison.png"
    csv_path = target / "reference_trend_data.csv"
    note_path = target / "README.md"
    _write_plot(rows, plot_path)
    fields = [
        "source",
        "pressure_bar",
        "co2_mole_fraction",
        "n2_mole_fraction",
        "co2_mmol_g",
        "n2_mmol_g",
        "selectivity",
        "co2_ci95_lower_mmol_g",
        "co2_ci95_upper_mmol_g",
        "n2_ci95_lower_mmol_g",
        "n2_ci95_upper_mmol_g",
        "selectivity_ci95_lower",
        "selectivity_ci95_upper",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    note_path.write_text(
        "# Qualitative reference trend comparison\n\n"
        "The figure overlays the local 15/85 absolute-uptake simulation with two NIST datasets: "
        "a 10/90 simulation reported as excess adsorption (DOI 10.1039/b807470d) and an "
        "approximately 50/50 experimental series with unspecified adsorption basis "
        "(DOI 10.1016/j.seppur.2018.01.013). NIST points are shown as tabulated, without "
        "composition correction, basis conversion or interpolation. The selectivity values "
        "are computed from each dataset's own component uptake and gas composition. Use the "
        "figure only to inspect broad trends; it is not a quantitative validation.\n\n"
        "Simulation bands show the local convergence report's 95% intervals. No uncertainty "
        "was available for the digitized/reference points.\n",
        encoding="utf-8",
    )
    return {"plot": str(plot_path), "data": str(csv_path), "notes": str(note_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_run = (
        PROJECT_ROOT
        / "outputs"
        / "module_B_mixture_selectivity"
        / "runs"
        / "mof5_co2_n2_validation_pilot_2_longer"
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=str(default_run),
        help=f"Module-B run directory (default: {default_run})",
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    outputs = create_reference_trend_plot(args.run_dir, args.output_dir)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())