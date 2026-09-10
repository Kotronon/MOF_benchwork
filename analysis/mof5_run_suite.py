from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


RUN_ROOT = Path("outputs/module_A_co2_isotherm/runs")
REFERENCE_PATH = Path("CRAFTED-2.0.0/ISOTHERM_FILES/DDEC_IRMOF-1_UFF_CO2_298.csv")

DEFAULT_RUNS = {
    "seed_12345": RUN_ROOT / "tao2022_conventional_seed12345",
    "seeds_104729_209759": RUN_ROOT / "mof5_co2_conventional_2seeds",
    "crafted_like": RUN_ROOT / "crafted_like_shift_pppm1e-6",
    "pppm_1e-4": RUN_ROOT / "pppm_1e-4_sensitivity",
    "pppm_1e-5": RUN_ROOT / "pppm_1e-5_sensitivity",
    "pppm_1e-6": RUN_ROOT / "pppm_1e-6_sensitivity",
}


def analyze(output_dir: str | Path, runs: dict[str, Path] | None = None) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_map = runs or DEFAULT_RUNS

    run_rows = []
    seed_rows = []
    for label, run_dir in run_map.items():
        if not (run_dir / "evaluated_isotherm.csv").exists():
            continue
        run_rows.extend(_load_evaluated_rows(label, run_dir))
        seed_rows.extend(_load_seed_rows(label, run_dir))

    reference_rows = _load_crafted_reference(REFERENCE_PATH)
    reference_by_pressure = {row["pressure_bar"]: row for row in reference_rows}
    seed_summary_rows = _seed_summary(seed_rows, reference_by_pressure)
    deviation_rows = _deviation_rows(run_rows, reference_by_pressure)

    outputs = {
        "run_table": output / "mof5_run_table.csv",
        "seed_table": output / "mof5_seed_table.csv",
        "seed_summary": output / "mof5_seed_summary.csv",
        "deviation_table": output / "mof5_deviation_to_crafted.csv",
        "report": output / "mof5_run_suite_report.md",
        "absolute_plot": output / "mof5_absolute_vs_crafted.png",
        "excess_plot": output / "mof5_excess_vs_crafted.png",
        "seed_plot": output / "mof5_seed_reproducibility.png",
        "seed_delta_plot": output / "mof5_seed_delta_to_crafted.png",
        "kspace_plot": output / "mof5_kspace_sensitivity.png",
        "kspace_delta_plot": output / "mof5_kspace_delta_to_pppm_1e-6.png",
        "crafted_like_plot": output / "mof5_crafted_like_vs_crafted.png",
    }

    _write_csv(run_rows, outputs["run_table"])
    _write_csv(seed_rows, outputs["seed_table"])
    _write_csv(seed_summary_rows, outputs["seed_summary"])
    _write_csv(deviation_rows, outputs["deviation_table"])
    _write_report(outputs["report"], run_rows, seed_summary_rows, deviation_rows)
    _write_reference_comparison_plot(run_rows, reference_rows, outputs["absolute_plot"], "absolute")
    _write_reference_comparison_plot(run_rows, reference_rows, outputs["excess_plot"], "excess")
    _write_seed_plot(seed_rows, seed_summary_rows, reference_rows, outputs["seed_plot"])
    _write_seed_delta_plot(seed_summary_rows, outputs["seed_delta_plot"])
    _write_kspace_plot(run_rows, outputs["kspace_plot"])
    _write_kspace_delta_plot(run_rows, outputs["kspace_delta_plot"], baseline_label="pppm_1e-6")
    _write_crafted_like_only_plot(run_rows, reference_rows, outputs["crafted_like_plot"])

    return {
        "status": "completed",
        "run_rows": len(run_rows),
        "seed_rows": len(seed_rows),
        "seed_summary_rows": len(seed_summary_rows),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def _load_evaluated_rows(label: str, run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    metadata = _load_run_metadata(run_dir)
    with (run_dir / "evaluated_isotherm.csv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "run_label": label,
                    "run_directory": str(run_dir),
                    "pressure_bar": _float(row.get("pressure_bar")),
                    "absolute_mol_per_kg": _float(row.get("loading_absolute_mol_per_kg")),
                    "excess_mol_per_kg": _float(row.get("loading_excess_mol_per_kg")),
                    "reference_mol_per_kg": _float(row.get("reference_mol_per_kg")),
                    "relative_error_percent": _float(row.get("comparison_relative_error_percent")),
                    "pore_volume_cm3_g": _float(row.get("pore_volume_cm3_g")),
                    "gas_density_mol_per_m3": _float(row.get("gas_density_mol_per_m3")),
                    "kspace_style": metadata.get("kspace_style", ""),
                    "kspace_accuracy": metadata.get("kspace_accuracy", ""),
                    "cycles": metadata.get("cycles", ""),
                    "initialization_cycles": metadata.get("initialization_cycles", ""),
                }
            )
    return sorted(rows, key=lambda row: (row["run_label"], row["pressure_bar"]))


def _load_seed_rows(label: str, run_dir: Path) -> list[dict[str, Any]]:
    summary_path = run_dir / "isotherm_summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evaluated = {
        row["pressure_bar"]: row
        for row in _load_evaluated_rows(label, run_dir)
        if row["pressure_bar"] is not None
    }
    rows = []
    for replicate in summary.get("replicate_results", []):
        pressure = _float(replicate.get("pressure_bar"))
        seed = replicate.get("seed")
        mean_adsorbates = _float((replicate.get("summary") or {}).get("mean_adsorbates"))
        evaluated_row = evaluated.get(pressure)
        if pressure is None or seed is None or mean_adsorbates is None or evaluated_row is None:
            continue
        framework_mass = _framework_mass_amu(run_dir, pressure)
        if framework_mass is None:
            continue
        absolute = mean_adsorbates * 1000.0 / framework_mass
        pore_volume = _float(evaluated_row.get("pore_volume_cm3_g"))
        gas_density = _float(evaluated_row.get("gas_density_mol_per_m3"))
        excess = None
        if pore_volume is not None and gas_density is not None:
            excess = absolute - gas_density * pore_volume * 1e-3
        rows.append(
            {
                "run_label": label,
                "run_directory": str(run_dir),
                "pressure_bar": pressure,
                "seed": seed,
                "mean_adsorbates_per_cell": mean_adsorbates,
                "absolute_mol_per_kg": absolute,
                "excess_mol_per_kg": excess,
            }
        )
    return sorted(rows, key=lambda row: (row["pressure_bar"], row["seed"]))


def _framework_mass_amu(run_dir: Path, pressure: float) -> float | None:
    csv_path = run_dir / "evaluated_isotherm.csv"
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _float(row.get("pressure_bar")) == pressure:
                return _float(row.get("framework_mass_amu"))
    return None


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "prepare_summary.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("result", {}).get("parameters") or data.get("prepare_plan", {}).get("parameters", {})


def _load_crafted_reference(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            pressure_pa, loading, error = [float(value) for value in line.split(",")]
            rows.append(
                {
                    "pressure_bar": pressure_pa / 100000.0,
                    "absolute_mol_per_kg": loading,
                    "error_mol_per_kg": error,
                }
            )
    return rows


def _seed_summary(seed_rows: list[dict[str, Any]], reference_by_pressure: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in seed_rows:
        if row["run_label"] not in {"seed_12345", "seeds_104729_209759"}:
            continue
        grouped.setdefault(row["pressure_bar"], []).append(row)

    rows = []
    for pressure, values in sorted(grouped.items()):
        if len(values) < 2:
            continue
        absolutes = [row["absolute_mol_per_kg"] for row in values]
        excesses = [row["excess_mol_per_kg"] for row in values if row["excess_mol_per_kg"] is not None]
        reference = reference_by_pressure.get(pressure, {}).get("absolute_mol_per_kg")
        abs_mean = mean(absolutes)
        abs_sd = stdev(absolutes) if len(absolutes) > 1 else 0.0
        rel_sd = abs_sd / abs_mean * 100.0 if abs_mean else None
        delta = abs_mean - reference if reference is not None else None
        delta_pct = delta / reference * 100.0 if reference not in (None, 0.0) else None
        rows.append(
            {
                "pressure_bar": pressure,
                "seed_count": len(values),
                "seeds": " ".join(str(row["seed"]) for row in values),
                "absolute_mean_mol_per_kg": abs_mean,
                "absolute_sd_mol_per_kg": abs_sd,
                "absolute_relative_sd_percent": rel_sd,
                "excess_mean_mol_per_kg": mean(excesses) if excesses else "",
                "crafted_reference_mol_per_kg": reference if reference is not None else "",
                "absolute_delta_to_crafted_mol_per_kg": delta if delta is not None else "",
                "absolute_delta_to_crafted_percent": delta_pct if delta_pct is not None else "",
            }
        )
    return rows


def _deviation_rows(run_rows: list[dict[str, Any]], reference_by_pressure: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in run_rows:
        pressure = row["pressure_bar"]
        reference = reference_by_pressure.get(pressure, {}).get("absolute_mol_per_kg")
        if reference in (None, 0.0) or row["absolute_mol_per_kg"] is None:
            continue
        delta = row["absolute_mol_per_kg"] - reference
        rows.append(
            {
                "run_label": row["run_label"],
                "pressure_bar": pressure,
                "absolute_mol_per_kg": row["absolute_mol_per_kg"],
                "crafted_reference_mol_per_kg": reference,
                "absolute_delta_mol_per_kg": delta,
                "absolute_delta_percent": delta / reference * 100.0,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, run_rows: list[dict[str, Any]], seed_summary_rows: list[dict[str, Any]], deviation_rows: list[dict[str, Any]]) -> None:
    seed_rows_by_pressure = {row["pressure_bar"]: row for row in seed_summary_rows}
    lines = [
        "# MOF-5 CO2 Run Suite",
        "",
        "## Seed reproducibility",
        "",
        "| pressure / bar | seeds | mean absolute / mol kg^-1 | rel. SD / % | delta to CRAFTED / % |",
        "|---:|:---|---:|---:|---:|",
    ]
    for pressure in sorted(seed_rows_by_pressure):
        row = seed_rows_by_pressure[pressure]
        lines.append(
            f"| {pressure:g} | {row['seeds']} | {_fmt(row['absolute_mean_mol_per_kg'])} | "
            f"{_fmt(row['absolute_relative_sd_percent'])} | {_fmt(row['absolute_delta_to_crafted_percent'])} |"
        )

    lines.extend(["", "## Run deviations to CRAFTED", ""])
    interesting = [row for row in deviation_rows if row["run_label"] in {"seed_12345", "seeds_104729_209759", "crafted_like"}]
    lines.extend(
        [
            "| run | pressure / bar | absolute / mol kg^-1 | CRAFTED / mol kg^-1 | delta / % |",
            "|:---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(interesting, key=lambda item: (item["pressure_bar"], item["run_label"])):
        lines.append(
            f"| {row['run_label']} | {row['pressure_bar']:g} | {_fmt(row['absolute_mol_per_kg'])} | "
            f"{_fmt(row['crafted_reference_mol_per_kg'])} | {_fmt(row['absolute_delta_percent'])} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Seed statistics combine seed 12345 with the two-seed conventional run when pressures overlap.",
            "- CRAFTED-like and PPPM sensitivity runs are single-seed checks and should be interpreted as method sensitivity, not sampling uncertainty.",
            "- Excess loading is computed from the evaluated run files using the configured pore volume and gas density.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reference_comparison_plot(run_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], path: Path, basis: str) -> None:
    import matplotlib.pyplot as plt

    key = f"{basis}_mol_per_kg"
    labels = ["seed_12345", "seeds_104729_209759", "crafted_like"]
    gas_correction = _gas_correction_by_pressure(run_rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    plotted_reference = _reference_for_basis(reference_rows, basis, gas_correction)
    ax.errorbar(
        [row["pressure_bar"] for row in plotted_reference],
        [row[f"{basis}_mol_per_kg"] for row in plotted_reference],
        yerr=[row["error_mol_per_kg"] for row in plotted_reference],
        color="black",
        linewidth=1.5,
        marker="o",
        markersize=3,
        label=f"CRAFTED reference ({basis})",
    )
    for label in labels:
        points = sorted(
            (row for row in run_rows if row["run_label"] == label and row.get(key) is not None),
            key=lambda row: row["pressure_bar"],
        )
        if not points:
            continue
        ax.plot(
            [row["pressure_bar"] for row in points],
            [row[key] for row in points],
            marker="o",
            linewidth=1.3,
            label=label,
        )
    _finish_pressure_plot(ax, f"MOF-5 CO2 {basis} loading", "Loading / mol kg$^{-1}$")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _gas_correction_by_pressure(run_rows: list[dict[str, Any]]) -> dict[float, float]:
    corrections = {}
    for row in run_rows:
        pressure = row.get("pressure_bar")
        pore_volume = row.get("pore_volume_cm3_g")
        gas_density = row.get("gas_density_mol_per_m3")
        if pressure is None or pore_volume is None or gas_density is None:
            continue
        corrections.setdefault(float(pressure), float(gas_density) * float(pore_volume) * 1e-3)
    return corrections


def _reference_for_basis(reference_rows: list[dict[str, Any]], basis: str, gas_correction: dict[float, float]) -> list[dict[str, Any]]:
    rows = []
    for row in reference_rows:
        pressure = row["pressure_bar"]
        if basis == "absolute":
            rows.append({**row, "absolute_mol_per_kg": row["absolute_mol_per_kg"]})
        elif pressure in gas_correction:
            rows.append({**row, "excess_mol_per_kg": row["absolute_mol_per_kg"] - gas_correction[pressure]})
    return rows


def _write_seed_plot(seed_rows: list[dict[str, Any]], seed_summary_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    relevant = [row for row in seed_rows if row["run_label"] in {"seed_12345", "seeds_104729_209759"}]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.errorbar(
        [row["pressure_bar"] for row in reference_rows],
        [row["absolute_mol_per_kg"] for row in reference_rows],
        yerr=[row["error_mol_per_kg"] for row in reference_rows],
        color="black",
        linewidth=1.4,
        marker="o",
        markersize=3,
        label="CRAFTED reference",
    )
    for seed in sorted({row["seed"] for row in relevant}):
        points = sorted(
            (row for row in relevant if row["seed"] == seed),
            key=lambda row: row["pressure_bar"],
        )
        ax.plot(
            [row["pressure_bar"] for row in points],
            [row["absolute_mol_per_kg"] for row in points],
            marker=".",
            linewidth=1.0,
            alpha=0.7,
            label=f"seed {seed}",
        )
    summary_points = sorted(seed_summary_rows, key=lambda row: row["pressure_bar"])
    ax.errorbar(
        [row["pressure_bar"] for row in summary_points],
        [row["absolute_mean_mol_per_kg"] for row in summary_points],
        yerr=[row["absolute_sd_mol_per_kg"] for row in summary_points],
        color="tab:red",
        linewidth=2.0,
        marker="o",
        label="3-seed mean +/- SD",
    )
    _finish_pressure_plot(ax, "MOF-5 CO2 seed reproducibility", "Absolute loading / mol kg$^{-1}$")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_seed_delta_plot(seed_summary_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in seed_summary_rows if row["absolute_delta_to_crafted_percent"] != ""]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot(
        [row["pressure_bar"] for row in rows],
        [row["absolute_delta_to_crafted_percent"] for row in rows],
        marker="o",
        linewidth=1.8,
        label="3-seed mean",
    )
    ax.fill_between(
        [row["pressure_bar"] for row in rows],
        [
            row["absolute_delta_to_crafted_percent"] - row["absolute_relative_sd_percent"]
            for row in rows
        ],
        [
            row["absolute_delta_to_crafted_percent"] + row["absolute_relative_sd_percent"]
            for row in rows
        ],
        alpha=0.2,
        label="approx. +/- relative SD",
    )
    _finish_pressure_plot(ax, "3-seed deviation to CRAFTED", "Delta / %")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_kspace_plot(run_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["pppm_1e-4", "pppm_1e-5", "pppm_1e-6"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for label in labels:
        points = sorted(
            (row for row in run_rows if row["run_label"] == label),
            key=lambda row: row["pressure_bar"],
        )
        if not points:
            continue
        ax.plot(
            [row["pressure_bar"] for row in points],
            [row["absolute_mol_per_kg"] for row in points],
            marker="o",
            linewidth=1.5,
            label=label,
        )
    _finish_pressure_plot(ax, "PPPM sensitivity", "Absolute loading / mol kg$^{-1}$")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_kspace_delta_plot(run_rows: list[dict[str, Any]], path: Path, baseline_label: str) -> None:
    import matplotlib.pyplot as plt

    baseline = {
        row["pressure_bar"]: row["absolute_mol_per_kg"]
        for row in run_rows
        if row["run_label"] == baseline_label and row["absolute_mol_per_kg"] not in (None, 0.0)
    }
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.axhline(0.0, color="black", linewidth=1.0)
    for label in ["pppm_1e-4", "pppm_1e-5"]:
        points = []
        for row in run_rows:
            if row["run_label"] != label:
                continue
            base = baseline.get(row["pressure_bar"])
            if base in (None, 0.0):
                continue
            points.append((row["pressure_bar"], (row["absolute_mol_per_kg"] - base) / base * 100.0))
        points.sort()
        if points:
            ax.plot([p for p, _ in points], [delta for _, delta in points], marker="o", linewidth=1.5, label=f"{label} vs {baseline_label}")
    _finish_pressure_plot(ax, "PPPM deviation to 1e-6", "Delta / %")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_crafted_like_only_plot(run_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    gas_correction = _gas_correction_by_pressure(run_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharex=False)
    for ax, basis in zip(axes, ["absolute", "excess"], strict=True):
        reference = _reference_for_basis(reference_rows, basis, gas_correction)
        key = f"{basis}_mol_per_kg"
        crafted_like = sorted(
            (
                row
                for row in run_rows
                if row["run_label"] == "crafted_like" and row.get(key) is not None
            ),
            key=lambda row: row["pressure_bar"],
        )
        ax.errorbar(
            [row["pressure_bar"] for row in reference],
            [row[key] for row in reference],
            yerr=[row["error_mol_per_kg"] for row in reference],
            color="black",
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=f"CRAFTED reference ({basis})",
        )
        ax.plot(
            [row["pressure_bar"] for row in crafted_like],
            [row[key] for row in crafted_like],
            marker="o",
            linewidth=1.7,
            label="crafted_like LAMMPS",
        )
        _finish_pressure_plot(ax, basis.capitalize(), "Loading / mol kg$^{-1}$")
    fig.suptitle("MOF-5 CO2 crafted-like comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _finish_pressure_plot(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Pressure / bar")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    _use_plain_pressure_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend()


def _use_plain_pressure_ticks(ax: Any) -> None:
    pressures = set()
    for line in ax.lines:
        for value in line.get_xdata():
            if value is None:
                continue
            value = float(value)
            if value > 0:
                pressures.add(value)
    if not pressures:
        return
    ticks = sorted(pressures)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in ticks])
    if len(ticks) > 8:
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    ax.minorticks_off()


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _fmt(value: Any) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.6g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze completed MOF-5 CO2 run suite.")
    parser.add_argument("--output-dir", default="reports/mof5_run_suite", help="Directory for CSV, report, and PNG outputs.")
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
