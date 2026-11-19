"""Plot a returned guarded-pilot pressure screening against its frozen reference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.plotting import decimal_pressure_axis


STYLES = {
    "umof_mh_pretrained": ("uMOF-MH pretrained", "#2676a6", "o"),
    "umof_polar_pretrained": ("uMOF-POLAR pretrained", "#438b69", "s"),
    "umof_polar_stage1": ("uMOF-POLAR Stage 1", "#d27031", "^"),
    "umof_mh_finetuned_025pct": ("uMOF-MH fine-tuned (0.25%)", "#8b6cab", "P"),
}


def create_plot(bundle: Path, output: Path) -> None:
    config = json.loads((bundle / "pilot.json").read_text())
    assessment = json.loads((bundle / "report/assessment.json").read_text())
    if len(config["seeds"]) != 1:
        raise ValueError("This figure is specifically labelled for single-seed screening")
    if (assessment["missing"] or assessment["completed_jobs"] != assessment["expected_jobs"]):
        raise ValueError("Run guarded_pilot report after completing all pressure points")
    with (bundle / "report/points.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    pressures = sorted(config["pressures_bar"])
    models = [spec["label"] for spec in config["models"]]
    expected = {(model, pressure) for model in models for pressure in pressures}
    identities = [(row["model"], float(row["pressure_bar"])) for row in rows]
    if set(identities) != expected or len(identities) != len(expected):
        raise ValueError("Incomplete or duplicate model/pressure grid")
    reference_path = bundle / config["reference"]["path"]
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != config["reference"]["sha256"]:
        raise ValueError("Frozen reference checksum mismatch")
    reference = np.loadtxt(reference_path, delimiter=",", ndmin=2)
    reference = reference[np.argsort(reference[:, 0])]
    reference[:, 0] /= 1e5  # CRAFTED pressure: Pa -> bar; mol/kg equals mmol/g.
    if pressures[0] < reference[0, 0] or pressures[-1] > reference[-1, 0]:
        raise ValueError("Reference extrapolation is forbidden")
    for row in rows:
        pressure, loading = float(row["pressure_bar"]), float(row["mean"])
        matched = float(np.interp(pressure, reference[:, 0], reference[:, 1]))
        if (int(row["replicates"]) != 1 or not np.isfinite(loading)
                or not np.isclose(matched, float(row["reference_mmol_g"]), rtol=1e-10)):
            raise ValueError("Report values or reference comparison are inconsistent")
    visible = reference[(reference[:, 0] >= pressures[0]) & (reference[:, 0] <= pressures[-1])]
    # Include interpolated endpoints if the pressure range cuts between reference points.
    ref_pressure = np.unique(np.r_[pressures[0], visible[:, 0], pressures[-1]])
    ref_loading = np.interp(ref_pressure, reference[:, 0], reference[:, 1])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
    fig.subplots_adjust(left=.075, right=.98, bottom=.20, top=.70, wspace=.12)
    for model in models:
        label, color, marker = STYLES[model]
        selected = sorted((r for r in rows if r["model"] == model),
                          key=lambda r: float(r["pressure_bar"]))
        x = [float(r["pressure_bar"]) for r in selected]
        y = [float(r["mean"]) for r in selected]
        for ax in axes:
            ax.plot(x, y, color=color, marker=marker, label=label,
                    markersize=6, linewidth=2)
    for ax in axes:
        ax.plot(ref_pressure, ref_loading, "D--", color="#252c35", markersize=4,
                linewidth=1.8, label="CRAFTED UFF/DDEC reference", zorder=4)
        ax.set_xlabel("Pressure (bar)", labelpad=9)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_title("Linear pressure axis", loc="left", fontsize=12, pad=12)
    axes[0].set_xlim(0, pressures[-1] * 1.025)
    axes[0].set_ylabel("Absolute CO₂ loading (mmol/g)", labelpad=10)
    decimal_pressure_axis(axes[0])
    axes[1].set_title("Logarithmic pressure axis", loc="left", fontsize=12, pad=12)
    axes[1].set_xscale("log")
    axes[1].set_xlim(pressures[0] / 1.12, pressures[-1] * 1.12)
    decimal_pressure_axis(axes[1], pressures)
    fig.suptitle(f"MOF-5 / CO₂ adsorption at {config['temperature_K']:g} K",
                 x=.075, y=.97, ha="left", fontsize=20, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.068, .89),
               ncol=2, frameon=False, columnspacing=3, handlelength=2.8)
    fig.text(.075, .065,
             "Single seed per model and pressure; no between-seed confidence intervals.\n"
             "Screening results: equilibrium is not established; some chain screens fail. "
             "CRAFTED is a classical simulation reference.",
             fontsize=9, color="#52606c", linespacing=1.7)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        path = output / f"isotherm_comparison.{extension}"
        fig.savefig(path, dpi=240, facecolor="white")
        print(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="Returned guarded-pilot bundle")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    create_plot(args.directory, args.output_dir or args.directory / "report")


if __name__ == "__main__":
    main()
