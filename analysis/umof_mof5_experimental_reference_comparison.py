"""Compare the four MOF-5/CO2 uMOF screening curves with experimental isotherms."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from analysis.plotting import decimal_pressure_axis


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "reports/umof_mof5_single_seed_isotherm"
OUT = ROOT / "reports/umof_mof5_experimental_reference_comparison"
LIB = ROOT / "isodb-library/Library"
MOLAR_MASS_CO2_MG_PER_MMOL = 44.0095

MODELS = {
    "umof_mh_pretrained": ("uMOF-MH pretrained", "#2676a6", "o"),
    "umof_polar_pretrained": ("uMOF-POLAR pretrained", "#438b69", "s"),
    "umof_polar_stage1": ("uMOF-POLAR Stage 1", "#d27031", "^"),
    "umof_mh_finetuned_025pct": ("uMOF-MH fine-tuned (0.25%)", "#8b6cab", "P"),
}

REFERENCES = [
    {
        "filename": "10.1016j.micromeso.2008.04.033.isotherm2.json",
        "label": "Microwave-synthesized MOF-5 (2008, 298 K)",
        "short": "Microwave MOF-5, 298 K [1]",
        "citation": "J.-S. Choi et al., Microporous and Mesoporous Materials 116 (2008), DOI: 10.1016/j.micromeso.2008.04.033",
        "doi": "10.1016/j.micromeso.2008.04.033",
        "temperature_K": 298,
        "reported_basis": "Not specified in the NIST record; source reports gravimetric uptake",
    },
    {
        "filename": "10.1038srep12045.Isotherm2.json",
        "label": "MOF-5 (2015, 298 K)",
        "short": "MOF-5, 298 K [2]",
        "citation": "J. H. Park et al., Scientific Reports 5, 12045 (2015), DOI: 10.1038/srep12045",
        "doi": "10.1038/srep12045",
        "temperature_K": 298,
        "reported_basis": "Not specified in the NIST record; source reports gravimetric uptake",
    },
    {
        "filename": "10.1016j.seppur.2018.01.013.Isotherm11.json",
        "label": "MOF-5 (2018, 297 K)",
        "short": "MOF-5, 297 K [3]",
        "citation": "F. A. Kloutse et al., Separation and Purification Technology 197 (2018), DOI: 10.1016/j.seppur.2018.01.013",
        "doi": "10.1016/j.seppur.2018.01.013",
        "temperature_K": 297,
        "reported_basis": "Not specified in the NIST record; paper reports volumetric pure-component isotherm",
    },
    {
        # Walton et al. Fig. 1: NIST Isotherm3 is the experimental curve;
        # Isotherms1 and 2 are their GCMC curves (with/without electrostatics).
        "filename": "10.1021Ja076595g.Isotherm3.json",
        "label": "Walton et al. experiment (2008, 298 K)",
        "short": "Walton experiment, 298 K [4]",
        "citation": "K. S. Walton et al., Journal of the American Chemical Society 130 (2008), DOI: 10.1021/ja076595g",
        "doi": "10.1021/ja076595g",
        "temperature_K": 298,
        "reported_basis": "NIST record does not identify absolute versus excess uptake",
    },
]


def _read_reference(spec: dict) -> dict:
    matches = list(LIB.rglob(spec["filename"]))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one mirrored NIST record named {spec['filename']}")
    path = matches[0]
    data = json.loads(path.read_text())
    if (data.get("category") == "sim" or data.get("adsorbent", {}).get("name", "").lower() != "irmof-1"
            or data.get("temperature") != spec["temperature_K"]
            or [a.get("name") for a in data.get("adsorbates", [])] != ["Carbon Dioxide"]):
        raise ValueError(f"Unexpected NIST isotherm metadata: {path}")
    pressure_unit = data["pressureUnits"]
    if pressure_unit.lower() != "bar":
        raise ValueError(f"Unexpected pressure units: {pressure_unit}")
    uptake_unit = data["adsorptionUnits"].lower()
    factor = {"mmol/g": 1.0, "mg/g": 1.0 / MOLAR_MASS_CO2_MG_PER_MMOL}.get(uptake_unit)
    if factor is None:
        raise ValueError(f"Unsupported experimental uptake unit: {uptake_unit}")
    points = data["isotherm_data"]
    p = np.asarray([float(x["pressure"]) for x in points])
    q = np.asarray([float(x["species_data"][0]["adsorption"]) * factor for x in points])
    order = np.argsort(p)
    return {**spec, "source_path": str(path.relative_to(ROOT)), "raw_units": data["adsorptionUnits"],
            "pressures_bar": p[order].tolist(), "loading_mmol_g": q[order].tolist(),
            "uptake_basis_metadata": data.get("isotherm_type") or "not specified in NIST record"}


def _log_interpolate(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    positive = (xp > 0) & (fp > 0)
    xp, fp = xp[positive], fp[positive]
    if x < xp.min() or x > xp.max():
        raise ValueError("Reference extrapolation is forbidden")
    return float(np.exp(np.interp(np.log(x), np.log(xp), np.log(fp))))


def main() -> None:
    assessment = json.loads((BUNDLE / "report/assessment.json").read_text())
    if assessment["missing"] or assessment["completed_jobs"] != assessment["expected_jobs"]:
        raise ValueError("The four-model single-seed campaign is incomplete")
    with (BUNDLE / "report/points.csv").open(newline="") as f:
        input_rows = list(csv.DictReader(f))
    models = {}
    for row in input_rows:
        models.setdefault(row["model"], []).append((float(row["pressure_bar"]), float(row["mean"])))
    if set(models) != set(MODELS) or any(len(v) != 5 for v in models.values()):
        raise ValueError("Expected all four models on the five-point pressure grid")
    references = [_read_reference(spec) for spec in REFERENCES]
    walton_gcmc = []
    walton_file = LIB / "10.1021ja076595g"
    for number, label in ((1, "Walton GCMC, LJ + electrostatics"),
                          (2, "Walton GCMC, LJ only")):
        record = json.loads((walton_file / f"10.1021Ja076595g.Isotherm{number}.json").read_text())
        wx = np.asarray([float(p["pressure"]) for p in record["isotherm_data"]])
        wy = np.asarray([float(p["species_data"][0]["adsorption"]) for p in record["isotherm_data"]])
        for pressure in (5.0, 10.0):
            walton_gcmc.append({"series": label, "pressure_bar": pressure,
                                "uptake_mmol_g": _log_interpolate(pressure, wx, wy)})
    with (OUT / "walton_gcmc_crosscheck.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(walton_gcmc[0]))
        writer.writeheader(); writer.writerows(walton_gcmc)
    crafted = np.loadtxt(BUNDLE / "assets/reference.csv", delimiter=",")
    crafted = crafted[np.argsort(crafted[:, 0])]
    crafted_p = crafted[:, 0] / 1e5
    crafted_q = crafted[:, 1]

    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = []
    for model, values in models.items():
        values.sort()
        for p, q in values:
            all_rows.append({"series": MODELS[model][0], "source_type": "MLIP-MC single-seed screening",
                             "pressure_bar": p, "loading_mmol_g": q, "temperature_K": 298,
                             "source": "guarded-pilot result"})
        model_p = np.asarray([v[0] for v in values])
        model_q = np.asarray([v[1] for v in values])
        # Compare only pressure points bracketed by each reference; never extrapolate.
        for ref in references:
            xp = np.asarray(ref["pressures_bar"])
            fp = np.asarray(ref["loading_mmol_g"])
            usable = (xp > 0) & (fp > 0)
            pairs = [(p, q) for p, q in values if p >= xp[usable].min() and p <= xp[usable].max()]
            errors = []
            for p, q in pairs:
                ref_q = _log_interpolate(p, xp, fp)
                errors.append(q - ref_q)
                summary.append({"model": model, "reference": ref["label"], "pressure_bar": p,
                                "model_mmol_g": q, "reference_mmol_g": ref_q,
                                "difference_mmol_g": q-ref_q,
                                "relative_difference_percent": 100*(q-ref_q)/ref_q})
    with (OUT / "reference_deviations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    for ref in references:
        for p, q in zip(ref["pressures_bar"], ref["loading_mmol_g"]):
            all_rows.append({"series": ref["label"], "source_type": "experiment (NIST/ARPA-E record)",
                             "pressure_bar": p, "loading_mmol_g": q, "temperature_K": ref["temperature_K"],
                             "source": ref["doi"]})
    for p, q in zip(crafted_p, crafted_q):
        if 0 <= p <= 10:
            all_rows.append({"series": "CRAFTED UFF/DDEC (simulation)", "source_type": "classical simulation",
                             "pressure_bar": p, "loading_mmol_g": q, "temperature_K": 298,
                             "source": "frozen MOF-5 pilot reference.csv"})
    with (OUT / "isotherm_data.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.subplots_adjust(left=.075, right=.98, bottom=.27, top=.72, wspace=.12)
    for model, vals in models.items():
        label, color, marker = MODELS[model]
        vals.sort()
        x, y = zip(*vals)
        axes[0].plot(x, y, color=color, marker=marker, markersize=5.5,
                     linewidth=1.8, label=label, zorder=4)
        positive = np.asarray(x) > 0
        axes[1].plot(np.asarray(x)[positive], np.asarray(y)[positive],
                     color=color, marker=marker, markersize=5.5,
                     linewidth=1.8, label=label, zorder=4)
    ref_colors = ["#008c95", "#bb5a35", "#8064a2", "#b28a18"]
    for ref, color in zip(references, ref_colors):
        p = np.asarray(ref["pressures_bar"]); q = np.asarray(ref["loading_mmol_g"])
        axes[0].plot(p, q, linestyle="--", marker="D", markersize=4.2,
                     linewidth=1.55, color=color, label=ref["short"], zorder=3)
        positive = p > 0
        axes[1].plot(p[positive], q[positive], linestyle="--", marker="D",
                     markersize=4.2, linewidth=1.55, color=color,
                     label=ref["short"], zorder=3)
    mask = (crafted_p > 0) & (crafted_p <= 10)
    for ax in axes:
        ax.plot(crafted_p[mask], crafted_q[mask], color="#30343b", linestyle=":",
                linewidth=2, label="CRAFTED UFF/DDEC (simulation)", zorder=2)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.19); ax.set_axisbelow(True)
    axes[0].set_title("Linear pressure axis", loc="left", pad=10)
    axes[0].set_xlim(0, 10.2)
    axes[0].set_ylabel("Reported CO₂ uptake (mmol/g)", labelpad=10)
    decimal_pressure_axis(axes[0])
    axes[1].set_title("Logarithmic pressure axis", loc="left", pad=10)
    axes[1].set_xscale("log"); axes[1].set_xlim(.07, 10.5)
    decimal_pressure_axis(axes[1], [0.1, 0.5, 1, 5, 10])
    fig.supxlabel("Pressure (bar)", y=.205)
    fig.suptitle("MOF-5 / CO₂ isotherm comparison at 297–298 K", x=.075, y=.97,
                 ha="left", fontsize=18, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.055, .885),
               ncol=3, frameon=False, columnspacing=1.5, handlelength=2.1, fontsize=9.0)
    fig.text(.075, .125,
             "Experimental points digitized or curated in NIST/ARPA-E. Reported loading basis is not identified in the NIST records;\n"
             "comparisons to absolute GCMC loading are therefore indicative. The 2018 dataset is at 297 K; the others are at 298 K.\n"
             "Single seed per MLIP/pressure; equilibrium is not established and several chain screens fail.",
             fontsize=8.4, color="#4e5b66", linespacing=1.55)
    fig.text(.075, .035,
             "[1] Choi et al. (2008), DOI: 10.1016/j.micromeso.2008.04.033.  "
             "[2] Park et al. (2015), DOI: 10.1038/srep12045.\n"
             "[3] Kloutse et al. (2018), DOI: 10.1016/j.seppur.2018.01.013; "
             "[4] Walton et al. (2008), DOI: 10.1021/ja076595g.\n"
             "Experimental records: NIST/ARPA-E SRD 205; CRAFTED is a simulation reference.",
             fontsize=8.0, color="#4e5b66", linespacing=1.5)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"isotherm_comparison_experimental.{ext}", dpi=240, facecolor="white")
    plt.close(fig)

    # Focused matched-pressure view: all three experimental records bracket 5 and 10 bar;
    # only the Kloutse record brackets 1 bar. Do not imply experimental coverage at 0.1/0.5 bar.
    target_pressures = np.asarray([1.0, 5.0, 10.0])
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    fig.subplots_adjust(left=.12, right=.98, bottom=.27, top=.83)
    for model, vals in models.items():
        label, color, marker = MODELS[model]
        by_pressure = dict(vals)
        y = [by_pressure[p] for p in target_pressures]
        ax.plot(target_pressures, y, color=color, marker=marker, markersize=7,
                linewidth=1.8, label=label, zorder=4)
    ref_markers = ["D", "o", "v", "s"]
    for ref, color, marker in zip(references, ref_colors, ref_markers):
        xp = np.asarray(ref["pressures_bar"]); fp = np.asarray(ref["loading_mmol_g"])
        available_p, available_q = [], []
        for pressure in target_pressures:
            positive = (xp > 0) & (fp > 0)
            if xp[positive].min() <= pressure <= xp[positive].max():
                available_p.append(pressure)
                available_q.append(_log_interpolate(pressure, xp, fp))
        ax.scatter(available_p, available_q, s=62, marker=marker, facecolors="white",
                   edgecolors=color, linewidths=1.8, label=ref["short"], zorder=5)
    crafted_at_targets = [_log_interpolate(p, crafted_p[(crafted_p > 0)],
                                             crafted_q[(crafted_p > 0)])
                          for p in target_pressures]
    ax.plot(target_pressures, crafted_at_targets, color="#30343b", linestyle=":",
            marker="x", markersize=7, linewidth=2, label="CRAFTED UFF/DDEC (simulation)", zorder=3)
    ax.set_xscale("log"); ax.set_xlim(.82, 12)
    decimal_pressure_axis(ax, target_pressures)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("CO₂ uptake (mmol/g)")
    ax.grid(alpha=.2); ax.set_axisbelow(True)
    fig.supxlabel("Pressure (bar)", y=.205)
    ax.set_title("MOF-5 / CO₂ comparison at shared pressures", loc="left", fontsize=16, weight="bold", pad=16)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.01), ncol=2,
              frameon=False, fontsize=8.3, columnspacing=1.5, handlelength=2)
    fig.text(.12, .135,
             "Experimental values use log–log interpolation from measurements bracketing each target pressure.\n"
             "Four references are available at 5 and 10 bar; only Kloutse et al. at 1 bar. NIST does not specify absolute/excess uptake basis.\n"
             "MLIP results are single-seed screening values with failed chain screens; treat the comparison as indicative.",
             fontsize=7.8, color="#4e5b66", linespacing=1.45)
    fig.text(.12, .035,
             "Sources: Choi et al. (2008), DOI 10.1016/j.micromeso.2008.04.033; Park et al. (2015), DOI 10.1038/srep12045;\n"
             "Kloutse et al. (2018), DOI 10.1016/j.seppur.2018.01.013; Walton et al. (2008), DOI 10.1021/ja076595g. NIST/ARPA-E SRD 205.",
             fontsize=8, color="#4e5b66", linespacing=1.5)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"matched_pressure_comparison.{ext}", dpi=240, facecolor="white")
    plt.close(fig)

    report = ["# MOF-5 / CO₂: comparison against experimental references", "",
              "Four uMOF model curves are compared with four experimental IRMOF-1/MOF-5 records and the original CRAFTED UFF/DDEC simulated baseline.", "",
              "## Experimental references", ""]
    for i, ref in enumerate(references, 1):
        report.append(f"{i}. {ref['citation']}. {ref['temperature_K']} K; {len(ref['pressures_bar'])} NIST points; "
                      f"{min(ref['pressures_bar']):g}–{max(ref['pressures_bar']):g} bar; original uptake units {ref['raw_units']}. "
                      f"{ref['reported_basis']}.")
    report += ["", "## Interpretation", "",
               "The experiment records provide a stronger empirical anchor than CRAFTED, which is a classical GCMC simulation baseline. "
               "Their curves differ from each other, reflecting sample/protocol variation. The NIST records do not declare absolute versus excess uptake; "
               "the plotted experimental values are retained on their reported uptake scale after unit conversion. Relative deviations are supplied as exploratory comparisons, "
               "not definitive absolute-loading errors. The 2018 pure-component record is at 297 K and is kept at its measured temperature; no temperature correction was applied.", "",
               "At the shared 5 and 10 bar points, MH pretrained is below all four experiments, while POLAR pretrained, POLAR Stage 1, and MH fine-tuned are above them. "
               "This supports a qualitative screening conclusion only; the unknown uptake basis and single-seed, non-equilibrated runs prevent a defensible accuracy ranking.", "",
               "Walton et al. also report two classical GCMC curves at 298 K: including CO₂ electrostatics reproduces the experimental inflection and loading better, while Lennard–Jones-only interactions underpredict. "
               "Only Walton's experimental curve is included in the experimental series here; their GCMC curves are simulation comparisons, not additional experiments.", "",
               "Walton et al. at common pressures (NIST curve points, log–log interpolated):", "",
               "| Walton series | 5 bar (mmol/g) | 10 bar (mmol/g) |",
               "|---|---:|---:|",
               f"| Experiment | {_log_interpolate(5.0, np.asarray(next(r for r in references if r['doi'] == '10.1021/ja076595g')['pressures_bar']), np.asarray(next(r for r in references if r['doi'] == '10.1021/ja076595g')['loading_mmol_g'])):.2f} | {_log_interpolate(10.0, np.asarray(next(r for r in references if r['doi'] == '10.1021/ja076595g')['pressures_bar']), np.asarray(next(r for r in references if r['doi'] == '10.1021/ja076595g')['loading_mmol_g'])):.2f} |",
               f"| GCMC, LJ + electrostatics | {walton_gcmc[0]['uptake_mmol_g']:.2f} | {walton_gcmc[1]['uptake_mmol_g']:.2f} |",
               f"| GCMC, LJ only | {walton_gcmc[2]['uptake_mmol_g']:.2f} | {walton_gcmc[3]['uptake_mmol_g']:.2f} |", "",
               "Schmid and Cheng (2023; DOI: 10.1063/5.0146711) is a separate computational cross-check, not an experimental reference. They use a fully flexible MOF-FF model with the S₀–TIFEP chemical-potential method at 218 and 300 K. Their 300 K isotherm is slightly above experiment while reproducing its overall shape; they attribute differences from earlier GCMC in part to force-field choice. The paper shows the curve graphically but does not tabulate its isotherm values, so it is not used for a pointwise error calculation here.", "",
               "The MLIP runs use one seed and do not establish equilibrium; failed chain screens further limit the affected pressure points. "
               "The graph is therefore a screening comparison, not a validated accuracy ranking.", "",
               "## Files", "",
              "- `matched_pressure_comparison.png`, `.svg`, `.pdf`: focused comparison at 1, 5 and 10 bar; use this for the main result.",
              "- `isotherm_comparison_experimental.png`, `.svg`, `.pdf`: full measured-isotherm overview; use as supplementary context.",
               "- `isotherm_data.csv`: all plotted points after unit conversion, with source labels.",
               "- `reference_deviations.csv`: exploratory model-minus-experiment values at target pressures bracketed by each reference; log–log interpolation, no extrapolation.", "",
              "- `walton_gcmc_crosscheck.csv`: two Walton GCMC curves at 5 and 10 bar; log–log interpolation, no extrapolation.", "",
               "Experimental data were retrieved from the local mirror of NIST/ARPA-E SRD 205 (`isodb-library/Library`)."]
    (OUT / "REPORT.md").write_text("\n".join(report)+"\n")
    print(f"Wrote comparison to {OUT}")


if __name__ == "__main__":
    main()
