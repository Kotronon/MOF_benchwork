"""Consolidate archived Module C evidence without training or simulation.

Run: python -m analysis.module_c_closure
Outputs are deliberately separate from the source campaigns.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MH = "outputs/module_C_umof_finetuning_matrix_v2"
POLAR = "outputs/umof_polar_24h_float32"
POTENTIAL = "outputs/module_C_potential_benchmark"
MH_ISO = POTENTIAL + "/campaigns/mof5_co2_umof_mh_025pct_stability_v1/comparison"
POLAR_ISO = "outputs/umof_polar_followup_evaluation/crafted_zif8"
COLORS = ["#2676a6", "#d27031", "#4c9277", "#8b6cab"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def error_metrics(predicted, reference):
    """Signed errors are prediction minus reference; never hide nonfinite data."""
    predicted, reference = np.asarray(predicted, float), np.asarray(reference, float)
    if predicted.shape != reference.shape or not predicted.size:
        raise ValueError("Nonempty matched arrays required")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite values in benchmark")
    delta = predicted - reference
    return {"mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt((delta**2).mean())),
            "bias": float(delta.mean())}


def seed_interval(values):
    """Student-t interval across independent seed means, never across MC steps."""
    from scipy.stats import t
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite seed values required")
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values) >= 3 else None
    return {"mean": mean, "replicates": len(values), "ci95_half_width": half,
            "relative_ci95_half_width": half/abs(mean) if half is not None and mean else None}


class Sources:
    def __init__(self, root):
        self.root, self.files = Path(root), {}

    def path(self, relative):
        path = self.root / relative
        self.files[str(relative)] = digest(path)
        return path

    def json(self, relative):
        return json.loads(self.path(relative).read_text())

    def csv(self, relative):
        with self.path(relative).open() as stream:
            return list(csv.DictReader(stream))


def audit_frames(sources):
    """Byte hashes may differ because POLAR adds charge/spin metadata."""
    from ase.io import read
    audit = []
    for split in ("train", "valid", "test"):
        frames = [read(sources.path(f"{root}/025pct/seed-12345/datasets/{split}.extxyz"), index=":")
                  for root in (MH, POLAR)]
        left, right = frames
        matched = len(left) == len(right) and all(
            a.info["umof_frame_id"] == b.info["umof_frame_id"]
            and np.array_equal(a.numbers, b.numbers) and np.array_equal(a.positions, b.positions)
            and np.array_equal(a.cell.array, b.cell.array) and np.array_equal(a.pbc, b.pbc)
            and a.info["REF_energy"] == b.info["REF_energy"]
            and np.array_equal(a.arrays["REF_forces"], b.arrays["REF_forces"])
            for a, b in zip(left, right))
        if not matched:
            raise ValueError(f"MH/POLAR {split} geometry or reference-label mismatch")
        audit.append({"split": split, "entries": len(left),
                      "unique_geometries": len({a.info['umof_frame_id'] for a in left}),
                      "same_geometry_order_and_labels": matched})
    return audit


def collect(sources):
    metrics, runs = [], []
    for label, root, fraction in (("MH 10 %", MH, "010pct"), ("MH 25 %", MH, "025pct"),
                                  ("POLAR 25 %", POLAR, "025pct")):
        prefix = f"{root}/{fraction}/seed-12345"
        status = sources.json(prefix + "/status.json")
        if status["status"] != "completed":
            raise ValueError(f"Selected result is not complete: {prefix}")
        selection = sources.json(prefix + "/results/selection.json")
        if selection["selection_split"] != "validation" or selection["test_used"]:
            raise ValueError("Checkpoint selection used test labels")
        manifest = sources.json(prefix + "/datasets/dataset_manifest.json")
        protocol = sources.json(root + "/campaign.json")["protocol"]
        for row in sources.csv(prefix + "/results/metrics.csv"):
            metrics.append({"run": label, "profile": "MH" if root == MH else "POLAR",
                "split": "test", "training_seed": 12345, "model": row["model"], "group": row["group"],
                "configuration_count": int(row["configuration_count"]),
                **{k.replace("_ev_", "_meV_"): float(row[k])*1000 for k in row if k.startswith(("energy_", "force_"))}})
        runs.append({"run": label, "status": status["status"], "selected_stage": selection["selected"]["stage"],
            "selected_sha256": selection["selected"]["sha256"], "training_frames": manifest["train_configuration_count"],
            "actual_training_fraction": manifest["actual_training_fraction"],
            "training_seconds": status.get("training_seconds"), "protocol": protocol})
    # Validation metrics remain explicitly separated from test results.
    valid = sources.csv("outputs/umof_polar_followup_evaluation/validation_metrics.csv")
    for row in valid:
        row["split"] = "validation"
    stages = sources.csv(POLAR + "/diagnostics/stage_review_2026-10-29/metrics.csv")
    gold_path = POTENTIAL + "/runs/golddac_co2_dispersion_ablation_production_001/reports/potential_metrics.csv"
    raw_gold = sources.csv(gold_path)
    gold = []
    for model in dict.fromkeys(r["candidate_backend"] for r in raw_gold):
        for region in ["all", "equilibrium", "weak_attraction", "repulsive"]:
            rows = [r for r in raw_gold if r["candidate_backend"] == model and (region == "all" or r["region"] == region)]
            values = [float(r["energy_difference_ev"]) for r in rows]
            e = error_metrics(values, np.zeros(len(values)))
            gold.append({"model": model, "region": region, "configurations": len(rows),
                "energy_mae_eV": e["mae"], "energy_rmse_eV": e["rmse"], "energy_bias_eV": e["bias"],
                "mean_configuration_force_mae_eV_A": float(np.mean([float(r["force_mae_ev_per_angstrom"]) for r in rows])),
                "mean_runtime_seconds": float(np.mean([float(r["candidate_runtime_seconds"]) for r in rows]))})
    if len({r["configuration_id"] for r in raw_gold}) != 156 or any(r["configurations"] != 156 for r in gold if r["region"] == "all"):
        raise ValueError("Incomplete four-way GoldDAC comparison")
    widom = []
    for folder, label in (("mace_mp0a_273K", "MACE-MP-0a"), ("mace_dac_273K", "MACE-DAC"),
                          ("nequip_finetuned_273K", "NequIP"), ("nequip_finetuned_298K", "NequIP")):
        prefix = f"{POTENTIAL}/benchmarks/zif8_co2/{folder}"
        d = sources.json(prefix + "/widom_replicate_summary.json")
        ref = sources.json(prefix + "/reference_comparison/widom_reference_comparison.json")
        widom.append({"model": label, "temperature_K": d["temperature_K"], "replicates": d["replicate_count"],
            "attempts_per_seed": d["attempts_per_replicate"], "seed_precision_passed": d["seed_reproducibility_converged"],
            "pooled_block_precision_passed": d["pooled_block_sampling_converged"],
            "checkpoint_stable": d["checkpoint_stable"],
            "henry": d["metrics"]["henry_coefficient_mmol_g_bar"],
            "qst": d["metrics"]["isosteric_heat_zero_loading_kj_mol"], "references": ref["references"]})
    return {"fine_tuning": metrics, "runs": runs, "polar_validation": valid, "polar_stages": stages,
        "gold": gold, "widom": widom, "geometry_audit": audit_frames(sources),
        "mh_isotherm": sources.csv(MH_ISO + "/isotherm_summary.csv"),
        "mh_assessment": sources.json(MH_ISO + "/final_assessment.json"),
        "mh_sampling": sources.json(MH_ISO + "/sampling_details.json"),
        "polar_isotherm": sources.json(POLAR_ISO + "/comparison.json"),
        "polar_isotherm_metrics": sources.csv(POLAR_ISO + "/reference_metrics.csv"),
        "polar_runtime": sources.csv(POLAR_ISO + "/runtime.csv"),
        "mg_refinement": sources.json("outputs/umof_polar_adsorption_refinement/report/comparison.json"),
        "cp2k": sources.json(POTENTIAL + "/campaigns/zif8_co2_active_learning/active_learning_summary.json"),
        "mh_dft": sources.json("outputs/module_C_umof_dft_reference_025pct_mof5/run_status.json")}


def reference_points(sources, name):
    with sources.path("CRAFTED-2.0.0/ISOTHERM_FILES/" + name).open() as stream:
        points = np.loadtxt(stream, delimiter=",")
    return points[:, 0]/1e5, points[:, 1]


def plot_all(data, sources, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.labelcolor": "#243344", "text.color": "#243344", "axes.edgecolor": "#abb5bd",
        "savefig.facecolor": "white", "svg.fonttype": "none"})
    files = []

    def save(fig, name, title, note):
        fig.suptitle(title, x=.07, ha="left", fontsize=17, weight="bold", y=.98)
        fig.text(.07, .025, note, fontsize=9, color="#5a6570", va="bottom")
        fig.subplots_adjust(top=.82, bottom=.22, left=.10, right=.96, wspace=.33)
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180)
        files.append(name)
        plt.close(fig)

    labels = [r["run"] for r in data["runs"]]
    display_labels = [label.replace(" %", "%") for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    for ax, key, title in zip(axes, ("energy_rmse_meV_per_atom", "force_rmse_meV_per_A"),
                             ("Energy RMSE (meV/atom)", "Force RMSE (meV/Å)")):
        x = np.arange(3)
        for i, model in enumerate(("baseline", "finetuned")):
            values = [next(r[key] for r in data["fine_tuning"] if r["run"] == label and r["group"] == "all" and r["model"] == model) for label in labels]
            bars = ax.bar(x + (i-.5)*.34, values, width=.32, color=COLORS[i], label=["Pretrained", "Fine-tuned"][i])
            ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=4)
        if key.startswith("energy"):
            ax.set_yscale("log"); ax.set_ylim(.1, 300)
        else:
            ax.set_ylim(0, 60)
        ax.set(xticks=x, xticklabels=display_labels, ylabel=title)
        ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    save(fig, "01_finetuning_test", "Fine-tuning does not consistently reduce test errors",
         "Same 38 test entries / 36 unique geometries; whole loaded configurations; one training seed.\nNominal fractions: MH 10% = 109/774; MH/POLAR 25% = 211/774. Logarithmic energy axis.")

    groups = ["mof:MOF-303-adsorbates", "mof:MgMOF-74-adsorbates", "components:CO2", "components:N2", "components:CH4"]
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2))
    for ax, key, title in zip(axes, ("energy_rmse_meV_per_atom", "force_rmse_meV_per_A"), ("Energy error", "Force error")):
        ratios = np.array([[next(r[key] for r in data["fine_tuning"] if r["run"] == run and r["group"] == group and r["model"] == "finetuned") /
                            next(r[key] for r in data["fine_tuning"] if r["run"] == run and r["group"] == group and r["model"] == "baseline")
                            for group in groups] for run in labels])
        ax.imshow(np.log2(ratios), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4), aspect="auto")
        ax.set(xticks=range(5), xticklabels=["MOF-303 (16)", "MgMOF-74 (22)", "CO₂ (13)", "N₂ (12)", "CH₄ (13)"], yticks=range(3), yticklabels=display_labels, title=title)
        for i in range(3):
            for j in range(5):
                ax.text(j, i, f"{ratios[i,j]:.2f}×", ha="center", va="center", color="white" if abs(np.log2(ratios[i,j])) > 2.4 else "#243344", weight="bold")
    fig.subplots_adjust(hspace=.65)
    save(fig, "02_material_gas_effects", "Fine-tuning effects depend on the MOF and the metric",
         "Cell values: fine-tuned / pretrained RMSE. Blue < 1: lower error; red > 1: higher error.\nGas groups describe whole configurations, including all atoms. MOF and gas groups overlap; parentheses show entry counts.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    xref, yref = reference_points(sources, "DDEC_IRMOF-1_UFF_CO2_298.csv")
    axes[0].plot(xref, yref, "--", color="#343c45", label="CRAFTED UFF/DDEC")
    for i, model in enumerate(("umof_mh_pretrained", "umof_mh_finetuned_025pct")):
        rows = sorted([r for r in data["mh_isotherm"] if r["model"] == model], key=lambda r: float(r["pressure_bar"]))
        axes[0].plot([float(r["pressure_bar"]) for r in rows], [float(r["mean_loading_mol_per_kg"]) for r in rows], "o-", color=COLORS[i], label=["MH pretrained", "MH fine-tuned 25%"][i])
    axes[0].set(title="MOF-5 · uMOF-MH", xlim=(.07, 12))
    for ff, style in (("UFF", "--"), ("DREIDING", ":")):
        xref, yref = reference_points(sources, f"DDEC_ZIF-8_{ff}_CO2_298.csv")
        axes[1].plot(xref, yref, style, color="#343c45", label=f"CRAFTED {ff}/DDEC")
    for i, model in enumerate(("baseline", "parent_stage1", "followup_selected")):
        rows = sorted([r for r in data["polar_isotherm"]["summary"] if r["model"] == model], key=lambda r:r["pressure_bar"])
        axes[1].plot([r["pressure_bar"] for r in rows], [r["mean_loading_mol_per_kg"] for r in rows], "o-", color=COLORS[i], label=["POLAR pretrained", "POLAR Stage 1", "POLAR follow-up"][i])
    axes[1].set(title="ZIF-8 · uMOF-POLAR", xlim=(.07, 6), ylim=(0, 8))
    for ax in axes:
        ax.set(xscale="log", xlabel="Pressure (bar)", ylabel="Absolute loading (mmol/g)", ylim=(0, None))
        ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.18)
    save(fig, "03_isotherms", "Isotherms before and after fine-tuning · 298 K",
         "One seed per comparison; sampling limits quantitative rankings. No seed confidence intervals.\nDifferent MOFs: no direct MH/POLAR comparison. CRAFTED is a classical simulation reference.")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    rows = [r for r in data["widom"] if r["temperature_K"] == 273]
    for ax, key, ylabel in zip(axes, ("henry", "qst"), ("Henry coefficient (mmol/g/bar)", "Zero-loading Qst (kJ/mol)")):
        for i, row in enumerate(rows):
            ax.errorbar(i, row[key]["mean"], yerr=row[key]["ci95_half_width"], fmt="o", color=COLORS[i], capsize=5, markersize=8)
        for j, ref in enumerate(rows[0]["references"][:2]):
            value = ref["henry_coefficient_mmol_g_bar"] if key == "henry" else ref.get("isosteric_heat_low_pressure_kj_mol", ref["isosteric_heat_zero_loading_kj_mol"])
            ax.axhline(value, linestyle=["--", ":"][j], color="#525c65", label=ref["label"])
        ax.set(xticks=range(3), xticklabels=[r["model"] for r in rows], ylabel=ylabel, xlim=(-.5, 2.5))
        if key == "henry": ax.set_yscale("log")
        ax.grid(axis="y", alpha=.18)
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "04_widom_potential_classes", "Potential classes yield different low-pressure predictions",
         "ZIF-8 / CO₂, 273 K. 3 seeds × 100,000 trials; error bars: 95% Student-t intervals across seed means.\nNequIP misses the archived seed-precision target. CRAFTED Qst: low-pressure value at 100 Pa, not an exact zero-loading limit.")

    gold = [r for r in data["gold"] if r["region"] == "all"]
    names = ["MP", "MP + D3", "DAC", "DAC + D3"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.8))
    for ax, key, ylabel in zip(axes, ("energy_mae_eV", "energy_bias_eV", "mean_runtime_seconds"),
                               ("Interaction energy MAE (eV)", "Mean signed energy error (eV)", "Time per configuration (s)")):
        values = [r[key] for r in gold]
        bars = ax.bar(range(4), values, color=COLORS, width=.65)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=4)
        ax.set(xticks=range(4), xticklabels=names, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
        ax.margins(y=.25)
        if key == "energy_bias_eV": ax.axhline(0, color="#555", linewidth=.8)
    save(fig, "05_golddac_dispersion", "DFT benchmark: assess the potential and dispersion together",
         "The same 156 GoldDAC CO₂ test configurations per variant. Energy error = model − DFT.\nNegative error: interaction energy too favorable. Runtimes are archived measurements, not MC step times.")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    mh = sorted(data["mh_sampling"], key=lambda r:(r["model"], r["pressure_bar"]))
    y = np.arange(len(mh))
    values = [r["screen"]["effective_sample_count_estimate"] for r in mh]
    axes[0].barh(y, values, color=[COLORS[1] if "finetuned" in r["model"] else COLORS[0] for r in mh])
    axes[0].axvline(20, color="#343c45", linestyle="--", label="Screen: at least 20")
    axes[0].set(yticks=y, yticklabels=[("FT" if "finetuned" in r["model"] else "PT")+f" · {r['pressure_bar']:g} bar" for r in mh], xlabel="Estimated effective sample size", title="MH / MOF-5: autocorrelation")
    axes[0].legend(frameon=False, fontsize=9)
    for i, model in enumerate(("baseline", "parent_stage1", "followup_selected")):
        rows = sorted([r for r in data["polar_isotherm"]["summary"] if r["model"] == model], key=lambda r:r["pressure_bar"])
        drift = [100*abs(r["second_half_loading_mol_per_kg"]-r["first_half_loading_mol_per_kg"])/r["mean_loading_mol_per_kg"] for r in rows]
        axes[1].plot(range(4), drift, "o-", color=COLORS[i], label=["Pretrained", "Stage 1", "Follow-up"][i])
    axes[1].set(xticks=range(4), xticklabels=["0.1", "0.5", "1", "5"], xlabel="Pressure (bar)", ylabel="Difference between production halves / mean (%)", title="POLAR / ZIF-8: drift")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes: ax.grid(alpha=.18); ax.set_axisbelow(True)
    save(fig, "06_sampling_limits", "Completed simulations do not establish a reliable ranking",
         "PT = pretrained; FT = fine-tuned. MH: 4 of 6 points fail the full stability screen.\nPOLAR: production halves are dependent, not independent replicates. These diagnostics do not establish equilibrium.")
    return files


def table(rows, fields):
    lines = ["| " + " | ".join(label for _, label in fields) + " |", "|" + "---|"*len(fields)]
    for row in rows:
        def fmt(v):
            return f"{v:.4g}" if isinstance(v, float) else str(v)
        lines.append("| " + " | ".join(fmt(row.get(key, "—")) for key, _ in fields) + " |")
    return "\n".join(lines)


def report(data, output, figures):
    summary = [r for r in data["fine_tuning"] if r["group"] == "all"]
    fields = [("run", "Lauf"), ("model", "Modell"), ("configuration_count", "Einträge"),
              ("energy_rmse_meV_per_atom", "Energie-RMSE meV/Atom"), ("force_rmse_meV_per_A", "Kraft-RMSE meV/Å")]
    gold = [r for r in data["gold"] if r["region"] == "all"]
    mh = data["mh_assessment"]["reference_metrics"]
    text = f"""# Abschlussauswertung für Modul C

Stand: 30. Oktober 2026. Auswertung des lokal vorliegenden Datenbestands.

Modul C liegt als dokumentierter Potentialbenchmark mit abgeschlossenen Einzeluntersuchungen vor. Die vorhandenen Ergebnisse belegen große Potentialunterschiede und materialabhängige Fine-Tuning-Effekte. Eine allgemeine Verbesserung durch Fine-Tuning folgt daraus nicht. Quantitative Isothermenrangfolgen bleiben wegen unzureichend abgesicherten Samplings offen. Dieser Bericht erklärt Modul C deshalb nicht pauschal für wissenschaftlich vollständig validiert.

## Forschungsfrage und Umfang

Wie verändern Potentialklasse und Fine-Tuning die Adsorptionsenergetik, Henry-Konstante, Adsorptionswärme, Isotherme und den Rechenaufwand? Der Kern umfasst den ZIF-8-Widom-Vergleich bei 273 K, MH-Fine-Tuning mit 10/25 % und POLAR-Fine-Tuning mit 25 % nominellem Datenanteil sowie ihre jeweiligen Adsorptionspiloten. MH/MOF-5 und POLAR/ZIF-8 bei 298 K sind zwei Vorher-nachher-Untersuchungen; sie isolieren keinen Architekturunterschied. Eine vollständige Reproduktion aller Systeme der Literatur ist nicht Gegenstand dieses Abschlussumfangs.

## Energie und Kräfte vor und nach Fine-Tuning

{table(summary, fields)}

In `fine_tuning_metrics_meV.csv`, `results.json` und dieser Tabelle sind alle vier Fehlergrößen ausdrücklich mit 1000 multipliziert und mit `meV` bezeichnet, also meV/Atom und meV/Å. Die ergänzenden originalen Validierungs-/Stage-Tabellen behalten ihre eV-Einheiten. Die Modellbezeichnung `baseline` bedeutet jeweils die unveränderte Foundation desselben Profils.

MH 10 % und MH 25 % sowie POLAR 25 % sind abgeschlossen; jeweils Stage 1 wurde mit Validierungsdaten ausgewählt. Der nominelle MH-10-%-Lauf verwendet 109/774 = 14,08 % der Trainingskonfigurationen, die 25-%-Läufe 211/774 = 27,26 %. Stratifizierung erklärt die Abweichung. Eine vollständige Datenmengen-Lernkurve wird aus zwei MH-Läufen nicht behauptet. MH 50 % wurde unterbrochen; 100 % ist vorbereitet und gehört nicht zu den ausgewerteten Resultaten.

MH und POLAR verwenden bei 25 % nach Prüfung dieselben Geometrien in derselben Reihenfolge und identische Energie-/Kraftlabels: 211 Trainingseinträge, 46 Validierungseinträge und 38 Testeinträge. POLAR ergänzt Ladungs-/Spinmetadaten; deshalb unterscheiden sich die Dateihashes. Der Testsatz enthält nur 36 einzigartige Geometrien. Die archivierten Metriken sind nach Einträgen gewichtet; es wurde kein nachträglicher, scheinbar unabhängiger Testsatz erzeugt. Unabhängigkeit vom Foundation-Pretraining ist nicht nachgewiesen.

Die aggregierten Testfehler steigen bei beiden MH-Läufen und beim ausgewählten POLAR-Modell. Für POLAR sinkt der Energiefehler von MOF-303, während er für MgMOF-74 steigt. Bei MH sinken diese beiden Energie-RMSEs gegenüber der eigenen Foundation nicht. Gasgruppen bezeichnen vollständige beladene Konfigurationen, nicht isolierte Kräfte der Gastmoleküle. Überlappende MOF-/Gasgruppen werden nicht als unabhängige Stichproben behandelt.

![Testfehler](01_finetuning_test.png)

![Material- und Gasgruppen](02_material_gas_effects.png)

## Validierung und zusätzlicher POLAR-Folgelauf

`polar_validation_metrics.csv` enthält sämtliche bereits ausgewerteten Folgelauf-Checkpoints auf dem Validierungssatz. `polar_stage_metrics.csv` ergänzt Stage 1/2 mit ausdrücklich erhaltenen Splitbezeichnungen. Validierung und Test bleiben getrennt; Stage 2 wurde nicht als zusätzlicher unabhängiger Testlauf behandelt. Der ausgewählte Folgekandidat verbessert gegenüber Stage 1 den Validierungs-Kraft-RMSE von 58,39 auf 52,54 meV/Å, erhöht aber den Energie-RMSE von 63,51 auf 70,61 meV/Atom. Die Auswahl erfolgte mit Validierungskräften; der CRAFTED-Vergleich war nicht Auswahlkriterium. Ein Trainingsseed und wiederholt betrachtete Validierungsdaten begrenzen Verallgemeinerungen.

MH und POLAR unterscheiden sich in Architektur, Replay-Daten und numerischem Protokoll. Gleiche Testlabels erlauben einen deskriptiven Fehlervergleich, aber keinen isolierten kausalen Nachweis, dass die Architektur einen Unterschied verursacht. Vollständige Trainingsprotokolle sind in `results.json` erhalten.

## Isothermen und systematische Abweichungen

**MH / MOF-5, 298 K:** drei Druckpunkte (0,1 / 1 / 10 bar), jeweils ein Seed. Gegen CRAFTED UFF/DDEC beträgt der deskriptive RMSE pretrained {mh[0]['rmse_mol_per_kg']:.3f}, fine-tuned {mh[1]['rmse_mol_per_kg']:.3f} mmol/g. Das Fine-Tuning liefert insbesondere bei 10 bar eine wesentlich höhere Beladung. Vier der sechs Modell-Druckpunkte bestehen den archivierten Stabilitätsscreen nicht; Status `inconclusive_sampling`. Die kleinere Zahl des Ausgangsmodells ist keine abgesicherte endgültige Rangfolge.

**POLAR / ZIF-8, 298 K:** vier Druckpunkte (0,1 / 0,5 / 1 / 5 bar), ein Seed, 1000 Equilibrierungs- und 2000 Produktionsschritte je Punkt. Stage 1 liegt gegen UFF im absoluten Isothermen-RMSE vor der Foundation (0,543 gegenüber 1,576 mmol/g). Der Folgekandidat liegt bei 0,776 mmol/g. Bei mittlerem relativem Fehler sowie gegen DREIDING verändert sich die Bewertung; eine referenzunabhängige Überlegenheit ist damit nicht belegt. Teilweise starke Messwertdrift verhindert eine definitive quantitative Rangfolge.

{table(data['polar_isotherm_metrics'], [('reference','Referenz'),('model','POLAR-Modell'),('rmse_mol_per_kg','RMSE mmol/g'),('mean_signed_deviation_mol_per_kg','Bias mmol/g'),('mean_absolute_relative_deviation_percent','Mittlere relative Abweichung %')])}

CRAFTED ist eine klassische Simulationsreferenz. Nähe zu CRAFTED ist kein Nachweis experimenteller oder DFT-Genauigkeit. Referenzen werden nur im überlappenden Druckbereich verwendet. Absolute Adsorption wird nicht mit unbekannter Excess-Basis gleichgesetzt.

Der ergänzende MgMOF-74/POLAR-Vergleich enthält einen kurzen Drei-Modell-Versuch und vier längere Baseline/Stage-1-Jobs mit 5000+10000 Schritten. Er bleibt vorläufig: ein Seed, begrenzte Druckauswahl und experimentelle Quelle mit unbekannter Absolute-/Excess-Basis. Er wird nicht zur quantitativen Bestätigung der ZIF-8-Rangfolge benutzt.

![Isothermen](03_isotherms.png)

![Sampling](06_sampling_limits.png)

## Widom, Henry-Bereich und Adsorptionswärme

Der bereits abgeschlossene ZIF-8-Vergleich umfasst MACE-MP-0a, MACE-DAC und eine adsorptionstrainierte NequIP-Referenz, bei 273 K jeweils drei Seeds mit 100000 Einfügungsversuchen. NequIP liegt zusätzlich bei 298,15 K vor; diese Ergebnisse werden nicht mit 273 K zusammengefasst. Die Fehlerbalken der Abbildung zeigen Student-t-Intervalle über Seedmittel. Die gespeicherten NequIP-Berichte melden ausreichende gepoolte Blockpräzision, aber ein verfehltes Seed-Präzisionsziel. Ein übergeordnetes `converged: true` wird deshalb nicht als Bestehen aller Einzelkriterien übernommen.

Die klassischen Henry-Referenzen sind Niederdruckfits; die CRAFTED-Wärmen bei 100 Pa sind keine exakt bestimmten Nullbeladungsgrenzwerte. POLAR/ZIF-8 besitzt ergänzend nur 2000 Widomversuche je Modell und effektive Boltzmann-Stichproben von etwa 9–12. Diese Schätzungen bleiben orientierend und belegen keinen sicheren Fine-Tuning-Vorteil. Die Werte sind vollständig in `results.json` erhalten.

![Potentialklassen](04_widom_potential_classes.png)

## Statischer DFT-Test und Dispersion

{table(gold, [('model','Potential'),('configurations','Konfigurationen'),('energy_mae_eV','Energie-MAE eV'),('energy_rmse_eV','Energie-RMSE eV'),('energy_bias_eV','Bias eV'),('mean_configuration_force_mae_eV_A','Mittlerer Konfigurations-Kraft-MAE eV/Å'),('mean_runtime_seconds','Zeit s/Konfiguration')])}

Alle vier Varianten wurden an denselben 156 GoldDAC-CO₂-Testkonfigurationen geprüft. Regionale Werte für Gleichgewicht, schwache Anziehung und Repulsion sind separat in `gold_metrics.csv` archiviert. Die Energiefehler beziehen sich auf Interaktionsenergien. Der Kraftwert ist der Mittelwert konfigurationsweiser MAEs der kombinierten Systeme, kein atomzahlgewichteter Gesamt-MAE und kein isolierter Gastfehler. Ein einfacher Mittelwert über diesen gezielt zusammengesetzten Testsatz ist keine thermodynamisch gewichtete Adsorptionsstatistik.

![GoldDAC und Dispersion](05_golddac_dispersion.png)

## Rechenaufwand

`runtime.csv` trennt GoldDAC-Konfigurationen, Widomversuche und GCMC-Schritte. Für POLAR/ZIF-8 wurden etwa 0,155 s/MC-Schritt für Foundation und Stage 1 und 0,164 s/MC-Schritt für den Folgekandidaten archiviert. Die Modelle liefen auf verschiedenen GPUs und durchliefen unterschiedliche Beladungen; daraus folgt kein isolierter Inferenz-Geschwindigkeitsvorteil. GoldDAC-Zeiten sind Messwerte auf identischen Konfigurationen und bleiben als eigener Block erhalten. Wiederverwendete MH-Ketten und CPU-/GPU-Trainingszeiten werden nicht zu einer scheinbar einheitlichen Geschwindigkeitsrangfolge vermischt. Training und Evaluation stehen separat in den Laufprotokollen.

## Abgleich mit den Guidelines

| Geforderter Nachweis | Stand | Abschlussaussage |
|---|---|---|
| Widom-Einfügungsenergetik | GoldDAC-Energiefehler und archivierte Widom-Samplingauswertung vorhanden | Potentialabhängigkeit belegt; statische und thermische Stichproben getrennt |
| Henry-Bereich / Qst | Drei-Seed-Vergleich bei 273 K vorhanden | Deskriptiver Vergleich mit einzelnen Präzisionsgrenzen |
| Isothermenabweichung | MH/MOF-5 und POLAR/ZIF-8 ausgewertet | Pilotresultate; definitive quantitative Rangfolge offen |
| Fehler zu Referenzen | Klassische Referenzen und DFT-Tests getrennt vorhanden | Keine Gleichsetzung mit experimenteller Wahrheit |
| Laufzeit pro Schritt | Vorhandene Messwerte zusammengeführt | Messumfang und Vergleichsgrenzen explizit |
| Über-/Unterschätzung | Vorzeichenbehaftete Energie-/Beladungsabweichungen vorhanden | Druck-, Material- und Referenzabhängigkeit erhalten |
| Modell- und Datenprovenienz | Hashes, Splitprüfung, Auswahlregel und Protokolle erfasst | Ein Trainingsseed; Foundation-Überlappung nicht ausgeschlossen |

## Verbleibende Clusterrechnungen

Das separate Clusterpaket ergänzt die quantitative Adsorptionsabsicherung. Es enthält MH pretrained/25-%-Stage-1 auf MOF-5 bei 0,1/1/10 bar sowie POLAR pretrained/25-%-Stage-1 auf ZIF-8 bei 0,1/0,5/1/5 bar, jeweils drei neue Seeds. Das sind 18 + 24 = 42 Modell-Druck-Seed-Jobs. Jeder Job hat ein festes Budget von 20000 Equilibrierungs- und 40000 Produktionsschritten. Das sind insgesamt 2520000 MC-Schritte; die Schrittzahl garantiert keine Konvergenz.

Die neuen Läufe verwenden einen gemeinsamen, ausdrücklich archivierten MC-Adapter mit derselben Überlappungsregel für beide Profile; sie überschreiben und verlängern keine historischen Pilotketten. Auch danach wird wegen unterschiedlicher MOFs keine MH/POLAR-Architekturrangfolge behauptet. Nach Abschluss werden Seedintervalle, Messhälften, Blockmittel, Autokorrelationsdiagnostik und CRAFTED-Residuen gemeinsam bewertet. Das vorgeschlagene Präzisionsziel beträgt max(20 % der Beladung, 0,02 mmol/g) für die Seed-Intervallhalbbreite, ergänzt durch eine Driftprüfung. Diese projektspezifischen Kriterien sind kein Beweis für Gleichgewicht. Fehlgeschlagene Punkte bleiben offen; es gibt keine automatische Endlosschleife oder automatische Nachtrainierung.

Der CP2K/MACE-AL-Zweig bleibt `adaptation_not_converged`; der gezielte MH/MOF-5-DFT-Diagnostiklauf wurde nach zwei von zwanzig Rechnungen gestoppt. Beide werden transparent als offene Nebenuntersuchungen geführt. Zusätzliche 50/100-%-Trainingsläufe oder neue DFT-Daten sind für die hier formulierten deskriptiven Aussagen nicht erforderlich.

## Belastbare Schlussfolgerung

Potentialwahl und Dispersion verändern die Adsorptionsenergetik deutlich. Fine-Tuning liefert bei MH und POLAR keinen durchgängigen Gewinn auf den untersuchten Transfer-Testkonfigurationen. Änderungen hängen von Material und Messgröße ab. Der POLAR-Folgelauf verbessert Validierungskräfte, liefert im kurzen ZIF-8-Vergleich aber keinen zusätzlichen Isothermenvorteil gegenüber Stage 1. Die vorhandenen GCMC-Ergebnisse rechtfertigen wegen Sampling und Referenzabhängigkeit keine allgemeine Genauigkeitsrangfolge. Die noch ausstehenden Clusterläufe dienen genau dieser Absicherung.

## Reproduzieren und Quellen prüfen

Vom Projektverzeichnis: `MPLCONFIGDIR=/tmp/mof-c-mpl python -m analysis.module_c_closure`.
Der Aufruf trainiert kein Modell und startet keine Simulation. `provenance.json` enthält SHA-256 aller gelesenen Quellen, den Skripthash, Laufzeitversionen und die Aussagegrenzen. `results.json` enthält die vollständigen konsolidierten Daten. Alle sechs Abbildungen liegen als PNG und editierbares SVG vor. Die Quellkampagnen bleiben unverändert.
"""
    (output / "REPORT.md").write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/module_c_closure")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = Sources(ROOT)
    data = collect(sources)
    for source in ("guidelines_for_project_developement/Results_Progress.pdf",
                   "guidelines_for_project_developement/Literature research for MOF simulations.docx",
                   "guidelines_for_project_developement/Pipeline proposal.docx"):
        sources.path(source)
    figures = plot_all(data, sources, output)
    write_csv(output / "fine_tuning_metrics_meV.csv", data["fine_tuning"])
    write_csv(output / "gold_metrics.csv", data["gold"])
    write_csv(output / "polar_validation_metrics.csv", data["polar_validation"])
    write_csv(output / "polar_stage_metrics.csv", data["polar_stages"])
    write_csv(output / "mh_isotherm.csv", data["mh_isotherm"])
    write_csv(output / "polar_isotherm_metrics.csv", data["polar_isotherm_metrics"])
    runtime = [{"block": "POLAR/ZIF-8", **r} for r in data["polar_runtime"]]
    runtime += [{"block": "GoldDAC", "model": r["model"], "phase": "static_configuration", "seconds_per_attempt_or_step": r["mean_runtime_seconds"]} for r in data["gold"] if r["region"] == "all"]
    write_csv(output / "runtime.csv", runtime)
    data["fine_tuning_units"] = {"energy": "meV/atom", "force": "meV/Angstrom"}
    data["status"] = {"existing_evidence_consolidated": True, "quantitative_gcmc_ranking": "open_sampling",
                      "new_training": False, "new_simulations": False}
    dump(output / "results.json", data)
    report(data, output, figures)
    import matplotlib, scipy, ase
    for path, expected in sources.files.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Source changed during consolidation: {path}")
    dump(output / "provenance.json", {"date": "2026-10-30", "source_sha256": sources.files,
        "script_sha256": digest(__file__), "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__, "scipy": scipy.__version__, "ase": ase.__version__},
        "figures": figures, "source_files_unchanged": True, "geometry_audit": data["geometry_audit"]})
    print(f"Created {output / 'REPORT.md'} and {len(figures)} PNG/SVG figure pairs; source hashes verified.")


if __name__ == "__main__":
    main()
