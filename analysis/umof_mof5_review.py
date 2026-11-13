"""Audit and compare the returned, fixed-wall MOF-5 three-model pilot.

No training, model inference or simulation. Historical protocols are not pooled.
Figures use English labels and decimal pressure values.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import zipfile

import numpy as np
from ase.io import read

from analysis.module_c_closure import ROOT, Sources, digest, dump, seed_interval, table, write_csv
from engines.mlip_mc_runner import _read_gcmc_transition_log, _reconstruct_gcmc_samples
from modules.module_c_mlips.guarded_pilot import allowed, radii_array
from modules.module_c_mlips.umof_isotherm_stability import assess

BUNDLE = "reports/umof_mof5_pilot"
LABELS = {"umof_mh_pretrained": "uMOF-MH pretrained", "umof_polar_pretrained": "uMOF-POLAR pretrained",
          "umof_polar_stage1": "uMOF-POLAR Stage 1"}
COLORS = {"umof_mh_pretrained": "#2676a6", "umof_polar_pretrained": "#438b69", "umof_polar_stage1": "#d27031"}
CRITERIA = ("at_least_20000_samples", "at_least_20_effective_samples",
            "precision_within_tolerance", "half_drift_within_tolerance")


def close(actual, expected, context):
    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-10):
        raise ValueError(f"Archived and reconstructed values differ: {context}")


def collect(sources, bundle):
    def relative(path):
        return str(Path(path).relative_to(ROOT))

    prefix = relative(bundle)
    manifest_path = sources.path(prefix + "/bundle_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["file_sha256"].items():
        if digest(sources.path(prefix + "/" + name)) != expected:
            raise ValueError(f"Bundle asset or implementation changed: {name}")
    config = sources.json(prefix + "/pilot.json")
    if (config["material"] != "MOF-5" or config["temperature_K"] != 298.
            or config["pressure_bar"] != 1. or len(config["models"]) != 3
            or len(config["seeds"]) != 3 or len(set(config["seeds"])) != 3):
        raise ValueError("Expected the frozen three-model, three-seed MOF-5/CO2 pilot at 298 K, 1 bar")
    if set(m["label"] for m in config["models"]) != set(LABELS) or any(m["dtype"] != "float64" for m in config["models"]):
        raise ValueError("Model set or common float64 protocol changed")
    for spec in (config["framework"], config["adsorbate"], config["reference"], *config["models"]):
        if digest(sources.path(prefix + "/" + spec["path"])) != spec["sha256"]:
            raise ValueError("Input identity mismatch")
    host = read(sources.path(prefix + "/" + config["framework"]["path"]))
    guest = read(sources.path(prefix + "/" + config["adsorbate"]["path"]))
    factor = 1000 / host.get_masses().sum()
    reference = np.loadtxt(sources.path(prefix + "/" + config["reference"]["path"]), delimiter=",")
    p = config["pressure_bar"]
    if not reference[:, 0].min()/1e5 <= p <= reference[:, 0].max()/1e5:
        raise ValueError("Reference extrapolation is forbidden")
    ref = float(np.interp(p, reference[:,0]/1e5, reference[:,1]))
    old = sources.json(prefix + "/report/assessment.json")
    g = config["gcmc"]
    total = g["equilibration_steps"] + g["production_steps"]
    rows, series, checks, gates = [], {}, [], []
    environments = set()

    def result_for(stage, model, seed):
        job = {"id": f"{model}_s{seed}", "model": model, "seed": seed}
        directory = bundle / "results" / stage / job["id"]
        marker = sources.json(relative(directory / "completed.json"))
        identity = marker["identity"]
        if identity != {"bundle_sha256": digest(manifest_path), "stage": stage, "job": job, "runtime": identity["runtime"]}:
            raise ValueError("Result stage/model/seed/bundle identity mismatch")
        path = directory / marker["result_path"]
        if digest(sources.path(relative(path))) != marker["result_sha256"]:
            raise ValueError("Result checksum mismatch")
        d = json.loads(path.read_text())
        if d["stage"] != stage or d["model"] != model or d.get("gate_passed") is not True:
            raise ValueError("Missing or failed technical gate")
        environments.add(json.dumps(identity["runtime"], sort_keys=True))
        return path, d

    for model in config["models"]:
        label = model["label"]
        for stage in ("scan", "smoke"):
            path, d = result_for(stage, label, config["seeds"][0])
            gates.append({"model": label, "stage": stage, "gate_passed": True,
                "scope": "Technical screening only, no physical accuracy validation",
                "allowed_probe_coverage": d.get("coverage"), "source": relative(path)})
        for seed in config["seeds"]:
            path, d = result_for("pilot", label, seed)
            raw = d["raw"]
            if raw["status"] != "completed" or d["seed"] != seed or raw["seed"] != seed:
                raise ValueError("Pilot status/seed mismatch")
            close(raw["temperature_K"], config["temperature_K"], "temperature")
            if len(raw["pressure_points"]) != 1:
                raise ValueError("Expected one isolated pressure point")
            point = raw["pressure_points"][0]
            for key, expected in (("pressure_bar", p), ("seed", seed),
                                  ("equilibration_steps", g["equilibration_steps"]),
                                  ("production_steps", g["production_steps"]),
                                  ("production_sample_count", g["production_steps"])):
                close(point[key], expected, key)
            if point["fugacity_source"] != "peng-robinson":
                raise ValueError("Fugacity convention changed")
            matches = list(path.parent.rglob(Path(point["transition_log_path"]).name))
            if len(matches) != 1:
                raise ValueError("Expected one transition log per result")
            trace_path = sources.path(relative(matches[0]))
            transitions = _read_gcmc_transition_log(trace_path)
            trace = _reconstruct_gcmc_samples(transitions, total_steps=total)
            counts = np.array([r["uptake"] for r in trace], float)
            energies = np.array([r["interaction_energy_ev"] for r in trace], float)
            if np.any(counts < 0) or not np.isfinite(energies).all():
                raise ValueError("Invalid accepted-state population or energy")
            values = counts*factor
            production = values[g["equilibration_steps"]:]
            prod_energy = energies[g["equilibration_steps"]:]
            screen = assess(production)
            close(production.mean(), d["loading_mmol_g"], "result mean")
            close(production.mean(), point["mean_uptake_molecules_per_unit_cell"]*factor, "raw mean")
            close(prod_energy.mean(), d["mean_interaction_eV_all_production_steps"], "mean interaction energy")
            for key in ("mean_mmol_g", "effective_sample_count_estimate", "half_difference_mmol_g", "statistical_inefficiency_estimate"):
                close(screen[key], d["screen"][key], "chain screen " + key)
            if screen["status"] != d["screen"]["status"] or screen["criteria"] != d["screen"]["criteria"]:
                raise ValueError("Archived chain screen does not reproduce")
            prod_csv = sources.path(relative(path.parent / "production.csv"))
            exported = np.loadtxt(prod_csv, delimiter=",", skiprows=1)
            close(exported[:,0], np.arange(g["equilibration_steps"]+1, total+1), "CSV steps")
            close(exported[:,1], production, "CSV all-step loading")
            close(exported[:,2], prod_energy, "CSV interaction energies")
            restarts = list(trace_path.parent.glob("restart/restart_*.json"))
            snapshots = list(trace_path.parent.glob("restart/restart_*.xyz"))
            if len(restarts) != 1 or len(snapshots) != 1:
                raise ValueError("Missing final restart state")
            restart = sources.json(relative(restarts[0]))
            close(restart["n_equil_completed"], g["equilibration_steps"], "restart equilibration")
            close(restart["n_prod_completed"], g["production_steps"], "restart production")
            final = read(sources.path(relative(snapshots[0])))
            nh, ng = len(host), len(guest)
            if (not np.array_equal(final.numbers[:nh], host.numbers)
                    or not np.allclose(final.positions[:nh], host.positions, rtol=0, atol=2e-7)
                    or not np.allclose(final.cell.array, host.cell.array, rtol=0, atol=1e-9)
                    or len(final) != nh+int(counts[-1])*ng
                    or not np.array_equal(final.numbers[nh:], np.tile(guest.numbers, int(counts[-1])))):
                raise ValueError("Final framework, cell or molecule counts changed")
            if not allowed(final, nh, radii_array(config)):
                raise ValueError("Final snapshot violates the frozen overlap rule")
            checkpoints = list(path.parent.glob("engine/**/checkpoint_*/traj.xyz"))
            if len(checkpoints) != d["checkpoint_structures_checked"] or len(checkpoints) != total//g["checkpoint_interval"]:
                raise ValueError("Incomplete checkpoint set")
            moves = point["move_statistics"]
            rows.append({"model": label, "seed": seed, "loading_mmol_g": float(production.mean()),
                "chain_status": screen["status"], "effective_samples": screen["effective_sample_count_estimate"],
                "first_half_mmol_g": screen["first_half_mean_mmol_g"], "second_half_mmol_g": screen["second_half_mean_mmol_g"],
                "half_difference_mmol_g": screen["half_difference_mmol_g"],
                "drift_tolerance_mmol_g": screen["absolute_tolerance_mmol_g"],
                "drift_over_tolerance": screen["half_difference_mmol_g"]/screen["absolute_tolerance_mmol_g"],
                "runtime_hours": raw["runtime_seconds"]/3600, "seconds_per_mc_step": raw["runtime_seconds"]/total,
                "mean_interaction_eV": float(prod_energy.mean()),
                **{f"{kind}_acceptance_percent": 100*v["accepted"]/v["attempted"] for kind,v in moves.items()},
                **screen["criteria"]})
            series[f"{label}_{seed}"] = values
            checks.append({"model": label, "seed": seed, "result_hash_verified": True,
                "production_steps_reconstructed": len(production), "production_csv_matches_trace": True,
                "archived_screen_reproduced": True, "final_snapshot_verified": True,
                "available_checkpoint_count": len(checkpoints),
                "all_checkpoint_wall_checks": "60 checks recorded by the archived worker; final snapshot independently rechecked"})
    if len(environments) != 1:
        raise ValueError("Scientific/gate runtime environments differ")
    summary = []
    for model in config["models"]:
        label = model["label"]
        selected = [r for r in rows if r["model"] == label]
        stats = seed_interval([r["loading_mmol_g"] for r in selected])
        target = max(config["precision"]["absolute_floor_mmol_g"], config["precision"]["relative_seed_ci95_target"]*abs(stats["mean"]))
        bias = stats["mean"]-ref
        seed_pass = stats["ci95_half_width"] <= target
        chain_pass = sum(r["chain_status"] == "passes_operational_screen" for r in selected)
        row = {"model": label, **stats, "ci95_low": stats["mean"]-stats["ci95_half_width"],
            "ci95_high": stats["mean"]+stats["ci95_half_width"], "reference_mmol_g": ref,
            "bias_mmol_g": bias, "relative_bias_percent": 100*bias/ref, "absolute_deviation_mmol_g": abs(bias),
            "reference_ratio": stats["mean"]/ref, "passed_chains": chain_pass, "seed_precision_passed": seed_pass,
            "all_screens_passed": seed_pass and chain_pass == len(selected),
            "seed_precision_tolerance_mmol_g": target,
            "effective_samples_min": min(r["effective_samples"] for r in selected),
            "effective_samples_max": max(r["effective_samples"] for r in selected),
            "mean_runtime_hours": float(np.mean([r["runtime_hours"] for r in selected])),
            "total_runtime_hours": sum(r["runtime_hours"] for r in selected),
            "mean_seconds_per_mc_step": float(np.mean([r["seconds_per_mc_step"] for r in selected]))}
        archived = next(x for x in old["points"] if x["model"] == label)
        for key in ("mean", "ci95_half_width", "relative_ci95_half_width"):
            close(row[key], archived[key], "archived summary " + key)
        if row["all_screens_passed"] != archived["all_screens_passed"]:
            raise ValueError("Combined screen differs from archived assessment")
        summary.append(row)
    if old["completed_jobs"] != len(rows) or old["expected_jobs"] != len(rows) or old["missing"]:
        raise ValueError("Pilot completion counts disagree")
    contrasts = []
    for a,b in combinations([m["label"] for m in config["models"]], 2):
        by_key = {(r["model"],r["seed"]):r["loading_mmol_g"] for r in rows}
        stats = seed_interval([by_key[b,s]-by_key[a,s] for s in config["seeds"]])
        contrasts.append({"model_a": a, "model_b": b, "difference": "B minus A", **stats,
            "ci95_low": stats["mean"]-stats["ci95_half_width"], "ci95_high": stats["mean"]+stats["ci95_half_width"],
            "scope": "Descriptive matched-seed contrast only; nonstationary chains invalidate a definitive equilibrium interpretation"})
    return {"config": config, "summary": summary, "seeds": rows, "series": series, "audits": checks,
        "gates": gates, "runtime": json.loads(next(iter(environments))), "paired_seed_contrasts": contrasts,
        "bundle_files_verified": len(manifest["file_sha256"])}


def figures(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
    models = [m["label"] for m in data["config"]["models"]]
    short = ["MH pretrained", "POLAR pretrained", "POLAR Stage 1"]
    seeds = data["config"]["seeds"]
    seed_colors = ["#2676a6", "#d27031", "#8b6cab"]
    ref = data["summary"][0]["reference_mmol_g"]
    files = []

    def save(fig, name, title, note, bottom=.20, top=.83):
        fig.suptitle(title, fontsize=17, weight="bold", x=.07, y=.98, ha="left")
        fig.text(.07, .025, note, fontsize=9, color="#586674", va="bottom")
        fig.subplots_adjust(left=.10, right=.96, top=top, bottom=bottom, wspace=.35)
        for ext in ("png", "svg"):
            fig.savefig(output/f"{name}.{ext}", dpi=190, facecolor="white")
        plt.close(fig); files.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, rows, names, ymax, title in (
            (axes[0], data["summary"][:1], ["Pretrained"], 1.2, "uMOF-MH · range 0–1.2 mmol/g"),
            (axes[1], data["summary"][1:], ["Pretrained", "Stage 1"], 14., "uMOF-POLAR · range 0–14 mmol/g")):
        for i, row in enumerate(rows):
            color = COLORS[row["model"]]
            ax.errorbar(i, row["mean"], yerr=row["ci95_half_width"], fmt="D", capsize=8,
                        markersize=9, color=color, linewidth=2.3, zorder=3)
            values = [r["loading_mmol_g"] for r in data["seeds"] if r["model"] == row["model"]]
            ax.scatter(i+np.array([-.12,0,.12]), values, s=45, facecolor="white", edgecolor=color, linewidth=1.5, zorder=4)
            lower_panel = ymax < 2
            anchor = row["ci95_low"] if lower_panel else row["ci95_high"]
            ax.annotate(f"{row['mean']:.3f} ± {row['ci95_half_width']:.3f}", (i,anchor),
                        xytext=(0,-24 if lower_panel else 13), textcoords="offset points", ha="center", fontsize=13, weight="bold")
        ax.set(xticks=range(len(rows)), xticklabels=names, ylabel="Absolute loading (mmol/g)",
               ylim=(0,ymax), xlim=(-.5,len(rows)-.5), title=title)
        ax.tick_params(labelsize=12)
        ax.axhline(ref, color="#303b45", linestyle="--", label=f"CRAFTED UFF/DDEC: {ref:.3f}")
        ax.grid(axis="y", alpha=.2); ax.legend(frameon=False, loc="upper left", fontsize=10)
    save(fig, "01_loading_reference", "MH and POLAR predictions shown on separate loading scales",
         "Different y-axis ranges: compare the numerical values, not marker heights across panels. MOF-5 / CO₂, 298 K, 1 bar.\nDiamonds: model means; open circles: three seed means; error bars: 95% Student-t intervals. CRAFTED is a classical reference.\nEvery model misses the 20% seed-precision target; POLAR chain screens also fail.", bottom=.24)

    fig, axes = plt.subplots(3, 2, figsize=(13.5,10))
    eq = data["config"]["gcmc"]["equilibration_steps"]
    prod = data["config"]["gcmc"]["production_steps"]
    for row,model in enumerate(models):
        for i,seed in enumerate(seeds):
            values = data["series"][f"{model}_{seed}"]
            blocks = values.reshape(-1,1000).mean(axis=1)
            axes[row,0].plot(np.arange(500,eq+prod,1000)/1000, blocks, color=seed_colors[i], label=f"Seed {seed}", linewidth=1.3)
            tail = values[eq:]
            axes[row,1].plot(np.arange(eq+1,eq+prod+1)/1000, np.cumsum(tail)/np.arange(1,prod+1), color=seed_colors[i], linewidth=1.5)
        axes[row,0].axvspan(0,eq/1000,color="#8899aa",alpha=.13)
        axes[row,0].axvline(eq/1000,color="#6b7785",linestyle=":",linewidth=1)
        axes[row,0].set(title=short[row]+": 1,000-step block means", ylabel="Loading (mmol/g)", xlabel="MC step (×1,000)", xlim=(0,60), ylim=(0,None))
        axes[row,1].set(title=short[row]+": cumulative production mean", ylabel="Loading (mmol/g)", xlabel="MC step (×1,000)", xlim=(20,60), ylim=(0,None))
        for ax in axes[row]: ax.grid(alpha=.16)
    handles, names = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(.10,.94))
    fig.subplots_adjust(hspace=.65)
    save(fig, "02_loading_histories", "POLAR histories retain stronger drift than MH histories",
         "Same 20,000 equilibration + 40,000 production moves per run. Shading marks equilibration; rejected moves enter the means.\nRows use different y-axis ranges. Cumulative averaging smooths drift and does not prove equilibrium.", bottom=.13, top=.86)

    fig,axes = plt.subplots(1,3,figsize=(14.5,6))
    for i,model in enumerate(models):
        rows = [r for r in data["seeds"] if r["model"]==model]
        for j,r in enumerate(rows):
            axes[0].scatter(i+(j-1)*.11,r["effective_samples"],color=COLORS[model],s=45)
            axes[1].scatter(i+(j-1)*.11,r["drift_over_tolerance"],color=COLORS[model],s=45)
        r=data["summary"][i]
        axes[2].bar(i,r["relative_ci95_half_width"]*100,color=COLORS[model],width=.55)
        axes[2].text(i,r["relative_ci95_half_width"]*100+1,f"{r['relative_ci95_half_width']*100:.1f}%",ha="center",fontsize=10)
    axes[0].axhline(20,color="#303b45",linestyle="--",label="Screen minimum: 20")
    axes[0].set(yscale="log",ylabel="Estimated effective samples",title="Within-chain correlation")
    axes[1].axhline(1,color="#303b45",linestyle="--",label="Screen maximum: 1")
    axes[1].set(ylabel="Half-to-half drift / tolerance",title="Within-chain drift",ylim=(0,None))
    axes[2].axhline(20,color="#303b45",linestyle="--",label="Target maximum: 20%")
    axes[2].set(ylabel="95% interval half-width / mean (%)",title="Between-seed precision",ylim=(0,50))
    for ax, legend_location in zip(axes, ("upper right", "lower right", "upper right")):
        ax.set(xticks=range(3),xticklabels=["MH\npretrained","POLAR\npretrained","POLAR\nStage 1"])
        ax.legend(frameon=False,fontsize=8,loc=legend_location);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    save(fig,"03_sampling_quality","MH passes all chain screens, but no model passes every precision check",
         "Dots in the first two panels represent three individual seeds. Chain tolerance: max(15% of loading, 0.02 mmol/g).\nBetween-seed target: max(20% of loading, 0.02 mmol/g). These operational criteria do not establish equilibrium.")

    fig,axes=plt.subplots(1,2,figsize=(13,6))
    reference_deviations=[r["relative_bias_percent"] for r in data["summary"]]
    bars=axes[0].bar(range(3),reference_deviations,color=[COLORS[m] for m in models],width=.6)
    axes[0].bar_label(bars,labels=[f"{v:+.1f}%" for v in reference_deviations],padding=5,fontsize=10)
    axes[0].axhline(0,color="#4c5966",linewidth=.8)
    axes[0].set(ylabel="Signed deviation from CRAFTED (%)",title="Descriptive reference agreement")
    times=[r["mean_runtime_hours"] for r in data["summary"]]
    bars=axes[1].bar(range(3),times,color=[COLORS[m] for m in models],width=.6)
    axes[1].bar_label(bars,labels=[f"{v:.2f} h" for v in times],padding=5,fontsize=10)
    for i,m in enumerate(models):
        axes[1].scatter([i-.1,i,i+.1],[r["runtime_hours"] for r in data["seeds"] if r["model"]==m],s=24,facecolor="white",edgecolor="#303b45",zorder=4)
    axes[1].set(ylabel="Mean runtime per 60,000-move run (hours)",title="Observed computational cost")
    for ax in axes:
        ax.set(xticks=range(3),xticklabels=["MH\npretrained","POLAR\npretrained","POLAR\nStage 1"])
        ax.grid(axis="y",alpha=.16);ax.set_axisbelow(True);ax.margins(y=.20)
    save(fig,"04_reference_and_cost","MH combines smaller reference deviation with lower observed runtime",
         "Identical rigid MOF-5 system, 298 K, 1 bar, float64 and MC budgets; archived GPU: NVIDIA GeForce RTX 2080 Ti.\nLoading-dependent workloads differ. Runtimes include MC overhead and are not pure inference benchmarks; reference deviations are provisional.")
    return files


def write_report(data, output, as_of):
    summary=data["summary"]
    chosen=min(summary,key=lambda r:(-r["passed_chains"],r["absolute_deviation_mmol_g"]))
    spec=next(m for m in data["config"]["models"] if m["label"]==chosen["model"])
    selection={"status":"recommended_candidate_not_quantitatively_validated","model":spec,
        "source_bundle":BUNDLE,"scope":{"material":"MOF-5","component":"CO2","temperature_K":298.,"pressure_bar":1.,"framework":"rigid"},
        "basis":["3/3 single-chain screens passed","smallest descriptive absolute deviation from CRAFTED UFF/DDEC","lowest observed runtime"],
        "seed_precision_passed":chosen["seed_precision_passed"],"equilibrium_established":False,
        "new_training":False,"new_simulations":False,"new_inference":False,
        "not_evaluated_in_this_pilot":["MH fine-tuned","other pressures, temperatures, gases or MOFs","flexible-framework dynamics"]}
    dump(output/"recommended_model.json",selection)
    cirows=[{"model":LABELS[r["model"]],"loading":f"{r['mean']:.3f} ± {r['ci95_half_width']:.3f}",
        "bias":f"{r['relative_bias_percent']:+.1f} %","chains":f"{r['passed_chains']}/3",
        "precision":f"{r['relative_ci95_half_width']*100:.1f} %","hours":f"{r['mean_runtime_hours']:.2f}"} for r in summary]
    polar_diff=next(r for r in data["paired_seed_contrasts"] if r["model_a"]=="umof_polar_pretrained" and r["model_b"]=="umof_polar_stage1")
    text=f"""# Modellempfehlung aus dem neuen MOF-5 Pilot

Stand: {as_of}. **Für weitere Untersuchungen von starrem MOF-5/CO₂ bei 298 K empfehle ich uMOF-MH pretrained mit dem Head `r2scan+d4` als derzeit besten Kandidaten unter den drei verglichenen Modellen.** Der Pilot liefert diese Entscheidung am einzigen untersuchten Druckpunkt 1 bar. MH liegt näher an der klassischen Referenz, zeigt die stabileren Ketten und benötigt weniger Rechenzeit. Die Ergebnisse reichen noch nicht für die Freigabe als quantitativ validiertes Produktionsmodell.

## Verglichene Ergebnisse

Alle neun wissenschaftlichen Läufe sind vollständig: drei Modelle × drei neue MC-Seeds, jeweils 20000 Equilibrierungs- und 40000 Produktionsschritte. Alle Modelle verwenden dieselbe MOF-5-Struktur, starres CO₂, 298 K, 1 bar, float64, Peng–Robinson-Fugazität und dieselbe vorab festgelegte Überlappungsregel. CRAFTED UFF/DDEC liefert am Vergleichspunkt **{summary[0]['reference_mmol_g']:.7f} mmol/g**. Dies ist eine klassische Simulationsreferenz.

{table(cirows,[("model","Modell"),("loading","Beladung ± 95-%-Halbbreite mmol/g"),("bias","Abweichung zu CRAFTED"),("chains","Kettenscreens bestanden"),("precision","Relative 95-%-Halbbreite"),("hours","Zeit je Lauf h")])}

Die Intervalle sind zweiseitige 95-%-Student-t-Intervalle der drei unabhängigen Seedmittel (zwei Freiheitsgrade), keine Standardabweichungen der MC-Einzelschritte. Sie enthalten weder die Unsicherheit der klassischen Referenz noch systematische Modellfehler oder einen möglichen Nichtgleichgewichtsfehler. Alle drei Modelle verfehlen das vorab festgelegte Seed-Präzisionsziel von max(20 % des Mittelwerts, 0,02 mmol/g).

![Modellvergleich mit Referenz](01_loading_reference.png)

## Warum MH derzeit die zweckmäßige Wahl ist

MH liefert {summary[0]['mean']:.3f} mmol/g und unterschätzt die CRAFTED-Beladung deskriptiv um {abs(summary[0]['relative_bias_percent']):.1f} %. POLAR pretrained und POLAR Stage 1 liegen bei {summary[1]['mean']:.3f} bzw. {summary[2]['mean']:.3f} mmol/g, also etwa dem {summary[1]['reference_ratio']:.1f}- bzw. {summary[2]['reference_ratio']:.1f}-Fachen der klassischen Referenz. Die referenznähere MH-Vorhersage ist damit ein wesentlich besserer Ausgangspunkt innerhalb dieses Protokolls. Die bereits etwa 35 % große MH-Abweichung verhindert eine Behauptung hoher quantitativer Genauigkeit.

Alle drei MH-Ketten bestehen die vorhandenen operationalen Einzelkettenscreens. Sie besitzen ungefähr 220–226 geschätzte effektive Stichproben. POLAR pretrained besitzt nur ungefähr 10–17, POLAR Stage 1 ungefähr 20–27. Keine der sechs POLAR-Ketten besteht sämtliche Kriterien gemeinsam. Teilweise verändern sich die Mittelwerte zwischen den Produktionshälften weiter deutlich. Dass ein Modell eine technisch vollständige Rechnung liefert oder eine einzelne Präzisionsprüfung besteht, validiert seine Adsorptionsvorhersage nicht.

![Zeitverläufe](02_loading_histories.png)

![Samplingqualität](03_sampling_quality.png)

Die drei MH-Seedmittel liegen bei 0,582/0,439/0,587 mmol/g (gerundet). Trotz bestandener Einzelkettenscreens ergibt die Streuung dieser drei Mittelwerte mit dem Student-t-Faktor eine relative Intervallhalbbreite von {summary[0]['relative_ci95_half_width']*100:.1f} %. Dies ist der konkrete verbleibende MH-Präzisionspunkt. Das Ergebnis ist kein Widerspruch: Kettenscreen und Präzision zwischen unabhängigen Seeds prüfen unterschiedliche Aspekte.

## Was sich über POLAR-Fine-Tuning sagen lässt

Das Stage-1-Modell liefert deskriptiv eine höhere Beladung und eine größere Abweichung zu CRAFTED als POLAR pretrained. Eine Verbesserung der Referenzübereinstimmung ist in diesem Pilot somit nicht sichtbar. Die Intervalle beider Modelle überlappen, und beide Gruppen enthalten unzureichend stabile Ketten. Der gepaarte deskriptive Unterschied Stage 1 minus pretrained beträgt {polar_diff['mean']:.3f} mmol/g mit einem Seed-t-Intervall von [{polar_diff['ci95_low']:.3f}, {polar_diff['ci95_high']:.3f}] mmol/g. Die Paarung verwendet gleiche Seednummern; sie erzeugt keine zusätzlichen unabhängigen Daten. Daraus wird keine statistisch abgesicherte Verschlechterung des Gleichgewichtsmodells behauptet.

MH fine-tuned ist in diesem neuen Drei-Modell-Pilot nicht enthalten. Seine alten Ergebnisse werden wegen anderer Protokolle nicht zu dieser Rangfolge hinzugefügt. Ebenso belegt dieser MOF-5-Punkt keine allgemeine Überlegenheit von MH gegenüber POLAR bei anderen Materialien und keine Eignung für flexible Framework-MD in Modul D.

## Rechenaufwand

Die mittleren Laufzeiten betragen MH {summary[0]['mean_runtime_hours']:.2f} h, POLAR pretrained {summary[1]['mean_runtime_hours']:.2f} h und POLAR Stage 1 {summary[2]['mean_runtime_hours']:.2f} h pro 60000-Schritt-Lauf. Insgesamt wurden {sum(r['total_runtime_hours'] for r in summary):.2f} GPU-Laufzeitstunden archiviert. Alle Marker nennen dieselbe GPU-Baureihe und dieselben Paketversionen. Die POLAR-Läufe sind hier etwa {summary[1]['mean_runtime_hours']/summary[0]['mean_runtime_hours']:.1f}–{summary[2]['mean_runtime_hours']/summary[0]['mean_runtime_hours']:.1f} Mal so aufwendig. Unterschiedliche Beladungen und MC-Akzeptanzen verändern die Zahl und Größe der Energieauswertungen. Die Zeiten sind deshalb beobachtete Workflowkosten, keine isolierten Inferenzzeiten und kein kontrollierter Architekturbenchmark.

![Referenzabweichung und Kosten](04_reference_and_cost.png)

## Konkrete nächste Entscheidung

Für weitere MOF-5/CO₂-Rechnungen die unveränderte **MH-Foundation** (`assets/umof_mh_pretrained.model`, Head `r2scan+d4`, float64) innerhalb des festgeschriebenen Pilotprotokolls priorisieren. `recommended_model.json` dokumentiert Gewichts-Hash, Head und begrenzten Gültigkeitsbereich. Es wird keine Modellregistrierung geändert und keine Rechnung automatisch gestartet.

Zunächst die MH-Präzision bei 1 bar verbessern: unabhängige Seeds ergänzen oder deren stationäre Produktionsabschnitte mit geprüftem Restart-Verfahren verlängern. Die erforderliche Länge ergibt sich aus den Diagnosen; dieser Bericht behauptet keine erforderliche feste Anzahl. Vor der Auswertung das Präzisionsziel und das Rechenbudget festlegen und alle vorhandenen Seeds beibehalten. Danach weitere Druckpunkte gegen passende Referenzen prüfen. Für die aktuelle Entscheidung ist weiteres POLAR-Training ohne gezielte Ursachenanalyse der Adsorptionsabweichung nicht die erste Priorität.

Das ist eine Priorisierung von Kandidaten, keine Freigabe beliebiger Druckreihen. Für experimentelle Genauigkeit sind geeignete experimentelle Referenzen nötig. Für eine physikalische Bewertung der Potentiale an den neu zugelassenen Zuständen fehlen hier unabhängige DFT-Interaktionsenergie-Labels.

## Technische Prüfung und Protokollgrenzen

- Alle eingefrorenen Bundle-Dateien einschließlich Modellgewichte wurden über SHA-256 geprüft. Scan-, Smoke- und Pilotmarker stimmen hinsichtlich Modell, Seed, Stage und Bundle-Identität überein. Alle drei Modelle bestehen die endlichen Energieproben und technischen Kurztests.
- Alle neun Produktionsmittel und Kettenscreens wurden aus binären akzeptierten Übergängen neu berechnet. Abgelehnte MC-Schritte bleiben als unveränderte Zustände enthalten. Die jeweiligen 40000 rekonstruierten Produktionsschritte stimmen mit der archivierten `production.csv` einschließlich Wechselwirkungsenergien überein.
- Finale Restart-Zähler, starres Framework, Zelle, Molekülzählung und die Einhaltung der eingefrorenen Überlappungsregel am Endzustand wurden geprüft. Pro Lauf liegen die erwarteten 60 Checkpoint-Strukturen vor; deren vollständige Wandprüfung ist im archivierten Worker dokumentiert, der finale Zustand wurde hier unabhängig erneut geprüft.
- Die Ausschlussregel verwendet die Summe der ASE-vdW-Radien minus 0,35 Å pro Atom. Sie wurde vorab festgelegt und nicht anhand dieser Resultate an CRAFTED angepasst. Der Energieguard bei −2 eV/Gast ist ein grober Anomalieabbruch und kein Genauigkeitskriterium.
- Gemeinsame float64-Präzision und diese Ausschlussregel unterscheiden den neuen Pilot von den historischen MH/POLAR-Kampagnen. Alte Isothermen werden nicht mit den neuen Werten zu einer gemeinsamen Druckreihe verbunden.
- Dies ist ein Ein-Druckpunkt-Vergleich. Daten sind unzureichend für eine vollständige Isothermenvalidierung, eine Architekturaussage über andere Systeme oder eine Entscheidung zur Flexibilitätsmodellierung.

Die Analyse startet weder Inferenz noch Training oder Simulation. Quellen bleiben unverändert. Vollständige Daten, Einzelkettenprüfungen, Kontraste und SHA-256 stehen in den CSV-/JSON-Dateien. Abbildungen sind auf Englisch als PNG und SVG exportiert.

## Für eine Präsentation

1. `01_loading_reference`: “MH is the closest current candidate to the classical reference.”
2. `03_sampling_quality`: “MH passes all chain screens, but between-seed precision remains unresolved.”
3. `04_reference_and_cost`: “POLAR is more expensive here without improving agreement with the classical reference.”

Schlusssatz: **“We prioritize pretrained uMOF-MH for further MOF-5 validation. The current pilot does not establish quantitative adsorption accuracy.”** Die Zeitverläufe als Backup zeigen, warum abgeschlossene POLAR-Läufe noch keine stabilen Gleichgewichtsmittel liefern.
"""
    (output/"REPORT.md").write_text(text)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,default=ROOT/BUNDLE)
    parser.add_argument("--as-of",default="2026-11-13")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    output=(args.output or ROOT/f"reports/umof_mof5_review_{args.as_of}").resolve()
    output.mkdir(parents=True,exist_ok=True)
    sources=Sources(ROOT)
    data=collect(sources,args.bundle.resolve())
    files=figures(data,output)
    write_csv(output/"model_summary.csv",data["summary"])
    write_csv(output/"seed_diagnostics.csv",data["seeds"])
    write_csv(output/"paired_seed_contrasts.csv",data["paired_seed_contrasts"])
    write_csv(output/"source_checks.csv",data["audits"])
    np.savez_compressed(output/"loading_histories.npz",**data.pop("series"))
    dump(output/"results.json",data)
    write_report(data,output,args.as_of)
    for name,expected in sources.files.items():
        if digest(ROOT/name)!=expected:raise ValueError(f"Source changed during analysis: {name}")
    import matplotlib,scipy,ase
    dump(output/"provenance.json",{"as_of":args.as_of,"source_sha256":sources.files,
        "analysis_sha256":digest(__file__),"reused_diagnostics_sha256":digest(ROOT/'modules/module_c_mlips/umof_isotherm_stability.py'),
        "trace_reader_sha256":digest(ROOT/'engines/mlip_mc_runner.py'),
        "versions":{"numpy":np.__version__,"matplotlib":matplotlib.__version__,"scipy":scipy.__version__,"ase":ase.__version__},
        "figures":files,"sources_unchanged":True,"model_inference":False,"new_simulations":False})
    with zipfile.ZipFile(output/"figures_and_results.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for pattern in ("*.png","*.svg","*.csv","*.json","REPORT.md"):
            for path in sorted(output.glob(pattern)):archive.write(path,path.name)
    print(table(data["summary"],[("model","Model"),("mean","Mean mmol/g"),("relative_bias_percent","Deviation %"),("passed_chains","Passed chains"),("seed_precision_passed","Seed precision"),("mean_runtime_hours","Hours/run")]))
    print(f"Created {output/'REPORT.md'} and {len(files)} PNG/SVG figure pairs; source audit passed.")


if __name__=="__main__":main()
