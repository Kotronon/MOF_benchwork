"""Longer GCMC comparison of baseline and Stage 1, without repeating Widom.

Reuse the inspected potential/MC adapter and keep the original campaign intact.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from . import umof_adsorption_comparison as original
from .umof_finetuning import sha256_file
from .umof_workflow import atomic_json, campaign_lock, read_json, write_csv

DEFAULT_CONFIG = original.PROJECT / "input_json_files/umof_polar_mgmof74_co2_refinement.json"


def load_config(path):
    recipe = read_json(path)
    source = original.resolve(recipe["source_config"]["path"])
    if sha256_file(source) != recipe["source_config"]["sha256"]:
        raise ValueError("Original configuration changed")
    config = original.load_config(source)
    if recipe["models"] != ["baseline", "stage1"] or len(recipe["seeds"]) != 1:
        raise ValueError("Refinement requires baseline, Stage 1 and exactly one seed")
    if any(p not in config["pressures_bar"] for p in recipe["pressures_bar"]) or len(set(recipe["pressures_bar"])) != len(recipe["pressures_bar"]) or not recipe["pressures_bar"]:
        raise ValueError("Distinct pressures from original comparison required")
    if any(type(s) is not int or s <= 0 for s in recipe["seeds"]):
        raise ValueError("Positive seed required")
    for key in ("equilibration_steps", "production_steps", "checkpoint_interval"):
        if type(recipe["gcmc"][key]) is not int or recipe["gcmc"][key] <= 0:
            raise ValueError(f"Positive {key} required")
    if not recipe["gpu_ids"] or any(type(g) is not int or g < 0 for g in recipe["gpu_ids"]) or len(set(recipe["gpu_ids"])) != len(recipe["gpu_ids"]):
        raise ValueError("Distinct GPU IDs required")
    pilot = original.resolve(recipe["pilot_summary"]["path"])
    if sha256_file(pilot) != recipe["pilot_summary"]["sha256"]:
        raise ValueError("Pilot summary changed")
    config["models"] = [m for m in config["models"] if m["label"] in recipe["models"]]
    for key in ("seeds", "pressures_bar", "gpu_ids", "gcmc", "output_directory"):
        config[key] = recipe[key]
    config["refinement_recipe"] = recipe
    config["phases"] = ["gcmc"]
    return config


def jobs(config, output):
    for pressure in config["pressures_bar"]:
        for model in config["models"]:
            seed = config["seeds"][0]
            yield model["label"], seed, pressure, original.job_directory(output, model["label"], seed, "gcmc", pressure)


def report(config, output):
    from ase.io import read
    from analysis.module_c_isotherm_report import compare_reference
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = output / "report"
    destination.mkdir(parents=True, exist_ok=True)
    host = read(original.resolve(config["framework"]["path"]))
    factor = 1000/host.get_masses().sum()
    with original.resolve(config["refinement_recipe"]["pilot_summary"]["path"]).open() as handle:
        pilot = {(r["model"], float(r["pressure_bar"])): float(r["mean_loading_mol_per_kg"]) for r in csv.DictReader(handle)}
    rows, blocks = [], []
    for label, seed, pressure, directory in jobs(config, output):
        result = original.get_completed(directory, config)
        if result is None:
            continue
        point = result["pressure_points"][0]
        d = point["production_diagnostics"]
        rows.append({"model": label, "seed": seed, "pressure_bar": pressure,
            "mean_loading_mol_per_kg": point["loading_mol_per_kg"],
            "pilot_loading_mol_per_kg": pilot[label, pressure],
            "first_half_loading_mol_per_kg": d["first_half_mean_molecules"]*factor,
            "second_half_loading_mol_per_kg": d["second_half_mean_molecules"]*factor,
            "block_sem_mol_per_kg": d["block_sem_molecules"]*factor,
            "runtime_seconds": result["runtime_seconds"]})
        for i, value in enumerate(d["production_block_means_molecules"], 1):
            blocks.append({"model": label, "pressure_bar": pressure, "block": i, "loading_mol_per_kg": value*factor})
    reference = read_json(original.resolve(config["reference"]["path"]))
    points = [{"pressure_bar": float(r["pressure"]), "loading_mol_per_kg": float(r["total_adsorption"])} for r in reference["isotherm_data"] if r["pressure"] > 0 and r["total_adsorption"] >= 0]
    points.sort(key=lambda r: r["pressure_bar"])
    common = [p for p in config["pressures_bar"] if sum(r["pressure_bar"] == p for r in rows) == len(config["models"])]
    deviations, metrics = compare_reference([r for r in rows if r["pressure_bar"] in common], {"points": points}) if common else ([], [])
    for name, records in (("isotherm_summary.csv", rows), ("production_blocks.csv", blocks),
                          ("reference_deviations.csv", deviations), ("reference_metrics.csv", metrics)):
        write_csv(destination / name, records)
    atomic_json(destination / "comparison.json", {"completed_jobs": len(rows), "total_jobs": len(list(jobs(config, output))),
        "summary": rows, "production_blocks": blocks, "reference_metrics": metrics,
        "reference": config["reference"], "convergence": "unestablished; compare block means and measurement halves",
        "uncertainty": "one seed; block SEM is diagnostic and can underestimate uncertainty when correlated or drifting"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    colors = {"baseline": "#0072B2", "stage1": "#D55E00"}
    axes[0].plot([p["pressure_bar"] for p in points], [p["loading_mol_per_kg"] for p in points], "k.--", label="Experiment (reported uptake)")
    for label in colors:
        selected = [r for r in rows if r["model"] == label]
        if selected:
            axes[0].plot([r["pressure_bar"] for r in selected], [r["mean_loading_mol_per_kg"] for r in selected], "o-", color=colors[label], label=label)
            axes[0].scatter([r["pressure_bar"] for r in selected], [r["pilot_loading_mol_per_kg"] for r in selected], marker="x", color=colors[label], label=label+" short pilot")
        for i, pressure in enumerate(config["pressures_bar"]):
            selected_blocks = [r for r in blocks if r["model"] == label and r["pressure_bar"] == pressure]
            if selected_blocks:
                axes[1].plot([r["block"] for r in selected_blocks], [r["loading_mol_per_kg"] for r in selected_blocks],
                    linestyle=["-", "--"][i % 2], color=colors[label], label=f"{label}, {pressure:g} bar")
    axes[0].set(xlabel="Pressure (bar)", ylabel="Loading (mol/kg)", ylim=(0, None))
    axes[1].set(xlabel="Production block", ylabel="Block mean loading (mol/kg)")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(f"Mg-MOF-74 / CO2, 298 K — longer comparison ({len(rows)}/4 jobs)\nOne seed; convergence unestablished")
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"comparison.{suffix}", dpi=160)
    plt.close(fig)
    (destination / "README.md").write_text(
        "# Longer baseline / Stage 1 comparison\n\n"
        f"Completed: {len(rows)}/{len(list(jobs(config, output)))} GCMC jobs. "
        f"Per job: {config['gcmc']['equilibration_steps']} equilibration + {config['gcmc']['production_steps']} production steps.\n\n"
        "The same seed, rigid framework, CO2 geometry, same-model vacuum gas references, overlap rule and fugacity treatment as the pilot are used. "
        "These are fresh runs from the empty framework; the short pilot chains are not extended. Widom is not repeated.\n\n"
        "comparison.png/.svg show the longer-run uptake, short pilot values, reported experimental uptake and production block means. "
        "isotherm_summary.csv compares first and second measurement halves. Inspect drift before interpreting model rankings. "
        "One seed provides no between-seed uncertainty, and completion alone does not prove convergence. "
        "The experimental absolute/excess basis remains unspecified; agreement is provisional.\n")


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    with campaign_lock(output):
        effective = output / "effective_config.json"
        if effective.exists() and read_json(effective) != config:
            raise ValueError("Output has different settings; choose a new --output")
        atomic_json(effective, config)
        provenance = {"refinement_sha256": sha256_file(__file__), "base_adapter_sha256": sha256_file(original.__file__),
            "engine_sha256": sha256_file(original.PROJECT / "engines/mlip_mc_runner.py")}
        if (output / "implementation.json").exists() and read_json(output / "implementation.json") != provenance:
            raise ValueError("Implementation changed; choose a new --output")
        atomic_json(output / "implementation.json", provenance)
        slots = config["gpu_ids"] if config["device"].startswith("cuda") else [None]
        pending = [j for j in jobs(config, output) if original.get_completed(j[-1], config) is None]
        for start in range(0, len(pending), len(slots)):
            active = []
            try:
                for gpu, (label, seed, pressure, directory) in zip(slots, pending[start:start+len(slots)]):
                    directory.mkdir(parents=True, exist_ok=True)
                    attempt = directory / f"attempt-{len(list(directory.glob('attempt-*')))+1:03d}"
                    attempt.mkdir()
                    command = [sys.executable, "-m", "modules.module_c_mlips.umof_adsorption_refinement", "worker",
                        "--config", str(effective), "--model", label, "--pressure", str(pressure), "--output", str(attempt)]
                    env = dict(os.environ)
                    if gpu is not None:
                        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    with (attempt / "worker.log").open("w") as handle:
                        process = subprocess.Popen(command, cwd=original.PROJECT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                    active.append((process, directory, attempt))
                    print(f"Start {label}, {pressure:g} bar, seed {seed}, GPU {gpu}; log {attempt/'worker.log'}", flush=True)
                atomic_json(output / "status.json", {"status": "running", "active_jobs": len(active)})
                while active:
                    for record in list(active):
                        process, directory, attempt = record
                        code = process.poll()
                        if code is None:
                            continue
                        if code:
                            raise RuntimeError(f"Job exited {code}; inspect {attempt/'worker.log'}")
                        path = attempt / "result.json"
                        atomic_json(directory / "completed.json", {"config_sha256": original.fingerprint(config),
                            "result_path": str(path.relative_to(directory)), "result_sha256": sha256_file(path)})
                        active.remove(record)
                        print(f"Done {directory}", flush=True)
                        report(config, output)
                    if active:
                        time.sleep(.5)
            except BaseException as exc:
                atomic_json(output / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error": str(exc)})
                raise
            finally:
                for process, _, _ in active:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
        atomic_json(output / "status.json", {"status": "completed", "completed_jobs": len(list(jobs(config, output)))})
        print(f"Report: {output/'report/README.md'}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run", "check", "status", "report", "smoke", "worker"], nargs="?", default="run")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path)
    p.add_argument("--model", choices=["baseline", "stage1"])
    p.add_argument("--pressure", type=float)
    args = p.parse_args(argv)
    if args.action == "worker":
        config = read_json(args.config)
        if args.model is None or args.pressure is None or args.output is None:
            p.error("worker requires model, pressure and output")
        original.worker(config, args.model, config["seeds"][0], "gcmc", args.pressure, args.output)
        return 0
    config = load_config(args.config)
    if args.action == "smoke":
        config["gcmc"] = {"equilibration_steps": 16, "production_steps": 64, "checkpoint_interval": 40}
        config["output_directory"] += "_smoke"
    output = (args.output or original.resolve(config["output_directory"])).resolve()
    if args.action in {"run", "smoke", "check"}:
        checked = original.check(config)
        checked.update(jobs=len(list(jobs(config, output))),
            steps_per_model=len(config["pressures_bar"])*(config["gcmc"]["equilibration_steps"]+config["gcmc"]["production_steps"]),
            phases=["gcmc"])
        print(json.dumps(checked, indent=2), flush=True)
    if args.action in {"run", "smoke"}:
        run(config, output)
    elif args.action == "status":
        print(json.dumps({"completed": sum(original.get_completed(j[-1], config) is not None for j in jobs(config, output)),
            "total": len(list(jobs(config, output))), "output": str(output)}, indent=2))
    elif args.action == "report":
        report(config, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
