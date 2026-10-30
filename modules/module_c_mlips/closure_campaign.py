"""Prepare, transfer and execute bounded MH/POLAR adsorption confirmation jobs.

No training, download or scheduler submission is performed. Cluster workers use
the existing, inspected Metropolis adapter. Finished results require matching
input AND implementation identities. Interrupted jobs restart in new attempts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np

from . import umof_adsorption_comparison as adapter
from .umof_workflow import atomic_json, campaign_lock
from .umof_finetuning import sha256_file

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT = PROJECT / "reports/module_c_closure/cluster_bundle"


def job_list(config):
    return [{"id": f"{model['label']}_p{pressure:g}_s{seed}", "model": model["label"],
             "seed": seed, "pressure_bar": pressure}
            for pressure in config["pressures_bar"] for seed in config["seeds"] for model in config["models"]]


def validate(config, root):
    if config["schema_version"] != 1 or config["component"] != "CO2" or config["device"] != "cuda":
        raise ValueError("Expected schema-1 rigid CO2 CUDA campaign")
    if len(config["models"]) != 2 or {m["label"] for m in config["models"]} != {"pretrained", "finetuned"}:
        raise ValueError("Frozen pretrained/finetuned pair required")
    seeds, pressures = config["seeds"], config["pressures_bar"]
    minimum = 1 if config.get("technical_smoke") is True else 3
    if len(seeds) < minimum or len(set(seeds)) != len(seeds) or any(type(s) is not int or s <= 0 for s in seeds):
        raise ValueError("Three distinct positive integer seeds required for scientific runs")
    if not pressures or len(set(pressures)) != len(pressures) or any(not np.isfinite(p) or p <= 0 for p in pressures):
        raise ValueError("Distinct positive finite pressures required")
    for key in ("equilibration_steps", "production_steps", "checkpoint_interval"):
        if type(config["gcmc"][key]) is not int or config["gcmc"][key] <= 0:
            raise ValueError(f"Positive integer {key} required")
    if config["dtype"] not in ("float32", "float64") or not 0 < config["overlap_covalent_radius_scale"] <= 1:
        raise ValueError("Invalid numerical/overlap settings")
    for row in [config["framework"], config["adsorbate"], *config["models"], *config["references"]]:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Bundle inputs must use contained relative paths")
        if sha256_file(root / path) != row["sha256"]:
            raise ValueError(f"Changed asset: {path}")
    for model in config["models"]:
        if model["profile"] not in ("mh", "polar") or model["head"] not in ("Default", "r2scan+d4"):
            raise ValueError("Unknown frozen model profile/head")
    return job_list(config)


def verify_bundle(root):
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    for name, expected in manifest["file_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Bundle file changed: {name}")
    return manifest


def completed_result(directory, identity):
    marker = directory / "completed.json"
    if not marker.exists():
        return None
    stored = json.loads(marker.read_text())
    if stored["identity"] != identity:
        raise ValueError("Completed job has different inputs/implementation")
    result = directory / stored["result_path"]
    if sha256_file(result) != stored["result_sha256"]:
        raise ValueError("Completed result checksum mismatch")
    return json.loads(result.read_text())


def identity_for(root, system):
    # Bundle includes exact Python code and assets. Runtime versions are added at execution.
    identity = {"bundle_sha256": sha256_file(root / "bundle_manifest.json"),
                "config_sha256": sha256_file(root / f"{system}.json")}
    return identity


def check(root, system=None):
    verify_bundle(root)
    totals = {}
    for name in ([system] if system else ["mh", "polar"]):
        config = json.loads((root / f"{name}.json").read_text())
        jobs = validate(config, root)
        totals[name] = {"jobs": len(jobs), "mc_steps": len(jobs)*(config["gcmc"]["equilibration_steps"]+config["gcmc"]["production_steps"])}
    print(json.dumps(totals, indent=2))
    return totals


def prepare(destination):
    """Create a self-contained code/asset bundle, without loading model weights."""
    from ase.io import read, write
    from .datasets.common import load_host_guest_system
    from pipeline.config import normalize_config
    from pipeline.planning import build_run_plan
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Bundle already exists; use check/run or choose a new --directory: {destination}")
    destination.mkdir(parents=True)
    assets = destination / "assets"
    assets.mkdir()
    origins = {}

    def asset(source, name):
        source = Path(source)
        target = assets / name
        shutil.copy2(source, target)
        origins[str(target.relative_to(destination))] = str(source.relative_to(PROJECT))
        return {"path": str(target.relative_to(destination)), "sha256": sha256_file(target)}

    mh_source = json.loads((PROJECT / "input_json_files/benchmark_mof5_co2_umof_025pct_isotherm.json").read_text())
    plan = build_run_plan(normalize_config(mh_source))
    ff = plan["resources"]["forcefield"]
    host, guest, _ = load_host_guest_system(plan["resources"]["cif_path"], ff["adsorbates"]["CO2"], ff["files"]["pseudo_atoms"],
        cell_representation=plan["simulation"]["cell_representation"], unit_cells=plan["simulation"]["unit_cells"],
        cutoff_A=6., minimum_image_policy="error")
    write(assets / "mof5.extxyz", adapter.clean_atoms(host), format="extxyz")
    write(assets / "co2_mh.extxyz", adapter.clean_atoms(guest), format="extxyz")
    sources = {
        "mh": ("MOF-5", [.1, 1., 10.], "float64", "mof5.extxyz", "co2_mh.extxyz",
            [(m["model"], m["head"]) for m in mh_source["benchmark"]["potential_benchmark"]["campaign"]["models"]],
            ["DDEC_IRMOF-1_UFF_CO2_298.csv"]),
        "polar": ("ZIF-8", [.1, .5, 1., 5.], "float32", "zif8.extxyz", "co2_polar.extxyz",
            [("outputs/umof_polar_24h_float32/assets/uMOF-POLAR.model", "r2scan+d4"),
             ("outputs/umof_polar_24h_float32/025pct/seed-12345/models/umof_polar_adsorption_finetuned.model", "Default")],
            ["DDEC_ZIF-8_UFF_CO2_298.csv", "DDEC_ZIF-8_DREIDING_CO2_298.csv"])}
    asset(PROJECT / "outputs/umof_polar_followup_evaluation/crafted_zif8/framework.extxyz", "zif8.extxyz")
    asset(PROJECT / "outputs/umof_polar_followup_evaluation/crafted_zif8/co2.extxyz", "co2_polar.extxyz")
    for profile, (material, pressures, dtype, framework, gas, models, refs) in sources.items():
        model_rows = [{"label": label, "profile": profile, "head": head,
                       **asset(PROJECT / path, f"{profile}_{label}.model")}
                      for label, (path, head) in zip(("pretrained", "finetuned"), models)]
        if profile == "mh":
            expected = [m["sha256"] for m in mh_source["benchmark"]["potential_benchmark"]["campaign"]["models"]]
        else:
            archived = json.loads((PROJECT / "outputs/umof_polar_followup_evaluation/crafted_zif8/config.json").read_text())
            expected = [m["sha256"] for m in archived["models"] if m["label"] in ("baseline", "parent_stage1")]
        if [m["sha256"] for m in model_rows] != expected:
            raise ValueError(f"{profile}: model weights differ from archived comparison")
        references = [{"label": name, "class": "classical_simulation", "basis": "absolute",
                       **asset(PROJECT / "CRAFTED-2.0.0/ISOTHERM_FILES" / name, name)} for name in refs]
        local_asset = lambda name: {"path": "assets/" + name, "sha256": sha256_file(assets / name)}
        config = {"schema_version": 1, "material": material, "component": "CO2", "temperature_K": 298.,
            "device": "cuda", "dtype": dtype, "models": model_rows, "framework": local_asset(framework),
            "adsorbate": local_asset(gas), "references": references, "pressures_bar": pressures,
            "seeds": [32452843, 32452867, 32452889], "overlap_covalent_radius_scale": .8,
            "gcmc": {"equilibration_steps": 20000, "production_steps": 40000, "checkpoint_interval": 1000},
            "precision": {"relative_seed_ci95_target": .20, "absolute_floor_mmol_g": .02},
            "scope": "Fresh bounded confirmation; same-model vacuum gas references; neutral singlet conditioning; no external dispersion. Fixed overlap wall = 0.8 times sums of ASE covalent radii. No MH/POLAR architecture ranking across different MOFs."}
        atomic_json(destination / f"{profile}.json", config)
    for directory in ("analysis", "modules", "engines", "pipeline", "converter", "resolvers", "parsers"):
        for path in (PROJECT / directory).rglob("*.py"):
            target = destination / path.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for name in ("requirements.txt", "requirements-mlip-mc.txt", "requirements-module-c-umof.txt"):
        shutil.copy2(PROJECT / name, destination / name)
    # Preserve the original runtime metadata rather than inventing GPU wheel pins.
    for profile, source in (("mh", "outputs/module_C_umof_finetuning_matrix_v2/environment.json"),
                            ("polar", "outputs/umof_polar_24h_float32/environment.json")):
        asset(PROJECT / source, f"{profile}_source_environment.json")
    readme = """# GPU confirmation for Module C

This bundle is self-contained: selected weights, fixed host/CO2 structures, classical reference CSVs and project Python code. No CRAFTED tree, original training data, CP2K, scheduler account or network download is needed to run it in a compatible environment.

Extract and enter this directory on the cluster. Activate the existing MOF_sim environment with CUDA PyTorch, ASE, SciPy, MACE 0.3.16, graph_longrange 0.4.0 and the pinned MLIP-MC dependency. See requirements-mlip-mc.txt and assets/*_source_environment.json. Install dependencies on the login node according to your cluster's CUDA setup; this bundle does not install them automatically.

```
python -m modules.module_c_mlips.closure_campaign check --directory .
# Inside an allocation with ONE visible GPU (e.g. CUDA_VISIBLE_DEVICES=0):
python -m modules.module_c_mlips.closure_campaign smoke --directory . --system mh
python -m modules.module_c_mlips.closure_campaign smoke --directory . --system polar
python -m modules.module_c_mlips.closure_campaign run --directory . --system mh --max-jobs 18
python -m modules.module_c_mlips.closure_campaign run --directory . --system polar --max-jobs 24
python -m modules.module_c_mlips.closure_campaign report --directory . --system mh
python -m modules.module_c_mlips.closure_campaign report --directory . --system polar
```

Jobs execute serially on the first GPU visible to the process; the launcher does not override CUDA_VISIBLE_DEVICES. Request resources through your site's scheduler. For a first bounded batch use --max-jobs 1. Repeating a command reuses matching completed jobs. For independent scheduler-array jobs use `run --system mh --job-index INDEX --max-jobs 1` (indices 0–17), or polar indices 0–23. Different indices may run concurrently; each has its own writer lock. Do not launch an unindexed full campaign concurrently with an array against the same outputs. `status` reports progress without running anything.

Each smoke uses two separate jobs (one per model), one seed and the lowest pressure, with 2 equilibration + 4 production moves each. Its configurations/results use the separate mh_smoke/polar_smoke names and never contribute to scientific summaries. Passing smoke establishes technical execution only.

MH: 18 jobs (2 models × 3 pressures × 3 new seeds). POLAR: 24 jobs (2 × 4 × 3). Each job has 20,000 equilibration + 40,000 production steps. Total budget: 2,520,000 MC steps. Completion is not evidence of equilibrium. Interrupted jobs restart from the original seed in a new attempt; partial trajectories are preserved but never treated as independent replicas. There is no automatic extension or retraining.

The adapter uses a same-model vacuum CO2 reference and the full GCMC acceptance prefactor; rejected production steps contribute to means. Both systems use the explicitly frozen covalent-radius overlap approximation. For MH this differs from the historical MLIP-MC default wall: these are separate fresh results, not continuations or direct replicas of the pilot protocol. Native MH float64 and POLAR float32 are retained. Different MOFs/protocols prohibit an architecture-speed or accuracy ranking.

`check` verifies bundle files and input hashes. `run` also checks CUDA and runtime versions, and stores exact package versions with each result. Package, code and config changes invalidate reuse. Final reports are under results/{mh,polar}/report and contain reference errors, seed Student-t intervals, correlation/batch diagnostics and sampling flags. A passed operational screen is not proof of equilibrium. Return those result directories together with mh.json, polar.json and bundle_manifest.json for the final local update. `report` works without CUDA.
"""
    (destination / "START_HERE.md").write_text(readme)
    atomic_json(destination / "bundle_manifest.json", {"schema_version": 1, "origins": origins,
        "file_sha256": {str(p.relative_to(destination)): sha256_file(p) for p in sorted(destination.rglob("*")) if p.is_file()},
        "new_simulations_started": False})
    check(destination)
    archive = destination.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(destination, arcname="module_c_cluster")
    archive.with_suffix(archive.suffix + ".sha256").write_text(sha256_file(archive) + "  " + archive.name + "\n")
    print(f"Cluster archive: {archive}")


def execution_versions():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: execute inside the cluster GPU allocation")
    versions = {key: importlib.metadata.version(key) for key in ("torch", "mace-torch", "mlip-mc", "ase", "numpy", "scipy", "graph_longrange")}
    if versions["mace-torch"] != "0.3.16" or versions["graph_longrange"] != "0.4.0" or versions["mlip-mc"] != "0.1.3":
        raise RuntimeError("Expected mace-torch 0.3.16, graph_longrange 0.4.0 and mlip-mc 0.1.3")
    versions["cuda_runtime"] = str(torch.version.cuda)
    versions["gpu_model"] = torch.cuda.get_device_name(0)
    return versions


def smoke(root, system):
    if system not in ("mh", "polar"):
        raise ValueError("Smoke expects mh or polar")
    verify_bundle(root)
    config = json.loads((root / f"{system}.json").read_text())
    config.update(technical_smoke=True, seeds=config["seeds"][:1], pressures_bar=config["pressures_bar"][:1],
                  gcmc={"equilibration_steps": 2, "production_steps": 4, "checkpoint_interval": 1})
    path = root / f"{system}_smoke.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Existing smoke inputs changed")
    atomic_json(path, config)
    run(root, system + "_smoke", 2)
    print("Technical smoke only: two models × six MC moves; no accuracy or convergence claim.")


def run(root, system, max_jobs, index=None):
    verify_bundle(root)
    config_path = root / f"{system}.json"
    config = json.loads(config_path.read_text())
    jobs = validate(config, root)
    versions = execution_versions()
    identity = {**identity_for(root, system), "runtime_versions": versions}
    completed_count = 0
    for job in ([jobs[index]] if index is not None else jobs):
        directory = root / "results" / system / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        with campaign_lock(directory):
            if completed_result(directory, identity) is not None:
                print(f"Reuse {job['id']}", flush=True)
                continue
            if completed_count >= max_jobs:
                break
            attempts = [int(p.name.split("-")[-1]) for p in directory.glob("attempt-*")]
            attempt = directory / f"attempt-{max(attempts, default=0)+1:03d}"
            attempt.mkdir()
            atomic_json(directory / "running.json", {"identity": identity, "job": job})
            cmd = [sys.executable, "-m", "modules.module_c_mlips.closure_campaign", "worker", "--directory", str(root),
                   "--system", system, "--job-index", str(jobs.index(job)), "--attempt", str(attempt)]
            print(f"Run {job['id']}; {attempt / 'worker.log'}", flush=True)
            with (attempt / "worker.log").open("w") as log:
                process = subprocess.Popen(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                    raise
            if code:
                raise RuntimeError(f"Worker failed ({code}); inspect {attempt / 'worker.log'}")
            result = attempt / "result.json"
            atomic_json(directory / "completed.json", {"identity": identity,
                "result_path": str(result.relative_to(directory)), "result_sha256": sha256_file(result)})
            completed_count += 1
    print(f"Finished {completed_count} new jobs; regenerate with the report command.")


def status(root, system):
    verify_bundle(root)
    config = json.loads((root / f"{system}.json").read_text())
    jobs = validate(config, root)
    identity = identity_for(root, system)
    n = 0
    for job in jobs:
        directory = root / "results" / system / job["id"]
        marker = directory / "completed.json"
        if marker.exists():
            stored = json.loads(marker.read_text())
            if any(stored["identity"].get(k) != v for k, v in identity.items()):
                raise ValueError("Changed bundle/config in completed result")
            completed_result(directory, stored["identity"])
            n += 1
    print(f"{system}: {n}/{len(jobs)} completed jobs")


def report(root, system):
    """Never turn incomplete seed groups into a final accuracy assessment."""
    from analysis.module_c_closure import seed_interval, write_csv, error_metrics
    from .umof_isotherm_stability import assess
    from engines.mlip_mc_runner import _read_gcmc_transition_log, _reconstruct_gcmc_samples
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    verify_bundle(root)
    config = json.loads((root / f"{system}.json").read_text())
    jobs = validate(config, root)
    result_root = root / "results" / system
    output = result_root / "report"
    output.mkdir(parents=True, exist_ok=True)
    seeds, records, identities = [], [], set()
    for job in jobs:
        directory = result_root / job["id"]
        marker = directory / "completed.json"
        if not marker.exists(): continue
        stored = json.loads(marker.read_text())
        if any(stored["identity"].get(k) != v for k, v in identity_for(root, system).items()):
            raise ValueError("Changed input/implementation identity")
        identities.add(json.dumps(stored["identity"]["runtime_versions"], sort_keys=True))
        d = completed_result(directory, stored["identity"])
        point = d["pressure_points"][0]
        # Trace paths archived inside transferred results are resolved by basename.
        attempt = directory / Path(stored["result_path"]).parent
        matches = list(attempt.rglob(Path(point["transition_log_path"]).name))
        if len(matches) != 1: raise ValueError("Expected one transition log per isolated job")
        g = config["gcmc"]
        from ase.io import read
        host = read(root / config["framework"]["path"])
        factor = 1000 / host.get_masses().sum()
        trace = _reconstruct_gcmc_samples(_read_gcmc_transition_log(matches[0]), total_steps=g["equilibration_steps"]+g["production_steps"])
        values = np.array([r["uptake"] for r in trace[g["equilibration_steps"]:]])*factor
        if len(values) != g["production_steps"]: raise ValueError("Incomplete production history")
        screen = assess(values)
        if not np.isclose(values.mean(), point["loading_mol_per_kg"], rtol=1e-8, atol=1e-8):
            raise ValueError("Saved mean disagrees with reconstructed production trace")
        seeds.append({"model": job["model"], "pressure_bar": job["pressure_bar"], "seed": job["seed"],
            "loading_mmol_g": float(values.mean()), "chain_screen": screen["status"],
            "effective_samples": screen["effective_sample_count_estimate"], "drift_mmol_g": screen["half_difference_mmol_g"]})
    if len(identities) > 1: raise ValueError("Mixed runtime environments in one confirmation campaign")
    for model in ("pretrained", "finetuned"):
        for p in config["pressures_bar"]:
            selected = [r for r in seeds if r["model"] == model and r["pressure_bar"] == p]
            if not selected: continue
            stats = seed_interval([r["loading_mmol_g"] for r in selected])
            tolerance = max(.02, .20*abs(stats["mean"]))
            complete = len(selected) == len(config["seeds"])
            passed = complete and stats["ci95_half_width"] is not None and stats["ci95_half_width"] <= tolerance and all(r["chain_screen"] == "passes_operational_screen" for r in selected)
            records.append({"model": model, "pressure_bar": p, **stats, "complete_seed_group": complete,
                "passes_precision_and_chain_screens": passed, "absolute_tolerance_mmol_g": tolerance})
    reference_metrics = []
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for ref in config["references"]:
        points = np.loadtxt(root / ref["path"], delimiter=",")
        x, y = points[:,0]/1e5, points[:,1]
        ax.plot(x, y, "--", label=ref["label"])
        for model in ("pretrained", "finetuned"):
            selected = sorted([r for r in records if r["model"] == model and r["complete_seed_group"] and x.min() <= r["pressure_bar"] <= x.max()], key=lambda r:r["pressure_bar"])
            if len(selected) != len(config["pressures_bar"]): continue
            errors = error_metrics([r["mean"] for r in selected], np.interp([r["pressure_bar"] for r in selected], x, y))
            reference_metrics.append({"model": model, "reference": ref["label"], **errors,
                                      "interpretation": "descriptive classical-reference agreement; sampling flags apply"})
    for model in ("pretrained", "finetuned"):
        selected = [r for r in records if r["model"] == model]
        if selected:
            ax.errorbar([r["pressure_bar"] for r in selected], [r["mean"] for r in selected],
                yerr=[r["ci95_half_width"] if r["ci95_half_width"] is not None else np.nan for r in selected], fmt="o-", capsize=4, label=model)
    ax.set(xscale="log", xlim=(min(config["pressures_bar"])*.7, max(config["pressures_bar"])*1.3), xlabel="Pressure (bar)", ylabel="Absolute loading (mmol/g)",
           title=f"{config['material']} / CO2, 298 K — {len(seeds)}/{len(jobs)} jobs\nSeed intervals and chain screens do not prove equilibrium")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    for ext in ("png", "svg"): fig.savefig(output / f"isotherms.{ext}", dpi=180)
    plt.close(fig)
    write_csv(output / "seed_results.csv", seeds)
    write_csv(output / "point_summary.csv", records)
    if reference_metrics: write_csv(output / "reference_metrics.csv", reference_metrics)
    all_pass = len(seeds) == len(jobs) and all(r["passes_precision_and_chain_screens"] for r in records)
    atomic_json(output / "assessment.json", {"completed_jobs": len(seeds), "expected_jobs": len(jobs),
        "status": "technical_smoke_only" if config.get("technical_smoke") else ("operational_screens_passed" if all_pass else "incomplete_or_sampling_unresolved"),
        "equilibrium_established": False, "points": records, "reference_metrics": reference_metrics,
        "chain_screen": "Existing autocorrelation/batch screen: at least 20 effective samples and 8 batches; within-chain 15%/0.02 mmol/g tolerances. Independent-seed interval uses 20%/0.02 mmol/g."})
    print(f"Report: {output}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "check", "run", "worker", "report", "status", "smoke"])
    p.add_argument("--directory", type=Path, default=DEFAULT)
    p.add_argument("--system", choices=["mh", "polar", "mh_smoke", "polar_smoke"])
    p.add_argument("--max-jobs", type=int, default=1)
    p.add_argument("--job-index", type=int)
    p.add_argument("--attempt", type=Path)
    args = p.parse_args(argv)
    root = args.directory.resolve()
    if args.max_jobs <= 0: p.error("--max-jobs must be positive")
    if args.action == "prepare": return prepare(root)
    if args.action == "check": return check(root, args.system)
    if not args.system: p.error("--system required")
    if args.action == "smoke": return smoke(root, args.system)
    if args.action == "status": return status(root, args.system)
    if args.action == "report": return report(root, args.system)
    config = json.loads((root / f"{args.system}.json").read_text())
    jobs = job_list(config)
    if args.job_index is not None and not 0 <= args.job_index < len(jobs): p.error("--job-index out of range")
    if args.action == "run": return run(root, args.system, args.max_jobs, args.job_index)
    if args.job_index is None or args.attempt is None: p.error("worker requires --job-index and --attempt")
    verify_bundle(root); validate(config, root)
    # resolve() in the shared adapter is anchored at the extracted bundle root.
    if root != PROJECT:
        raise ValueError("Execute worker from the extracted bundle's own Python code")
    execution_versions()
    job = jobs[args.job_index]
    adapter.worker(config, job["model"], job["seed"], "gcmc", job["pressure_bar"], args.attempt)


if __name__ == "__main__":
    main()
