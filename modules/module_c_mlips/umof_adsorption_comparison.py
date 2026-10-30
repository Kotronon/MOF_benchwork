"""Offline, restartable adsorption comparison of the three uMOF-POLAR exports.

Completed model/seed/pressure jobs are reused. An interrupted job starts again
from its seed in a new attempt directory; MLIP-MC 0.1.3 does not save RNG state.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

from .umof_finetuning import sha256_file
from .umof_workflow import atomic_json, campaign_lock, read_json, write_csv

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT / "input_json_files/umof_polar_mgmof74_co2_adsorption.json"


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT / path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_config(path):
    config = read_json(path)
    if config["schema_version"] != 1 or config["component"] != "CO2":
        raise ValueError("This comparison expects schema 1 and rigid CO2")
    if config["temperature_K"] <= 0 or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Invalid temperature or duplicate seeds")
    if not config["seeds"] or any(type(s) is not int or s <= 0 for s in config["seeds"]):
        raise ValueError("Positive integer seeds required")
    p = np.asarray(config["pressures_bar"], dtype=float)
    if not len(p) or not np.isfinite(p).all() or np.any(p <= 0) or len(set(p)) != len(p):
        raise ValueError("Distinct, positive finite pressures required")
    for key in ("trials", "block_size"):
        if type(config["widom"][key]) is not int or config["widom"][key] <= 0:
            raise ValueError(f"Positive Widom {key} required")
    for key in ("equilibration_steps", "production_steps"):
        if type(config["gcmc"][key]) is not int or config["gcmc"][key] < (key == "production_steps"):
            raise ValueError(f"Invalid GCMC {key}")
    if config["dtype"] not in {"float32", "float64"}:
        raise ValueError("Unsupported dtype")
    if not 0 < config["overlap_covalent_radius_scale"] <= 1:
        raise ValueError("Invalid overlap radius scale")
    if sorted(m["label"] for m in config["models"]) != ["baseline", "stage1", "stage2"]:
        raise ValueError("Exactly baseline, stage1 and stage2 required")
    for row in [config["framework"], config["adsorbate"], config["reference"], *config["models"]]:
        path = resolve(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Input hash mismatch: {path}")
    return config


def clean_atoms(atoms):
    from ase import Atoms
    return Atoms(numbers=atoms.numbers, positions=atoms.positions,
                 cell=atoms.cell, pbc=atoms.pbc)


def prepare(config_path=DEFAULT_CONFIG):
    """Freeze one host, one rigid guest, the experimental reference and exports."""
    from ase import Atoms
    from ase.io import read, write
    from ase.neighborlist import natural_cutoffs, neighbor_list
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    config_path = Path(config_path)
    if config_path.exists():
        return load_config(config_path)
    assets = PROJECT / "data/module_c/umof_adsorption_comparison"
    assets.mkdir(parents=True, exist_ok=True)
    source = PROJECT / "external/umof_data/MgMOF-74-adsorbates-train.extxyz"
    # Frame 0 has a close guest-host contact merged by the legacy connectivity
    # heuristic. Frame 1 has an unambiguous CH4 component and intact host.
    atoms = read(source, index=1)
    i, j = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=1.15), self_interaction=False)
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(atoms), len(atoms))).tocsr()
    n, labels = connected_components(graph, directed=False)
    hosts = [np.flatnonzero(labels == k) for k in range(n) if 12 in atoms.numbers[labels == k]]
    if len(hosts) != 1:
        raise ValueError("Expected one connected Mg-containing framework")
    indices = hosts[0]
    host = clean_atoms(atoms[indices])
    removed = atoms[[i for i in range(len(atoms)) if i not in set(indices)]]
    if Counter(host.get_chemical_symbols()) != Counter({"Mg": 18, "C": 72, "H": 18, "O": 54}) or removed.get_chemical_formula() != "CH4":
        raise ValueError("Framework extraction disagrees with Mg18C72H18O54 + CH4")
    host = host.repeat((1, 1, 2))
    guest = Atoms("OCO", positions=[[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]])
    write(assets / "mgmof74_framework.extxyz", host, format="extxyz")
    write(assets / "co2.extxyz", guest, format="extxyz")
    ref = PROJECT / "isodb-library/Library/10.1039c5dt03137k/10.1039c5dt03137k.Isotherm9.json"
    reference = read_json(ref)
    if (reference["adsorbent"]["name"] != "Mg-MOF-74" or reference["category"] != "exp"
            or reference["temperature"] != 298 or reference["pressureUnits"] != "bar"
            or reference["adsorptionUnits"] != "mmol/g" or len(reference["adsorbates"]) != 1
            or reference["adsorbates"][0]["name"] != "Carbon Dioxide"):
        raise ValueError("Unexpected experimental reference")
    atomic_json(assets / "experimental_source.json", reference)
    root = PROJECT / "outputs/umof_polar_24h_float32"
    model_paths = [("baseline", root / "assets/uMOF-POLAR.model", "r2scan+d4"),
        ("stage1", root / "025pct/seed-12345/models/umof_polar_adsorption_finetuned.model", "Default"),
        ("stage2", root / "025pct/seed-12345/models/umof_polar_adsorption_finetuned_stagetwo.model", "Default")]
    def asset(path):
        return {"path": str(path.relative_to(PROJECT)), "sha256": sha256_file(path)}
    config = {"schema_version": 1, "material": "Mg-MOF-74", "component": "CO2",
        "temperature_K": 298.0, "device": "cuda", "dtype": "float32",
        "seeds": [12345], "gpu_ids": [0, 2, 3], "pressures_bar": [.0175, .1015, .4655, .9905],
        "widom": {"trials": 2000, "block_size": 200},
        "gcmc": {"equilibration_steps": 1000, "production_steps": 2000,
                 "checkpoint_interval": 500},
        "overlap_covalent_radius_scale": .8,
        "framework": {**asset(assets / "mgmof74_framework.extxyz"),
            "source": str(source.relative_to(PROJECT)), "source_sha256": sha256_file(source),
            "source_index": 1, "source_framework_indices": indices.tolist(),
            "unit_cells": [1, 1, 2], "atoms": len(host),
            "scope": "rigid framework extracted from released adsorption geometry; no relaxation"},
        "adsorbate": {**asset(assets / "co2.extxyz"), "bond_length_A": 1.16,
                      "gas_reference": "same-model single rigid molecule in vacuum, nonperiodic 30 A cell"},
        "models": [{"label": label, **asset(path), "head": head} for label, path, head in model_paths],
        "reference": {**asset(assets / "experimental_source.json"), "doi": reference["DOI"],
            "url": "https://doi.org/10.1039/C5DT03137K", "article_source": reference["articleSource"],
            "reference_class": "experiment", "reported_loading_basis": "unspecified",
            "original_path": str(ref.relative_to(PROJECT)), "original_sha256": sha256_file(ref),
            "note": "Reported mmol/g equals mol/kg numerically. Absolute/excess basis is unspecified; comparison is provisional."},
        "output_directory": "outputs/umof_polar_adsorption_comparison"}
    atomic_json(config_path, config)
    return config


def overlap_radii(scale):
    from ase.data import covalent_radii
    # Gross clashes only. Standard vdW-0.35 radii exclude close Mg-O adsorption
    # contacts; use the same explicitly archived hard wall for every model.
    return float(scale) * np.asarray(covalent_radii, dtype=float)


def log_acceptance(delta_energy, beta, prefactor):
    """Metropolis probability with the entire prefactor in log space."""
    if prefactor <= 0 or not np.isfinite([delta_energy, beta, prefactor]).all():
        raise ValueError("Invalid Metropolis inputs")
    return min(0.0, math.log(prefactor) - beta * delta_energy)


def engine_classes(scale):
    from mlip_mc.src.widom import MLP_Widom
    from mlip_mc.src.gcmc import MLP_GCMC

    class PairedWidom(MLP_Widom):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.vdw = overlap_radii(scale)

    class StableGCMC(MLP_GCMC):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.vdw = overlap_radii(scale)

        def _accept(self, kind, delta_energy, prefactor):
            log_p = log_acceptance(delta_energy, self.beta, prefactor)
            accepted = math.log(max(float(np.random.rand()), np.finfo(float).tiny)) < log_p
            if accepted:
                self.moves[kind]["accepted"] += 1
            return accepted

        def _insertion_acceptance(self, e_trial, e):
            return self._accept("insertion", e_trial-e, self.V*self.beta*self.fugacity/self.Z_ads)

        def _deletion_acceptance(self, e_trial, e):
            return self._accept("deletion", e_trial-e, (self.Z_ads+1)/(self.V*self.beta*self.fugacity))

    return PairedWidom, StableGCMC


def consistent_calculator(base, guest):
    """Use one same-model vacuum gas reference for every rigid-gas query."""
    from ase.calculators.calculator import Calculator, all_changes
    from .umof_profiles import polar_conditioning
    gas = clean_atoms(guest)
    gas.set_cell([30, 30, 30]); gas.set_pbc(False); gas.center()
    polar_conditioning(gas)
    gas.calc = base
    gas_energy = float(gas.get_potential_energy())
    if not math.isfinite(gas_energy):
        raise ValueError("Nonfinite vacuum gas reference")
    reference_distances = guest.get_all_distances()

    class RigidGasReference(Calculator):
        implemented_properties = ["energy"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            if len(atoms) == len(guest) and np.array_equal(atoms.numbers, guest.numbers):
                if not np.allclose(atoms.get_all_distances(mic=True), reference_distances, rtol=0, atol=1e-5):
                    raise ValueError("Gas query violates the rigid CO2 geometry")
                energy = gas_energy
            else:
                current = clean_atoms(atoms)
                polar_conditioning(current)
                current.calc = base
                energy = float(current.get_potential_energy())
            if not math.isfinite(energy):
                raise ValueError("Nonfinite potential energy")
            self.results = {"energy": energy}

    calc = RigidGasReference()
    calc.gas_energy_eV = gas_energy
    return calc


def widom_pose_hash(path):
    """Hash accepted trial indices and geometries, excluding model energies."""
    digest, trials = hashlib.sha256(), []
    with Path(path).open("rb") as handle:
        size = struct.calcsize("iddi")
        while payload := handle.read(size):
            if len(payload) != size:
                raise ValueError("Truncated Widom header")
            trial, _, _, n = struct.unpack("iddi", payload)
            geometry = handle.read(n*4 + n*3*8 + 9*8)
            if len(geometry) != n*4 + n*3*8 + 9*8:
                raise ValueError("Truncated Widom geometry")
            digest.update(struct.pack("ii", trial, n)); digest.update(geometry)
            trials.append(trial)
    return {"sha256": digest.hexdigest(), "valid_trial_count": len(trials)}


def production_diagnostics(transition_log, equilibration, production):
    from engines.mlip_mc_runner import _read_gcmc_transition_log, _reconstruct_gcmc_samples
    rows = _reconstruct_gcmc_samples(_read_gcmc_transition_log(Path(transition_log)),
                                    total_steps=equilibration+production)[equilibration:]
    uptake = np.asarray([r["uptake"] for r in rows], dtype=float)
    blocks = [float(a.mean()) for a in np.array_split(uptake, min(20, len(uptake))) if len(a)]
    return {"production_samples": len(uptake), "production_block_means_molecules": blocks,
        "block_sem_molecules": float(np.std(blocks, ddof=1)/np.sqrt(len(blocks))) if len(blocks)>1 else None,
        "first_half_mean_molecules": float(uptake[:max(1, len(uptake)//2)].mean()),
        "second_half_mean_molecules": float(uptake[len(uptake)//2:].mean()),
        "convergence": "unestablished; inspect block means, seed spread and equilibration"}


def job_directory(output, model, seed, phase, pressure=None):
    suffix = phase if pressure is None else f"gcmc-{pressure:g}bar"
    return output / "jobs" / model / f"seed-{seed}" / suffix


def worker(config, model_label, seed, phase, pressure, directory):
    from ase.io import read
    from .umof_profiles import calculator
    from engines.mlip_mc_runner import run_mlip_mc_widom, run_mlip_mc_gcmc
    from analysis.mlip_widom import read_widom_attempt_trace, BOLTZMANN_EV_PER_K
    host, guest = (read(resolve(config[key]["path"])) for key in ("framework", "adsorbate"))
    row = next(m for m in config["models"] if m["label"] == model_label)
    base = calculator(resolve(row["path"]), model="polar", device=config["device"],
                      head=row["head"], dtype=config["dtype"])
    if base.head != row["head"]:
        raise ValueError("Calculator silently selected an unexpected head")
    # Supercell dimensions must accommodate the model's local cutoff.
    heights = 1/np.linalg.norm(np.linalg.inv(host.cell.array), axis=0)
    if heights.min() < 2*base.r_max:
        raise ValueError(f"Cell heights {heights} below twice local cutoff {base.r_max}")
    calc = consistent_calculator(base, guest)
    widom_class, gcmc_class = engine_classes(config["overlap_covalent_radius_scale"])
    if phase == "widom":
        w = config["widom"]
        result = run_mlip_mc_widom(calc, host, guest, temperature_K=config["temperature_K"],
            trial_count=w["trials"], seed=seed, device=config["device"], output_directory=directory,
            block_size=w["block_size"], engine_class=widom_class)
        result["paired_poses"] = widom_pose_hash(result["trial_log_path"])
        values = read_widom_attempt_trace(result["trial_log_path"], attempts=w["trials"])
        finite = np.asarray([e for e in values if e is not None])
        if not len(finite):
            raise RuntimeError("No non-overlapping Widom insertions; increase trials")
        log_weights = -finite/(BOLTZMANN_EV_PER_K*config["temperature_K"])
        weights = np.exp(log_weights-log_weights.max())
        result["boltzmann_weight_ess"] = float(weights.sum()**2/np.square(weights).sum())
        result["insertion_energy_range_eV"] = [float(finite.min()), float(finite.max())]
    else:
        g = config["gcmc"]
        result = run_mlip_mc_gcmc(calc, host, guest, component=config["component"],
            temperature_K=config["temperature_K"], pressures_bar=[pressure],
            equilibration_steps=g["equilibration_steps"], production_steps=g["production_steps"],
            seed=seed, device=config["device"], output_directory=directory,
            checkpoint_interval=g["checkpoint_interval"], engine_class=gcmc_class)
        point = result["pressure_points"][0]
        factor = 1000/host.get_masses().sum()
        point["loading_mol_per_kg"] = float(point["mean_uptake_molecules_per_unit_cell"]*factor)
        point["loading_std_mol_per_kg"] = float(point["uptake_std_molecules_per_unit_cell"]*factor)
        point["production_diagnostics"] = production_diagnostics(point["transition_log_path"],
            g["equilibration_steps"], g["production_steps"])
    result.update(model=model_label, material=config["material"], component=config["component"],
        gas_reference_eV=calc.gas_energy_eV, config_sha256=fingerprint(config),
        sampling_status="pilot; convergence unestablished")
    atomic_json(directory / "result.json", result)
    return result


def jobs(config, output):
    # Finish all Widom comparisons before starting any GCMC pressure point.
    for phase in ("widom", "gcmc"):
        for seed in config["seeds"]:
            for pressure in ([None] if phase == "widom" else config["pressures_bar"]):
                for model in config["models"]:
                    yield model["label"], seed, phase, pressure, job_directory(output, model["label"], seed, phase, pressure)


def get_completed(directory, config):
    path = directory / "completed.json"
    if not path.exists():
        return None
    marker = read_json(path)
    if marker["config_sha256"] != fingerprint(config):
        raise ValueError(f"Changed settings for completed job: {directory}; use a new output directory")
    result_path = directory / marker["result_path"]
    if sha256_file(result_path) != marker["result_sha256"]:
        raise ValueError(f"Completed result changed: {result_path}")
    return read_json(result_path)


def run(config, config_path, output):
    output.mkdir(parents=True, exist_ok=True)
    with campaign_lock(output):
        snapshot = output / "comparison_config.json"
        if snapshot.exists() and read_json(snapshot) != config:
            raise ValueError("Existing output has different settings; use a new --output")
        atomic_json(snapshot, config)
        versions = {p: importlib.metadata.version(p) for p in
                    ("torch", "mace-torch", "mlip-mc", "graph_longrange", "ase")}
        provenance = {"packages": versions, "adapter_sha256": sha256_file(__file__),
            "engine_adapter_sha256": sha256_file(PROJECT / "engines/mlip_mc_runner.py"),
            "gpu_ids": config["gpu_ids"], "config_sha256": fingerprint(config)}
        provenance_path = output / "implementation.json"
        if provenance_path.exists() and read_json(provenance_path) != provenance:
            raise ValueError("Implementation or package versions changed; use a new --output")
        atomic_json(provenance_path, provenance)
        slots = config["gpu_ids"] if config["device"].startswith("cuda") else [None]
        all_jobs = list(jobs(config, output))
        for phase in ("widom", "gcmc"):
            pending = []
            for job in all_jobs:
                if job[2] != phase:
                    continue
                if get_completed(job[-1], config) is None:
                    pending.append(job)
                else:
                    print(f"Reuse {job[:4]}", flush=True)
            for offset in range(0, len(pending), len(slots)):
                active = []
                try:
                    for gpu, (model, seed, _, pressure, directory) in zip(slots, pending[offset:offset+len(slots)]):
                        identity = {"model": model, "seed": seed, "phase": phase, "pressure_bar": pressure, "gpu": gpu}
                        directory.mkdir(parents=True, exist_ok=True)
                        attempt = directory / f"attempt-{len(list(directory.glob('attempt-*')))+1:03d}"
                        attempt.mkdir()
                        command = [sys.executable, "-m", "modules.module_c_mlips.umof_adsorption_comparison", "worker", "--config", str(config_path),
                            "--model", model, "--seed", str(seed), "--phase", phase, "--output", str(attempt)]
                        if pressure is not None:
                            command += ["--pressure", str(pressure)]
                        env = dict(os.environ)
                        if gpu is not None:
                            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                        with (attempt / "worker.log").open("w") as handle:
                            process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                        active.append({"process": process, "directory": directory, "attempt": attempt, "identity": identity})
                        print(f"Start {identity}; log: {attempt/'worker.log'}", flush=True)
                    atomic_json(output / "status.json", {"status": "running", "active": [a["identity"] for a in active]})
                    while active:
                        for record in list(active):
                            process, attempt, directory = (record[k] for k in ("process", "attempt", "directory"))
                            code = process.poll()
                            if code is None:
                                continue
                            if code:
                                raise RuntimeError(f"Job exited {code}; inspect {attempt/'worker.log'}")
                            result_path = attempt / "result.json"
                            atomic_json(directory / "completed.json", {"config_sha256": fingerprint(config),
                                "result_path": str(result_path.relative_to(directory)), "result_sha256": sha256_file(result_path)})
                            active.remove(record)
                            print(f"Done {record['identity']}", flush=True)
                            report(config, output)
                        if active:
                            time.sleep(.5)
                except BaseException as exc:
                    atomic_json(output / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error": str(exc)})
                    raise
                finally:
                    for record in active:
                        process = record["process"]
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill(); process.wait()
            if phase == "widom":
                for seed in config["seeds"]:
                    hashes = [get_completed(job_directory(output, m["label"], seed, "widom"), config)["paired_poses"] for m in config["models"]]
                    if any(h != hashes[0] for h in hashes):
                        raise RuntimeError("Widom trial geometries differ between models")
        atomic_json(output / "status.json", {"status": "completed", "completed_jobs": len(all_jobs)})
        print(f"Completed. Report: {output/'report/README.md'}", flush=True)


def report(config, output):
    from analysis.module_c_isotherm_report import compare_reference
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = output / "report"
    destination.mkdir(parents=True, exist_ok=True)
    reference = read_json(resolve(config["reference"]["path"]))
    points = [{"pressure_bar": float(r["pressure"]), "loading_mol_per_kg": float(r["total_adsorption"])}
              for r in reference["isotherm_data"] if r["pressure"] > 0 and r["total_adsorption"] >= 0]
    points.sort(key=lambda r: r["pressure_bar"])
    groups, widom, completed_count = defaultdict(list), [], 0
    for model, seed, phase, pressure, directory in jobs(config, output):
        result = get_completed(directory, config)
        if not result:
            continue
        completed_count += 1
        if phase == "widom":
            stats = result["corrected_statistics"]
            widom.append({"model": model, "seed": seed,
                "henry_mol_kg_Pa": stats["henry_coefficient_mol_kg_pa"],
                "henry_mmol_g_bar": stats["henry_coefficient_mmol_g_bar"],
                "qst_kJ_mol": stats["isosteric_heat_zero_loading_kj_mol"],
                "boltzmann_weight_ess": result["boltzmann_weight_ess"],
                "valid_fraction": result["valid_fraction"], "gas_reference_eV": result["gas_reference_eV"]})
        else:
            p = result["pressure_points"][0]
            groups[model, pressure].append({"seed": seed, "loading": p["loading_mol_per_kg"],
                "production_diagnostics": p["production_diagnostics"], "move_statistics": p["move_statistics"]})
    rows = []
    for (model, pressure), replicas in sorted(groups.items()):
        values = [r["loading"] for r in replicas]
        rows.append({"model": model, "pressure_bar": pressure, "replicates": len(values),
            "mean_loading_mol_per_kg": float(np.mean(values)),
            "between_seed_sd_mol_per_kg": float(np.std(values, ddof=1)) if len(values)>1 else None})
    # Rank only fully sampled, common pressures with all models and seeds.
    labels = [m["label"] for m in config["models"]]
    common = [p for p in config["pressures_bar"] if all(len(groups[m, p]) == len(config["seeds"]) for m in labels)]
    matched = [r for r in rows if r["pressure_bar"] in common]
    deviations, metrics = compare_reference(matched, {"points": points}) if matched else ([], [])
    for name, records in (("widom_seeds.csv", widom), ("isotherm_summary.csv", rows),
                          ("reference_deviations.csv", deviations), ("reference_metrics.csv", metrics)):
        write_csv(destination / name, records)
    atomic_json(destination / "comparison.json", {"completed_jobs": completed_count,
        "total_jobs": sum(1 for _ in jobs(config, output)), "widom": widom, "gcmc": rows,
        "production_diagnostics": [{"model": m, "pressure_bar": p, "replicas": r} for (m, p), r in groups.items() if r],
        "reference": config["reference"], "reference_metrics": metrics,
        "comparison_scope": "provisional agreement with digitized experimental reported uptake; source absolute/excess basis unspecified",
        "convergence": "unestablished; error bars show between-seed sample SD"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for j, label in enumerate(labels):
        selected = [r for r in rows if r["model"] == label]
        if selected:
            for ax in axes:
                ax.errorbar([r["pressure_bar"] for r in selected], [r["mean_loading_mol_per_kg"] for r in selected],
                    yerr=[r["between_seed_sd_mol_per_kg"] or 0 for r in selected], fmt="o-", capsize=3, label=label)
    for ax in axes:
        ax.plot([p["pressure_bar"] for p in points], [p["loading_mol_per_kg"] for p in points], "k.--", label="Experiment (reported uptake)")
        ax.set(xlabel="Pressure (bar)", ylabel="Loading (mol/kg = mmol/g)", ylim=(0, None)); ax.grid(alpha=.2)
    axes[1].set_xscale("log"); axes[0].legend(fontsize=8)
    uncertainty_label = "one seed; no between-seed uncertainty" if len(config["seeds"]) == 1 else "error bars: seed SD"
    fig.suptitle(f"Mg-MOF-74 / CO2, {config['temperature_K']:g} K — {completed_count}/{sum(1 for _ in jobs(config, output))} jobs\nPilot; {uncertainty_label}; convergence unestablished")
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"isotherm_comparison.{suffix}", dpi=160)
    plt.close(fig)
    lines = ["# uMOF-POLAR adsorption comparison", "", f"Completed jobs: {completed_count}/{sum(1 for _ in jobs(config, output))}.", "",
        "Baseline, Stage 1 and Stage 2; same rigid Mg-MOF-74 host, rigid CO2 (1.16 A), 298 K, pressure grid and seeds.",
        "Widom accepted trial indices and geometries are hashed and required to agree between models. All attempted insertions enter the estimator; gross overlaps contribute zero.",
        "Each model uses its own nonperiodic vacuum CO2 reference, held constant across poses in both Widom and GCMC. Gas and host are neutral with spin multiplicity 1.", "",
        "The shared hard-overlap rule is 0.8 times ASE covalent radii (pairwise sums), rather than the MLIP-MC vdW-0.35 rule that excludes close Mg-O contacts. This is an explicit excluded-volume approximation. Inspect Widom energy tails and ESS for extrapolation or rare-event domination.", "",
        "GCMC uses Peng-Robinson fugacity. Insertion/deletion acceptance includes the full fugacity/volume/population prefactor in log space. Means include every production MC step, including rejected moves. Production block means, first/second-half means and acceptance counts are archived in comparison.json.", "",
        "Completed jobs are hash-checked and reused. Interrupted jobs restart from the original seed in a new attempt directory; partial attempts are retained. This avoids relying on missing RNG state in MLIP-MC's restart files.", "",
        "## Reference", "", "Digitized experimental bare Mg-MOF-74 / CO2 at 298 K, Figure 4, [Bernini et al.](https://doi.org/10.1039/C5DT03137K), local NIST ISODB mirror. Source snapshot and hashes are archived. Reported mmol/g equals mol/kg numerically. The source does not specify absolute versus excess uptake; comparison remains provisional. No uptake rescaling or reference fitting is performed.", "",
        "## Results", "", "Plots: isotherm_comparison.png and .svg. CSV/JSON retain Henry constants, zero-loading heats, per-seed uptake, diagnostics and experimental deviations. Experimental metrics use only pressures completed for all three models and all seeds, with no extrapolation. With one seed there are no between-seed error bars; inspect within-run block means and drift. For multiple seeds error bars are sample SD, not confidence intervals. Completion does not establish equilibration or convergence.", ""]
    if metrics:
        lines += ["| Model | Matched pressures | Uptake RMSE (mol/kg) |", "|---|---:|---:|"]
        lines += [f"| {r['model']} | {r['matched_pressure_count']} | {r['rmse_mol_per_kg']:.5g} |" for r in metrics]
    (destination / "README.md").write_text("\n".join(lines)+"\n")


def check(config):
    import torch
    from ase.io import read
    packages = {p: importlib.metadata.version(p) for p in
                ("torch", "mace-torch", "mlip-mc", "graph_longrange", "ase", "scipy", "matplotlib")}
    if packages["mlip-mc"] != "0.1.3":
        raise ValueError("This adapter requires the inspected MLIP-MC 0.1.3 API")
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate the MOF_sim environment or use --device cpu")
    if config["device"].startswith("cuda") and not config["gpu_ids"]:
        raise ValueError("At least one GPU ID required")
    host = read(resolve(config["framework"]["path"]))
    for row in config["models"]:
        model = torch.load(resolve(row["path"]), map_location="cpu", weights_only=False)
        if type(model).__name__ != "PolarMACE" or row["head"] not in model.heads:
            raise ValueError(f"Unexpected model/head for {row['label']}")
        if str(next(model.parameters()).dtype) != "torch."+config["dtype"]:
            raise ValueError("Archived model dtype differs from config")
        heights = 1/np.linalg.norm(np.linalg.inv(host.cell.array), axis=0)
        if heights.min() < 2*float(model.r_max):
            raise ValueError("Framework supercell too small")
    from engines.mlip_mc_runner import _load_preos_factory
    from ase.units import bar
    fugacity = _load_preos_factory()(config["component"]).calculate_fugacity(config["temperature_K"], float(config["pressures_bar"][0])*bar)
    if not math.isfinite(fugacity) or fugacity <= 0:
        raise ValueError("Invalid Peng-Robinson fugacity")
    return {"status": "ready", "framework_atoms": len(host), "packages": packages,
            "gpu_ids": config["gpu_ids"], "seeds": config["seeds"],
            "steps_per_model": config["widom"]["trials"]+len(config["pressures_bar"])*(config["gcmc"]["equilibration_steps"]+config["gcmc"]["production_steps"]),
            "config_sha256": fingerprint(config), "jobs": sum(1 for _ in jobs(config, Path(".")))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "check", "run", "smoke", "status", "report", "worker"], nargs="?", default="run")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path)
    p.add_argument("--device")
    p.add_argument("--gpus", help="Comma-separated physical CUDA GPU IDs, e.g. 0,2,3")
    p.add_argument("--model", choices=["baseline", "stage1", "stage2"])
    p.add_argument("--seed", type=int)
    p.add_argument("--phase", choices=["widom", "gcmc"])
    p.add_argument("--pressure", type=float)
    args = p.parse_args(argv)
    if args.action == "prepare":
        prepare(args.config); print(f"Prepared {args.config}"); return 0
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.gpus:
        config["gpu_ids"] = [int(v) for v in args.gpus.split(",")]
    if not config.get("gpu_ids") or len(set(config["gpu_ids"])) != len(config["gpu_ids"]) or any(g < 0 for g in config["gpu_ids"]):
        p.error("Distinct nonnegative GPU IDs required")
    output = (args.output or resolve(config["output_directory"])).resolve()
    if args.action == "worker":
        if args.model is None or args.seed is None or args.phase is None or (args.phase == "gcmc" and args.pressure is None):
            p.error("worker requires model, seed, phase and GCMC pressure")
        worker(config, args.model, args.seed, args.phase, args.pressure, output)
        return 0
    if args.action == "smoke":
        config = copy.deepcopy(config)
        config["seeds"] = [config["seeds"][0]]
        config["pressures_bar"] = [.1015]
        config["widom"] = {"trials": 64, "block_size": 16}
        config["gcmc"] = {"equilibration_steps": 16, "production_steps": 64, "checkpoint_interval": 40}
        output = (args.output or resolve(config["output_directory"]+"_smoke")).resolve()
    if args.action in {"run", "smoke", "check"}:
        print(json.dumps(check(config), indent=2), flush=True)
    if args.action in {"run", "smoke"}:
        output.mkdir(parents=True, exist_ok=True)
        effective = output / "effective_config.json"
        if effective.exists() and read_json(effective) != config:
            raise ValueError("Changed config for existing output; choose a new --output")
        atomic_json(effective, config)
        run(config, effective, output)
    elif args.action == "report":
        report(config, output)
    elif args.action == "status":
        count = sum(get_completed(d, config) is not None for *_, d in jobs(config, output))
        print(json.dumps({"completed": count, "total": sum(1 for _ in jobs(config, output)),
                          "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
