"""Offline, manifest-checked uMOF fine-tuning lifecycle.

Only ``setup --download-missing`` may access the network. Production training
and smoke data live in separate campaigns. Paths in manifests are relocatable.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import csv
import fcntl
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

import numpy as np

from . import umof_finetuning as data
from .umof_profiles import (profile, campaign_profile, calculator, POLAR_REPLAY_FILENAME,
    POLAR_REPLAY_SHA256, POLAR_REPLAY_URL, reduced_composition, adsorption_framework_compositions,
    polar_conditioning, prepare_polar_datasets)

REPLAY_URL = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/matpes-r2scan-replay-data.extxyz"
REPLAY_SOURCE = Path("external/umof_replay/matpes-r2scan-replay-data.extxyz")
REPLAY_SHA256 = "1963d8ab3a902dc1867709f0a42f13581271645ecb30aea0e6b8288456c59e20"
NAME = "umof_mh_adsorption_finetuned"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def campaign_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another writer owns {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def environment():
    packages = {p: importlib.metadata.version(p) for p in
                ("torch", "mace-torch", "ase", "numpy", "scipy", "matplotlib")}
    try:
        packages["graph_longrange"] = importlib.metadata.version("graph_longrange")
    except importlib.metadata.PackageNotFoundError:
        pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages,
            "installed_packages": {dist.metadata["Name"]: dist.version
                                   for dist in importlib.metadata.distributions() if dist.metadata["Name"]},
            "threads": {p: os.environ.get(p, "1") for p in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def model_info(checkpoint):
    import torch
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {"sha256": data.sha256_file(checkpoint), "heads": list(model.heads),
            "atomic_numbers": [int(z) for z in model.atomic_numbers],
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "class": type(model).__name__}


def setup(checkpoint, replay_source=None, download=False, *,
          data_directory=data.DEFAULT_DATA_DIRECTORY, output_directory=None, model="mh"):
    selected_profile = profile(model)
    if model == "polar":
        from .dependencies import ensure_mace_polar_dependencies
        ensure_mace_polar_dependencies()
    checkpoint, directory = Path(checkpoint), Path(data_directory)
    replay_source = Path(replay_source) if replay_source is not None else (
        REPLAY_SOURCE if model == "mh" else directory / POLAR_REPLAY_FILENAME)
    needed = {}
    if not checkpoint.is_file():
        needed["umof_models.zip"] = [("umof_models/" + selected_profile.filename, checkpoint)]
    if not list(directory.glob("*-adsorbates-train.extxyz")) or (model == "polar" and not Path(replay_source).is_file()):
        needed["umof_data.zip"] = None
    for archive_name, members in needed.items():
        archive = directory.parent / archive_name
        if not archive.is_file():
            if not download:
                raise FileNotFoundError(f"Missing {archive}; use setup --download-missing")
            from .model_assets import _figshare_files, UMOF_FIGSHARE_ARTICLE_ID
            files = _figshare_files(UMOF_FIGSHARE_ARTICLE_ID)
            asset = next(f for f in files if f["name"] == archive_name)
            archive.parent.mkdir(parents=True, exist_ok=True)
            temporary = archive.with_suffix(".download")
            try:
                urllib.request.urlretrieve(asset["download_url"], temporary)
                os.replace(temporary, archive)
            finally:
                temporary.unlink(missing_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            if members is None:
                members = [(n, directory / Path(n).name) for n in bundle.namelist()
                           if re.fullmatch(r"[^/]+-adsorbates-(train|valid|test)\.extxyz", Path(n).name)
                           or (model == "polar" and Path(n).name == POLAR_REPLAY_FILENAME)]
            if not members:
                raise ValueError(f"No expected adsorption assets in {archive}")
            for member, destination in members:
                if model == "polar" and Path(member).name == POLAR_REPLAY_FILENAME:
                    destination = replay_source
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    info = model_info(checkpoint)
    selected_profile.validate(info)
    source = Path(replay_source)
    provenance = source.with_suffix(".provenance.json")
    if provenance.exists() and source.exists():
        previous = read_json(provenance)
        if previous["replay_source"]["sha256"] != data.sha256_file(source):
            raise ValueError("Replay source differs from its recorded checksum")
    if not source.is_file():
        if model == "polar":
            raise FileNotFoundError(f"Missing published POLAR replay source: {source}")
        if not download:
            raise FileNotFoundError(f"Missing {source}; use setup --download-missing")
        source.parent.mkdir(parents=True, exist_ok=True)
        temporary = source.with_suffix(".download")
        try:
            urllib.request.urlretrieve(REPLAY_URL, temporary)
            os.replace(temporary, source)
        finally:
            temporary.unlink(missing_ok=True)
    digest = data.sha256_file(source)
    expected = REPLAY_SHA256 if model == "mh" else POLAR_REPLAY_SHA256
    if digest != expected:
        raise ValueError(f"Replay source is not the audited {model} release")
    for split in ("train", "valid", "test"):
        if not list(directory.glob(f"*-adsorbates-{split}.extxyz")):
            raise FileNotFoundError(f"Missing released adsorption {split} files in {directory}")
    result = {"status": "ready", "model": info, "environment": environment(),
              "adsorption_source": {p.name: data.sha256_file(p) for p in sorted(directory.glob("*-adsorbates-*.extxyz"))},
              "replay_source": {"model": model, "url": REPLAY_URL if model == "mh" else POLAR_REPLAY_URL, "sha256": data.sha256_file(source),
                                "label_mapping": {"dft_energy": "REF_energy", "dft_forces": "REF_forces"},
                                "note": "DFT labels only; stored predictor outputs are ignored"}}
    atomic_json(provenance, result)
    if output_directory is not None:
        atomic_json(Path(output_directory) / "offline_assets.json", result)
    return result


def prepare_replay(source, output, *, seed, count=1000, valid_count=100, weight=.25,
                   supported_elements=None, max_atoms=None, model="mh", excluded_compositions=()):
    """Index raw extxyz without loading a gigabyte of ASE frames into memory."""
    from ase.io import read, write
    if count < 1 or valid_count < 1 or not math.isfinite(weight) or weight <= 0:
        raise ValueError("Positive replay counts and weight required")
    source, output = Path(source), Path(output)
    offsets = []
    with source.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            n = int(line)
            if n < 1 or not handle.readline():
                raise ValueError("Malformed replay extxyz")
            for _ in range(n):
                if not handle.readline():
                    raise ValueError("Truncated replay extxyz")
            if max_atoms is None or n <= max_atoms:
                offsets.append((offset, handle.tell() - offset))
    random.Random(seed).shuffle(offsets)
    frames, identities, selected = [], set(), []
    excluded = 0
    with source.open("rb") as handle:
        for offset, length in offsets:
            handle.seek(offset)
            atoms = read(io.StringIO(handle.read(length).decode()), format="extxyz")
            if supported_elements is not None and set(atoms.numbers) - supported_elements:
                continue
            if model == "polar":
                from collections import Counter
                if reduced_composition(Counter(atoms.get_chemical_symbols())) in excluded_compositions:
                    excluded += 1
                    continue
            identity = data.geometry_id(atoms)
            if identity in identities:
                continue
            energy = float(atoms.info["dft_energy"])
            forces = np.asarray(atoms.arrays["dft_forces"])
            if not np.isfinite(energy) or forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
                raise ValueError("Invalid replay DFT labels")
            # Do not accidentally replay the predictor outputs stored as energy/forces.
            current = atoms.copy()
            current.calc = None
            if model == "polar":
                polar_conditioning(current)
            current.info.update(REF_energy=energy, head="pt_head", config_weight=weight,
                                replay_geometry_id=identity, replay_source_offset=offset)
            current.set_array("REF_forces", forces.copy())
            frames.append(current); identities.add(identity); selected.append(offset)
            if len(frames) == count + valid_count:
                break
    if len(frames) != count + valid_count:
        raise ValueError("Insufficient distinct supported replay configurations; no duplicate padding")
    output.mkdir(parents=True, exist_ok=True)
    for split, subset in (("train", frames[:count]), ("valid", frames[count:])):
        write(output / f"{split}.extxyz", subset, format="extxyz")
    result = {"source_url": REPLAY_URL if model == "mh" else POLAR_REPLAY_URL, "source_sha256": data.sha256_file(source),
              "seed": seed, "train_count": count, "valid_count": valid_count, "weight": weight,
              "head": "pt_head", "reference": "MatPES r2SCAN DFT (not source MACE predictions)",
              "source_offsets": selected, "geometry_ids": [a.info["replay_geometry_id"] for a in frames],
              "file_sha256": {s: data.sha256_file(output / f"{s}.extxyz") for s in ("train", "valid")}}
    if model == "polar":
        result.update(model="polar", reference="uMOF r2SCAN-D4 DFT",
            replay_kind="new uMOF retention head; not restored OMol replay",
            excluded_framework_compositions=[list(c) for c in sorted(excluded_compositions)],
            excluded_composition_frames=excluded)
    atomic_json(output / "manifest.json", result)
    return result


def relative(path, root):
    return os.path.relpath(Path(path).resolve(), Path(root).resolve())


def verify_files(root, hashes):
    for path, expected in hashes.items():
        full = Path(root) / path
        if not full.is_file() or data.sha256_file(full) != expected:
            raise ValueError(f"Missing or changed file: {full}")


def protocol(**kwargs):
    return {"foundation_head": "r2scan+d4", "adsorption_head": "Default", "replay_head": "pt_head",
            "dtype": "float64", "batch_size": 1, "start_swa": 40, "patience": 20,
            "real_pt_data_ratio_threshold": 0, "adapter_version": 1,
            "scheduler": "ReduceLROnPlateau", "lr_factor": .8, "scheduler_patience": 50,
            "stress_training": False, **kwargs}


def prepare_campaign(data_directory, checkpoint, output_directory, *, fractions=data.DEFAULT_TRAIN_FRACTIONS,
                     seeds=(12345,), max_epochs=50, learning_rate=1e-4, replay_weight=.25,
                     replay_samples=1000, replay_valid_count=100, device="cpu", gases=None,
                     gas_definitions=None, validation_mofs=None, test_mofs=None,
                     replay_source=None, purpose="production", model="mh"):
    selected_profile = profile(model)
    replay_source = Path(replay_source) if replay_source is not None else (
        REPLAY_SOURCE if model == "mh" else Path(data_directory) / POLAR_REPLAY_FILENAME)
    root = Path(output_directory).resolve()
    gases, definitions = data.validate_gases(gases, gas_definitions)
    if not fractions or len({f"{round(float(f)*100):03d}" for f in fractions}) != len(fractions):
        raise ValueError("Fractions must have distinct percentage directory names")
    if max_epochs < 1 or learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("Invalid epoch limit or learning rate")
    settings = protocol(max_epochs=max_epochs, learning_rate=learning_rate, replay_weight=replay_weight,
                        replay_samples=replay_samples, replay_valid_count=replay_valid_count,
                        device=device, fractions=sorted(fractions), seeds=sorted(seeds),
                        gases=sorted(gases), gas_definitions=definitions,
                        validation_mofs=sorted(data.DEFAULT_VALIDATION_MOFS if validation_mofs is None else validation_mofs),
                        test_mofs=sorted(data.DEFAULT_TEST_MOFS if test_mofs is None else test_mofs), purpose=purpose)
    if model != "mh":
        settings.update(model=model, replay_recipe="umof_r2scan_d4",
            polar_conditioning="source values where present; otherwise charge=0, spin=1 (MACE multiplicity default); not inferred spin labels")
    with campaign_lock(root):
        path = root / "campaign.json"
        if path.exists():
            previous = read_json(path)
            compare_protocol(previous["protocol"], settings)
            if data.sha256_file(checkpoint) != previous["foundation"]["sha256"]:
                raise ValueError("Foundation checkpoint changed")
            if any(m["source_sha256"] != data.sha256_file(replay_source)
                   for m in previous.get("replay", {}).values()):
                raise ValueError("Replay source changed")
            verify_files(root, previous["files"])
            if previous["protocol"]["max_epochs"] != max_epochs:
                previous["protocol"]["max_epochs"] = max_epochs
                atomic_json(path, previous)
            return previous
        if list(root.glob("*/seed-*/status.json")) or list(root.glob("*/seed-*/checkpoints/*.pt")):
            raise ValueError("Refusing to overwrite untracked prior training; choose a new campaign directory")
        info = model_info(checkpoint)
        selected_profile.validate(info)
        if purpose == "production" and data.sha256_file(replay_source) != (REPLAY_SHA256 if model == "mh" else POLAR_REPLAY_SHA256):
            raise ValueError("Production replay must match the audited model-specific release")
        assets = root / "assets"
        assets.mkdir(exist_ok=True)
        shutil.copy2(checkpoint, assets / selected_profile.filename)
        manifests = data.prepare_finetuning_matrix(data_directory, root, fractions=fractions, seeds=seeds,
            validation_mofs=validation_mofs, test_mofs=test_mofs, gases=gases,
            gas_definitions=definitions, supported_elements=set(info["atomic_numbers"]))
        if model == "polar":
            prepare_polar_datasets(manifests)
        excluded_compositions = adsorption_framework_compositions(manifests) if model == "polar" else set()
        replay_manifests = {}
        for seed in sorted(seeds):
            replay_manifests[str(seed)] = prepare_replay(replay_source, assets / f"replay-{seed}",
                seed=seed, count=replay_samples, valid_count=replay_valid_count,
                weight=replay_weight, supported_elements=set(info["atomic_numbers"]),
                model=model, excluded_compositions=excluded_compositions)
        runs, hashes = [], {selected_profile.asset: info["sha256"]}
        for manifest in manifests:
            run = Path(manifest["train_file"]).parent.parent
            for split in ("train", "valid", "test"):
                hashes[relative(manifest[f"{split}_file"], root)] = manifest["file_sha256"][split]
                manifest[f"{split}_file"] = f"{split}.extxyz"
            atomic_json(run / "datasets/dataset_manifest.json", manifest)
            hashes[relative(run / "datasets/dataset_manifest.json", root)] = data.sha256_file(run / "datasets/dataset_manifest.json")
            runs.append({"directory": relative(run, root), "fraction": manifest["training_data_fraction"],
                         "actual_fraction": manifest["actual_training_fraction"],
                         "seed": manifest["seed"], "train_count": manifest["train_configuration_count"]})
        for seed in seeds:
            for name in ("train.extxyz", "valid.extxyz", "manifest.json"):
                rel = f"assets/replay-{seed}/{name}"
                hashes[rel] = data.sha256_file(root / rel)
        campaign = {"schema_version": 2, "protocol": settings, "foundation": info,
                    "files": hashes, "runs": runs, "environment": environment(),
                    "source_directory": str(Path(data_directory).resolve()),
                    "replay": replay_manifests}
        atomic_json(path, campaign)
        atomic_json(root / "environment.json", campaign["environment"])
        (root / "requirements-lock.txt").write_text("".join(
            f"{package}=={version}\n" for package, version in campaign["environment"]["packages"].items()))
        atomic_json(root / "coverage.json", manifests[-1]["coverage"])
        for run in runs:
            command = training_command(root, campaign, run)
            # Rebuild executable/absolute paths at invocation; stored command is portable documentation.
            atomic_json(root / run["directory"] / "training_command.json",
                        {"command": ["python"] + [relative_token(t, root) for t in command[1:]],
                         "working_directory": run["directory"], "paths_relative_to": "campaign root"})
        import shlex
        location = relative(root, Path(__file__).resolve().parents[2])
        (root / "START.txt").write_text(
            "From the transferred repository, activate its Python/MACE environment.\n" +
            "".join(f"bash scripts/run_module_c_umof_finetuning.sh {action} --model {model} "
                    f"--output-directory {shlex.quote(location)}\n"
                    for action in ("status", "train", "resume", "evaluate", "report")))
        return campaign


def relative_token(token, root):
    return relative(token, root) if token.startswith(str(root)) else token


def compare_protocol(old, new):
    a, b = dict(old), dict(new)
    old_epochs, new_epochs = a.pop("max_epochs"), b.pop("max_epochs")
    if a != b or new_epochs < old_epochs:
        raise ValueError("Incompatible campaign protocol; only an increased epoch limit with unchanged schedule is allowed")


def training_command(root, campaign, run, *, epochs=None):
    folder = root / run["directory"]
    p = campaign["protocol"]
    manifest = {f"{s}_file": str(folder / "datasets" / f"{s}.extxyz") for s in ("train", "valid")}
    manifest.update(replay_train_file=str(root / f"assets/replay-{run['seed']}/train.extxyz"),
                    replay_valid_file=str(root / f"assets/replay-{run['seed']}/valid.extxyz"))
    selected_profile = campaign_profile(campaign)
    return data.build_umof_training_command(manifest, root / selected_profile.asset, folder,
        model=selected_profile.key,
        max_epochs=epochs or p["max_epochs"], learning_rate=p["learning_rate"],
        replay_weight=p["replay_weight"], replay_sample_count=p["replay_samples"],
        device=p["device"], seed=run["seed"], start_swa=p["start_swa"])


def training_identity(campaign, run):
    p = dict(campaign["protocol"]); p.pop("max_epochs")
    # Test data never affect training/restart/checkpoint selection.
    relevant = {k: v for k, v in campaign["files"].items()
                if k == campaign_profile(campaign).asset or
                k in {f"assets/replay-{run['seed']}/{s}.extxyz" for s in ("train", "valid")} or
                k in {f"{run['directory']}/datasets/{s}.extxyz" for s in ("train", "valid")}}
    return hashlib.sha256(json.dumps([p, relevant], sort_keys=True).encode()).hexdigest()


def resume_checkpoint(folder, identity):
    candidates = []
    for path in (folder / "checkpoints").glob("*.pt"):
        sidecar = Path(str(path) + ".manifest.json")
        if not sidecar.is_file():
            raise ValueError(f"Untracked/incomplete checkpoint: {path}")
        saved = read_json(sidecar)
        if saved["training_identity"] != identity or saved["sha256"] != data.sha256_file(path):
            raise ValueError(f"Incompatible or altered checkpoint: {path}")
        candidates.append((saved["epoch"], path))
    return str(max(candidates)[1]) if candidates else None


def release_cuda_memory():
    """Release evaluation models and cached allocations before child training."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def execute_campaign(output_directory, action="train", *, max_epochs=None, device=None):
    root = Path(output_directory).resolve()
    with campaign_lock(root):
        campaign = read_json(root / "campaign.json")
        if campaign_profile(campaign).key == "polar":
            from .dependencies import ensure_mace_polar_dependencies
            ensure_mace_polar_dependencies()
        if campaign.get("environment") and importlib.metadata.version("mace-torch") != campaign["environment"]["packages"]["mace-torch"]:
            raise ValueError("MACE version differs from prepared campaign; use the archived environment")
        if max_epochs is not None:
            changed = dict(campaign["protocol"], max_epochs=max_epochs)
            compare_protocol(campaign["protocol"], changed)
            campaign["protocol"] = changed
            atomic_json(root / "campaign.json", campaign)
        verify_files(root, campaign["files"])
        # Keep archived identities intact when moving existing checkpoints between devices.
        identity_campaign = campaign
        if device is not None:
            if device not in {"cpu", "cuda", "mps"}:
                raise ValueError("device must be one of: cpu, cuda, mps")
            campaign = dict(campaign, protocol=dict(campaign["protocol"], device=device))
        for run in campaign["runs"]:
            folder = root / run["directory"]
            status_path = folder / "status.json"
            state = read_json(status_path) if status_path.exists() else {}
            identity = training_identity(identity_campaign, run)
            if state and state.get("training_identity") != identity:
                raise ValueError(f"Foreign or incompatible checkpoint state in {folder}")
            if not state and list((folder / "checkpoints").glob("*")):
                raise ValueError(f"Untracked checkpoints in {folder}")
            if state.get("selected_model"):
                selected = state["selected_model"]
                if data.sha256_file(root / selected["path"]) != selected["sha256"]:
                    raise ValueError(f"Selected checkpoint changed in {folder}")
            if action in ("train", "resume"):
                resume_checkpoint(folder, identity)
            state["training_identity"] = identity
            stage = "training"
            try:
                target = campaign["protocol"]["max_epochs"]
                if action in ("train", "resume") and state.get("trained_to_limit", 0) < target:
                    before = time.perf_counter()
                    state.update(status="training", log="training.log")
                    atomic_json(status_path, state)
                    command = training_command(root, campaign, run)
                    env = os.environ.copy()
                    env["UMOF_TRAINING_IDENTITY"] = identity
                    latest = resume_checkpoint(folder, identity)
                    if state.get("trained_to_limit", 0) and not latest:
                        raise ValueError("Previously trained campaign has no resumable checkpoint")
                    if latest:
                        env["UMOF_RESUME_CHECKPOINT"] = latest
                    else:
                        env.pop("UMOF_RESUME_CHECKPOINT", None)
                    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
                    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                        env.setdefault(key, "1")
                    if campaign["protocol"]["device"] == "cuda":
                        release_cuda_memory()
                    with (folder / "training.log").open("a") as log:
                        completed = subprocess.run(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
                    state["training_seconds"] = state.get("training_seconds", 0) + time.perf_counter() - before
                    if completed.returncode:
                        raise RuntimeError(f"MACE exited {completed.returncode}; inspect {folder / 'training.log'}")
                    resume_checkpoint(folder, identity)
                    with (folder / "export.log").open("a") as export_log:
                        log_text = (folder / "training.log").read_text()
                        export_log.write(log_text.rsplit("===========RESULTS===========", 1)[-1])
                    atomic_json(folder / "exports.json", {p.name: model_info(p)
                        for p in (folder / "models").glob("*.model")})
                    state.update(status="trained", trained_to_limit=target)
                    state.pop("evaluation", None); state.pop("selected_model", None)
                    atomic_json(status_path, state)
                if state.get("trained_to_limit", 0) == 0:
                    raise ValueError(f"No completed training available in {folder}")
                stage = "selection"
                if "selected_model" not in state:
                    with (folder / "selection.log").open("a") as log, redirect_stdout(log), redirect_stderr(log):
                        state["selected_model"] = select_model(root, campaign, run)
                    state["status"] = "selected"
                    atomic_json(status_path, state)
                stage = "evaluation"
                if "evaluation" not in state:
                    before = time.perf_counter()
                    with (folder / "evaluation.log").open("a") as log, redirect_stdout(log), redirect_stderr(log):
                        evaluate_run(root, campaign, run, state["selected_model"])
                    state["evaluation_seconds"] = time.perf_counter() - before
                    state["evaluation"] = "results/evaluation.json"
                state.update(status="completed"); state.pop("error", None); state.pop("failed_stage", None)
                atomic_json(status_path, state)
            except BaseException as exc:
                state.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                             failed_stage=stage, error=f"{type(exc).__name__}: {exc}")
                atomic_json(status_path, state)
                raise
        return report_campaign(root)


def predict(dataset, checkpoint, head, cache, device="cpu", model="mh"):
    from ase.io import read
    identity = [data.sha256_file(dataset), data.sha256_file(checkpoint),
        head, "float64", device, importlib.metadata.version("mace-torch")]
    # Preserve the cache identity of existing MH campaigns.
    if model == "polar":
        identity += [model, importlib.metadata.version("graph_longrange")]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{key}.npz"
    if path.exists():
        with np.load(path) as saved:
            return saved["energies"], saved["forces"]
    calc = calculator(checkpoint, model=model, device=device, head=head)
    energies, forces = [], []
    for atoms in read(dataset, index=":"):
        atoms.calc = calc
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces()))
    energies, forces = np.asarray(energies), np.concatenate(forces)
    if not np.isfinite(energies).all() or not np.isfinite(forces).all():
        raise ValueError("Non-finite model predictions")
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, energies=energies, forces=forces)
    os.replace(temporary, path)
    return energies, forces


def reference_arrays(frames):
    return (np.asarray([a.info["REF_energy"] for a in frames]),
            np.concatenate([a.arrays["REF_forces"] for a in frames]),
            np.asarray([len(a) for a in frames]))


def select_model(root, campaign, run):
    from ase.io import read
    folder = root / run["directory"]
    valid = folder / "datasets/valid.extxyz"
    frames = read(valid, index=":")
    energy, force, sizes = reference_arrays(frames)
    candidates = []
    for stage, suffix in (("stage1", ""), ("stage2", "_stagetwo")):
        path = folder / "models" / f"{campaign_profile(campaign).name}{suffix}.model"
        if not path.exists():
            continue
        info = model_info(path)
        if "class" in info and info["class"] != campaign_profile(campaign).model_type:
            raise ValueError(f"Export architecture does not match campaign: {path}")
        if "Default" not in info["heads"] or "pt_head" not in info["heads"]:
            raise ValueError(f"Export missing actual adsorption/replay heads: {path}")
        pe, pf = predict(valid, path, "Default", root / "prediction_cache", campaign["protocol"]["device"], model=campaign_profile(campaign).key)
        metrics = data._error_metrics((pe-energy)/sizes, (pf-force).ravel())
        candidates.append({"stage": stage, "path": relative(path, root), "sha256": info["sha256"],
                           "head": "Default", "validation": metrics})
    if not candidates:
        raise FileNotFoundError(f"No exported stage-1 or stage-2 model in {folder}")
    chosen = min(candidates, key=lambda c: (c["validation"]["force_rmse_ev_per_A"],
                 c["validation"]["energy_rmse_ev_per_atom"], c["stage"] != "stage1"))
    atomic_json(folder / "results/selection.json", {"selected": chosen, "candidates": candidates,
                "selection_split": "validation", "test_used": False})
    atomic_json(folder / "module_c_model.json", {"backend": "mace-torch", "name": f"umof_{campaign_profile(campaign).key}_finetuned",
        "model_type": campaign_profile(campaign).model_type,
        "model": relative(root / chosen["path"], folder), "path_base": "specification_directory",
        "sha256": chosen["sha256"], "head": chosen["head"], "device": campaign["protocol"]["device"],
        "default_dtype": "float64", "dispersion": False,
        "validated_scope": "selected on adsorption validation; test pending; mixtures and flexibility remain open"})
    return chosen


def grouped_metrics(frames, prediction):
    energy, force, sizes = reference_arrays(frames)
    pe, pf = prediction
    energy_error, force_error = (pe-energy)/sizes, pf-force
    bounds = np.cumsum(np.r_[0, sizes])
    groups = {"all": list(range(len(frames)))}
    for index, atoms in enumerate(frames):
        groups.setdefault("mof:" + atoms.info.get("umof_mof_id", "unknown"), []).append(index)
        counts = json.loads(atoms.info["umof_guest_counts"])
        groups.setdefault("components:" + "+".join(sorted(counts)), []).append(index)
    metrics = {}
    for name, ids in groups.items():
        selected_forces = np.concatenate([force_error[bounds[i]:bounds[i+1]] for i in ids])
        metrics[name] = {"configuration_count": len(ids),
                         **data._error_metrics(energy_error[ids], selected_forces.ravel())}
    return metrics


def load_model_specification(path):
    """Resolve a portable exported specification for the existing Module-C backend."""
    path = Path(path).resolve()
    spec = read_json(path)
    model = (path.parent / spec["model"]).resolve()
    if data.sha256_file(model) != spec["sha256"]:
        raise ValueError("Exported model checksum mismatch")
    info = model_info(model)
    if spec["head"] not in info["heads"]:
        raise ValueError("Exported model head mismatch")
    if spec.get("model_type", "ScaleShiftMACE") != info["class"]:
        raise ValueError("Exported model architecture mismatch")
    if spec["default_dtype"] != info["dtype"]:
        raise ValueError("Exported model datatype mismatch")
    return {key: value for key, value in dict(spec, model=str(model)).items()
            if key in {"backend", "name", "model", "head", "device", "default_dtype", "dispersion", "model_type"}}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate_run(root, campaign, run, selected):
    from ase.io import read
    folder = root / run["directory"]
    path = folder / "datasets/test.extxyz"
    frames = read(path, index=":")
    checkpoint = root / selected["path"]
    if data.sha256_file(checkpoint) != selected["sha256"]:
        raise ValueError("Selected checkpoint changed after validation")
    predictions = {}
    for label, model, head in (("baseline", root / campaign_profile(campaign).asset, "r2scan+d4"),
                              ("finetuned", checkpoint, selected["head"])):
        predictions[label] = predict(path, model, head, root / "prediction_cache", campaign["protocol"]["device"], model=campaign_profile(campaign).key)
    metrics = {label: grouped_metrics(frames, pred) for label, pred in predictions.items()}
    result = {"metrics": metrics, "test_sha256": data.sha256_file(path), "selected_model": selected,
              "grouping": "Whole-configuration errors by component set; mixtures are not pure-gas observations",
              "mixture_validation": "open", "flexible_framework_validation": "open"}
    (folder / "results").mkdir(exist_ok=True)
    atomic_json(folder / "results/evaluation.json", result)
    write_csv(folder / "results/metrics.csv", [dict(model=label, group=group, **row)
        for label, groupmetrics in metrics.items() for group, row in groupmetrics.items()])
    re, rf, sizes = reference_arrays(frames)
    write_csv(folder / "results/energy_parity.csv", [{"frame_id": a.info["umof_frame_id"],
        "mof": a.info["umof_mof_id"], "reference_eV_per_atom": re[i]/sizes[i],
        "baseline_eV_per_atom": predictions["baseline"][0][i]/sizes[i],
        "finetuned_eV_per_atom": predictions["finetuned"][0][i]/sizes[i]} for i, a in enumerate(frames)])
    write_csv(folder / "results/force_parity.csv", [{"atom_index": i, "axis": axis,
        "reference_eV_per_A": rf[i, j], "baseline_eV_per_A": predictions["baseline"][1][i, j],
        "finetuned_eV_per_A": predictions["finetuned"][1][i, j]}
        for i in range(len(rf)) for j, axis in enumerate("xyz")])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, (pe, pf) in predictions.items():
        axes[0].scatter(re/sizes, pe/sizes, s=12, alpha=.65, label=label)
        step = max(1, rf.size // 20000)
        axes[1].scatter(rf.ravel()[::step], pf.ravel()[::step], s=2, alpha=.25, label=label)
    for ax, units in zip(axes, ("Energy (eV/atom)", "Force component (eV/Å)")):
        lo, hi = ax.get_xlim(); ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(xlabel=f"DFT reference: {units}", ylabel=f"Prediction: {units}")
        ax.legend()
    fig.suptitle("Held-out MOFs: energy and force transfer test")
    save_figure(fig, folder / "results/parity")
    spec_path = folder / "module_c_model.json"
    specification = read_json(spec_path)
    specification["validated_scope"] = "energy/force transfer test completed; mixture and flexibility validation remain open"
    atomic_json(spec_path, specification)
    return result


def save_figure(fig, path):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(str(path) + "." + suffix, dpi=160)
    plt.close(fig)


def status_campaign(root):
    root = Path(root)
    campaign = read_json(root / "campaign.json")
    verify_files(root, campaign["files"])
    return {"model": campaign_profile(campaign).key, "purpose": campaign["protocol"]["purpose"], "assets_verified": True, "runs": [dict(run,
        **(read_json(root / run["directory"] / "status.json") if
           (root / run["directory"] / "status.json").exists() else {"status": "prepared"}))
        for run in campaign["runs"]]}


def report_campaign(root):
    root = Path(root)
    status = status_campaign(root)
    rows, learning = [], []
    for run in status["runs"]:
        folder = root / run["directory"]
        row = {"fraction": run["fraction"], "seed": run["seed"], "training_frames": run["train_count"],
               "actual_training_fraction": run.get("actual_fraction", run["fraction"]),
               "status": run["status"], "training_seconds": run.get("training_seconds", 0),
               "evaluation_seconds": run.get("evaluation_seconds", 0)}
        path = folder / "results/evaluation.json"
        for label in ("baseline", "finetuned"):
            metrics = read_json(path)["metrics"][label]["all"] if path.exists() else {}
            for metric in ("energy_rmse_ev_per_atom", "force_rmse_ev_per_A"):
                row[f"{label}_{metric}"] = metrics.get(metric)
        rows.append(row)
        for logfile in (folder / "results").glob("*_train.txt"):
            for line in logfile.read_text().splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("epoch") is not None and entry.get("mode") == "eval":
                    learning.append({"fraction": run["fraction"], "seed": run["seed"],
                        "head": entry.get("head", "Default"), "epoch": entry["epoch"],
                        "loss": entry.get("loss"), "force_RMSE_eV_per_A": entry.get("rmse_f"),
                        "energy_RMSE_eV_per_atom": entry.get("rmse_e_per_atom")})
    write_csv(root / "matrix_summary.csv", rows)
    write_csv(root / "learning_curves.csv", learning)
    atomic_json(root / "matrix_summary.json", status)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for label in ("baseline", "finetuned"):
        for ax, metric in zip(axes[:2], ("energy_rmse_ev_per_atom", "force_rmse_ev_per_A")):
            valid = [r for r in rows if r[f"{label}_{metric}"] is not None]
            if valid:
                ax.plot([r["fraction"]*100 for r in valid], [r[f"{label}_{metric}"] for r in valid], "o-", label=label)
    axes[0].set(ylabel="Test energy RMSE (eV/atom)")
    axes[1].set(ylabel="Test force RMSE (eV/Å)")
    axes[2].bar([r["fraction"]*100 for r in rows], [r["training_seconds"]/3600 for r in rows], width=5)
    axes[2].set(ylabel="Training runtime (hours)")
    for ax in axes:
        ax.set(xlabel="Nominal training data fraction (%)")
        if ax.lines:
            ax.legend()
    save_figure(fig, root / "matrix_comparison")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["fraction"]*100 for r in rows], [r["actual_training_fraction"]*100 for r in rows], "o-")
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set(xlabel="Requested training fraction (%)", ylabel="Actual selected fraction (%)")
    save_figure(fig, root / "data_fractions")
    if learning:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for fraction, seed, head in sorted({(r["fraction"], r["seed"], r["head"]) for r in learning}):
            series = [r for r in learning if (r["fraction"], r["seed"], r["head"]) == (fraction, seed, head)]
            for ax, key in zip(axes, ("force_RMSE_eV_per_A", "energy_RMSE_eV_per_atom")):
                valid = [r for r in series if r[key] is not None]
                ax.plot([r["epoch"] for r in valid], [r[key] for r in valid], label=f"{fraction:.0%}, {head}")
        axes[0].set(ylabel="Validation force RMSE (eV/Å)")
        axes[1].set(ylabel="Validation energy RMSE (eV/atom)")
        for ax in axes:
            ax.set(xlabel="Epoch"); ax.legend(fontsize=7)
        save_figure(fig, root / "learning_curves")
    (root / "REPORT.md").write_text(
        "# uMOF fine-tuning transfer test\n\n"
        f"Purpose: **{status['purpose']}**. Completed runs: {sum(r['status']=='completed' for r in rows)}/{len(rows)}.\n\n"
        "See matrix_summary.csv, learning_curves.csv and per-run results/metrics.csv. "
        "Gas groups describe whole configurations, not isolated guest force errors. "
        "Nominal and actual data fractions differ because each occupied stratum is retained.\n\n"
        "Mixture validation and flexible-framework validation remain **open**. "
        "Energy/force transfer errors do not establish adsorption accuracy or independence from foundation pretraining.\n")
    return status


def smoke_campaign(production_directory, smoke_directory):
    """Use real published configurations; never put smoke observations in production."""
    from ase.io import read, write
    import torch
    production, root = Path(production_directory).resolve(), Path(smoke_directory).resolve()
    if root == production or production in root.parents:
        raise ValueError("Smoke must use a separate directory outside the production campaign")
    original = read_json(production / "campaign.json")
    selected_profile = campaign_profile(original)
    if original["protocol"]["purpose"] != "production":
        raise ValueError("Smoke source must be a prepared production campaign")
    verify_files(production, original["files"])
    with campaign_lock(root):
        if (root / "campaign.json").exists():
            existing = read_json(root / "campaign.json")
            if existing["protocol"]["purpose"] != "smoke":
                raise ValueError("Refusing to use production directory for smoke")
            if (campaign_profile(existing).key != selected_profile.key or
                existing.get("production_source_sha256") != data.sha256_file(production / "campaign.json")):
                raise ValueError("Smoke source/model changed; use a new smoke directory")
        else:
            full = max(original["runs"], key=lambda r: r["fraction"])
            folder = root / "100pct/seed-12345"
            (folder / "datasets").mkdir(parents=True, exist_ok=True)
            hashes, chosen = {}, {}
            for split, per_gas in (("train", 3), ("valid", 1), ("test", 1)):
                frames = read(production / full["directory"] / "datasets" / f"{split}.extxyz", index=":")
                selected = []
                for gas in ("CO2", "N2", "CH4"):
                    candidates = [a for a in frames if set(json.loads(a.info["umof_guest_counts"])) == {gas}]
                    unique = {a.info["umof_frame_id"]: a for a in candidates}
                    ordered = sorted(unique.values(), key=lambda a: (len(a), a.info["umof_frame_id"]))
                    if len(ordered) < per_gas:
                        raise ValueError(f"Not enough real {gas} {split} frames for smoke")
                    selected.extend(ordered[:per_gas])
                for atoms in selected:
                    atoms.info["umof_usage"] = "smoke_only"
                write(folder / "datasets" / f"{split}.extxyz", selected, format="extxyz")
                rel = relative(folder / "datasets" / f"{split}.extxyz", root)
                hashes[rel] = data.sha256_file(root / rel)
                chosen[split] = selected
            assets = root / "assets"; assets.mkdir(exist_ok=True)
            shutil.copy2(production / selected_profile.asset, assets / selected_profile.filename)
            # Use the smallest real, already selected replay structures; train/valid remain disjoint.
            replay_original = production / f"assets/replay-{full['seed']}"
            for split, count in (("train", 8), ("valid", 2)):
                frames = sorted(read(replay_original / f"{split}.extxyz", index=":"),
                                key=lambda a: (len(a), a.info["replay_geometry_id"]))[:count]
                replay_dir = assets / "replay-12345"; replay_dir.mkdir(exist_ok=True)
                for a in frames:
                    a.info["umof_usage"] = "smoke_only"
                write(replay_dir / f"{split}.extxyz", frames, format="extxyz")
                hashes[relative(replay_dir / f"{split}.extxyz", root)] = data.sha256_file(replay_dir / f"{split}.extxyz")
            hashes[selected_profile.asset] = data.sha256_file(assets / selected_profile.filename)
            settings = dict(original["protocol"], max_epochs=2, start_swa=1, fractions=[1.0],
                            seeds=[12345], purpose="smoke", replay_samples=8, replay_valid_count=2)
            campaign = {"schema_version": 2, "protocol": settings, "foundation": original["foundation"],
                        "files": hashes, "runs": [{"directory": "100pct/seed-12345", "seed": 12345,
                                                   "fraction": 1.0, "train_count": 9}],
                        "environment": environment(), "production_source_sha256": data.sha256_file(production / "campaign.json")}
            atomic_json(root / "campaign.json", campaign)
            atomic_json(root / "coverage.json", data.coverage_report(chosen))
    # These calls independently acquire/release the lock, as separate cluster invocations do.
    current = read_json(root / "campaign.json")
    if current["protocol"]["max_epochs"] == 2:
        execute_campaign(root, "train")
        state = read_json(root / "100pct/seed-12345/status.json")
        atomic_json(root / "two_epoch_checkpoint.json", state)
    execute_campaign(root, "resume", max_epochs=3)
    campaign = read_json(root / "campaign.json")
    folder = root / "100pct/seed-12345"
    model_path = root / read_json(folder / "status.json")["selected_model"]["path"]
    trained = torch.load(model_path, map_location="cpu", weights_only=False)
    foundation = torch.load(root / selected_profile.asset, map_location="cpu", weights_only=False)
    common = [(key, value, trained.state_dict()[key]) for key, value in foundation.state_dict().items()
              if key in trained.state_dict() and value.shape == trained.state_dict()[key].shape
              and key.startswith(("interactions.", "products.")) and value.is_floating_point()]
    changed = any(not torch.equal(a, b) for _, a, b in common)
    if not changed or set(trained.heads) != {"pt_head", "Default"}:
        raise AssertionError("Smoke did not change shared weights or preserve both heads")
    log = (folder / "training.log").read_text()
    if "Loading checkpoint" not in log and "Loaded checkpoint" not in log:
        raise AssertionError("Smoke resume log does not demonstrate checkpoint loading")
    completed_epochs = [int(value) for value in re.findall(r"Epoch (\d+): head: Default", log)]
    if completed_epochs != [0, 1, 2]:
        raise AssertionError(f"Smoke must train exactly three epochs, without repeating on resume: {completed_epochs}")
    resume_checkpoint(folder, training_identity(campaign, campaign["runs"][0]))
    losses = []
    for path in (folder / "results").glob("*_train.txt"):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "loss" in row:
                losses.append(float(row["loss"]))
    if not losses or not np.isfinite(losses).all():
        raise AssertionError("Missing or non-finite smoke losses")
    from .potential_backends.mace import MaceBackend
    from .models import InteractionConfiguration
    atoms = read(folder / "datasets/test.extxyz", index=0)
    spec = load_model_specification(folder / "module_c_model.json")
    backend = MaceBackend(model=spec["model"], device=spec["device"],
                          default_dtype=spec["default_dtype"], head=spec["head"], model_type=spec.get("model_type"))
    configuration = InteractionConfiguration(configuration_id="umof-smoke-backend", material=atoms.info["umof_mof_id"],
        adsorbate="whole_configuration", atoms=atoms, framework_indices=list(range(len(atoms))), adsorbate_indices=[])
    result = backend.evaluate(configuration)
    if not np.isfinite(result.energy_ev) or not np.isfinite(result.forces_ev_per_angstrom).all():
        raise AssertionError("Module-C backend smoke failed")
    report = {"status": "passed", "model": selected_profile.key, "purpose": "technical smoke only", "train": 9, "valid": 3, "test": 3,
              "replay_train": 8, "replay_valid": 2, "epochs": [2, 3], "resume_verified": True,
              "shared_weights_changed": changed, "heads": list(trained.heads), "finite_losses": len(losses),
              "completed_zero_based_epochs": completed_epochs,
              "checkpoint_hashes_verified": True,
              "backend_energy_eV": result.energy_ev, "selected_checkpoint_sha256": data.sha256_file(model_path),
              "production_data_import": "forbidden", "environment": environment()}
    atomic_json(root / "smoke_result.json", report)
    return report


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("setup", "prepare", "train", "resume", "status", "evaluate", "report", "smoke"))
    parser.add_argument("--matrix", action="store_true", help="Compatibility flag: prepare/train a fraction matrix")
    parser.add_argument("--train", action="store_true", help="Compatibility alias for train")
    parser.add_argument("--data-directory", type=Path, default=data.DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--model", choices=("mh", "polar"), help="Model profile; defaults to MH, or inherits an existing campaign")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--replay-source", type=Path)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--fractions", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--seeds", default="12345")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--replay-weight", type=float, default=.25)
    parser.add_argument("--replay-samples", type=int, default=1000)
    parser.add_argument("--replay-valid-samples", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--gases", default="CO2,N2,CH4")
    parser.add_argument("--gas-definitions", type=Path, help="JSON mapping stable component IDs to elemental formulas; ambiguous isomers require explicit atom assignments")
    parser.add_argument("--validation-mofs", default=",".join(sorted(data.DEFAULT_VALIDATION_MOFS)))
    parser.add_argument("--test-mofs", default=",".join(sorted(data.DEFAULT_TEST_MOFS)))
    parser.add_argument("--smoke-directory", type=Path)
    args = parser.parse_args(argv)
    action = args.action or ("train" if args.train else "prepare")
    selected_profile = profile(args.model or "mh")
    args.output_directory = args.output_directory or selected_profile.output_directory
    existing_path = args.output_directory / "campaign.json"
    if existing_path.exists():
        archived = campaign_profile(read_json(existing_path))
        if args.model is not None and args.model != archived.key:
            raise ValueError("--model conflicts with archived campaign; use a separate output directory")
        selected_profile = archived
    args.model = selected_profile.key
    args.checkpoint = args.checkpoint or selected_profile.checkpoint
    args.replay_source = args.replay_source or (REPLAY_SOURCE if args.model == "mh" else args.data_directory / POLAR_REPLAY_FILENAME)
    suffix = "" if args.model == "mh" else "_polar"
    args.smoke_directory = args.smoke_directory or Path(f"outputs/module_C_umof_finetuning_smoke{suffix}_v2")
    if args.download_missing and action != "setup":
        parser.error("Downloads are allowed only in setup")
    if action == "setup":
        with campaign_lock(args.output_directory):
            result = setup(args.checkpoint, args.replay_source, args.download_missing,
                           data_directory=args.data_directory, output_directory=args.output_directory, model=args.model)
    elif action == "status":
        result = status_campaign(args.output_directory)
    elif action == "report":
        with campaign_lock(args.output_directory):
            result = report_campaign(args.output_directory)
    elif action == "smoke":
        result = smoke_campaign(args.output_directory, args.smoke_directory)
    else:
        if action == "prepare" or (action == "train" and not (args.output_directory / "campaign.json").exists()):
            result = prepare_campaign(args.data_directory, args.checkpoint, args.output_directory,
                fractions=[float(x) for x in args.fractions.split(",")] if args.matrix or args.action else [args.train_fraction],
                seeds=[int(x) for x in args.seeds.split(",")] if args.matrix or args.action else [args.seed],
                max_epochs=args.max_epochs or 50, learning_rate=args.learning_rate,
                replay_weight=args.replay_weight, replay_samples=args.replay_samples,
                replay_valid_count=args.replay_valid_samples, device=args.device,
                gases=set(filter(None, args.gases.split(","))),
                gas_definitions=read_json(args.gas_definitions) if args.gas_definitions else None,
                validation_mofs=set(filter(None, args.validation_mofs.split(","))),
                test_mofs=set(filter(None, args.test_mofs.split(","))), replay_source=args.replay_source, model=args.model)
        if action in ("train", "resume", "evaluate"):
            campaign = read_json(args.output_directory / "campaign.json")
            # Omitted options inherit the archived protocol; explicit overrides must agree.
            checks = {
                "--learning-rate": ("learning_rate", args.learning_rate),
                "--replay-weight": ("replay_weight", args.replay_weight),
                "--replay-samples": ("replay_samples", args.replay_samples),
                "--replay-valid-samples": ("replay_valid_count", args.replay_valid_samples),
                "--gases": ("gases", sorted(filter(None, args.gases.split(",")))),
                "--fractions": ("fractions", sorted(float(x) for x in args.fractions.split(","))),
                "--seeds": ("seeds", sorted(int(x) for x in args.seeds.split(","))),
                "--validation-mofs": ("validation_mofs", sorted(filter(None, args.validation_mofs.split(",")))),
                "--test-mofs": ("test_mofs", sorted(filter(None, args.test_mofs.split(",")))),
            }
            explicit = {token.split("=", 1)[0] for token in raw_args if token.startswith("--")}
            for flag, (key, value) in checks.items():
                if flag in explicit and campaign["protocol"][key] != value:
                    raise ValueError(f"{flag} conflicts with archived campaign protocol")
            if args.gas_definitions:
                _, definitions = data.validate_gases(set(campaign["protocol"]["gases"]), read_json(args.gas_definitions))
                if definitions != campaign["protocol"]["gas_definitions"]:
                    raise ValueError("Gas definitions conflict with archived campaign")
            if "--checkpoint" in explicit and data.sha256_file(args.checkpoint) != campaign["foundation"]["sha256"]:
                raise ValueError("Explicit foundation checkpoint differs from archived campaign")
            if "--replay-source" in explicit and any(m["source_sha256"] != data.sha256_file(args.replay_source)
                                                       for m in campaign.get("replay", {}).values()):
                raise ValueError("Explicit replay source differs from archived campaign")
            result = execute_campaign(args.output_directory, action, max_epochs=args.max_epochs,
                device=args.device if "--device" in explicit else None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
