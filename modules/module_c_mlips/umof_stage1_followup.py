"""Short, isolated weight warm start from the selected Stage-1 adsorption head.

This starts a new optimizer and replay head; it is not an exact optimizer resume
of the earlier campaign. Subsequent resumes within this branch are strict.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from . import umof_workflow as workflow
from .umof_finetuning import build_umof_training_command, sha256_file

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "outputs/umof_polar_24h_float32"
DEFAULT_OUTPUT = PROJECT / "outputs/umof_polar_stage1_followup"


def extract_stage1_head(model):
    """Work around MACE 0.3.16 slicing head-independent POLAR cl_scale buffers."""
    from unittest.mock import patch
    from mace.tools.scripts_utils import remove_pt_head
    cls = type(model)
    original_load = cls.load_state_dict
    source_state = model.state_dict()
    shared = {k: v for k, v in source_state.items() if k.endswith(".cl_scale")}

    def load(self, state_dict, strict=True, assign=False):
        # Upstream remove_pt_head slices all names containing "scale". These
        # electrostatic basis tensors have no head axis and must remain intact.
        restored = dict(state_dict)
        restored.update(shared)
        # Upstream intentionally omits empty bias tensors; retain the target's
        # empty tensors while requiring every nonempty parameter to load.
        for name, value in self.state_dict().items():
            if name not in restored and value.numel() == 0:
                restored[name] = value
        return original_load(self, restored, strict=True, assign=assign)

    with patch.object(cls, "load_state_dict", load):
        selected = remove_pt_head(model, "Default")
    return selected.to(dtype=next(model.parameters()).dtype)


def add_single_head_asset(output, config):
    import torch
    from ase.io import read
    from .umof_profiles import calculator
    single_path = output / "assets/stage1_singlehead.model"
    parent_path = output / "assets/stage1.model"
    selected = extract_stage1_head(torch.load(parent_path, map_location="cpu", weights_only=False))
    torch.save(selected, single_path)
    frames = read(output / "assets/valid.extxyz", index=":")
    probe = min(frames, key=len)
    values = []
    for path in (parent_path, single_path):
        atoms = probe.copy()
        atoms.calc = calculator(path, model="polar", head="Default", device=config["device"], dtype=config["dtype"])
        values.append((float(atoms.get_potential_energy()), atoms.get_forces()))
    delta_e = abs(values[0][0]-values[1][0])
    delta_f = float(np.max(np.abs(values[0][1]-values[1][1])))
    if delta_e > .002 or not np.allclose(values[0][1], values[1][1], rtol=2e-5, atol=2e-5):
        raise ValueError(f"Head extraction changed predictions: energy {delta_e}, force {delta_f}")
    workflow.atomic_json(output / "head_extraction_check.json", {"energy_difference_eV": delta_e,
        "max_force_difference_eV_per_A": delta_f, "probe": "smallest validation frame", "status": "passed"})
    config["foundation"] = "assets/stage1_singlehead.model"
    config["files"][config["foundation"]] = sha256_file(single_path)
    return config


def identity(config):
    import hashlib
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def prepare(output, smoke=False):
    from ase.io import read, write
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "followup.json"
    if config_path.exists():
        config = workflow.read_json(config_path)
        if config["smoke"] != smoke:
            raise ValueError("Use separate output directories for smoke and full follow-up")
        workflow.verify_files(output, config["files"])
        if "foundation" not in config:
            if list((output / "checkpoints").glob("*.pt")):
                raise ValueError("Cannot change an already trained branch; use a new output")
            config = add_single_head_asset(output, config)
            workflow.atomic_json(config_path, config)
            workflow.atomic_json(output / "status.json", {"status": "prepared", "training_identity": identity(config)})
            workflow.atomic_json(output / "training_command.json", {"command": command(output, config)})
        return config
    campaign = workflow.read_json(SOURCE / "campaign.json")
    workflow.verify_files(SOURCE, campaign["files"])
    source_run = SOURCE / campaign["runs"][0]["directory"]
    parent = workflow.read_json(source_run / "results/selection.json")["selected"]
    if parent["stage"] != "stage1" or parent["head"] != "Default":
        raise ValueError("This follow-up expects the previously selected Stage 1 / Default export")
    parent_path = SOURCE / parent["path"]
    if sha256_file(parent_path) != parent["sha256"]:
        raise ValueError("Selected Stage-1 parent changed")
    assets = output / "assets"; assets.mkdir(exist_ok=True)
    shutil.copy2(parent_path, assets / "stage1.model")
    files = {"assets/stage1.model": parent["sha256"]}
    paths = {"train": source_run / "datasets/train.extxyz", "valid": source_run / "datasets/valid.extxyz",
             "replay_train": SOURCE / "assets/replay-12345/train.extxyz", "replay_valid": SOURCE / "assets/replay-12345/valid.extxyz"}
    counts = {}
    for key, path in paths.items():
        target = assets / f"{key}.extxyz"
        frames = read(path, index=":")
        if smoke:
            frames = sorted(frames, key=lambda a: len(a))[:2 if key.endswith("train") else 1]
            write(target, frames, format="extxyz")
        else:
            shutil.copy2(path, target)
        files[str(target.relative_to(output))] = sha256_file(target)
        counts[key] = len(frames)
    config = {"model": "polar", "parent": {"path": str(parent_path), "sha256": parent["sha256"], "head": "Default"},
        "seed": 12345, "max_epochs": 1 if smoke else 5, "learning_rate": 3e-5,
        "dtype": "float32", "device": "cuda", "swa": False, "replay_weight": campaign["protocol"]["replay_weight"],
        "counts": counts, "files": files, "smoke": smoke,
        "initialization": "weights from Stage-1 Default head; fresh optimizer and replay head; unchanged stage-one loss weights",
        "selection_scope": "validation only; no test or isotherm evaluation",
        "source_campaign_sha256": sha256_file(SOURCE / "campaign.json")}
    config = add_single_head_asset(output, config)
    workflow.atomic_json(config_path, config)
    workflow.atomic_json(output / "status.json", {"status": "prepared", "training_identity": identity(config)})
    workflow.atomic_json(output / "training_command.json", {"command": command(output, config)})
    return config


def command(output, config):
    output = output.resolve()
    manifest = {"train_file": output / "assets/train.extxyz", "valid_file": output / "assets/valid.extxyz",
        "replay_train_file": output / "assets/replay_train.extxyz", "replay_valid_file": output / "assets/replay_valid.extxyz"}
    tokens = build_umof_training_command(manifest, output / config["foundation"], output, model="polar",
        max_epochs=config["max_epochs"], learning_rate=config["learning_rate"],
        replay_weight=config["replay_weight"], replay_sample_count=config["counts"]["replay_train"],
        device=config["device"], seed=config["seed"], dtype=config["dtype"])
    tokens[tokens.index("--foundation_head")+1] = "Default"
    result, skip = [], False
    for token in tokens:
        if skip:
            skip = False
            continue
        if token == "--swa":
            continue
        if token in {"--swa_lr", "--start_swa"}:
            skip = True
            continue
        result.append(str(token))
    return result


def validate(output, config):
    from ase.io import read
    from .umof_profiles import calculator
    new = output / "models/umof_polar_adsorption_finetuned.model"
    dataset = output / "assets/valid.extxyz"
    frames = read(dataset, index=":")
    candidates = [("parent_stage1", output / "assets/stage1.model", "Default"),
                  ("followup", new, "Default")]
    groups, rows = {}, []
    for name, path, head in candidates:
        predictions = workflow.predict(dataset, path, head, output / "prediction_cache", config["device"], model="polar", dtype=config["dtype"])
        groups[name] = workflow.grouped_metrics(frames, predictions)
        for group, metrics in groups[name].items():
            rows.append({"model": name, "group": group, **metrics})
    workflow.write_csv(output / "validation_metrics.csv", rows)
    workflow.atomic_json(output / "validation.json", {"metrics": groups,
        "test_used": False, "smoke": config["smoke"], "model_sha256": sha256_file(new)})
    (output / "README.md").write_text(
        "# Stage-1 follow-up\n\n"
        "Weight warm start from the selected Stage-1 Default head, fresh optimizer and replay head. "
        "One seed, learning rate 3e-5, no SWA/Stage Two; same Stage-One loss weights. "
        "Original checkpoints and campaigns are unchanged.\n\n"
        "validation_metrics.csv compares parent and follow-up by MOF and gas group. "
        "This is validation of whole-configuration energies/forces, not proof of improved adsorption. "
        "No test or isotherm evaluation is run.\n")
    return groups


def check(output, config):
    import importlib.metadata
    import torch
    workflow.verify_files(output, config["files"])
    if importlib.metadata.version("mace-torch") != "0.3.16":
        raise ValueError("This POLAR head-extraction adapter was verified with MACE 0.3.16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    info = workflow.model_info(output / "assets/stage1.model")
    if info["class"] != "PolarMACE" or "Default" not in info["heads"] or info["dtype"] != config["dtype"]:
        raise ValueError("Unexpected parent model architecture, head or dtype")
    return {"status": "ready", "max_epochs": config["max_epochs"], "learning_rate": config["learning_rate"],
        "swa": False, "seed": config["seed"], "counts": config["counts"], "output": str(output)}


def run(output, config):
    with workflow.campaign_lock(output):
        state = workflow.read_json(output / "status.json")
        if state["training_identity"] != identity(config):
            raise ValueError("Branch identity changed")
        if state["status"] == "completed":
            print(f"Already completed: {output}"); return
        latest = workflow.resume_checkpoint(output, identity(config))
        env = dict(os.environ, UMOF_TRAINING_IDENTITY=identity(config))
        if latest:
            env["UMOF_RESUME_CHECKPOINT"] = latest
        else:
            env.pop("UMOF_RESUME_CHECKPOINT", None)
        env["PYTHONPATH"] = str(PROJECT)+os.pathsep+env.get("PYTHONPATH", "")
        state.update(status="training")
        workflow.atomic_json(output / "status.json", state)
        print(f"Training log: {output/'training.log'}", flush=True)
        started = time.perf_counter()
        try:
            with (output / "training.log").open("a") as handle:
                result = subprocess.run(command(output, config), cwd=output, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"MACE exited {result.returncode}; see {output/'training.log'}")
            workflow.resume_checkpoint(output, identity(config))
            state.update(status="validating")
            workflow.atomic_json(output / "status.json", state)
            metrics = validate(output, config)
            state.update(status="completed", training_and_validation_seconds=time.perf_counter()-started)
            workflow.atomic_json(output / "status.json", state)
            print(json.dumps(metrics, indent=2), flush=True)
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            workflow.atomic_json(output / "status.json", state)
            raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "check", "run", "status", "smoke"], nargs="?", default="run")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    smoke = args.action == "smoke"
    output = (args.output or (Path(str(DEFAULT_OUTPUT)+"_smoke") if smoke else DEFAULT_OUTPUT)).resolve()
    config = prepare(output, smoke)
    if args.action == "status":
        print(json.dumps(workflow.read_json(output / "status.json"), indent=2)); return 0
    if args.action != "prepare":
        print(json.dumps(check(output, config), indent=2), flush=True)
    else:
        print(f"Prepared {output}")
    if args.action in {"run", "smoke"}:
        run(output, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
