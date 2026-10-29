"""Small audited adapter for MACE's permissive restart handling.

MACE 0.3.16 catches restart errors and may start at epoch zero. Here an explicit
resume checkpoint must load strictly; each saved checkpoint has a hash/identity
sidecar. The adapter also preserves SWA scheduler/averaging and RNG state.
CUDA training saves autograd intermediates in CPU RAM by default so large
POLAR configurations can retain float64 precision on smaller GPUs.
"""
from functools import wraps
import importlib
import logging
from pathlib import Path
import os
import random

import numpy as np
import torch

from .umof_finetuning import sha256_file
from .umof_workflow import atomic_json, read_json


def cpu_offloaded_step(step, *, gpu_only_max_edges=0):
    """Offload saved tensors, including the graph used for force derivatives.

    Keep both the model forward and loss backward inside the context: force
    training differentiates through an autograd.grad call in the model forward.
    Synchronous copies avoid locking large amounts of host RAM in pinned buffers.
    Model parameters, optimizer state and computations remain on the GPU.
    """
    reported = False

    @wraps(step)
    def wrapped(*args, **kwargs):
        nonlocal reported
        if torch.device(kwargs["device"]).type != "cuda":
            return step(*args, **kwargs)
        batch = kwargs.get("batch")
        # Optional fast path for small graphs measured to fit on the GPU. Keep
        # a separate atom cap because electrostatic work is not just edge based.
        if (gpu_only_max_edges and batch is not None and batch.num_nodes <= 128
                and batch.edge_index.shape[1] <= gpu_only_max_edges):
            return step(*args, **kwargs)
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            result = step(*args, **kwargs)
        if not reported:
            logging.info("uMOF CPU offload: first CUDA optimizer step completed successfully")
            reported = True
        return result
    return wrapped


def main():
    from mace import tools
    from mace.cli.run_train import main as mace_main
    from mace.tools.checkpoint import CheckpointHandler, CheckpointIO

    identity = os.environ["UMOF_TRAINING_IDENTITY"]
    resume = os.environ.get("UMOF_RESUME_CHECKPOINT")
    offload = os.environ.get("UMOF_CPU_OFFLOAD", "1")
    if offload not in ("0", "1"):
        raise ValueError("UMOF_CPU_OFFLOAD must be 0 or 1")
    gpu_only_max_edges = int(os.environ.get("UMOF_GPU_ONLY_MAX_EDGES", "0"))
    if gpu_only_max_edges < 0:
        raise ValueError("UMOF_GPU_ONLY_MAX_EDGES must be nonnegative")
    original_save = CheckpointIO.save
    original_load = CheckpointHandler.load_latest
    original_train = tools.train
    context = {}

    def save(self, checkpoint, epochs, keep_last=False):
        swa = context.get("swa")
        checkpoint["umof_runtime"] = {
            "identity": identity, "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "swa_scheduler": swa.scheduler.state_dict() if swa else None,
            "swa_model": swa.model.state_dict() if swa else None,
            "cpu_offload": context.get("cpu_offload", False),
            "gpu_only_max_edges": gpu_only_max_edges,
        }
        original_save(self, checkpoint, epochs, keep_last)
        path = Path(self.directory) / self._get_checkpoint_filename(epochs, self.swa_start)
        atomic_json(str(path) + ".manifest.json", {
            "sha256": sha256_file(path), "training_identity": identity, "epoch": epochs})

    first_load = True

    def load(self, state, swa=False, device=None, strict=False):
        nonlocal first_load
        if first_load:
            first_load = False
            if resume:
                try:
                    sidecar = read_json(resume + ".manifest.json")
                    if sidecar["training_identity"] != identity or sidecar["sha256"] != sha256_file(resume):
                        raise ValueError("Resume checkpoint identity/checksum mismatch")
                    payload = torch.load(resume, map_location="cpu", weights_only=False)
                    runtime = payload["umof_runtime"]
                    if runtime["identity"] != identity:
                        raise ValueError("Embedded checkpoint identity mismatch")
                    epoch = self.load(state, resume, strict=True, device=device)
                    context["restore"] = runtime
                    # MACE stores the completed zero-based epoch; do not repeat it.
                    return epoch + 1
                except Exception as exc:
                    # SystemExit deliberately bypasses MACE's 'except Exception' fallback.
                    raise SystemExit(f"Refusing incompatible resume checkpoint: {exc}") from exc
        return original_load(self, state, swa=swa, device=device, strict=strict)

    def train(**kwargs):
        context["swa"] = kwargs.get("swa")
        context["cpu_offload"] = offload == "1" and torch.device(kwargs["device"]).type == "cuda"
        runtime = context.get("restore")
        if runtime:
            torch.set_rng_state(runtime["torch_rng"])
            np.random.set_state(runtime["numpy_rng"])
            random.setstate(runtime["python_rng"])
            if context["swa"] and runtime["swa_scheduler"] is not None:
                context["swa"].scheduler.load_state_dict(runtime["swa_scheduler"])
                context["swa"].model.load_state_dict(runtime["swa_model"])
        if not context["cpu_offload"]:
            return original_train(**kwargs)
        logging.info("uMOF CPU offload enabled: saved autograd tensors use host RAM; dtype unchanged")
        if gpu_only_max_edges:
            logging.info("uMOF small-graph GPU fast path: at most %s edges and 128 atoms", gpu_only_max_edges)
        training = importlib.import_module("mace.tools.train")
        steps = {name: getattr(training, name) for name in ("take_step", "take_step_lbfgs")}
        try:
            for name, step in steps.items():
                setattr(training, name, cpu_offloaded_step(step, gpu_only_max_edges=gpu_only_max_edges))
            return original_train(**kwargs)
        finally:
            for name, step in steps.items():
                setattr(training, name, step)

    CheckpointIO.save = save
    CheckpointHandler.load_latest = load
    tools.train = train
    mace_main()


if __name__ == "__main__":
    main()
