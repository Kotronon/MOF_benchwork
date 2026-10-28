"""Small audited adapter for MACE's permissive restart handling.

MACE 0.3.16 catches restart errors and may start at epoch zero. Here an explicit
resume checkpoint must load strictly; each saved checkpoint has a hash/identity
sidecar. The adapter also preserves SWA scheduler/averaging and RNG state.
"""
from pathlib import Path
from contextlib import nullcontext
import logging
import os
import random

import numpy as np
import torch

from .umof_finetuning import sha256_file
from .umof_workflow import atomic_json, read_json


def main():
    from mace import tools
    from mace.cli.run_train import main as mace_main
    from mace.tools.checkpoint import CheckpointHandler, CheckpointIO

    identity = os.environ["UMOF_TRAINING_IDENTITY"]
    resume = os.environ.get("UMOF_RESUME_CHECKPOINT")
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
        runtime = context.get("restore")
        if runtime:
            torch.set_rng_state(runtime["torch_rng"])
            np.random.set_state(runtime["numpy_rng"])
            random.setstate(runtime["python_rng"])
            if context["swa"] and runtime["swa_scheduler"] is not None:
                context["swa"].scheduler.load_state_dict(runtime["swa_scheduler"])
                context["swa"].model.load_state_dict(runtime["swa_model"])
        # Offload tensors retained for backward without changing model precision.
        offload = os.environ.get("UMOF_CPU_OFFLOAD") == "1"
        if offload:
            logging.info("CPU offload enabled for autograd saved tensors")
        with torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext():
            return original_train(**kwargs)

    CheckpointIO.save = save
    CheckpointHandler.load_latest = load
    tools.train = train
    mace_main()


if __name__ == "__main__":
    main()
