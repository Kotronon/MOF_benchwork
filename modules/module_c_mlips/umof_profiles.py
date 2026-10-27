"""Model-specific uMOF assets and calculator selection (no downloads)."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    key: str
    filename: str
    model_type: str

    @property
    def checkpoint(self):
        return Path("external/umof_models") / self.filename

    @property
    def name(self):
        return f"umof_{self.key}_adsorption_finetuned"

    @property
    def output_directory(self):
        suffix = "" if self.key == "mh" else "_polar"
        return Path(f"outputs/module_C_umof_finetuning_matrix{suffix}_v2")

    @property
    def asset(self):
        return "assets/" + self.filename

    def validate(self, info):
        if info["class"] != self.model_type:
            raise ValueError(f"{self.key} requires {self.model_type}, got {info['class']}")
        if "r2scan+d4" not in info["heads"]:
            raise ValueError("Foundation checkpoint lacks r2scan+d4 head")


PROFILES = {
    "mh": ModelProfile("mh", "uMOF-MH.model", "ScaleShiftMACE"),
    "polar": ModelProfile("polar", "uMOF-POLAR.model", "PolarMACE"),
}

POLAR_REPLAY_FILENAME = "qmof_r2scan_gaussian_noise_train.extxyz"
POLAR_REPLAY_SHA256 = "310d9267720430280f2cf2a9448ec96d87e9df970e1908fa23925971eb517463"
POLAR_REPLAY_URL = "https://figshare.com/articles/dataset/33311829"


def polar_conditioning(atoms):
    """Make MACE training/calculator keys agree; never infer a DFT spin state."""
    import math
    provenance = {}
    for canonical, alias, default in (("total_charge", "charge", 0.0), ("total_spin", "spin", 1.0)):
        present = [float(atoms.info[k]) for k in (canonical, alias) if k in atoms.info]
        value = present[0] if present else default
        if not math.isfinite(value) or any(v != value for v in present):
            raise ValueError(f"Invalid or contradictory POLAR {canonical}")
        if canonical == "total_spin" and (value < 1 or not value.is_integer()):
            raise ValueError("POLAR total_spin is a positive integer spin multiplicity")
        atoms.info[canonical] = atoms.info[alias] = value
        provenance[canonical] = "source" if present else "MACE default; not a reference spin/charge label"
    import json
    atoms.info["umof_polar_conditioning"] = json.dumps(provenance, sort_keys=True)


def prepare_polar_datasets(manifests):
    from ase.io import read, write
    from .umof_finetuning import sha256_file
    for manifest in manifests:
        manifest["model_profile"] = "polar"
        manifest["replay_dataset"] = "umof_r2scan_d4_gaussian_noise_train"
        for split in ("train", "valid", "test"):
            path = manifest[f"{split}_file"]
            frames = read(path, index=":")
            for atoms in frames:
                polar_conditioning(atoms)
            write(path, frames, format="extxyz")
            manifest["file_sha256"][split] = sha256_file(path)


def reduced_composition(counts):
    """Conservative framework overlap filter, invariant to supercell size."""
    from math import gcd
    from functools import reduce
    values = {k: v for k, v in counts.items() if v}
    divisor = reduce(gcd, values.values())
    return tuple(sorted((k, v // divisor) for k, v in values.items()))


def adsorption_framework_compositions(manifests):
    import json
    from collections import Counter
    from ase.io import iread
    from .umof_finetuning import KNOWN_GAS_FORMULAS
    result = set()
    # Full fraction contains all smaller training sets; validation/test are shared.
    manifest = max(manifests, key=lambda m: m["training_data_fraction"])
    for split in ("train", "valid", "test"):
        for atoms in iread(manifest[f"{split}_file"]):
            counts = Counter(atoms.get_chemical_symbols())
            for gas, number in json.loads(atoms.info["umof_guest_counts"]).items():
                definition = manifest.get("gas_definitions", KNOWN_GAS_FORMULAS)[gas]
                for element, n in definition.items():
                    counts[element] -= n * number
            if any(n < 0 for n in counts.values()):
                raise ValueError("Invalid framework composition")
            result.add(reduced_composition(counts))
    return result


def profile(model="mh"):
    try:
        return PROFILES[model]
    except KeyError as exc:
        raise ValueError(f"Unknown uMOF model: {model}") from exc


def campaign_profile(campaign):
    # Old manifests and their training identities remain valid.
    return profile(campaign["protocol"].get("model", "mh"))


def calculator(checkpoint, *, model="mh", head="r2scan+d4", device="cpu"):
    from .potential_backends.calculators import build_ase_calculator
    selected = profile(model)
    return build_ase_calculator({"backend": "mace-torch", "name": f"umof_{model}_local",
        "model": str(Path(checkpoint).resolve()), "model_type": selected.model_type,
        "head": head, "device": device, "default_dtype": "float64", "dispersion": False})
