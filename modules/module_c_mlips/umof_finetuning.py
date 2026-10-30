"""Prepare and run uMOF MH/POLAR fine-tuning on released adsorption data."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np

from pipeline.config import save_benchmark_data


DEFAULT_DATA_DIRECTORY = Path("external/umof_data")
DEFAULT_CHECKPOINT = Path("external/umof_models/uMOF-MH.model")
DEFAULT_OUTPUT_DIRECTORY = Path("outputs/module_C_umof_finetuning_matrix_v2")
DEFAULT_TRAIN_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
DEFAULT_REPLAY_DATASET = "matpes_r2scan"
DEFAULT_VALIDATION_MOFS = {
    "HKUST-1-adsorbates",
    "MIL-53-Al-adsorbates",
}
DEFAULT_TEST_MOFS = {
    "MOF-303-adsorbates",
    "MgMOF-74-adsorbates",
}
TARGET_GASES = {"CO2", "N2", "CH4"}
KNOWN_GAS_FORMULAS = {
    "CO2": {"C": 1, "O": 2},
    "N2": {"N": 2},
    "CH4": {"C": 1, "H": 4},
    "H2": {"H": 2},
    "H2O": {"H": 2, "O": 1},
    "NH3": {"H": 3, "N": 1},
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_id(atoms: Any) -> str:
    digest = hashlib.sha256()
    for values in (atoms.numbers, atoms.positions, atoms.cell.array, atoms.pbc):
        digest.update(np.asarray(values).tobytes())
    return digest.hexdigest()


def prepare_adsorption_datasets(
    data_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *, train_fraction: float = 1.0, seed: int = 12345,
    validation_mofs: set[str] | None = None,
    test_mofs: set[str] | None = None,
    gases: set[str] | None = None,
    gas_definitions: dict[str, dict[str, int]] | None = None,
    supported_elements: set[int] | None = None,
) -> dict[str, Any]:
    """Prepare released labels, retaining full adsorbed composition and provenance."""
    from ase.io import read, write
    if not math.isfinite(train_fraction) or not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1].")
    gases, definitions = validate_gases(gases, gas_definitions)
    if supported_elements is not None:
        from ase.data import atomic_numbers
        if {atomic_numbers[e] for gas in gases for e in definitions[gas]} - supported_elements:
            raise ValueError("Selected gas definition uses elements unsupported by the checkpoint")
    root, output = Path(data_directory).expanduser(), Path(output_directory).expanduser()
    files = {split: {p.name.removesuffix(f"-{split}.extxyz"): p
                    for p in sorted(root.glob(f"*-adsorbates-{split}.extxyz"))
                    if p.is_file() and not p.name.startswith("._")}
             for split in ("train", "valid", "test")}
    if any(not group for group in files.values()):
        raise FileNotFoundError(f"Incomplete adsorption train/valid/test files in {root}")
    if set(files["train"]) != set(files["valid"]):
        raise ValueError("uMOF train/valid MOF groups differ")
    valid_mofs = set(DEFAULT_VALIDATION_MOFS if validation_mofs is None else validation_mofs)
    held_mofs = set(DEFAULT_TEST_MOFS if test_mofs is None else test_mofs)
    if not valid_mofs or not held_mofs or valid_mofs & held_mofs:
        raise ValueError("Validation and test MOF groups must be nonempty and disjoint")
    all_mofs = set(files["train"])
    if (valid_mofs | held_mofs) - all_mofs or held_mofs - set(files["test"]):
        raise ValueError("Unknown MOFs or missing official test files")
    groups = {"train": all_mofs - valid_mofs - held_mofs,
              "valid": valid_mofs, "test": held_mofs}
    if not groups["train"]:
        raise ValueError("At least one training MOF is required")
    rng = random.Random(seed)
    datasets, source_counts, exclusions, seen = {}, {}, [], {}
    for split, mofs in groups.items():
        selected_frames = []
        for mof in sorted(mofs):
            path = files[split][mof]
            frames = read(path, index=":", format="extxyz")
            prepared, source_indices = [], []
            for index, atoms in enumerate(frames):
                if atoms.info.get("umof_mof_id", mof) != mof:
                    raise ValueError(f"Contradictory MOF identity in {path}:{index}")
                try:
                    current = _prepare_frame(atoms, path, split, gases=gases,
                                             gas_definitions=definitions)
                except GuestAssignmentError as exc:
                    exclusions.append({"source": path.name, "index": index, "reason": str(exc)})
                    continue
                if current is None:
                    exclusions.append({"source": path.name, "index": index,
                                       "reason": "No allowed guests or excluded component present"})
                    continue
                if supported_elements is not None and set(current.numbers) - supported_elements:
                    raise ValueError(f"Checkpoint does not support elements in {path}:{index}")
                frame_id = geometry_id(current)
                if frame_id in seen and seen[frame_id] != split:
                    raise ValueError(f"Duplicate configuration across splits: {path}:{index}")
                seen[frame_id] = split
                current.info.update(umof_frame_id=frame_id, umof_source_index=index,
                                    umof_mof_id=mof)
                prepared.append(current)
                source_indices.append(index)
            if not prepared:
                raise ValueError(f"No unambiguously assigned allowed gas frames in {path}")
            indices = (_stratified_nested_indices(prepared, train_fraction=train_fraction, rng=rng,
                        gases=gases, gas_definitions=definitions)
                       if split == "train" else list(range(len(prepared))))
            selected_frames.extend(prepared[i] for i in indices)
            source_counts[path.name] = {"sha256": sha256_file(path), "available": len(prepared),
                "selected": len(indices), "source_indices": [source_indices[i] for i in indices],
                "selection_indices": indices,
                "frame_ids": [prepared[i].info["umof_frame_id"] for i in indices]}
        datasets[split] = selected_frames
    output.mkdir(parents=True, exist_ok=True)
    for split, frames in datasets.items():
        write(output / f"{split}.extxyz", frames, format="extxyz")
    manifest = {
        "schema_version": 2, "status": "prepared", "dataset": "uMOF released adsorbate-loaded frameworks",
        "energy_reference": "r2SCAN-D4", "target_gases": sorted(gases), "gas_definitions": definitions,
        "training_data_fraction": train_fraction, "seed": seed,
        **{f"{split}_file": str(output / f"{split}.extxyz") for split in datasets},
        "file_sha256": {split: sha256_file(output / f"{split}.extxyz") for split in datasets},
        "train_configuration_count": len(datasets["train"]),
        "validation_configuration_count": len(datasets["valid"]),
        "test_configuration_count": len(datasets["test"]),
        "actual_training_fraction": len(datasets["train"]) / sum(
            row["available"] for name, row in source_counts.items() if name.endswith("-train.extxyz")),
        "training_mofs": sorted(groups["train"]), "validation_mofs": sorted(valid_mofs),
        "test_mofs": sorted(held_mofs), "test_files_used_for_training": False,
        "source_file_counts": source_counts, "exclusions": exclusions,
        "replay_dataset": DEFAULT_REPLAY_DATASET,
        "label_mapping": {"dft_energy": "REF_energy", "dft_forces": "REF_forces"},
        "units": {"energy": "eV", "forces": "eV/Angstrom"},
        "stress": "Original dft_stress retained; source convention not inferred; unused in training",
        "coverage": coverage_report(datasets),
    }
    save_benchmark_data(output / "dataset_manifest.json", manifest)
    return manifest


def coverage_report(datasets: dict[str, list[Any]]) -> dict[str, Any]:
    from collections import Counter
    report = {}
    for split, frames in datasets.items():
        combinations, loadings, variations = Counter(), Counter(), Counter()
        structures = {}
        for atoms in frames:
            counts = json.loads(atoms.info["umof_guest_counts"])
            combinations["+".join(sorted(counts))] += 1
            loadings[str(sum(counts.values()))] += 1
            variations[str(atoms.info.get("config_type", "unspecified"))] += 1
            mof = atoms.info.get("umof_mof_id", "unknown")
            structures.setdefault(mof, set()).add(tuple(np.round(atoms.cell.array.ravel(), 6)))
        report[split] = {"component_sets": dict(combinations), "loading_counts": dict(loadings),
                         "source_config_types": dict(variations),
                         "distinct_cells_by_mof": {m: len(c) for m, c in structures.items()},
                         "framework_deformation_labels": "not established by source config_type or cell count",
                         "mixture_configuration_count": sum(v for k, v in combinations.items() if "+" in k)}
    report.update(composition_basis="adsorbed molecule counts, not gas-phase fractions",
                  available_gases=sorted({g for frames in datasets.values() for a in frames
                                          for g in json.loads(a.info["umof_guest_counts"])}),
                  configuration_scope="pure_gas_only" if all(
                      len(json.loads(a.info["umof_guest_counts"])) == 1
                      for frames in datasets.values() for a in frames) else "includes_real_coloaded_mixtures",
                  mixture_validation="open", flexible_framework_validation="open",
                  interpretation="Configuration coverage does not establish adsorption or flexibility accuracy")
    return report


def build_umof_mh_training_command(
    manifest: dict[str, Any],
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    max_epochs: int = 50,
    learning_rate: float = 1.0e-4,
    replay_weight: float = 0.25,
    replay_sample_count: int = 1000,
    replay_dataset: str = DEFAULT_REPLAY_DATASET,
    device: str = "cpu",
    seed: int = 12345,
    replay_valid_file: str | Path | None = None,
    start_swa: int = 40,
    model: str = "mh",
    dtype: str = "float64",
) -> list[str]:
    """Build a local multihead training command; MH remains the default profile."""
    from .umof_profiles import profile
    selected_profile = profile(model)
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"uMOF checkpoint not found: {checkpoint_path}")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite.")
    if not math.isfinite(replay_weight) or replay_weight <= 0.0:
        raise ValueError("replay_weight must be positive and finite.")
    if replay_sample_count <= 0:
        raise ValueError("replay_sample_count must be positive.")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: cpu, cuda, mps.")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64.")
    replay_path = Path(manifest.get("replay_train_file", replay_dataset)).expanduser()
    valid_replay = Path(replay_valid_file or manifest.get("replay_valid_file", replay_path.with_name("valid.extxyz")))
    if not replay_path.is_file() or not valid_replay.is_file():
        raise FileNotFoundError("Explicit local replay train/valid files required; run setup and prepare first")
    output = Path(output_directory).expanduser().resolve()
    return [
        sys.executable,
        "-m",
        "modules.module_c_mlips.umof_mace_train",
        "--name",
        selected_profile.name,
        "--restart_latest",
        "--save_all_checkpoints",
        "--train_file",
        str(Path(manifest["train_file"]).resolve()),
        "--valid_file",
        str(Path(manifest["valid_file"]).resolve()),
        "--model", selected_profile.model_type,
        "--foundation_model",
        str(checkpoint_path.resolve()),
        "--foundation_head",
        "r2scan+d4",
        "--foundation_model_elements",
        "True",
        "--multiheads_finetuning",
        "True",
        "--E0s",
        "foundation",
        "--pt_train_file",
        str(replay_path.resolve()),
        "--pt_valid_file",
        str(valid_replay.resolve()),
        "--real_pt_data_ratio_threshold",
        "0.0",
        "--num_samples_pt",
        str(replay_sample_count),
        "--weight_pt_head",
        str(replay_weight),
        "--subselect_pt",
        "random",
        "--filter_type_pt",
        "none",
        "--force_mh_ft_lr",
        "True",
        "--seed",
        str(seed),
        "--lr",
        str(learning_rate),
        "--swa_lr",
        str(learning_rate),
        "--energy_key",
        "REF_energy",
        "--forces_key",
        "REF_forces",
        "--stress_key", "UNUSED_stress",
        "--virials_key", "UNUSED_virials",
        "--compute_stress", "False",
        "--scheduler", "ReduceLROnPlateau",
        "--lr_factor", "0.8",
        "--scheduler_patience", "50",
        "--device",
        device,
        "--default_dtype",
        dtype,
        "--batch_size",
        "1",
        "--valid_batch_size",
        "1",
        "--max_num_epochs",
        str(max_epochs),
        "--patience",
        "20",
        "--swa",
        "--start_swa",
        str(start_swa),
        "--model_dir",
        str(output / "models"),
        "--checkpoints_dir",
        str(output / "checkpoints"),
        "--results_dir",
        str(output / "results"),
        "--log_dir",
        str(output / "logs"),
    ]


def build_umof_training_command(manifest, checkpoint=None, output_directory=None, *, model="mh", **kwargs):
    from .umof_profiles import profile
    selected = profile(model)
    return build_umof_mh_training_command(manifest,
        checkpoint if checkpoint is not None else selected.checkpoint,
        output_directory if output_directory is not None else selected.output_directory,
        model=model, **kwargs)


def evaluate_umof_mh_models(test_file, finetuned_checkpoint, *,
        baseline_checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
    """Compatibility evaluator; gas groups refer to complete component sets."""
    from ase.io import read, write
    import tempfile
    from .umof_workflow import predict, grouped_metrics, model_info
    frames = read(test_file, index=":")
    if not frames:
        raise ValueError("Empty evaluation dataset")
    for atoms in frames:
        counts = _guest_molecule_counts(atoms)
        atoms.info.setdefault("umof_guest_counts", json.dumps(counts, sort_keys=True))
        atoms.info.setdefault("umof_mof_id", str(atoms.info.get("umof_source_file", "unknown")).split("-adsorbates-")[0])
        atoms.info.setdefault("REF_energy", atoms.info.get("dft_energy"))
        if "REF_forces" not in atoms.arrays:
            atoms.set_array("REF_forces", atoms.arrays["dft_forces"])
    heads = model_info(finetuned_checkpoint)["heads"]
    if "Default" not in heads:
        raise ValueError("Finetuned checkpoint lacks Default head")
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "evaluation.extxyz"
        write(path, frames, format="extxyz")
        metrics = {label: grouped_metrics(frames, predict(path, model, head, Path(temporary)/"cache", device))
            for label, model, head in (("baseline", baseline_checkpoint, "r2scan+d4"),
                                      ("finetuned", finetuned_checkpoint, "Default"))}
    return {"status": "completed", "test_configuration_count": len(frames),
            "test_mof_count": len({a.info["umof_mof_id"] for a in frames}),
            "test_file": str(test_file), "baseline_checkpoint": str(baseline_checkpoint),
            "finetuned_checkpoint": str(finetuned_checkpoint),
            "aggregate": {label: values["all"] for label, values in metrics.items()},
            "by_mof": {key[4:]: {label: values[key] for label, values in metrics.items()}
                       for key in metrics["baseline"] if key.startswith("mof:")},
            "by_gas": {key[11:]: {label: values[key] for label, values in metrics.items()}
                       for key in metrics["baseline"] if key.startswith("components:")},
            "grouping": "Whole configurations by component set, not isolated guest forces"}


class GuestAssignmentError(ValueError):
    """A frame cannot be safely assigned to gas components."""


def validate_gases(gases=None, gas_definitions=None):
    from ase.data import atomic_numbers
    selected = set(TARGET_GASES if gases is None else gases)
    definitions = {k: dict(v) for k, v in KNOWN_GAS_FORMULAS.items()}
    definitions.update(gas_definitions or {})
    if not selected or selected - set(definitions):
        raise ValueError("Every selected gas requires a component/formula definition")
    for name, formula in definitions.items():
        if not name or not formula or any(e not in atomic_numbers or isinstance(n, bool)
                or not isinstance(n, int) or n <= 0 for e, n in formula.items()):
            raise ValueError(f"Invalid gas definition: {name}")
    return selected, definitions


def _target_gas_composition(atoms: Any, gases=None, gas_definitions=None) -> dict[str, int]:
    gases, definitions = validate_gases(gases, gas_definitions)
    counts = _guest_molecule_counts(atoms, definitions)
    return {} if not counts or set(counts) - gases else counts


def _stratified_nested_indices(frames: Sequence[Any], *, train_fraction: float,
        rng: random.Random, gases=None, gas_definitions=None) -> list[int]:
    strata = {}
    for index, atoms in enumerate(frames):
        counts = _target_gas_composition(atoms, gases, gas_definitions)
        names, total = tuple(sorted(counts)), sum(counts.values())
        bins = tuple(min(4, (5 * counts[g]) // total) for g in names) if len(names) > 1 else ()
        key = (names, "1" if total == 1 else "2-3" if total <= 3 else "4+", bins)
        strata.setdefault(key, []).append(index)
    selected = []
    for key in sorted(strata):
        indices = list(strata[key]); rng.shuffle(indices)
        selected.extend(indices[:max(1, math.ceil(len(indices) * train_fraction))])
    return sorted(selected)


def _is_replay_training_file(filename: str) -> bool:
    if not filename.endswith("-train.extxyz"):
        return False
    excluded_markers = (
        "adsorbate",
        "isolated_atoms",
        "mofgen",
    )
    normalized = filename.casefold()
    return not any(marker in normalized for marker in excluded_markers)


def _error_metrics(energy_errors: Sequence[float], force_errors: Sequence[float]) -> dict[str, float]:
    energy = np.asarray(energy_errors, dtype=float)
    forces = np.asarray(force_errors, dtype=float)
    return {
        "energy_mae_ev_per_atom": float(np.mean(np.abs(energy))),
        "energy_rmse_ev_per_atom": float(np.sqrt(np.mean(np.square(energy)))),
        "force_mae_ev_per_A": float(np.mean(np.abs(forces))),
        "force_rmse_ev_per_A": float(np.sqrt(np.mean(np.square(forces)))),
    }


def _prepare_frame(atoms: Any, source: Path, split: str, *, gases=None, gas_definitions=None) -> Any | None:
    if "dft_energy" not in atoms.info or "dft_forces" not in atoms.arrays:
        raise ValueError(
            f"uMOF frame in {source} lacks dft_energy or dft_forces labels."
        )
    energy = float(atoms.info["dft_energy"])
    forces = np.asarray(atoms.arrays["dft_forces"], dtype=float)
    if not math.isfinite(energy) or forces.shape != (len(atoms), 3):
        raise ValueError(f"Invalid uMOF energy/force labels in {source}.")
    if not np.isfinite(forces).all():
        raise ValueError(f"Non-finite uMOF force labels in {source}.")
    guest_counts = _target_gas_composition(atoms, gases, gas_definitions)
    if not guest_counts:
        return None
    prepared = atoms.copy()
    kind = "mixture" if len(guest_counts) > 1 else "pure_gas"
    if atoms.info.get("umof_configuration_kind", kind) != kind:
        raise ValueError("Stored configuration kind contradicts component identities")
    components = atoms.info.get("umof_component_ids")
    if components is not None and sorted(json.loads(components) if isinstance(components, str) else components) != sorted(guest_counts):
        raise ValueError("Stored component identities contradict atom assignments")
    composition = {gas: count / sum(guest_counts.values()) for gas, count in guest_counts.items()}
    stored = atoms.info.get("umof_gas_composition")
    if stored is not None:
        stored = json.loads(stored) if isinstance(stored, str) else stored
        if set(stored) != set(composition) or any(not math.isclose(stored[g], composition[g], abs_tol=1e-12) for g in stored):
            raise ValueError("Stored adsorbed composition contradicts molecule counts")
    if atoms.info.get("umof_composition_basis", "adsorbed_molecule_counts") != "adsorbed_molecule_counts":
        raise ValueError("Composition metadata must describe adsorbed molecule counts")
    prepared.info["REF_energy"] = energy
    if "REF_energy" in atoms.info and not math.isclose(atoms.info["REF_energy"], energy, abs_tol=1e-12):
        raise ValueError("Conflicting energy labels")
    if "REF_forces" in atoms.arrays and not np.array_equal(atoms.arrays["REF_forces"], forces):
        raise ValueError("Conflicting force labels")
    prepared.set_array("REF_forces", forces.copy())
    prepared.info.update(config_weight=1.0, config_energy_weight=1.0, config_forces_weight=1.0,
                         config_stress_weight=0.0, config_virials_weight=0.0)
    prepared.info["umof_source_file"] = source.name
    prepared.info["umof_source_split"] = split
    prepared.info["umof_guest_counts"] = json.dumps(guest_counts, sort_keys=True)
    prepared.info["umof_component_ids"] = json.dumps(sorted(guest_counts))
    prepared.info["umof_configuration_kind"] = kind
    prepared.info["umof_composition_basis"] = "adsorbed_molecule_counts"
    prepared.info["umof_assignment_source"] = "explicit_atom_assignment" if "umof_guest_assignments" in atoms.info else "legacy_connectivity_heuristic"
    prepared.info["head"] = "Default"
    prepared.info["umof_gas_composition"] = json.dumps(
        {
            gas: count / sum(guest_counts.values())
            for gas, count in sorted(guest_counts.items())
        },
        sort_keys=True,
    )
    return prepared


def prepare_finetuning_matrix(
    data_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    fractions: Sequence[float] = DEFAULT_TRAIN_FRACTIONS,
    seeds: Sequence[int] = (12345,),
    seed: int = 12345,
    validation_mofs: set[str] | None = None,
    test_mofs: set[str] | None = None,
    gases=None, gas_definitions=None, supported_elements=None,
) -> list[dict[str, Any]]:
    """Prepare reproducible, nested 10/25/50/100% datasets with shared splits."""
    if not fractions or any(
        not math.isfinite(float(fraction)) or not 0.0 < float(fraction) <= 1.0
        for fraction in fractions
    ):
        raise ValueError("All training fractions must be in (0, 1].")
    normalized_fractions = sorted({float(value) for value in fractions})
    normalized_seeds = sorted({int(value) for value in seeds})
    if not normalized_seeds:
        raise ValueError("At least one random seed is required.")
    manifests = []
    for current_seed in normalized_seeds:
        for fraction in normalized_fractions:
            manifests.append(
                prepare_adsorption_datasets(
                    data_directory,
                    Path(output_directory)
                    / f"{int(round(fraction * 100)):03d}pct"
                    / f"seed-{current_seed}"
                    / "datasets",
                    train_fraction=fraction,
                    seed=current_seed,
                    validation_mofs=validation_mofs,
                    test_mofs=test_mofs,
                    gases=gases, gas_definitions=gas_definitions, supported_elements=supported_elements,
                )
            )
    return manifests


def run_finetuning_matrix(data_directory=DEFAULT_DATA_DIRECTORY, checkpoint=None,
        output_directory=None, *, fractions=DEFAULT_TRAIN_FRACTIONS,
        seeds=(12345,), max_epochs=50, learning_rate=1e-4, replay_weight=.25,
        replay_sample_count=1000, seed=12345, device="cpu", validation_mofs=None,
        test_mofs=None, gases=None, gas_definitions=None, replay_source=None,
        replay_valid_sample_count=100, model="mh"):
    """Compatibility API routed through the manifest-checked offline lifecycle."""
    from .umof_workflow import prepare_campaign, execute_campaign
    from .umof_profiles import profile
    selected = profile(model)
    checkpoint = checkpoint if checkpoint is not None else selected.checkpoint
    output_directory = output_directory if output_directory is not None else selected.output_directory
    prepare_campaign(data_directory, checkpoint, output_directory, fractions=fractions,
        seeds=seeds, max_epochs=max_epochs, learning_rate=learning_rate,
        replay_weight=replay_weight, replay_samples=replay_sample_count, device=device,
        validation_mofs=validation_mofs, test_mofs=test_mofs, gases=gases,
        gas_definitions=gas_definitions, replay_source=replay_source,
        replay_valid_count=replay_valid_sample_count, model=model)
    return execute_campaign(output_directory, "train")


def _guest_molecule_counts(atoms: Any, definitions=None) -> dict[str, int]:
    from collections import Counter
    definitions = definitions or KNOWN_GAS_FORMULAS
    explicit = atoms.info.get("umof_guest_assignments")
    if explicit is not None:
        try:
            assignments = json.loads(explicit) if isinstance(explicit, str) else explicit
        except json.JSONDecodeError as exc:
            raise GuestAssignmentError("Malformed explicit molecule assignments") from exc
        if not isinstance(assignments, list) or any(not isinstance(row, dict) or
                set(row) != {"component_id", "atom_indices"} or not isinstance(row["atom_indices"], list)
                for row in assignments):
            raise GuestAssignmentError("Explicit assignments require component_id and atom_indices for each molecule")
        framework = atoms.info.get("umof_framework_indices")
        framework = json.loads(framework) if isinstance(framework, str) else framework
        if not isinstance(framework, list) or not framework:
            raise GuestAssignmentError("Explicit guests require framework atom indices")
        seen = set()
        for i in framework:
            if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(atoms) or i in seen:
                raise GuestAssignmentError("Invalid or duplicate framework atom assignment")
            seen.add(i)
        counts = Counter()
        for row in assignments:
            gas, indices = row["component_id"], row["atom_indices"]
            if gas not in definitions or not indices:
                raise GuestAssignmentError("Unknown or empty explicit component")
            for i in indices:
                if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(atoms) or i in seen:
                    raise GuestAssignmentError("Overlapping or invalid guest atom assignment")
                seen.add(i)
            if Counter(atoms[i].symbol for i in indices) != definitions[gas]:
                raise GuestAssignmentError("Explicit component formula disagrees with assigned atoms")
            counts[gas] += 1
        if seen != set(range(len(atoms))):
            raise GuestAssignmentError("Unassigned atoms in explicit component metadata")
    else:
        from ase.neighborlist import natural_cutoffs, neighbor_list
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        i, j = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=1.15), self_interaction=False)
        graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(atoms), len(atoms))).tocsr()
        n, labels = connected_components(graph, directed=False)
        counts = Counter()
        # Legacy MOF data: unmatched fragments must contain a metal framework atom.
        nonmetals = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr", "Te", "I", "Xe", "At", "Rn"}
        symbols = np.asarray(atoms.get_chemical_symbols())
        framework_fragments = []
        for label in range(n):
            formula = Counter(symbols[labels == label].tolist())
            matching = [gas for gas, expected in definitions.items() if formula == expected]
            if len(matching) > 1:
                raise GuestAssignmentError("Ambiguous gas formula; supply explicit atom assignments")
            if matching:
                counts[matching[0]] += 1
            elif not set(formula) - nonmetals:
                raise GuestAssignmentError(f"Unrecognized guest or disconnected organic fragment: {dict(formula)}")
            else:
                framework_fragments.append(formula)
        if len(framework_fragments) != 1:
            raise GuestAssignmentError("Ambiguous framework/metal-containing guest assignment; provide explicit atom assignments")
    stored = atoms.info.get("umof_guest_counts")
    if stored is not None:
        stored = json.loads(stored) if isinstance(stored, str) else stored
        if (not isinstance(stored, dict) or any(type(n) is not int or n <= 0 for n in stored.values())
                or stored != dict(counts)):
            raise ValueError("Stored molecule counts contradict atom assignments or are not positive integers")
    return dict(counts)


def main(argv: Sequence[str] | None = None) -> int:
    from .umof_workflow import main as workflow_main
    return workflow_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
