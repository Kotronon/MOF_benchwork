"""MACE fine-tuning preparation and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np

from modules.module_c_mlips.potential_backends.calculators import build_ase_calculator
from pipeline.config import save_benchmark_data


def write_mace_datasets(
    labels: Sequence[dict[str, Any]],
    *,
    training_ids: set[str],
    validation_ids: set[str],
    test_ids: set[str],
    output_directory: str | Path,
) -> dict[str, str]:
    """Write disjoint MACE extxyz datasets from parsed CP2K labels."""
    overlaps = {
        "training/validation": training_ids & validation_ids,
        "training/test": training_ids & test_ids,
        "validation/test": validation_ids & test_ids,
    }
    invalid = {
        name: sorted(values) for name, values in overlaps.items() if values
    }
    if invalid:
        raise ValueError(
            "Active-learning dataset IDs overlap: " + repr(invalid)
        )
    try:
        from ase.io import read, write
    except ImportError as exc:
        raise ImportError("ASE is required to build MACE datasets.") from exc
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output / "train.extxyz",
        "valid": output / "valid.extxyz",
        "test": output / "test.extxyz",
    }
    grouped = {"train": [], "valid": [], "test": []}
    for label in labels:
        candidate_id = str(label["candidate_id"])
        split = (
            "train"
            if candidate_id in training_ids
            else "valid"
            if candidate_id in validation_ids
            else "test"
            if candidate_id in test_ids
            else None
        )
        if split is None:
            continue
        atoms = read(str(label["structure_path"]))
        forces = np.asarray(label["forces_ev_per_A"], dtype=float)
        if forces.shape != (len(atoms), 3):
            raise ValueError(f"Force shape does not match {candidate_id}.")
        atoms.info["REF_energy"] = float(label["energy_ev"])
        atoms.info["candidate_id"] = candidate_id
        atoms.new_array("REF_forces", forces)
        stress = label.get("stress_GPa")
        if stress is not None:
            atoms.info["REF_stress_GPa"] = stress
        grouped[split].append(atoms)
    if any(not grouped[split] for split in ("train", "valid", "test")):
        raise ValueError(
            "Active-learning train, validation, and test datasets must all "
            "be non-empty."
        )
    for split, atoms_list in grouped.items():
        write(paths[split], atoms_list, format="extxyz")
    manifest = {
        "train": str(paths["train"]),
        "valid": str(paths["valid"]),
        "test": str(paths["test"]),
        "training_ids": sorted(training_ids),
        "validation_ids": sorted(validation_ids),
        "test_ids": sorted(test_ids),
    }
    save_benchmark_data(output / "dataset_manifest.json", manifest)
    return {key: str(value) for key, value in paths.items()}


def build_mace_finetune_commands(
    datasets: dict[str, str],
    output_directory: str | Path,
    *,
    committee_size: int,
    base_model: str,
    device: str,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Build explicit MACE commands for a reproducible committee."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    effective_seeds = list(seeds or (12345, 23456, 34567))
    if len(effective_seeds) < committee_size:
        raise ValueError("Not enough unique seeds for the MACE committee.")
    commands = []
    foundation = "small" if base_model == "mace_mp_0a_small" else base_model
    for index in range(committee_size):
        name = f"mace_adsorption_committee_{index}"
        member_dir = output / name
        model_dir = member_dir / "models"
        command = [
            "mace_run_train",
            "--name", name,
            "--train_file", datasets["train"],
            "--valid_file", datasets["valid"],
            "--foundation_model", foundation,
            "--energy_key", "REF_energy",
            "--forces_key", "REF_forces",
            "--E0s", "average",
            "--default_dtype", "float64",
            "--device", device,
            "--seed", str(effective_seeds[index]),
            "--model_dir", str(model_dir),
            "--checkpoints_dir", str(member_dir / "checkpoints"),
            "--results_dir", str(member_dir / "results"),
            "--logs_dir", str(member_dir / "logs"),
        ]
        commands.append(
            {
                "member": index,
                "seed": effective_seeds[index],
                "command": command,
                "working_directory": str(member_dir),
                "model_directory": str(model_dir),
                "status": "prepared",
            }
        )
    save_benchmark_data(output / "training_commands.json", {"commands": commands})
    return commands


def execute_mace_commands(commands: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for entry in commands:
        current = dict(entry)
        workdir = Path(current["working_directory"])
        workdir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(current["command"], capture_output=True, text=True)
        current["returncode"] = completed.returncode
        current["status"] = "completed" if completed.returncode == 0 else "failed"
        log_path = workdir / "training.log"
        log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        current["log_path"] = str(log_path)
        models = sorted(Path(current["model_directory"]).glob("*.model"))
        current["model_path"] = str(models[-1]) if models else None
        if completed.returncode != 0 or not models:
            raise RuntimeError(f"MACE fine-tuning failed; see {log_path}.")
        results.append(current)
    return results


def classify_validation_metrics(
    metrics: dict[str, float],
    settings: Any,
) -> dict[str, Any]:
    energy = float(metrics["energy_mae_ev"])
    force = float(metrics["force_mae_ev_per_A"])
    disagreement_energy = float(metrics.get("maximum_committee_energy_std_ev", 0.0))
    disagreement_force = float(metrics.get("maximum_committee_force_std_ev_per_A", 0.0))
    passed = (
        energy <= settings.energy_pass_mae_ev
        and force <= settings.force_pass_mae_ev_per_A
        and disagreement_energy <= settings.energy_pass_mae_ev
        and disagreement_force <= settings.force_pass_mae_ev_per_A
    )
    warning = (
        energy <= settings.energy_warning_mae_ev
        and force <= settings.force_warning_mae_ev_per_A
    )
    return {
        "status": "passed" if passed else "warning" if warning else "failed",
        "metrics": metrics,
        "thresholds": {
            "energy_pass_mae_ev": settings.energy_pass_mae_ev,
            "force_pass_mae_ev_per_A": settings.force_pass_mae_ev_per_A,
            "energy_warning_mae_ev": settings.energy_warning_mae_ev,
            "force_warning_mae_ev_per_A": settings.force_warning_mae_ev_per_A,
        },
    }


def evaluate_mace_committee(
    model_paths: Sequence[str],
    labels: Sequence[dict[str, Any]],
    *,
    test_ids: set[str],
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a MACE committee on the permanently held-out DFT split."""
    if not model_paths:
        raise ValueError("No committee models were provided.")
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError("ASE is required for committee validation.") from exc

    calculators = [
        build_ase_calculator(
            {
                "backend": "mace-torch",
                "model": str(path),
                "device": device,
                "default_dtype": "float64",
                "dispersion": False,
            }
        )
        for path in model_paths
    ]
    energy_errors = []
    force_errors = []
    energy_stds = []
    force_stds = []
    member_energy_errors: list[list[float]] = [
        [] for _ in calculators
    ]
    member_force_errors: list[list[float]] = [
        [] for _ in calculators
    ]
    evaluated = 0
    for label in labels:
        if str(label["candidate_id"]) not in test_ids:
            continue
        atoms = read(str(label["structure_path"]))
        energies = []
        forces = []
        for calculator in calculators:
            current = atoms.copy()
            current.calc = calculator
            energies.append(float(current.get_potential_energy()))
            forces.append(np.asarray(current.get_forces(), dtype=float))
        reference_forces = np.asarray(label["forces_ev_per_A"], dtype=float)
        mean_forces = np.mean(np.asarray(forces), axis=0)
        energy_errors.append(abs(float(np.mean(energies)) - float(label["energy_ev"])))
        force_errors.extend(np.linalg.norm(mean_forces - reference_forces, axis=1).tolist())
        for index, (energy, predicted_forces) in enumerate(
            zip(energies, forces, strict=True)
        ):
            member_energy_errors[index].append(
                abs(float(energy) - float(label["energy_ev"]))
            )
            member_force_errors[index].extend(
                np.linalg.norm(
                    predicted_forces - reference_forces,
                    axis=1,
                ).tolist()
            )
        energy_stds.append(float(np.std(energies, ddof=0)))
        force_stds.append(
            float(np.max(np.linalg.norm(np.std(np.asarray(forces), axis=0), axis=1)))
        )
        evaluated += 1
    if evaluated != len(test_ids):
        raise ValueError(
            f"Committee validation found {evaluated} of {len(test_ids)} test labels."
        )
    members = [
        {
            "member": index,
            "model_path": str(model_paths[index]),
            "energy_mae_ev": float(np.mean(member_energy_errors[index])),
            "force_mae_ev_per_A": float(np.mean(member_force_errors[index])),
        }
        for index in range(len(model_paths))
    ]
    best = min(
        members,
        key=lambda item: (
            item["energy_mae_ev"] / 0.043
            + item["force_mae_ev_per_A"] / 0.10
        ),
    )
    return {
        "test_configuration_count": float(evaluated),
        "energy_mae_ev": float(np.mean(energy_errors)),
        "force_mae_ev_per_A": float(np.mean(force_errors)),
        "maximum_committee_energy_std_ev": float(max(energy_stds, default=0.0)),
        "maximum_committee_force_std_ev_per_A": float(max(force_stds, default=0.0)),
        "committee_members": members,
        "best_model_index": int(best["member"]),
    }


def select_committee_candidates(
    candidates: Sequence[dict[str, Any]],
    model_paths: Sequence[str],
    *,
    count: int,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Select uncertain and structurally diverse candidates for DFT labeling."""
    if count <= 0 or count > len(candidates):
        raise ValueError("Candidate selection count is outside the available pool.")
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError("ASE is required for committee selection.") from exc
    calculators = [
        build_ase_calculator(
            {
                "backend": "mace-torch",
                "model": str(path),
                "device": device,
                "default_dtype": "float64",
                "dispersion": False,
            }
        )
        for path in model_paths
    ]
    scored = []
    for candidate in candidates:
        atoms = read(str(candidate["structure_path"]))
        energies = []
        forces = []
        for calculator in calculators:
            current = atoms.copy()
            current.calc = calculator
            energies.append(float(current.get_potential_energy()))
            forces.append(np.asarray(current.get_forces(), dtype=float))
        energy_std = float(np.std(energies, ddof=0))
        force_std = float(
            np.max(np.linalg.norm(np.std(np.asarray(forces), axis=0), axis=1))
        )
        descriptor = _selection_descriptor(atoms)
        scored.append((candidate, energy_std, force_std, descriptor))

    uncertainties = np.asarray(
        [energy + force for _, energy, force, _ in scored], dtype=float
    )
    scale = float(np.ptp(uncertainties))
    normalized = (
        np.ones_like(uncertainties)
        if scale <= 1.0e-15
        else (uncertainties - float(np.min(uncertainties))) / scale
    )
    selected: list[int] = [int(np.argmax(normalized))]
    descriptors = np.asarray([item[3] for item in scored], dtype=float)
    while len(selected) < count:
        distances = np.min(
            np.linalg.norm(descriptors[:, None, :] - descriptors[selected][None, :, :], axis=2),
            axis=1,
        )
        distance_scale = float(np.max(distances))
        if distance_scale > 0.0:
            distances /= distance_scale
        objective = 0.7 * normalized + 0.3 * distances
        objective[selected] = -1.0
        selected.append(int(np.argmax(objective)))

    result = []
    for index in selected:
        candidate, energy_std, force_std, _ = scored[index]
        result.append(
            {
                **candidate,
                "selection": {
                    "committee_energy_std_ev": energy_std,
                    "committee_force_std_ev_per_A": force_std,
                    "strategy": "committee_disagreement_plus_diversity",
                },
            }
        )
    return result


def load_labels(label_root: str | Path) -> list[dict[str, Any]]:
    labels = []
    for path in sorted(Path(label_root).glob("iteration_*/cp2k/*/label.json")):
        labels.append(json.loads(path.read_text(encoding="utf-8")))
    return labels


def _selection_descriptor(atoms: Any) -> np.ndarray:
    distances = atoms.get_all_distances(mic=True)
    upper = distances[np.triu_indices(len(atoms), 1)]
    histogram, _ = np.histogram(upper, bins=24, range=(0.0, 12.0))
    descriptor = histogram.astype(float)
    if descriptor.sum() > 0:
        descriptor /= descriptor.sum()
    return np.concatenate(
        [descriptor, [len(atoms) / 1000.0, atoms.get_volume() / max(len(atoms), 1)]]
    )
