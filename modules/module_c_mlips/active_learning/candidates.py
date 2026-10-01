"""Deterministic adsorption-configuration generation and diversity selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import save_benchmark_data


def generate_candidate_pool(
    framework: Any,
    adsorbate: Any,
    output_directory: str | Path,
    *,
    count: int,
    seed: int,
    fractions: dict[str, float],
    iteration: int = 0,
) -> list[dict[str, Any]]:
    """Generate a larger deterministic pool and retain diverse configurations."""
    if count <= 0:
        raise ValueError("Candidate count must be positive.")
    try:
        from ase.io import write
    except ImportError as exc:
        raise ImportError("ASE is required for active-learning candidates.") from exc

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + iteration * 100_003)
    quotas = _allocate_counts(count, fractions)
    generated: list[tuple[Any, str, int]] = []
    for candidate_type, quota in quotas.items():
        pool_size = max(quota, quota * 4)
        local_pool = []
        for local_index in range(pool_size):
            if candidate_type == "framework":
                atoms = _perturb_framework(framework, rng, local_index)
                adsorbate_count = 0
            elif candidate_type == "single_adsorbate":
                atoms = _host_guest_configuration(framework, adsorbate, rng, 1)
                adsorbate_count = 1
            else:
                adsorbate_count = (2, 4, 8)[local_index % 3]
                atoms = _host_guest_configuration(
                    framework,
                    adsorbate,
                    rng,
                    adsorbate_count,
                )
            local_pool.append((atoms, candidate_type, adsorbate_count))
        descriptors = [_configuration_descriptor(item[0], len(framework)) for item in local_pool]
        selected = _farthest_point_indices(descriptors, quota)
        generated.extend(local_pool[index] for index in selected)

    records = []
    for index, (atoms, candidate_type, adsorbate_count) in enumerate(generated):
        candidate_id = f"iter{iteration:02d}_{candidate_type}_{index:04d}"
        path = output / f"{candidate_id}.extxyz"
        atoms.info.update(
            {
                "candidate_id": candidate_id,
                "candidate_type": candidate_type,
                "adsorbate_count": adsorbate_count,
                "framework_atom_count": len(framework),
                "iteration": iteration,
            }
        )
        write(path, atoms, format="extxyz")
        records.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": candidate_type,
                "adsorbate_count": adsorbate_count,
                "framework_atom_count": len(framework),
                "iteration": iteration,
                "structure_path": str(path),
                "label_status": "pending",
            }
        )
    save_benchmark_data(output / "candidate_manifest.json", {"candidates": records})
    return records


def load_candidate_records(root: str | Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(Path(root).glob("iteration_*/**/candidate_manifest.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records.extend(data.get("candidates", []))
    return records


def _allocate_counts(count: int, fractions: dict[str, float]) -> dict[str, int]:
    keys = ("framework", "single_adsorbate", "multiple_adsorbates")
    raw = {key: count * float(fractions[key]) for key in keys}
    allocated = {key: int(np.floor(value)) for key, value in raw.items()}
    remaining = count - sum(allocated.values())
    order = sorted(keys, key=lambda key: raw[key] - allocated[key], reverse=True)
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def _perturb_framework(framework: Any, rng: Any, index: int) -> Any:
    atoms = framework.copy()
    sigma = (0.01, 0.03, 0.06, 0.10)[index % 4]
    atoms.positions += rng.normal(0.0, sigma, size=(len(atoms), 3))
    atoms.wrap()
    return atoms


def _host_guest_configuration(
    framework: Any,
    adsorbate: Any,
    rng: Any,
    molecule_count: int,
) -> Any:
    combined = framework.copy()
    for _ in range(molecule_count):
        oriented_guest = adsorbate.copy()
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        oriented_guest.rotate(float(rng.uniform(0.0, 360.0)), axis, center="COM")
        placed = False
        for _attempt in range(2_000):
            guest = oriented_guest.copy()
            fractional = rng.random(3)
            target = fractional @ framework.cell.array
            guest.translate(target - guest.get_center_of_mass())
            trial = combined.copy()
            trial += guest
            trial.set_cell(framework.cell)
            trial.set_pbc(True)
            distances = trial.get_all_distances(mic=True)
            start = len(trial) - len(guest)
            cross = distances[:start, start:]
            if cross.size == 0 or float(np.min(cross)) >= 1.2:
                combined = trial
                placed = True
                break
        if not placed:
            raise RuntimeError("Could not place an adsorbate without a hard overlap.")
    return combined


def _configuration_descriptor(atoms: Any, framework_atom_count: int) -> np.ndarray:
    distances = atoms.get_all_distances(mic=True)
    upper = distances[np.triu_indices(len(atoms), 1)]
    histogram, _ = np.histogram(upper, bins=24, range=(0.0, 12.0), density=False)
    histogram = histogram.astype(float)
    if histogram.sum() > 0:
        histogram /= histogram.sum()
    guest_fraction = max(0, len(atoms) - framework_atom_count) / max(len(atoms), 1)
    return np.concatenate([histogram, [guest_fraction, atoms.get_volume() / max(len(atoms), 1)]])


def _farthest_point_indices(descriptors: list[np.ndarray], count: int) -> list[int]:
    if count >= len(descriptors):
        return list(range(len(descriptors)))
    matrix = np.asarray(descriptors, dtype=float)
    selected = [int(np.argmax(np.linalg.norm(matrix - matrix.mean(axis=0), axis=1)))]
    minimum_distances = np.linalg.norm(matrix - matrix[selected[0]], axis=1)
    while len(selected) < count:
        next_index = int(np.argmax(minimum_distances))
        selected.append(next_index)
        distances = np.linalg.norm(matrix - matrix[next_index], axis=1)
        minimum_distances = np.minimum(minimum_distances, distances)
        minimum_distances[selected] = -1.0
    return selected
