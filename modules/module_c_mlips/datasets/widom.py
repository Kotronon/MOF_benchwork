"""Generate deterministic random-insertion configurations for Module C."""

from __future__ import annotations

from math import cos, pi, sin, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from modules.module_c_mlips.datasets.common import (
    combine_host_guest,
    load_host_guest_system,
)
from modules.module_c_mlips.models import InteractionConfiguration


def build_widom_configurations(
    cif_path: str | Path,
    molecule_def_path: str | Path,
    pseudo_atoms_path: str | Path | None = None,
    *,
    material: str = "IRMOF-1",
    adsorbate: str = "CO2",
    sample_count: int = 100,
    seed: int = 12345,
    minimum_distance_A: float = 0.0,
    maximum_attempts_per_sample: int = 10_000,
    random_orientations: bool = True,
    cell_representation: str = "source",
    unit_cells: list[int] | tuple[int, int, int] = (1, 1, 1),
    cutoff_A: float | None = None,
    minimum_image_policy: str = "ignore",
) -> list[InteractionConfiguration]:
    """Build reproducible single-guest random insertions.

    Positions are uniform in fractional cell coordinates. Orientations are
    uniform on SO(3) when enabled. A positive ``minimum_distance_A`` performs
    rejection sampling and therefore creates a filtered interaction dataset,
    not an unbiased thermodynamic Widom estimator.
    """
    _validate_settings(
        sample_count,
        seed,
        minimum_distance_A,
        maximum_attempts_per_sample,
        random_orientations,
    )
    framework, molecule, guest_bonds = load_host_guest_system(
        cif_path,
        molecule_def_path,
        pseudo_atoms_path,
        cell_representation=cell_representation,
        unit_cells=unit_cells,
        cutoff_A=cutoff_A,
        minimum_image_policy=minimum_image_policy,
    )
    rng = np.random.default_rng(seed)
    centered_positions = molecule.positions - molecule.get_center_of_mass()
    configurations: list[InteractionConfiguration] = []
    rejected_total = 0

    for sample_index in range(sample_count):
        for attempt in range(1, maximum_attempts_per_sample + 1):
            fractional_center = rng.random(3)
            rotation = (
                _uniform_rotation_matrix(rng)
                if random_orientations
                else np.identity(3)
            )
            guest = molecule.copy()
            guest.positions = centered_positions @ rotation.T
            cartesian_center = framework.cell.cartesian_positions(
                fractional_center
            )
            guest.translate(cartesian_center)
            minimum_distance = _minimum_host_guest_distance(framework, guest)
            if minimum_distance >= minimum_distance_A:
                break
            rejected_total += 1
        else:
            raise RuntimeError(
                "Could not generate a random insertion satisfying "
                f"minimum_distance_A={minimum_distance_A:g} after "
                f"{maximum_attempts_per_sample} attempts for sample "
                f"{sample_index}."
            )

        combined, framework_indices, adsorbate_indices, bonds = combine_host_guest(
            framework,
            guest,
            guest_bonds,
        )
        configurations.append(
            InteractionConfiguration(
                configuration_id=(
                    f"{material}_{adsorbate}_widom_{sample_index:05d}"
                ),
                material=material,
                adsorbate=adsorbate,
                atoms=combined,
                framework_indices=framework_indices,
                adsorbate_indices=adsorbate_indices,
                region=_distance_region(minimum_distance),
                source="generated_widom",
                bonds=bonds,
                metadata={
                    "generator": "widom_random_insertion",
                    "sample_index": sample_index,
                    "seed": seed,
                    "fractional_center": fractional_center.tolist(),
                    "rotation_matrix": rotation.tolist(),
                    "minimum_host_guest_distance_A": minimum_distance,
                    "minimum_distance_filter_A": minimum_distance_A,
                    "attempts_for_sample": attempt,
                    "rejected_before_sample": rejected_total,
                    "random_orientations": random_orientations,
                    "thermodynamically_unbiased": minimum_distance_A == 0.0,
                },
            )
        )

    return configurations


def _minimum_host_guest_distance(framework: Any, guest: Any) -> float:
    from ase.geometry import get_distances

    _vectors, distances = get_distances(
        guest.positions,
        framework.positions,
        cell=framework.cell,
        pbc=framework.pbc,
    )
    return float(np.min(distances))


def _uniform_rotation_matrix(rng: Any) -> np.ndarray:
    """Return a uniformly distributed rotation matrix from a unit quaternion."""
    u1, u2, u3 = rng.random(3)
    qx = sqrt(1.0 - u1) * sin(2.0 * pi * u2)
    qy = sqrt(1.0 - u1) * cos(2.0 * pi * u2)
    qz = sqrt(u1) * sin(2.0 * pi * u3)
    qw = sqrt(u1) * cos(2.0 * pi * u3)
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _distance_region(distance_A: float) -> str:
    if distance_A < 1.5:
        return "repulsive"
    if distance_A < 2.5:
        return "near_contact"
    return "pore"


def _validate_settings(
    sample_count: int,
    seed: int,
    minimum_distance_A: float,
    maximum_attempts_per_sample: int,
    random_orientations: bool,
) -> None:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise ValueError("seed must be a positive integer.")
    if (
        isinstance(minimum_distance_A, bool)
        or not isinstance(minimum_distance_A, (int, float))
        or minimum_distance_A < 0.0
    ):
        raise ValueError("minimum_distance_A must be non-negative.")
    if (
        isinstance(maximum_attempts_per_sample, bool)
        or not isinstance(maximum_attempts_per_sample, int)
        or maximum_attempts_per_sample <= 0
    ):
        raise ValueError("maximum_attempts_per_sample must be positive.")
    if not isinstance(random_orientations, bool):
        raise ValueError("random_orientations must be a boolean.")
