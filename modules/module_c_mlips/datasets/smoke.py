"""Generate deterministic MOF/adsorbate configurations for smoke tests."""

from __future__ import annotations

from pathlib import Path

from modules.module_c_mlips.datasets.common import (
    combine_host_guest,
    load_host_guest_system,
)
from modules.module_c_mlips.models import InteractionConfiguration


def build_smoke_configurations(
    cif_path: str | Path,
    molecule_def_path: str | Path,
    pseudo_atoms_path: str | Path | None = None,
    *,
    material: str = "IRMOF-1",
    adsorbate: str = "CO2",
    cell_representation: str = "source",
    unit_cells: list[int] | tuple[int, int, int] = (1, 1, 1),
    cutoff_A: float | None = None,
    minimum_image_policy: str = "ignore",
) -> list[InteractionConfiguration]:
    """Build three deterministic configurations for pipeline validation.

    The generated labels describe construction only. They are not validated
    interaction regimes and must not be interpreted as physical reference data.
    """
    framework, molecule_atoms, guest_bonds = load_host_guest_system(
        cif_path,
        molecule_def_path,
        pseudo_atoms_path,
        cell_representation=cell_representation,
        unit_cells=unit_cells,
        cutoff_A=cutoff_A,
        minimum_image_policy=minimum_image_policy,
    )
    centers = [
        (
            "cell_center",
            "unclassified",
            framework.cell.cartesian_positions((0.5, 0.5, 0.5)),
        ),
        (
            "eighth_cell",
            "unclassified",
            framework.cell.cartesian_positions((0.125, 0.125, 0.125)),
        ),
        (
            "overlap_probe",
            "technical_repulsive_probe",
            framework.positions[0],
        ),
    ]

    configurations = []
    for label, region, center in centers:
        guest = molecule_atoms.copy()
        guest.translate(center - guest.get_center_of_mass())
        combined, framework_indices, adsorbate_indices, bonds = combine_host_guest(
            framework,
            guest,
            guest_bonds,
        )

        configurations.append(
            InteractionConfiguration(
                configuration_id=f"{material}_{adsorbate}_{label}",
                material=material,
                adsorbate=adsorbate,
                atoms=combined,
                framework_indices=framework_indices.copy(),
                adsorbate_indices=adsorbate_indices.copy(),
                region=region,
                source="generated_smoke",
                bonds=bonds,
                metadata={
                    "generator": "smoke",
                    "fractional_center": (
                        framework.cell.scaled_positions([center])[0].tolist()
                    ),
                },
            )
        )

    return configurations
