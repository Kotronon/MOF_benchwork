"""Host-guest interaction energies and forces for Module C."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from modules.module_c_mlips.models import InteractionBond, InteractionConfiguration
from modules.module_c_mlips.potential_backends.base import (
    PotentialBackend,
    PotentialResult,
)


@dataclass
class InteractionResult:
    """Combined and isolated evaluations for one host-guest configuration."""

    configuration_id: str
    backend: str
    interaction_energy_ev: float
    interaction_forces_ev_per_angstrom: list[list[float]]
    combined: PotentialResult
    framework: PotentialResult
    adsorbate: PotentialResult

    def __post_init__(self) -> None:
        if not isfinite(self.interaction_energy_ev):
            raise ValueError("interaction_energy_ev must be finite.")
        if len(self.interaction_forces_ev_per_angstrom) != len(
            self.combined.forces_ev_per_angstrom
        ):
            raise ValueError(
                "Interaction force count must match the combined force count."
            )
        for force in self.interaction_forces_ev_per_angstrom:
            if len(force) != 3:
                raise ValueError(
                    "Each interaction force vector must have 3 components."
                )
            if any(not isfinite(component) for component in force):
                raise ValueError(
                    "All interaction force components must be finite."
                )

    @property
    def runtime_seconds(self) -> float:
        return (
            self.combined.runtime_seconds
            + self.framework.runtime_seconds
            + self.adsorbate.runtime_seconds
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "backend": self.backend,
            "interaction_energy_ev": self.interaction_energy_ev,
            "interaction_forces_ev_per_angstrom": (
                self.interaction_forces_ev_per_angstrom
            ),
            "runtime_seconds": self.runtime_seconds,
            "combined": self.combined.to_dict(),
            "framework": self.framework.to_dict(),
            "adsorbate": self.adsorbate.to_dict(),
        }


def build_framework_configuration(
    configuration: InteractionConfiguration,
) -> InteractionConfiguration:
    """Return the isolated framework in the unchanged periodic cell."""
    return _build_component_configuration(
        configuration,
        selected_indices=configuration.framework_indices,
        component="framework",
    )


def build_adsorbate_configuration(
    configuration: InteractionConfiguration,
) -> InteractionConfiguration:
    """Return the isolated adsorbate with local atom and bond indices."""
    return _build_component_configuration(
        configuration,
        selected_indices=configuration.adsorbate_indices,
        component="adsorbate",
    )


def evaluate_interaction(
    configuration: InteractionConfiguration,
    backend: PotentialBackend,
) -> InteractionResult:
    """Evaluate combined and isolated systems and subtract their results."""
    framework_configuration = build_framework_configuration(configuration)
    adsorbate_configuration = build_adsorbate_configuration(configuration)

    combined_result = backend.evaluate(configuration)
    framework_result = backend.evaluate(framework_configuration)
    adsorbate_result = backend.evaluate(adsorbate_configuration)

    _validate_force_count(
        "combined",
        combined_result,
        len(configuration.atoms),
    )
    _validate_force_count(
        "framework",
        framework_result,
        len(framework_configuration.atoms),
    )
    _validate_force_count(
        "adsorbate",
        adsorbate_result,
        len(adsorbate_configuration.atoms),
    )

    interaction_energy_ev = (
        combined_result.energy_ev
        - framework_result.energy_ev
        - adsorbate_result.energy_ev
    )
    interaction_forces = [
        force.copy() for force in combined_result.forces_ev_per_angstrom
    ]

    _subtract_component_forces(
        interaction_forces,
        combined_result.forces_ev_per_angstrom,
        framework_result.forces_ev_per_angstrom,
        configuration.framework_indices,
    )
    _subtract_component_forces(
        interaction_forces,
        combined_result.forces_ev_per_angstrom,
        adsorbate_result.forces_ev_per_angstrom,
        configuration.adsorbate_indices,
    )

    return InteractionResult(
        configuration_id=configuration.configuration_id,
        backend=backend.name,
        interaction_energy_ev=interaction_energy_ev,
        interaction_forces_ev_per_angstrom=interaction_forces,
        combined=combined_result,
        framework=framework_result,
        adsorbate=adsorbate_result,
    )


def _build_component_configuration(
    configuration: InteractionConfiguration,
    *,
    selected_indices: list[int],
    component: str,
) -> InteractionConfiguration:
    selected_set = set(selected_indices)
    atoms = configuration.atoms[selected_indices].copy()
    atoms.set_cell(configuration.atoms.cell)
    atoms.set_pbc(configuration.atoms.pbc)
    index_map = {
        original_index: local_index
        for local_index, original_index in enumerate(selected_indices)
    }

    remapped_bonds = []
    for bond in configuration.bonds:
        atom1_selected = bond.atom1_index in selected_set
        atom2_selected = bond.atom2_index in selected_set
        if atom1_selected != atom2_selected:
            raise ValueError(
                "Bonds between framework and adsorbate are not supported."
            )
        if atom1_selected:
            remapped_bonds.append(
                InteractionBond(
                    atom1_index=index_map[bond.atom1_index],
                    atom2_index=index_map[bond.atom2_index],
                    forcefield_type=bond.forcefield_type,
                )
            )

    is_framework = component == "framework"
    return InteractionConfiguration(
        configuration_id=f"{configuration.configuration_id}_{component}",
        material=configuration.material,
        adsorbate=configuration.adsorbate,
        atoms=atoms,
        framework_indices=list(range(len(atoms))) if is_framework else [],
        adsorbate_indices=[] if is_framework else list(range(len(atoms))),
        region=configuration.region,
        source=configuration.source,
        bonds=remapped_bonds,
    )


def _validate_force_count(
    label: str,
    result: PotentialResult,
    expected_count: int,
) -> None:
    actual_count = len(result.forces_ev_per_angstrom)
    if actual_count != expected_count:
        raise ValueError(
            f"The {label} backend result contains {actual_count} force vectors; "
            f"expected {expected_count}."
        )


def _subtract_component_forces(
    interaction_forces: list[list[float]],
    combined_forces: list[list[float]],
    isolated_forces: list[list[float]],
    combined_indices: list[int],
) -> None:
    for local_index, combined_index in enumerate(combined_indices):
        interaction_forces[combined_index] = [
            combined_value - isolated_value
            for combined_value, isolated_value in zip(
                combined_forces[combined_index],
                isolated_forces[local_index],
                strict=True,
            )
        ]
