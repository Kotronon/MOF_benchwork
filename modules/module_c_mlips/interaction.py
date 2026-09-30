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
    force_comparison_mode: str = "interaction"

    def __post_init__(self) -> None:
        if self.force_comparison_mode not in {"interaction", "combined_total"}:
            raise ValueError(
                "force_comparison_mode must be 'interaction' or "
                "'combined_total'."
            )
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

    @property
    def comparison_forces_ev_per_angstrom(self) -> list[list[float]]:
        """Return the force quantity required by the benchmark protocol."""
        if self.force_comparison_mode == "combined_total":
            return self.combined.forces_ev_per_angstrom
        return self.interaction_forces_ev_per_angstrom

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "backend": self.backend,
            "interaction_energy_ev": self.interaction_energy_ev,
            "interaction_forces_ev_per_angstrom": (
                self.interaction_forces_ev_per_angstrom
            ),
            "force_comparison_mode": self.force_comparison_mode,
            "comparison_forces_ev_per_angstrom": (
                self.comparison_forces_ev_per_angstrom
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
    *,
    framework_result: PotentialResult | None = None,
    force_comparison_mode: str = "interaction",
) -> InteractionResult:
    """Evaluate combined and isolated systems and subtract their results."""
    framework_configuration = build_framework_configuration(configuration)
    adsorbate_configuration = build_adsorbate_configuration(configuration)

    combined_result = backend.evaluate(configuration)
    if framework_result is None:
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
        force_comparison_mode=force_comparison_mode,
    )


def build_reference_interaction_result(
    configuration: InteractionConfiguration,
    *,
    backend_name: str,
) -> InteractionResult:
    """Build an immutable benchmark baseline from stored reference values."""
    energy = configuration.reference_interaction_energy_ev
    forces = configuration.reference_forces_ev_per_angstrom
    if energy is None or forces is None:
        raise ValueError(
            f"Configuration {configuration.configuration_id!r} does not "
            "contain reference energy and force values."
        )

    metadata = configuration.metadata
    total_energy = float(metadata.get("reference_total_energy_ev", energy))
    framework_energy = float(metadata.get("reference_framework_energy_ev", 0.0))
    adsorbate_energy = float(metadata.get("reference_adsorbate_energy_ev", 0.0))
    zero_framework_forces = [
        [0.0, 0.0, 0.0] for _ in configuration.framework_indices
    ]
    zero_adsorbate_forces = [
        [0.0, 0.0, 0.0] for _ in configuration.adsorbate_indices
    ]
    reference_metadata = {
        "source": configuration.source,
        "independent_reference": True,
        "configuration_id": configuration.configuration_id,
    }

    return InteractionResult(
        configuration_id=configuration.configuration_id,
        backend=backend_name,
        interaction_energy_ev=float(energy),
        interaction_forces_ev_per_angstrom=[force.copy() for force in forces],
        combined=PotentialResult(
            energy_ev=total_energy,
            forces_ev_per_angstrom=[force.copy() for force in forces],
            runtime_seconds=0.0,
            metadata=reference_metadata,
        ),
        framework=PotentialResult(
            energy_ev=framework_energy,
            forces_ev_per_angstrom=zero_framework_forces,
            runtime_seconds=0.0,
            metadata=reference_metadata,
        ),
        adsorbate=PotentialResult(
            energy_ev=adsorbate_energy,
            forces_ev_per_angstrom=zero_adsorbate_forces,
            runtime_seconds=0.0,
            metadata=reference_metadata,
        ),
        force_comparison_mode=str(
            metadata.get("reference_force_mode", "interaction")
        ),
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
        metadata=dict(configuration.metadata),
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
