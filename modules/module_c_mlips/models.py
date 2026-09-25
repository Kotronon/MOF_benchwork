from __future__ import annotations

from dataclasses import dataclass, field

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ase import Atoms


@dataclass
class InteractionConfiguration:
    configuration_id: str
    material: str
    adsorbate: str
    atoms: Atoms
    framework_indices: list[int]
    adsorbate_indices: list[int]
    reference_interaction_energy_ev: float | None = None
    reference_forces_ev_per_angstrom: list[list[float]] | None = None
    region: str | None = None
    source: str = "generated"
    energy_unit: str = "eV"
    force_unit: str = "eV/angstrom"
    bonds: list[InteractionBond] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        framework = set(self.framework_indices)
        adsorbate = set(self.adsorbate_indices)
        all_indices = framework | adsorbate

        if framework & adsorbate:
            raise ValueError(
                "Framework and adsorbate indices must not overlap."
            )

        expected = set(range(len(self.atoms)))
        if all_indices != expected:
            raise ValueError(
                "Framework and adsorbate indices must cover every atom."
            )

        if self.reference_forces_ev_per_angstrom is not None:
            if len(self.reference_forces_ev_per_angstrom) != len(self.atoms):
                raise ValueError(
                    "Reference force count must match the atom count."
                )

        for bond in self.bonds:
            if bond.atom1_index == bond.atom2_index:
                raise ValueError("A bond must connect two different atoms.")
            if not 0 <= bond.atom1_index < len(self.atoms):
                raise ValueError("Bond atom1_index is outside the atom range.")
            if not 0 <= bond.atom2_index < len(self.atoms):
                raise ValueError("Bond atom2_index is outside the atom range.")

@dataclass(frozen=True)
class InteractionBond:
    """Represents a bond between two atoms in an interaction configuration."""

    atom1_index: int
    atom2_index: int
    forcefield_type: str | None = None
