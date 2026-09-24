"""Common interface for potential-energy backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modules.module_c_mlips.models import InteractionConfiguration


@dataclass
class PotentialResult:
    """Result of a potential-energy evaluation."""

    energy_ev: float
    forces_ev_per_angstrom: list[list[float]]
    runtime_seconds: float
    stress_ev_per_angstrom_cubed: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isfinite(self.energy_ev):
            raise ValueError("energy_ev must be finite.")
        if not isfinite(self.runtime_seconds):
            raise ValueError("runtime_seconds must be finite.")
        if self.runtime_seconds < 0:
            raise ValueError("runtime_seconds must be non-negative.")

        for force in self.forces_ev_per_angstrom:
            if len(force) != 3:
                raise ValueError("Each force vector must have 3 components.")
            if any(not isfinite(component) for component in force):
                raise ValueError("All force components must be finite.")

        if self.stress_ev_per_angstrom_cubed is not None:
            if len(self.stress_ev_per_angstrom_cubed) != 6:
                raise ValueError(
                    "Stress must use the six-component Voigt representation."
                )
            if any(
                not isfinite(component)
                for component in self.stress_ev_per_angstrom_cubed
            ):
                raise ValueError("All stress components must be finite.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "energy_ev": self.energy_ev,
            "forces_ev_per_angstrom": self.forces_ev_per_angstrom,
            "runtime_seconds": self.runtime_seconds,
            "stress_ev_per_angstrom_cubed": self.stress_ev_per_angstrom_cubed,
            "metadata": self.metadata,
        }


class PotentialBackend(ABC):
    """Interface implemented by all classical and MLIP backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a stable backend identifier."""

    @abstractmethod
    def evaluate(self, configuration: InteractionConfiguration) -> PotentialResult:
        """Calculate total energy and atomic forces for a structure."""
