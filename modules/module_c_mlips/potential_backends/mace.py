"""MACE potential backend for Module C energy and force evaluations.
https://github.com/acesuit/mace

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from modules.module_c_mlips.potential_backends.base import (
    PotentialBackend,
    PotentialResult,
)
from modules.module_c_mlips.potential_backends.calculators import (
    build_ase_calculator,
)

if TYPE_CHECKING:
    from modules.module_c_mlips.models import InteractionConfiguration


@dataclass
class MaceBackend(PotentialBackend):
    """Evaluate fixed structures directly with a MACE-MP ASE calculator."""

    backend_name: str = "mace_mp"
    model: str | Path = "medium"
    device: str = "cpu"
    default_dtype: str = "float64"
    dispersion: bool = False
    calculator: Any | None = field(default=None, repr=False)
    _model_load_seconds: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.backend_name.strip():
            raise ValueError("backend_name must not be empty.")
        if not self.device:
            raise ValueError("device must not be empty.")
        if self.default_dtype not in {"float32", "float64"}:
            raise ValueError("default_dtype must be 'float32' or 'float64'.")

    @property
    def name(self) -> str:
        """Return a stable backend identifier."""
        return self.backend_name

    def evaluate(
        self,
        configuration: InteractionConfiguration,
    ) -> PotentialResult:
        """Calculate total energy and atomic forces for one configuration."""
        calculator = self._get_calculator()
        atoms = configuration.atoms.copy()
        atoms.calc = calculator

        started = perf_counter()
        try:
            energy_ev = float(atoms.get_potential_energy())
            forces = atoms.get_forces().tolist()
        except Exception as exc:
            raise RuntimeError(
                "MACE evaluation failed for configuration "
                f"{configuration.configuration_id!r}."
            ) from exc
        runtime_seconds = perf_counter() - started

        return PotentialResult(
            energy_ev=energy_ev,
            forces_ev_per_angstrom=forces,
            runtime_seconds=runtime_seconds,
            metadata={
                "backend": self.name,
                "configuration_id": configuration.configuration_id,
                "model": str(self.model),
                "device": self.device,
                "default_dtype": self.default_dtype,
                "dispersion": self.dispersion,
                "atom_count": len(atoms),
                "model_load_seconds": self._model_load_seconds,
            },
        )

    def _get_calculator(self) -> Any:
        """Return the cached calculator, loading MACE-MP only when needed."""
        if self.calculator is not None:
            return self.calculator

        started = perf_counter()
        self.calculator = build_ase_calculator(
            {
                "backend": "mace-torch",
                "model": self.model,
                "device": self.device,
                "dispersion": self.dispersion,
                "default_dtype": self.default_dtype,
            }
        )
        self._model_load_seconds = perf_counter() - started
        return self.calculator
