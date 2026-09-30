"""NequIP potential backend for fixed Module C configurations."""

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
class NequipBackend(PotentialBackend):
    """Evaluate structures with a deployed or compiled NequIP model."""

    model: str | Path
    backend_name: str = "nequip"
    device: str = "cpu"
    loader: str = "auto"
    species_to_type_name: dict[str, str] | bool | None = None
    energy_units_to_eV: float = 1.0
    length_units_to_A: float = 1.0
    energy_mode: str = "total_energy"
    calculator: Any | None = field(default=None, repr=False)
    _model_load_seconds: float = field(default=0.0, init=False, repr=False)

    @property
    def name(self) -> str:
        return self.backend_name

    def evaluate(
        self,
        configuration: InteractionConfiguration,
    ) -> PotentialResult:
        if self.energy_mode not in {"total_energy", "interaction_direct"}:
            raise ValueError(
                "NequIP energy_mode must be 'total_energy' or "
                "'interaction_direct'."
            )
        if self.energy_mode == "interaction_direct" and (
            not configuration.framework_indices
            or not configuration.adsorbate_indices
        ):
            return PotentialResult(
                energy_ev=0.0,
                forces_ev_per_angstrom=[
                    [0.0, 0.0, 0.0] for _ in configuration.atoms
                ],
                runtime_seconds=0.0,
                metadata={
                    "backend": self.name,
                    "configuration_id": configuration.configuration_id,
                    "energy_mode": self.energy_mode,
                    "baseline_short_circuit": True,
                },
            )
        calculator = self._get_calculator()
        atoms = configuration.atoms.copy()
        atoms.calc = calculator
        started = perf_counter()
        try:
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces().tolist()
        except Exception as exc:
            raise RuntimeError(
                "NequIP evaluation failed for configuration "
                f"{configuration.configuration_id!r}."
            ) from exc
        runtime = perf_counter() - started
        return PotentialResult(
            energy_ev=energy,
            forces_ev_per_angstrom=forces,
            runtime_seconds=runtime,
            metadata={
                "backend": self.name,
                "configuration_id": configuration.configuration_id,
                "model": str(self.model),
                "device": self.device,
                "loader": self.loader,
                "energy_mode": self.energy_mode,
                "atom_count": len(atoms),
                "model_load_seconds": self._model_load_seconds,
            },
        )

    def _get_calculator(self) -> Any:
        if self.calculator is not None:
            return self.calculator
        started = perf_counter()
        self.calculator = build_ase_calculator(
            {
                "backend": "nequip",
                "model": str(self.model),
                "device": self.device,
                "loader": self.loader,
                "species_to_type_name": self.species_to_type_name,
                "energy_units_to_eV": self.energy_units_to_eV,
                "length_units_to_A": self.length_units_to_A,
                "dispersion": False,
            }
        )
        self._model_load_seconds = perf_counter() - started
        return self.calculator
