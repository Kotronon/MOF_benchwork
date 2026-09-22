from dataclasses import dataclass
from pathlib import Path

from .base import PotentialBackend, PotentialResult

@dataclass
class ClassicalLAMMPSBackend(PotentialBackend):
    """Backend for evaluating potentials using LAMMPS."""

    lammps_command: str
    forcefield_file: Path
    working_directory: Path
    keep_working_directory: bool = False
    
    @property
    def name(self) -> str:
        return "uff_ddec_lammps"

    def evaluate(self, atoms) -> PotentialResult:
       raise NotImplementedError(
        "LAMMPS run-0 evaluation is not implemented yet."
        )