"""Parity checks between Module A inputs and Module C's classical backend."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import shlex
import subprocess
from time import perf_counter
from typing import Any

from modules.module_c_mlips.potential_backends.base import PotentialResult
from modules.module_c_mlips.potential_backends.classical_lammps import (
    KCAL_PER_MOL_TO_EV,
    parse_force_dump,
    parse_run_zero_thermo,
)
from pipeline.lammps_inputs import render_run0_input


@dataclass(frozen=True)
class ClassicalParityReport:
    """Numerical differences for two evaluations of the same configuration."""

    passed: bool
    atom_count: int
    energy_difference_ev: float
    absolute_energy_difference_ev: float
    maximum_force_difference_ev_per_angstrom: float
    force_rmse_ev_per_angstrom: float
    energy_component_differences_ev: dict[str, float]
    energy_tolerance_ev: float
    force_tolerance_ev_per_angstrom: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "passed": self.passed,
            "atom_count": self.atom_count,
            "energy_difference_ev": self.energy_difference_ev,
            "absolute_energy_difference_ev": self.absolute_energy_difference_ev,
            "maximum_force_difference_ev_per_angstrom": (
                self.maximum_force_difference_ev_per_angstrom
            ),
            "force_rmse_ev_per_angstrom": self.force_rmse_ev_per_angstrom,
            "energy_component_differences_ev": self.energy_component_differences_ev,
            "energy_tolerance_ev": self.energy_tolerance_ev,
            "force_tolerance_ev_per_angstrom": (
                self.force_tolerance_ev_per_angstrom
            ),
        }


def evaluate_module_a_static_files(
    *,
    data_file: str | Path,
    forcefield_file: str | Path,
    atom_count: int,
    working_directory: str | Path,
    lammps_command: str = "lmp",
    has_bonds: bool = False,
) -> PotentialResult:
    """Evaluate fixed atoms using materialized Module-A data and force field files."""
    workdir = Path(working_directory)
    workdir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_file).resolve()
    forcefield_path = Path(forcefield_file).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"LAMMPS data file does not exist: {data_path}")
    if not forcefield_path.is_file():
        raise FileNotFoundError(
            f"LAMMPS force-field include does not exist: {forcefield_path}"
        )

    input_path = workdir / "in.module_a_run_zero"
    log_path = workdir / "log.module_a_run_zero"
    force_dump = workdir / "module_a_forces.dump"
    input_path.write_text(
        render_run0_input(
            framework_data_path=data_path,
            molecule_templates={},
            forcefield_path=forcefield_path,
            extra_special_per_atom=0,
            has_bonds=has_bonds,
            force_dump=force_dump.name,
        ),
        encoding="utf-8",
    )

    command = [
        *shlex.split(lammps_command),
        "-in",
        input_path.name,
        "-log",
        log_path.name,
    ]
    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"LAMMPS executable was not found: {command[0]!r}."
        ) from exc
    runtime_seconds = perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            "Module-A static LAMMPS evaluation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDERR:\n{completed.stderr}\n"
            f"STDOUT:\n{completed.stdout}"
        )

    thermo = parse_run_zero_thermo(log_path)
    forces_kcal_per_mol_angstrom = parse_force_dump(force_dump)
    if len(forces_kcal_per_mol_angstrom) != atom_count:
        raise RuntimeError(
            f"LAMMPS returned {len(forces_kcal_per_mol_angstrom)} force vectors; "
            f"expected {atom_count}."
        )

    return PotentialResult(
        energy_ev=thermo["potential_energy"] * KCAL_PER_MOL_TO_EV,
        forces_ev_per_angstrom=[
            [component * KCAL_PER_MOL_TO_EV for component in force]
            for force in forces_kcal_per_mol_angstrom
        ],
        runtime_seconds=runtime_seconds,
        metadata={
            "backend": "module_a_static_lammps",
            "atom_count": atom_count,
            "energy_kcal_per_mol": thermo["potential_energy"],
            "energy_components_kcal_per_mol": thermo,
            "lammps_command": command,
            "data_file": str(data_path),
            "forcefield_file": str(forcefield_path),
            "working_directory": str(workdir),
        },
    )


def compare_classical_results(
    reference: PotentialResult,
    candidate: PotentialResult,
    *,
    energy_tolerance_ev: float = 1e-6,
    force_tolerance_ev_per_angstrom: float = 1e-6,
) -> ClassicalParityReport:
    """Compare energy components and atomwise forces from two LAMMPS paths."""
    if energy_tolerance_ev < 0.0 or force_tolerance_ev_per_angstrom < 0.0:
        raise ValueError("Parity tolerances must be non-negative.")
    if len(reference.forces_ev_per_angstrom) != len(
        candidate.forces_ev_per_angstrom
    ):
        raise ValueError("Parity results must contain the same number of forces.")

    component_differences = _component_differences_ev(reference, candidate)
    force_differences = [
        candidate_component - reference_component
        for reference_force, candidate_force in zip(
            reference.forces_ev_per_angstrom,
            candidate.forces_ev_per_angstrom,
            strict=True,
        )
        for reference_component, candidate_component in zip(
            reference_force,
            candidate_force,
            strict=True,
        )
    ]
    maximum_force_difference = max(
        (abs(value) for value in force_differences),
        default=0.0,
    )
    force_rmse = (
        sqrt(sum(value * value for value in force_differences) / len(force_differences))
        if force_differences
        else 0.0
    )
    energy_difference = candidate.energy_ev - reference.energy_ev
    maximum_component_difference = max(
        (abs(value) for value in component_differences.values()),
        default=0.0,
    )
    passed = (
        abs(energy_difference) <= energy_tolerance_ev
        and maximum_component_difference <= energy_tolerance_ev
        and maximum_force_difference <= force_tolerance_ev_per_angstrom
    )
    return ClassicalParityReport(
        passed=passed,
        atom_count=len(reference.forces_ev_per_angstrom),
        energy_difference_ev=energy_difference,
        absolute_energy_difference_ev=abs(energy_difference),
        maximum_force_difference_ev_per_angstrom=maximum_force_difference,
        force_rmse_ev_per_angstrom=force_rmse,
        energy_component_differences_ev=component_differences,
        energy_tolerance_ev=energy_tolerance_ev,
        force_tolerance_ev_per_angstrom=force_tolerance_ev_per_angstrom,
    )


def _component_differences_ev(
    reference: PotentialResult,
    candidate: PotentialResult,
) -> dict[str, float]:
    reference_components = reference.metadata.get(
        "energy_components_kcal_per_mol",
        {},
    )
    candidate_components = candidate.metadata.get(
        "energy_components_kcal_per_mol",
        {},
    )
    reference_names = set(reference_components)
    candidate_names = set(candidate_components)
    if reference_names != candidate_names:
        raise ValueError(
            "Parity results must contain the same energy components. "
            f"Reference-only: {sorted(reference_names - candidate_names)}; "
            f"candidate-only: {sorted(candidate_names - reference_names)}."
        )
    return {
        component: (
            float(candidate_components[component])
            - float(reference_components[component])
        )
        * KCAL_PER_MOL_TO_EV
        for component in sorted(reference_names)
    }
