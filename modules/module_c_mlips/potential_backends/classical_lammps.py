"""Classical LAMMPS backend for Module C energy and force evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import tempfile
from time import perf_counter
from typing import TYPE_CHECKING

from converter.forcefield_to_lammps import (
    build_lammps_forcefield,
    load_forcefield_parameters,
    write_lammps_forcefield_include,
)
from modules.module_c_mlips.potential_backends.base import (
    PotentialBackend,
    PotentialResult,
)
from modules.module_c_mlips.potential_backends.lammps_structure import (
    build_lammps_structure,
    write_lammps_data,
)
from parsers.lammps_log_parser import get_log, parse_lammps_log

if TYPE_CHECKING:
    from modules.module_c_mlips.models import InteractionConfiguration


KCAL_PER_MOL_TO_EV = 0.0433641153087705


@dataclass
class ClassicalLAMMPSBackend(PotentialBackend):
    """Evaluate one interaction configuration using UFF/DDEC in LAMMPS."""

    lammps_command: str
    pseudo_atoms_file: Path
    mixing_rules_file: Path
    working_directory: Path | None = None
    keep_working_directory: bool = False
    pair_style: str = "lj/cut/coul/long 12.8"
    kspace_style: str = "pppm 1e-5"

    @property
    def name(self) -> str:
        return "uff_ddec_lammps"

    def evaluate(
        self,
        configuration: InteractionConfiguration,
    ) -> PotentialResult:
        """Run LAMMPS at fixed coordinates and return energy and forces."""
        if self.working_directory is not None:
            workdir = Path(self.working_directory) / _safe_directory_name(
                configuration.configuration_id
            )
            workdir.mkdir(parents=True, exist_ok=True)
            return self._evaluate_in_directory(configuration, workdir)

        if self.keep_working_directory:
            workdir = Path(tempfile.mkdtemp(prefix="mof_module_c_"))
            return self._evaluate_in_directory(configuration, workdir)

        with tempfile.TemporaryDirectory(prefix="mof_module_c_") as tmpdir:
            return self._evaluate_in_directory(configuration, Path(tmpdir))

    def _evaluate_in_directory(
        self,
        configuration: InteractionConfiguration,
        workdir: Path,
    ) -> PotentialResult:
        parameters = load_forcefield_parameters(
            {
                "files": {
                    "pseudo_atoms": str(self.pseudo_atoms_file),
                    "mixing_rules": str(self.mixing_rules_file),
                }
            }
        )

        forcefield_types = configuration.atoms.arrays[
            "forcefield_type"
        ].tolist()
        framework_types = [
            forcefield_types[index]
            for index in configuration.framework_indices
        ]
        adsorbate_types = [
            forcefield_types[index]
            for index in configuration.adsorbate_indices
        ]
        forcefield = build_lammps_forcefield(
            framework_symbols=framework_types,
            adsorbate_atom_types=adsorbate_types,
            parameters=parameters,
            pair_style=self.pair_style,
            kspace_style=self.kspace_style,
        )
        type_ids = {
            atom_type.label: atom_type.type_id
            for atom_type in forcefield.atom_types
        }
        structure = build_lammps_structure(
            configuration.atoms,
            configuration.framework_indices,
            configuration.adsorbate_indices,
            type_ids,
            configuration.bonds,
        )

        data_path = write_lammps_data(structure, workdir / "combined.data")
        forcefield_path = write_lammps_forcefield_include(
            forcefield,
            workdir / "forcefield.inc",
        )
        input_path = workdir / "in.run_zero"
        input_path.write_text(
            render_run_zero_input(
                data_file=data_path.name,
                forcefield_file=forcefield_path.name,
                force_dump="forces.dump",
                has_bonds=bool(configuration.bonds),
            ),
            encoding="utf-8",
        )

        log_path = workdir / "log.lammps"
        command = [
            *shlex.split(self.lammps_command),
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
                "LAMMPS run-0 evaluation failed.\n"
                f"Command: {' '.join(command)}\n"
                f"STDERR:\n{completed.stderr}\n"
                f"STDOUT:\n{completed.stdout}"
            )

        thermo = parse_run_zero_thermo(log_path)
        energy_kcal_per_mol = thermo["potential_energy"]
        raw_forces = parse_force_dump(workdir / "forces.dump")
        if len(raw_forces) != len(configuration.atoms):
            raise RuntimeError(
                f"LAMMPS returned {len(raw_forces)} force vectors for "
                f"{len(configuration.atoms)} atoms."
            )

        return PotentialResult(
            energy_ev=energy_kcal_per_mol * KCAL_PER_MOL_TO_EV,
            forces_ev_per_angstrom=[
                [component * KCAL_PER_MOL_TO_EV for component in force]
                for force in raw_forces
            ],
            runtime_seconds=runtime_seconds,
            metadata={
                "backend": self.name,
                "configuration_id": configuration.configuration_id,
                "lammps_command": command,
                "pair_style": self.pair_style,
                "kspace_style": self.kspace_style,
                "atom_count": len(configuration.atoms),
                "energy_kcal_per_mol": energy_kcal_per_mol,
                "energy_components_kcal_per_mol": thermo,
                "working_directory": str(workdir),
            },
        )


def render_run_zero_input(
    *,
    data_file: str = "combined.data",
    forcefield_file: str = "forcefield.inc",
    force_dump: str = "forces.dump",
    has_bonds: bool = True,
) -> str:
    """Render a LAMMPS input for one fixed-coordinate evaluation."""
    lines = ["units real", "atom_style full", "boundary p p p"]
    if has_bonds:
        lines.append("bond_style zero")
    lines.append(f"read_data {data_file}")
    if has_bonds:
        lines.append("bond_coeff *")
    lines.extend(
        [
            "special_bonds lj/coul 0.0 0.0 0.0",
            f"include {forcefield_file}",
            "thermo 1",
            "thermo_style custom step atoms pe evdwl ecoul elong",
            (
                f"dump force_output all custom 1 {force_dump} "
                "id type fx fy fz"
            ),
            "dump_modify force_output sort id",
            "run 0",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_run_zero_energy(log_file: str | Path) -> float:
    """Return the last run-0 potential energy in kcal/mol."""
    return parse_run_zero_thermo(log_file)["potential_energy"]


def parse_run_zero_thermo(log_file: str | Path) -> dict[str, float]:
    """Return the last run-0 energy decomposition in kcal/mol."""
    rows = parse_lammps_log(get_log(log_file))["rows"]
    energy_rows = [row for row in rows if "PotEng" in row]
    if not energy_rows:
        raise ValueError(f"No PotEng thermo value found in {log_file}.")
    row = energy_rows[-1]
    columns = {
        "potential_energy": "PotEng",
        "vdw": "E_vdwl",
        "coulomb_short_range": "E_coul",
        "coulomb_long_range": "E_long",
    }
    missing = [column for column in columns.values() if column not in row]
    if missing:
        raise ValueError(
            f"Run-0 thermo data in {log_file} is missing: {', '.join(missing)}."
        )
    return {
        name: float(row[column])
        for name, column in columns.items()
    }


def parse_force_dump(dump_file: str | Path) -> list[list[float]]:
    """Return the last force frame sorted by one-based LAMMPS atom ID."""
    lines = Path(dump_file).read_text(encoding="utf-8").splitlines()
    frames: list[dict[int, list[float]]] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("ITEM: ATOMS"):
            index += 1
            continue

        columns = lines[index].split()[2:]
        required = {"id", "fx", "fy", "fz"}
        if not required.issubset(columns):
            raise ValueError(
                f"Force dump is missing columns: {', '.join(sorted(required - set(columns)))}."
            )
        column_indices = {name: columns.index(name) for name in required}
        index += 1
        frame: dict[int, list[float]] = {}
        while index < len(lines) and not lines[index].startswith("ITEM:"):
            parts = lines[index].split()
            if parts:
                atom_id = int(parts[column_indices["id"]])
                frame[atom_id] = [
                    float(parts[column_indices["fx"]]),
                    float(parts[column_indices["fy"]]),
                    float(parts[column_indices["fz"]]),
                ]
            index += 1
        if frame:
            frames.append(frame)

    if not frames:
        raise ValueError(f"No force frames found in {dump_file}.")
    last_frame = frames[-1]
    expected_ids = list(range(1, len(last_frame) + 1))
    if sorted(last_frame) != expected_ids:
        raise ValueError("Force dump atom IDs must be consecutive and start at 1.")
    return [last_frame[atom_id] for atom_id in expected_ids]


def _safe_directory_name(value: str) -> str:
    name = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value
    ).strip("._")
    return name or "configuration"
