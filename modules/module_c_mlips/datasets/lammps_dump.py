"""Load fixed host-guest configurations from Module-A LAMMPS dumps."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

from converter.forcefield_to_lammps import VIRTUAL_SITE_MASS_AMU, parse_pseudo_atoms
from converter.molecule_to_lammps_template import parse_crafted_molecule_def
from modules.module_c_mlips.models import InteractionBond, InteractionConfiguration


TYPE_MAP_PATTERN = re.compile(r"^#\s+(\d+):\s+(\S+)\s+->")


def parse_lammps_forcefield_type_map(
    forcefield_file: str | Path,
) -> dict[int, str]:
    """Parse ``type ID -> force-field label`` comments from an include file."""
    result: dict[int, str] = {}
    for line in Path(forcefield_file).read_text(encoding="utf-8").splitlines():
        match = TYPE_MAP_PATTERN.match(line.strip())
        if match:
            result[int(match.group(1))] = match.group(2)
    if not result:
        raise ValueError(f"No atom type map found in {forcefield_file}.")
    return result


def load_lammps_dump_configuration(
    dump_file: str | Path,
    forcefield_file: str | Path,
    molecule_definition: str | Path,
    pseudo_atoms_file: str | Path,
    *,
    frame_index: int = -1,
    material: str = "IRMOF-1",
    adsorbate: str = "CO2",
    framework_molecule_id: int = 1,
) -> InteractionConfiguration:
    """Load one sorted frame from a Module-A ``dump custom`` trajectory."""
    try:
        from ase import Atoms
        from ase.data import atomic_masses, atomic_numbers
    except ImportError as exc:
        raise ImportError("ASE is required to load LAMMPS dump snapshots.") from exc

    frames = _parse_dump_frames(Path(dump_file))
    try:
        frame = frames[frame_index]
    except IndexError as exc:
        raise IndexError(
            f"Frame index {frame_index} is outside a dump with {len(frames)} frames."
        ) from exc

    type_map = parse_lammps_forcefield_type_map(forcefield_file)
    pseudo_atoms = parse_pseudo_atoms(pseudo_atoms_file)
    records = sorted(frame["atoms"], key=lambda record: record["id"])
    expected_ids = list(range(1, len(records) + 1))
    if [record["id"] for record in records] != expected_ids:
        raise ValueError("LAMMPS dump atom IDs must be consecutive and start at 1.")

    labels: list[str] = []
    symbols: list[str] = []
    masses: list[float] = []
    positions: list[list[float]] = []
    charges: list[float] = []
    molecule_ids: list[int] = []
    origin = frame["origin"]
    for record in records:
        try:
            label = type_map[record["type"]]
        except KeyError as exc:
            raise ValueError(
                f"LAMMPS type {record['type']} has no force-field type mapping."
            ) from exc
        symbol = _chemical_symbol(label)
        labels.append(label)
        symbols.append(symbol)
        charges.append(record["q"])
        molecule_ids.append(record["mol"])
        positions.append(
            [
                record["x"] - origin[0],
                record["y"] - origin[1],
                record["z"] - origin[2],
            ]
        )
        pseudo_atom = pseudo_atoms.get(label)
        masses.append(
            (
                pseudo_atom.mass
                if pseudo_atom is not None and pseudo_atom.mass > 0.0
                else VIRTUAL_SITE_MASS_AMU
                if pseudo_atom is not None
                else float(atomic_masses[atomic_numbers[symbol]])
            )
        )

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=frame["cell"],
        pbc=True,
    )
    atoms.set_initial_charges(charges)
    atoms.set_masses(masses)
    atoms.new_array("forcefield_type", np.asarray(labels, dtype="U32"))

    framework_indices = [
        index
        for index, molecule_id in enumerate(molecule_ids)
        if molecule_id == framework_molecule_id
    ]
    adsorbate_indices = [
        index
        for index, molecule_id in enumerate(molecule_ids)
        if molecule_id != framework_molecule_id
    ]
    molecule = parse_crafted_molecule_def(molecule_definition)
    bonds: list[InteractionBond] = []
    adsorbate_molecule_ids = sorted(
        set(molecule_ids) - {framework_molecule_id}
    )
    for molecule_id in adsorbate_molecule_ids:
        indices = [
            index
            for index, current_id in enumerate(molecule_ids)
            if current_id == molecule_id
        ]
        if len(indices) != len(molecule.atoms):
            raise ValueError(
                f"Adsorbate molecule {molecule_id} contains {len(indices)} atoms; "
                f"expected {len(molecule.atoms)}."
            )
        for bond in molecule.bonds:
            bonds.append(
                InteractionBond(
                    atom1_index=indices[bond.atom1],
                    atom2_index=indices[bond.atom2],
                    forcefield_type=bond.bond_type,
                )
            )

    timestep = frame["timestep"]
    return InteractionConfiguration(
        configuration_id=f"{material}_{adsorbate}_dump_step_{timestep}",
        material=material,
        adsorbate=adsorbate,
        atoms=atoms,
        framework_indices=framework_indices,
        adsorbate_indices=adsorbate_indices,
        region="sampled_gcmc_snapshot",
        source=str(Path(dump_file)),
        bonds=bonds,
    )


def _parse_dump_frames(dump_file: Path) -> list[dict[str, Any]]:
    lines = dump_file.read_text(encoding="utf-8").splitlines()
    frames: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != "ITEM: TIMESTEP":
            index += 1
            continue
        timestep = int(lines[index + 1].strip())
        if lines[index + 2].strip() != "ITEM: NUMBER OF ATOMS":
            raise ValueError("Malformed LAMMPS dump: missing atom count.")
        atom_count = int(lines[index + 3].strip())
        box_header = lines[index + 4].split()
        if box_header[:3] != ["ITEM:", "BOX", "BOUNDS"]:
            raise ValueError("Malformed LAMMPS dump: missing box bounds.")
        bounds = [
            [float(value) for value in lines[index + offset].split()]
            for offset in (5, 6, 7)
        ]
        cell, origin = _restricted_triclinic_cell(bounds)
        atom_header = lines[index + 8].split()
        if atom_header[:2] != ["ITEM:", "ATOMS"]:
            raise ValueError("Malformed LAMMPS dump: missing atom columns.")
        columns = atom_header[2:]
        required = {"id", "mol", "type", "q", "x", "y", "z"}
        missing = required - set(columns)
        if missing:
            raise ValueError(
                "LAMMPS dump is missing columns: " + ", ".join(sorted(missing))
            )
        column_index = {name: columns.index(name) for name in required}
        atoms = []
        first_atom_line = index + 9
        for line in lines[first_atom_line : first_atom_line + atom_count]:
            values = line.split()
            atoms.append(
                {
                    "id": int(values[column_index["id"]]),
                    "mol": int(values[column_index["mol"]]),
                    "type": int(values[column_index["type"]]),
                    "q": float(values[column_index["q"]]),
                    "x": float(values[column_index["x"]]),
                    "y": float(values[column_index["y"]]),
                    "z": float(values[column_index["z"]]),
                }
            )
        frames.append(
            {
                "timestep": timestep,
                "cell": cell,
                "origin": origin,
                "atoms": atoms,
            }
        )
        index = first_atom_line + atom_count
    if not frames:
        raise ValueError(f"No frames found in LAMMPS dump {dump_file}.")
    return frames


def _restricted_triclinic_cell(
    bounds: list[list[float]],
) -> tuple[list[list[float]], list[float]]:
    if len(bounds) != 3 or any(len(values) < 2 for values in bounds):
        raise ValueError("LAMMPS dump box bounds must contain three rows.")
    xlo_bound, xhi_bound = bounds[0][:2]
    ylo_bound, yhi_bound = bounds[1][:2]
    zlo, zhi = bounds[2][:2]
    xy = bounds[0][2] if len(bounds[0]) > 2 else 0.0
    xz = bounds[1][2] if len(bounds[1]) > 2 else 0.0
    yz = bounds[2][2] if len(bounds[2]) > 2 else 0.0
    xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
    xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
    ylo = ylo_bound - min(0.0, yz)
    yhi = yhi_bound - max(0.0, yz)
    return (
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ],
        [xlo, ylo, zlo],
    )


def _chemical_symbol(forcefield_type: str) -> str:
    symbol = forcefield_type.split("_", 1)[0]
    if not symbol or not symbol[0].isalpha():
        raise ValueError(
            f"Cannot infer a chemical symbol from {forcefield_type!r}."
        )
    return symbol[0].upper() + symbol[1:].lower()
