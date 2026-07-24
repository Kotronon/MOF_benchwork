from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any
import warnings


CELL_REPRESENTATIONS = {"source", "primitive", "conventional", "auto"}
MINIMUM_IMAGE_POLICIES = {"error", "warn", "ignore"}


@dataclass(frozen=True)
class CifAtomRecord:
    label: str
    fractional: tuple[float, float, float]
    charge: float


@dataclass(frozen=True)
class FrameworkStructure:
    atom_count: int
    symbols: tuple[str, ...]
    charges: tuple[float, ...]
    cell_lengths: tuple[float, float, float]
    cell_angles: tuple[float, float, float]
    positions: tuple[tuple[float, float, float], ...]
    scaled_positions: tuple[tuple[float, float, float], ...]
    cell_vectors: tuple[tuple[float, float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_cif_to_lammps_data(
    cif_path: str | Path,
    output_path: str | Path,
    atom_style: str = "full",
    cell_representation: str = "source",
    unit_cells: list[int] | tuple[int, int, int] = (1, 1, 1),
    cutoff_A: float | None = None,
    minimum_image_policy: str = "error",
) -> dict[str, Any]:
    """Return the planned CIF-to-LAMMPS conversion without writing files."""
    return {
        "converter": "cif_to_lammps_data",
        "input_cif": str(cif_path),
        "output_data": str(output_path),
        "atom_style": atom_style,
        "cell_representation": cell_representation,
        "unit_cells": list(unit_cells),
        "cutoff_A": cutoff_A,
        "minimum_image_policy": minimum_image_policy,
        "status": "planned",
    }


def parse_cif_atom_site_charges(cif_path: str | Path) -> list[CifAtomRecord]:
    """Parse _atom_site_charge records from a simple CRAFTED CIF atom-site loop."""
    lines = Path(cif_path).read_text(encoding="utf-8").splitlines()
    records: list[CifAtomRecord] = []
    index = 0
    while index < len(lines):
        if lines[index].strip().lower() != "loop_":
            index += 1
            continue

        index += 1
        headers = []
        while index < len(lines) and lines[index].strip().startswith("_"):
            headers.append(lines[index].strip())
            index += 1

        if "_atom_site_charge" not in headers:
            continue

        required = ["_atom_site_label", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_charge"]
        missing = [header for header in required if header not in headers]
        if missing:
            raise ValueError(f"CIF atom-site loop is missing required columns: {', '.join(missing)}.")

        columns = {header: headers.index(header) for header in required}
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped or stripped.lower() == "loop_" or stripped.startswith("_"):
                break
            parts = stripped.split()
            if len(parts) >= len(headers):
                records.append(
                    CifAtomRecord(
                        label=parts[columns["_atom_site_label"]],
                        fractional=(
                            float(parts[columns["_atom_site_fract_x"]]),
                            float(parts[columns["_atom_site_fract_y"]]),
                            float(parts[columns["_atom_site_fract_z"]]),
                        ),
                        charge=float(parts[columns["_atom_site_charge"]]),
                    )
                )
            index += 1
        break

    if not records:
        raise ValueError(f"No _atom_site_charge records found in {cif_path}.")
    return records


def load_framework_structure(
    cif_path: str | Path,
    *,
    cell_representation: str = "source",
    unit_cells: list[int] | tuple[int, int, int] = (1, 1, 1),
    cutoff_A: float | None = None,
    minimum_image_policy: str = "error",
) -> FrameworkStructure:
    """Load framework geometry with ASE and charges from the CRAFTED CIF atom-site loop."""
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError("ASE is required to load CIF geometry. Install it in the active environment.") from exc

    representation = _normalize_cell_representation(cell_representation)
    repetitions = _validate_unit_cells(unit_cells)
    policy = _normalize_minimum_image_policy(minimum_image_policy)
    atoms = read(str(cif_path))
    charge_records = parse_cif_atom_site_charges(cif_path)
    if len(atoms) != len(charge_records):
        raise ValueError(
            f"ASE atom count and CIF charge count differ for {cif_path}: {len(atoms)} != {len(charge_records)}."
        )

    atoms.set_initial_charges([record.charge for record in charge_records])
    atoms = _apply_cell_representation(atoms, representation)
    if repetitions != (1, 1, 1):
        atoms = atoms.repeat(repetitions)
    _validate_minimum_image(atoms.cell.array, cutoff_A, policy)

    return FrameworkStructure(
        atom_count=len(atoms),
        symbols=tuple(atoms.get_chemical_symbols()),
        charges=tuple(float(value) for value in atoms.get_initial_charges()),
        cell_lengths=tuple(float(value) for value in atoms.cell.lengths()),
        cell_angles=tuple(float(value) for value in atoms.cell.angles()),
        positions=tuple(tuple(float(value) for value in position) for position in atoms.get_positions()),
        scaled_positions=tuple(tuple(float(value) for value in position) for position in atoms.get_scaled_positions()),
        cell_vectors=tuple(tuple(float(value) for value in vector) for vector in atoms.cell.array),
    )


def convert_cif_to_lammps_data(
    cif_path: str | Path,
    output_path: str | Path,
    atom_style: str = "full",
    extra_atom_types: dict[str, float] | None = None,
    extra_bond_types: int = 0,
    cell_representation: str = "source",
    unit_cells: list[int] | tuple[int, int, int] = (1, 1, 1),
    cutoff_A: float | None = None,
    minimum_image_policy: str = "error",
) -> Path:
    """Write a minimal LAMMPS data file with framework atoms and optional extra type masses."""
    if atom_style != "full":
        raise ValueError("Only atom_style='full' is supported by the V1 framework converter.")
    if extra_bond_types < 0:
        raise ValueError("extra_bond_types must be non-negative.")

    structure = load_framework_structure(
        cif_path,
        cell_representation=cell_representation,
        unit_cells=unit_cells,
        cutoff_A=cutoff_A,
        minimum_image_policy=minimum_image_policy,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    type_map = {symbol: index for index, symbol in enumerate(_unique(structure.symbols), 1)}
    masses = _atomic_masses(type_map)
    for label, mass in (extra_atom_types or {}).items():
        if label in type_map:
            raise ValueError(f"Extra atom type {label!r} duplicates an existing framework atom type.")
        type_map[label] = len(type_map) + 1
        masses[label] = float(mass)
    box = _lammps_triclinic_box(structure.cell_vectors)
    positions = _lammps_positions(structure.scaled_positions, box)

    lines = [
        f"LAMMPS data file generated from {Path(cif_path)}",
        "",
        f"{structure.atom_count} atoms",
    ]
    if extra_bond_types:
        lines.append("0 bonds")
    lines.append(f"{len(type_map)} atom types")
    if extra_bond_types:
        lines.append(f"{extra_bond_types} bond types")
    lines.extend(
        [
            "",
            f"0.0 {box['lx']:.8f} xlo xhi",
            f"0.0 {box['ly']:.8f} ylo yhi",
            f"0.0 {box['lz']:.8f} zlo zhi",
            f"{box['xy']:.8f} {box['xz']:.8f} {box['yz']:.8f} xy xz yz",
            "",
            "Masses",
            "",
        ]
    )
    for symbol, atom_type in type_map.items():
        lines.append(f"{atom_type} {masses[symbol]:.8f} # {symbol}")

    lines.extend(["", "Atoms # full", ""])
    for atom_id, (symbol, charge, position) in enumerate(
        zip(structure.symbols, structure.charges, positions, strict=True),
        1,
    ):
        x, y, z = position
        molecule_id = 1
        lines.append(f"{atom_id} {molecule_id} {type_map[symbol]} {charge:.8f} {x:.8f} {y:.8f} {z:.8f}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _normalize_cell_representation(value: str) -> str:
    normalized = str(value).strip().casefold()
    if normalized not in CELL_REPRESENTATIONS:
        raise ValueError(
            f"cell_representation must be one of {sorted(CELL_REPRESENTATIONS)}, got {value!r}."
        )
    return "source" if normalized == "auto" else normalized


def _normalize_minimum_image_policy(value: str) -> str:
    normalized = str(value).strip().casefold()
    if normalized not in MINIMUM_IMAGE_POLICIES:
        raise ValueError(
            f"minimum_image_policy must be one of {sorted(MINIMUM_IMAGE_POLICIES)}, got {value!r}."
        )
    return normalized


def _validate_unit_cells(values: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if len(values) != 3 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("unit_cells must contain three positive integers.")
    return tuple(values)


def _apply_cell_representation(atoms: Any, representation: str) -> Any:
    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    is_rhombohedral_primitive = (
        max(lengths) - min(lengths) < 1e-3
        and all(abs(float(angle) - 60.0) < 1e-3 for angle in angles)
    )
    is_orthogonal = all(abs(float(angle) - 90.0) < 1e-3 for angle in angles)
    if representation in {"source", "primitive"}:
        if representation == "primitive" and not is_rhombohedral_primitive:
            raise ValueError(
                "Requested primitive representation, but the source CIF is not the supported "
                "60-degree rhombohedral primitive cell."
            )
        return atoms
    if representation == "conventional":
        if is_orthogonal:
            return atoms
        if not is_rhombohedral_primitive:
            raise ValueError(
                "Conventional-cell conversion currently supports orthogonal source cells and "
                "60-degree rhombohedral primitive cells such as IRMOF-1."
            )
        try:
            from ase.build import make_supercell
        except ImportError as exc:
            raise ImportError("ASE is required for conventional-cell construction.") from exc
        transformation = ((-1, 1, 1), (1, -1, 1), (1, 1, -1))
        return make_supercell(atoms, transformation, wrap=True)
    raise AssertionError(f"Unhandled cell representation: {representation}")


def cell_perpendicular_widths(
    cell_vectors: Any,
) -> tuple[float, float, float]:
    """Return perpendicular periodic widths for a general triclinic cell."""
    a, b, c = [tuple(float(value) for value in vector) for vector in cell_vectors]
    volume = abs(_dot(a, _cross(b, c)))
    if volume <= 0:
        raise ValueError("Cell volume must be positive.")
    return (
        volume / _norm(_cross(b, c)),
        volume / _norm(_cross(c, a)),
        volume / _norm(_cross(a, b)),
    )


def _validate_minimum_image(cell_vectors: Any, cutoff_A: float | None, policy: str) -> None:
    if cutoff_A is None or policy == "ignore":
        return
    cutoff = float(cutoff_A)
    if cutoff <= 0:
        raise ValueError("cutoff_A must be positive.")
    widths = cell_perpendicular_widths(cell_vectors)
    required = 2.0 * cutoff
    if min(widths) + 1e-8 >= required:
        return
    message = (
        f"Minimum periodic cell width {min(widths):.6g} A is smaller than twice the "
        f"real-space cutoff ({required:.6g} A). Increase unit_cells or choose a larger "
        "cell representation."
    )
    if policy == "error":
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _atomic_masses(type_map: dict[str, int]) -> dict[str, float]:
    try:
        from ase.data import atomic_masses, atomic_numbers
    except ImportError as exc:
        raise ImportError("ASE is required to write atomic masses.") from exc
    return {symbol: float(atomic_masses[atomic_numbers[symbol]]) for symbol in type_map}


def _unique(values: Any) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _lammps_triclinic_box(cell_vectors: tuple[tuple[float, float, float], ...]) -> dict[str, float]:
    a, b, c = cell_vectors
    lx = _norm(a)
    ax = _scale(a, 1.0 / lx)
    xy = _dot(b, ax)
    xz = _dot(c, ax)
    ly_squared = _dot(b, b) - xy**2
    if ly_squared <= 0:
        raise ValueError("Invalid cell vectors: computed non-positive ly.")
    ly = math.sqrt(ly_squared)
    yz = (_dot(b, c) - xy * xz) / ly
    lz_squared = _dot(c, c) - xz**2 - yz**2
    if lz_squared <= 0:
        raise ValueError("Invalid cell vectors: computed non-positive lz.")
    lz = math.sqrt(lz_squared)
    return {"lx": lx, "ly": ly, "lz": lz, "xy": xy, "xz": xz, "yz": yz}


def _lammps_positions(
    scaled_positions: tuple[tuple[float, float, float], ...],
    box: dict[str, float],
) -> tuple[tuple[float, float, float], ...]:
    positions = []
    for sx, sy, sz in scaled_positions:
        x = sx * box["lx"] + sy * box["xy"] + sz * box["xz"]
        y = sy * box["ly"] + sz * box["yz"]
        z = sz * box["lz"]
        positions.append((x, y, z))
    return tuple(positions)


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _scale(vector: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return tuple(value * factor for value in vector)


def _cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )
