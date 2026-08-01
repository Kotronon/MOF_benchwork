from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MoleculeAtom:
    index: int
    atom_type: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class MoleculeBond:
    atom1: int
    atom2: int
    bond_type: str


@dataclass(frozen=True)
class CraftedMolecule:
    name: str
    critical_temperature_K: float
    critical_pressure_Pa: float
    acentric_factor: float
    atoms: tuple[MoleculeAtom, ...]
    bonds: tuple[MoleculeBond, ...]
    rigid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_molecule_template(molecule_def_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Return the planned molecule-template conversion without writing files."""
    return {
        "converter": "molecule_to_lammps_template",
        "input_definition": str(molecule_def_path),
        "output_template": str(output_path),
        "status": "planned",
    }


def parse_crafted_molecule_def(molecule_def_path: str | Path) -> CraftedMolecule:
    """Parse the subset of CRAFTED/RASPA molecule .def files used by the pipeline."""
    lines = Path(molecule_def_path).read_text(encoding="utf-8").splitlines()
    content = _content_lines(lines)
    cursor = 0

    critical_temperature = float(content[cursor])
    critical_pressure = float(content[cursor + 1])
    acentric_factor = float(content[cursor + 2])
    cursor += 3

    number_of_atoms = int(content[cursor])
    cursor += 1

    number_of_groups = int(content[cursor])
    cursor += 1
    if number_of_groups != 1:
        raise ValueError(f"Only single-group molecule definitions are supported, got {number_of_groups}.")

    if content[cursor].casefold() in {"rigid", "flexible"}:
        name = Path(molecule_def_path).stem
    else:
        group_name = content[cursor]
        cursor += 1
        name = group_name.removesuffix("-group")

    rigid = content[cursor].casefold() == "rigid"
    cursor += 1

    group_atom_count = int(content[cursor])
    cursor += 1
    if group_atom_count != number_of_atoms:
        raise ValueError(
            f"Atom count mismatch in {molecule_def_path}: header={number_of_atoms}, group={group_atom_count}."
        )

    atoms = []
    for _ in range(group_atom_count):
        parts = content[cursor].split()
        if len(parts) == 2:
            x = y = z = 0.0
        elif len(parts) == 5:
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
        else:
            raise ValueError(f"Invalid atomic position line in {molecule_def_path}: {content[cursor]!r}")
        atoms.append(
            MoleculeAtom(
                index=int(parts[0]),
                atom_type=parts[1],
                x=x,
                y=y,
                z=z,
            )
        )
        cursor += 1

    topology_counts = [int(value) for value in content[cursor].split()]
    cursor += 1
    bond_count = topology_counts[1]

    bonds = []
    for _ in range(bond_count):
        parts = content[cursor].split()
        if len(parts) != 3:
            raise ValueError(f"Invalid bond line in {molecule_def_path}: {content[cursor]!r}")
        bonds.append(MoleculeBond(atom1=int(parts[0]), atom2=int(parts[1]), bond_type=parts[2]))
        cursor += 1

    return CraftedMolecule(
        name=name,
        critical_temperature_K=critical_temperature,
        critical_pressure_Pa=critical_pressure,
        acentric_factor=acentric_factor,
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        rigid=rigid,
    )


def build_molecule_template(
    molecule_def_path: str | Path,
    output_path: str | Path,
    atom_type_ids: dict[str, int] | None = None,
    atom_charges: dict[str, float] | None = None,
) -> Path:
    """Build a minimal LAMMPS molecule template for a rigid linear CRAFTED molecule."""
    molecule = parse_crafted_molecule_def(molecule_def_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if atom_type_ids is None:
        type_ids = {atom_type: index for index, atom_type in enumerate(_unique(atom.atom_type for atom in molecule.atoms), 1)}
    else:
        missing = sorted({atom.atom_type for atom in molecule.atoms} - set(atom_type_ids))
        if missing:
            raise ValueError(f"Missing global atom type IDs for molecule atom types: {', '.join(missing)}.")
        type_ids = atom_type_ids
    if atom_charges is not None:
        missing_charges = sorted({atom.atom_type for atom in molecule.atoms} - set(atom_charges))
        if missing_charges:
            raise ValueError(f"Missing charges for molecule atom types: {', '.join(missing_charges)}.")
    bond_type_ids = {
        bond_type: index for index, bond_type in enumerate(_unique(bond.bond_type for bond in molecule.bonds), 1)
    }

    lines = [
        f"# LAMMPS molecule template generated from {Path(molecule_def_path)}",
        f"# Molecule: {molecule.name}",
        "",
        f"{len(molecule.atoms)} atoms",
        f"{len(molecule.bonds)} bonds",
        "",
        "Coords",
        "",
    ]
    for atom in molecule.atoms:
        atom_id = atom.index + 1
        lines.append(f"{atom_id} {atom.x:.8f} {atom.y:.8f} {atom.z:.8f}")

    lines.extend(["", "Types", ""])
    for atom in molecule.atoms:
        lines.append(f"{atom.index + 1} {type_ids[atom.atom_type]}")

    if atom_charges is not None:
        lines.extend(["", "Charges", ""])
        for atom in molecule.atoms:
            lines.append(f"{atom.index + 1} {atom_charges[atom.atom_type]:.8f}")

    if molecule.bonds:
        lines.extend(["", "Bonds", ""])
        for bond_index, bond in enumerate(molecule.bonds, 1):
            lines.append(f"{bond_index} {bond_type_ids[bond.bond_type]} {bond.atom1 + 1} {bond.atom2 + 1}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _content_lines(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _unique(values: Any) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
