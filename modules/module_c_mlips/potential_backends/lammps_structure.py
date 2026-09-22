"""Build and write structures for classical LAMMPS potential evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from converter.cif_to_lammps_data import lammps_positions, lammps_triclinic_box

if TYPE_CHECKING:
    from modules.module_c_mlips.models import InteractionBond


def build_lammps_structure(
    atoms: Any,
    framework_indices: list[int],
    adsorbate_indices: list[int],
    type_ids: dict[str, int],
    bonds: list[InteractionBond] | None = None,
) -> dict[str, Any]:
    """Build a serializable LAMMPS structure using global force-field types."""
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError("ASE is required to build a LAMMPS structure.") from exc

    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an instance of ase.Atoms.")

    framework_set = set(framework_indices)
    adsorbate_set = set(adsorbate_indices)
    if framework_set & adsorbate_set:
        raise ValueError("Framework and adsorbate indices must not overlap.")
    if framework_set | adsorbate_set != set(range(len(atoms))):
        raise ValueError("Framework and adsorbate indices must cover every atom.")

    if "forcefield_type" not in atoms.arrays:
        raise ValueError("ASE atoms must contain a 'forcefield_type' array.")
    if len(set(type_ids.values())) != len(type_ids):
        raise ValueError("LAMMPS type IDs must be unique.")
    if any(
        not isinstance(type_id, int) or type_id <= 0
        for type_id in type_ids.values()
    ):
        raise ValueError("LAMMPS type IDs must be positive integers.")

    forcefield_types = atoms.arrays["forcefield_type"].tolist()
    missing_types = set(forcefield_types) - set(type_ids)
    if missing_types:
        raise ValueError(
            "Missing LAMMPS type IDs for: " + ", ".join(sorted(missing_types))
        )

    charges = atoms.get_initial_charges()
    structure: dict[str, Any] = {
        "atoms": [],
        "bonds": [],
        "framework_indices": list(framework_indices),
        "adsorbate_indices": list(adsorbate_indices),
        "type_ids": dict(type_ids),
        "cell_vectors": atoms.cell.array.tolist(),
        "scaled_positions": atoms.get_scaled_positions(wrap=False).tolist(),
        "pbc": atoms.pbc.tolist(),
    }

    for index, atom in enumerate(atoms):
        forcefield_type = forcefield_types[index]
        structure["atoms"].append(
            {
                "id": index + 1,
                "element": atom.symbol,
                "charge": float(charges[index]),
                "mass": float(atom.mass),
                "forcefield_type": forcefield_type,
                "lammps_type_id": type_ids[forcefield_type],
                "component": (
                    "framework" if index in framework_set else "adsorbate"
                ),
            }
        )

    for bond in bonds or []:
        if not 0 <= bond.atom1_index < len(atoms):
            raise ValueError("Bond atom1_index is outside the atom range.")
        if not 0 <= bond.atom2_index < len(atoms):
            raise ValueError("Bond atom2_index is outside the atom range.")
        if bond.atom1_index == bond.atom2_index:
            raise ValueError("A bond must connect two different atoms.")
        structure["bonds"].append(
            {
                "atom1_index": bond.atom1_index,
                "atom2_index": bond.atom2_index,
                "forcefield_type": bond.forcefield_type or "bond",
            }
        )

    return structure


def write_lammps_data(
    structure: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a combined framework/adsorbate LAMMPS data file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    atoms = structure["atoms"]
    bonds = structure.get("bonds", [])
    type_ids = structure["type_ids"]
    bond_labels = list(dict.fromkeys(bond["forcefield_type"] for bond in bonds))
    bond_type_ids = {
        label: index for index, label in enumerate(bond_labels, start=1)
    }

    lines = [
        "LAMMPS data file generated for Module C",
        "",
        f"{len(atoms)} atoms",
        f"{len(bonds)} bonds",
        "",
        f"{len(type_ids)} atom types",
        f"{len(bond_type_ids)} bond types",
        "",
    ]

    cell_vectors = tuple(
        tuple(float(value) for value in vector)
        for vector in structure["cell_vectors"]
    )
    scaled_positions = tuple(
        tuple(float(value) for value in position)
        for position in structure["scaled_positions"]
    )
    box = lammps_triclinic_box(cell_vectors)
    positions = lammps_positions(scaled_positions, box)
    lines.extend(
        [
            f"0.0 {box['lx']:.10f} xlo xhi",
            f"0.0 {box['ly']:.10f} ylo yhi",
            f"0.0 {box['lz']:.10f} zlo zhi",
            (
                f"{box['xy']:.10f} {box['xz']:.10f} "
                f"{box['yz']:.10f} xy xz yz"
            ),
            "",
        ]
    )

    masses: dict[str, float] = {}
    for atom in atoms:
        atom_type = atom["forcefield_type"]
        mass = float(atom["mass"])
        if atom_type in masses and abs(masses[atom_type] - mass) > 1.0e-8:
            raise ValueError(f"Inconsistent mass for atom type {atom_type!r}.")
        masses[atom_type] = mass

    missing_masses = set(type_ids) - set(masses)
    if missing_masses:
        raise ValueError(
            "Missing masses for LAMMPS types: " + ", ".join(sorted(missing_masses))
        )

    lines.extend(["Masses", ""])
    for atom_type, type_id in sorted(type_ids.items(), key=lambda item: item[1]):
        lines.append(f"{type_id} {masses[atom_type]:.8f} # {atom_type}")

    lines.extend(["", "Atoms # full", ""])
    for atom, position in zip(atoms, positions, strict=True):
        molecule_id = 1 if atom["component"] == "framework" else 2
        x, y, z = position
        lines.append(
            f"{atom['id']} {molecule_id} {atom['lammps_type_id']} "
            f"{atom['charge']:.8f} {x:.10f} {y:.10f} {z:.10f}"
        )

    if bonds:
        lines.extend(["", "Bonds", ""])
        for bond_id, bond in enumerate(bonds, start=1):
            bond_type_id = bond_type_ids[bond["forcefield_type"]]
            lines.append(
                f"{bond_id} {bond_type_id} "
                f"{bond['atom1_index'] + 1} {bond['atom2_index'] + 1}"
            )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
