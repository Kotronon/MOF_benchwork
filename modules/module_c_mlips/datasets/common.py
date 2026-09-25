"""Shared host-guest structure construction for Module C datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from converter.cif_to_lammps_data import load_framework_structure
from converter.forcefield_to_lammps import (
    VIRTUAL_SITE_MASS_AMU,
    parse_pseudo_atoms,
)
from converter.molecule_to_lammps_template import parse_crafted_molecule_def
from modules.module_c_mlips.models import InteractionBond


def load_host_guest_system(
    cif_path: str | Path,
    molecule_def_path: str | Path,
    pseudo_atoms_path: str | Path | None = None,
    *,
    cell_representation: str = "source",
    unit_cells: list[int] | tuple[int, int, int] = (1, 1, 1),
    cutoff_A: float | None = None,
    minimum_image_policy: str = "ignore",
) -> tuple[Any, Any, list[InteractionBond]]:
    """Load a charged framework and one force-field-typed guest molecule."""
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError(
            "ASE is required to build Module C configurations."
        ) from exc

    cif = Path(cif_path)
    molecule_def = Path(molecule_def_path)
    pseudo_atoms_file = (
        Path(pseudo_atoms_path)
        if pseudo_atoms_path is not None
        else molecule_def.parent / "pseudo_atoms.def"
    )
    framework_structure = load_framework_structure(
        cif,
        cell_representation=cell_representation,
        unit_cells=unit_cells,
        cutoff_A=cutoff_A,
        minimum_image_policy=minimum_image_policy,
    )
    framework = Atoms(
        symbols=framework_structure.symbols,
        positions=framework_structure.positions,
        cell=framework_structure.cell_vectors,
        pbc=True,
    )
    framework.new_array(
        "forcefield_type",
        np.asarray(framework.get_chemical_symbols(), dtype="U32"),
    )
    framework.set_initial_charges(framework_structure.charges)

    molecule = parse_crafted_molecule_def(molecule_def)
    molecule_atoms = Atoms(
        symbols=[_chemical_symbol(atom.atom_type) for atom in molecule.atoms],
        positions=[(atom.x, atom.y, atom.z) for atom in molecule.atoms],
    )
    molecule_atoms.new_array(
        "forcefield_type",
        np.asarray([atom.atom_type for atom in molecule.atoms], dtype="U32"),
    )

    pseudo_atoms = parse_pseudo_atoms(pseudo_atoms_file)
    missing_types = sorted(
        {
            atom.atom_type
            for atom in molecule.atoms
            if atom.atom_type not in pseudo_atoms
        }
    )
    if missing_types:
        raise ValueError(
            "Missing pseudo-atom parameters for: " + ", ".join(missing_types)
        )
    molecule_atoms.set_initial_charges(
        [pseudo_atoms[atom.atom_type].charge for atom in molecule.atoms]
    )
    molecule_atoms.set_masses(
        [
            pseudo_atoms[atom.atom_type].mass
            if pseudo_atoms[atom.atom_type].mass > 0.0
            else VIRTUAL_SITE_MASS_AMU
            for atom in molecule.atoms
        ]
    )
    bonds = [
        InteractionBond(
            atom1_index=bond.atom1,
            atom2_index=bond.atom2,
            forcefield_type=bond.bond_type,
        )
        for bond in molecule.bonds
    ]
    return framework, molecule_atoms, bonds


def combine_host_guest(
    framework: Any,
    guest: Any,
    guest_bonds: list[InteractionBond],
) -> tuple[Any, list[int], list[int], list[InteractionBond]]:
    """Combine host and guest while remapping guest-local bond indices."""
    combined = framework.copy()
    combined += guest
    combined.set_cell(framework.cell)
    combined.set_pbc(True)

    offset = len(framework)
    framework_indices = list(range(offset))
    adsorbate_indices = list(range(offset, offset + len(guest)))
    bonds = [
        InteractionBond(
            atom1_index=offset + bond.atom1_index,
            atom2_index=offset + bond.atom2_index,
            forcefield_type=bond.forcefield_type,
        )
        for bond in guest_bonds
    ]
    return combined, framework_indices, adsorbate_indices, bonds


def _chemical_symbol(atom_type: str) -> str:
    """Extract an element symbol from CRAFTED types such as O_co2."""
    symbol = atom_type.split("_", 1)[0]
    if not symbol or not symbol[0].isalpha():
        raise ValueError(f"Cannot infer a chemical symbol from {atom_type!r}.")
    return symbol[0].upper() + symbol[1:].lower()
