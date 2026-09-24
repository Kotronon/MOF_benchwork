"""Generate deterministic MOF/adsorbate configurations for smoke tests."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from converter.cif_to_lammps_data import load_framework_structure
from converter.forcefield_to_lammps import (
    VIRTUAL_SITE_MASS_AMU,
    parse_pseudo_atoms,
)
from converter.molecule_to_lammps_template import parse_crafted_molecule_def
from modules.module_c_mlips.models import InteractionBond, InteractionConfiguration


def build_smoke_configurations(
    cif_path: str | Path,
    molecule_def_path: str | Path,
    pseudo_atoms_path: str | Path | None = None,
    *,
    material: str = "IRMOF-1",
    adsorbate: str = "CO2",
    cell_representation: str = "source",
    unit_cells: list[int] | tuple[int, int, int] = (1, 1, 1),
    cutoff_A: float | None = None,
    minimum_image_policy: str = "ignore",
) -> list[InteractionConfiguration]:
    """Build three deterministic configurations for pipeline validation.

    The generated labels describe construction only. They are not validated
    interaction regimes and must not be interpreted as physical reference data.
    """
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError(
            "ASE is required to build Module C smoke configurations."
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
    
    framework_types = framework.get_chemical_symbols()
    framework.new_array(
        "forcefield_type",
        np.asarray(framework_types, dtype="U32"),
    )

    framework.set_initial_charges(framework_structure.charges)

    molecule = parse_crafted_molecule_def(molecule_def)
    molecule_atoms = Atoms(
        symbols=[_chemical_symbol(atom.atom_type) for atom in molecule.atoms],
        positions=[(atom.x, atom.y, atom.z) for atom in molecule.atoms],
    )
    
    molecule_atoms.new_array(
        "forcefield_type",
        np.asarray(
            [atom.atom_type for atom in molecule.atoms],
            dtype="U32",
        ),
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
    
    centers = [
        (
            "cell_center",
            "unclassified",
            framework.cell.cartesian_positions((0.5, 0.5, 0.5)),
        ),
        (
            "eighth_cell",
            "unclassified",
            framework.cell.cartesian_positions((0.125, 0.125, 0.125)),
        ),
        (
            "overlap_probe",
            "technical_repulsive_probe",
            framework.positions[0],
        ),
    ]

    configurations = []
    framework_indices = list(range(len(framework)))
    adsorbate_indices = list(
        range(len(framework), len(framework) + len(molecule_atoms))
    )
    atom_offset = len(framework)
    bonds = [
        InteractionBond(
            atom1_index=atom_offset + bond.atom1,
            atom2_index=atom_offset + bond.atom2,
            forcefield_type=bond.bond_type,
        )
        for bond in molecule.bonds
    ]
    for label, region, center in centers:
        guest = molecule_atoms.copy()
        guest.translate(center - guest.get_center_of_mass())
        combined = framework.copy()
        combined += guest
        combined.set_cell(framework.cell)
        combined.set_pbc(True)

        configurations.append(
            InteractionConfiguration(
                configuration_id=f"{material}_{adsorbate}_{label}",
                material=material,
                adsorbate=adsorbate,
                atoms=combined,
                framework_indices=framework_indices.copy(),
                adsorbate_indices=adsorbate_indices.copy(),
                region=region,
                source="generated_smoke",
                bonds=bonds.copy(),
            )
        )

    return configurations


def _chemical_symbol(atom_type: str) -> str:
    """Extract an element symbol from CRAFTED types such as O_co2."""
    symbol = atom_type.split("_", 1)[0]
    if not symbol or not symbol[0].isalpha():
        raise ValueError(f"Cannot infer a chemical symbol from {atom_type!r}.")
    return symbol[0].upper() + symbol[1:].lower()
