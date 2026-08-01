from __future__ import annotations

from pathlib import Path
import math
from typing import Any

from converter.forcefield_to_lammps import load_forcefield_parameters
from converter.molecule_to_lammps_template import CraftedMolecule, parse_crafted_molecule_def


DEFAULT_DIAMETER_POLICY = {
    "source": "generated_from_lj_sigma",
    "sigma_scale": 1.10,
    "fallback_A": 3.5,
}


def infer_adsorbate_properties(
    component: str,
    forcefield: dict[str, Any],
    diameter_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer adsorbate properties from resolved CRAFTED forcefield files."""
    policy = {**DEFAULT_DIAMETER_POLICY, **(diameter_policy or {})}
    component_key = component.upper()
    adsorbate_defs = forcefield.get("adsorbates", {})
    molecule_path = adsorbate_defs.get(component_key)
    if molecule_path is None:
        raise LookupError(f"No molecule definition path is available for adsorbate {component_key!r}.")

    molecule = parse_crafted_molecule_def(molecule_path)
    parameters = load_forcefield_parameters(forcefield)
    atom_types = _unique(atom.atom_type for atom in molecule.atoms)
    geometry_extent = _molecule_extent_A(molecule)
    max_lj_sigma = _max_interacting_lj_sigma_A(atom_types, parameters.lj_parameters)
    diameter, diameter_source = _infer_access_diameter_A(
        geometry_extent_A=geometry_extent,
        max_lj_sigma_A=max_lj_sigma,
        policy=policy,
    )

    adsorbate_source = forcefield.get("adsorbate_sources", {}).get(
        component_key,
        forcefield.get("source", "generated_from_crafted_forcefield"),
    )

    return {
        "component": component_key,
        "source": str(adsorbate_source),
        "molecule_definition": str(molecule_path),
        "critical_temperature_K": molecule.critical_temperature_K,
        "critical_pressure_Pa": molecule.critical_pressure_Pa,
        "acentric_factor": molecule.acentric_factor,
        "rigid": molecule.rigid,
        "atom_count": len(molecule.atoms),
        "atom_types": atom_types,
        "molar_mass_g_mol": _molar_mass_g_mol(molecule, parameters.pseudo_atoms),
        "net_charge_e": _net_charge_e(molecule, parameters.pseudo_atoms),
        "geometry_extent_A": geometry_extent,
        "max_lj_sigma_A": max_lj_sigma,
        "access_diameter_A": diameter,
        "diameter_source": diameter_source,
    }


def _infer_access_diameter_A(
    *,
    geometry_extent_A: float,
    max_lj_sigma_A: float | None,
    policy: dict[str, Any],
) -> tuple[float, str]:
    if max_lj_sigma_A is not None:
        scale = float(policy.get("sigma_scale", DEFAULT_DIAMETER_POLICY["sigma_scale"]))
        return max(geometry_extent_A, max_lj_sigma_A * scale), f"max(geometry_extent_A, max_lj_sigma_A * {scale:g})"
    fallback = float(policy.get("fallback_A", DEFAULT_DIAMETER_POLICY["fallback_A"]))
    return max(geometry_extent_A, fallback), f"max(geometry_extent_A, fallback_A={fallback:g})"


def _molecule_extent_A(molecule: CraftedMolecule) -> float:
    max_distance = 0.0
    atoms = list(molecule.atoms)
    for index, left in enumerate(atoms):
        for right in atoms[index + 1 :]:
            max_distance = max(
                max_distance,
                math.dist((left.x, left.y, left.z), (right.x, right.y, right.z)),
            )
    return max_distance


def _max_interacting_lj_sigma_A(atom_types: list[str], lj_parameters: dict[str, Any]) -> float | None:
    sigmas = [
        lj_parameters[atom_type].sigma_A
        for atom_type in atom_types
        if atom_type in lj_parameters and lj_parameters[atom_type].epsilon_K > 0.0
    ]
    return max(sigmas) if sigmas else None


def _molar_mass_g_mol(molecule: CraftedMolecule, pseudo_atoms: dict[str, Any]) -> float:
    mass = 0.0
    for atom in molecule.atoms:
        pseudo_atom = pseudo_atoms.get(atom.atom_type)
        if pseudo_atom is not None and pseudo_atom.mass > 0.0:
            mass += pseudo_atom.mass
    return mass


def _net_charge_e(molecule: CraftedMolecule, pseudo_atoms: dict[str, Any]) -> float:
    charge = 0.0
    for atom in molecule.atoms:
        pseudo_atom = pseudo_atoms.get(atom.atom_type)
        if pseudo_atom is not None:
            charge += pseudo_atom.charge
    return charge


def _unique(values: Any) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
