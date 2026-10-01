"""Structure-based applicability assessment for extensible Module C runs."""

from __future__ import annotations

from math import ceil, isfinite
from pathlib import Path
from typing import Any

import numpy as np

from converter.cif_to_lammps_data import cell_perpendicular_widths
from converter.cif_to_lammps_data import parse_cif_atom_site_charges


REFERENCE_MATERIALS = {"zif8", "zif4", "mgmof74"}
OPEN_METAL_ELEMENTS = {
    "Mg", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
}


def assess_campaign_structure(
    config: dict[str, Any],
    resolved: dict[str, Any],
) -> dict[str, Any]:
    """Assess a campaign CIF without relying on a curated material whitelist."""
    components = config["adsorbates"]["components"]
    settings = config["benchmark"]["potential_benchmark"]
    validation_mode = settings["validation"]["mode"]
    cif_path = Path(resolved["material"]["cif_path"])
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    if components != ["CO2"]:
        return _unsupported(
            checks=[_check("co2_v1", False, ["CO2"], components)],
            reason="The extensible Module C V1 campaign currently supports CO2 only.",
        )

    try:
        from ase.data import covalent_radii, vdw_radii
        from ase.io import read
    except ImportError as exc:
        return _unsupported(
            checks=[_check("ase_available", False, True, False)],
            reason=f"ASE is required for Module C structure assessment: {exc}",
        )

    try:
        atoms = read(str(cif_path))
    except Exception as exc:
        return _unsupported(
            checks=[_check("readable_cif", False, "valid periodic CIF", str(exc))],
            reason=f"The framework CIF cannot be read: {exc}",
        )

    periodic = bool(all(atoms.pbc))
    cell_volume = float(atoms.get_volume()) if len(atoms) else 0.0
    checks.extend(
        [
            _check("readable_cif", True, "valid periodic CIF", str(cif_path)),
            _check("nonempty_structure", len(atoms) > 0, "> 0 atoms", len(atoms)),
            _check("periodic_structure", periodic, True, list(map(bool, atoms.pbc))),
            _check("positive_cell_volume", cell_volume > 0.0, "> 0 A^3", cell_volume),
        ]
    )
    if not len(atoms) or not periodic or cell_volume <= 0.0:
        return _unsupported(checks=checks, reason="The CIF is not a valid periodic structure.")

    symbols = sorted(set(atoms.get_chemical_symbols()))
    model_specs = _model_specs(settings)
    unsupported_by_model = {}
    for model in model_specs:
        supported = model.get("supported_elements")
        if supported is None:
            continue
        missing = sorted(set(symbols) - {str(value) for value in supported})
        if missing:
            unsupported_by_model[str(model.get("name", model.get("backend", "model")))] = missing
    checks.append(
        _check(
            "model_element_coverage",
            not unsupported_by_model,
            "all framework elements supported",
            {"elements": symbols, "unsupported": unsupported_by_model},
        )
    )
    if unsupported_by_model:
        return _unsupported(
            checks=checks,
            reason="At least one configured potential does not support all framework elements.",
        )

    forcefield = resolved.get("forcefield", {})
    adsorbate_paths = forcefield.get("adsorbates", {})
    missing_adsorbates = [
        component
        for component in components
        if not Path(str(adsorbate_paths.get(component, ""))).is_file()
    ]
    checks.append(
        _check(
            "adsorbate_definitions",
            not missing_adsorbates,
            "definition file for every adsorbate",
            {"missing": missing_adsorbates},
        )
    )
    if missing_adsorbates:
        return _unsupported(
            checks=checks,
            reason="At least one configured adsorbate definition is unavailable.",
        )

    primary_engine = settings.get("adsorption_engine")
    cross_engine = settings.get("cross_check_engine")
    needs_classical_charges = primary_engine == "lammps"
    checks_classical_charges = needs_classical_charges or cross_engine == "lammps"
    if checks_classical_charges:
        try:
            charge_count = len(parse_cif_atom_site_charges(cif_path))
        except (OSError, ValueError):
            charge_count = 0
        checks.append(
            _check(
                "classical_framework_charges",
                charge_count == len(atoms),
                f"{len(atoms)} explicit CIF charges",
                charge_count,
            )
        )
        if needs_classical_charges and charge_count != len(atoms):
            return _unsupported(
                checks=checks,
                reason=(
                    "The selected classical LAMMPS engine requires one explicit "
                    "framework charge per CIF atom."
                ),
            )
        if cross_engine == "lammps" and charge_count != len(atoms):
            warnings.append(
                "The optional LAMMPS cross-check is unavailable because the CIF "
                "does not contain explicit framework charges."
            )

    minimum_distance = _minimum_periodic_distance(atoms)
    checks.append(_check("minimum_atom_distance_A", minimum_distance >= 0.5, ">= 0.5", minimum_distance))
    if minimum_distance < 0.5:
        return _unsupported(checks=checks, reason="The CIF contains overlapping framework atoms.")

    cutoff = float(config["simulation"].get("cutoff_A", 6.0))
    repetitions = resolved["unit_cells"]
    repeated_cell = atoms.cell.array * np.asarray(repetitions)[:, None]
    widths = cell_perpendicular_widths(repeated_cell)
    minimum_image_ok = min(widths) + 1.0e-9 >= 2.0 * cutoff
    checks.append(
        _check(
            "minimum_image_supercell",
            minimum_image_ok,
            f"all perpendicular widths >= {2.0 * cutoff:g} A",
            {"unit_cells": repetitions, "widths_A": list(widths)},
        )
    )
    if not minimum_image_ok:
        return _unsupported(
            checks=checks,
            reason="The configured supercell is too small for the potential cutoff.",
        )

    accessible_fraction = _approximate_accessible_fraction(
        atoms,
        vdw_radii=vdw_radii,
        covalent_radii=covalent_radii,
    )
    pore_accessible = accessible_fraction > 0.001
    checks.append(
        _check(
            "approximate_co2_accessible_fraction",
            pore_accessible,
            "> 0.001",
            accessible_fraction,
        )
    )
    if not pore_accessible:
        return _unsupported(
            checks=checks,
            reason="No CO2-accessible grid volume was detected in the supplied rigid structure.",
        )

    open_metals = sorted(set(symbols) & OPEN_METAL_ELEMENTS)
    cp2k = settings.get("active_learning", {}).get("cp2k", {})
    dft_supported = cp2k.get("supported_elements")
    missing_dft_elements = (
        sorted(set(symbols) - {str(value) for value in dft_supported})
        if isinstance(dft_supported, list)
        else []
    )
    checks.append(
        _check(
            "dft_pseudopotential_coverage",
            not missing_dft_elements,
            "GTH pseudopotential configuration covers all elements",
            {"missing": missing_dft_elements},
        )
    )
    if validation_mode == "active_learning" and missing_dft_elements:
        return _unsupported(
            checks=checks,
            reason="The configured CP2K pseudopotentials do not cover all elements.",
        )
    dft_configuration_status = "not_required"
    if open_metals:
        if (
            validation_mode == "active_learning"
            and not bool(cp2k.get("configuration_reviewed", False))
        ):
            dft_configuration_status = "requires_dft_configuration"
            warnings.append(
                "Transition-metal sites are present. Set "
                "active_learning.cp2k.configuration_reviewed=true only after "
                "reviewing charge, multiplicity, pseudopotentials, and dispersion."
            )
        elif validation_mode == "active_learning":
            dft_configuration_status = "reviewed"
            warnings.append(
                "Transition-metal sites are present; the run declares its CP2K "
                "charge, spin, pseudopotential, and dispersion choices reviewed."
            )
        else:
            warnings.append(
                "Transition-metal sites are present; quantitative DFT adaptation "
                "would require explicit charge, spin, pseudopotential, and "
                "dispersion review."
            )
    elif validation_mode == "active_learning":
        dft_configuration_status = "default_closed_shell"
    checks.append(
        {
            "name": "open_metal_sites",
            "status": "warning" if open_metals else "passed",
            "expected": "review when transition metals are present",
            "actual": open_metals,
            **({"severity": "warning"} if open_metals else {}),
        }
    )

    material_key = _normalize_material_id(resolved["material"]["material_id"])
    known_reference = material_key in REFERENCE_MATERIALS
    if validation_mode == "reference" and not known_reference:
        warnings.append("No built-in Module C regression reference is registered for this MOF.")
    potential_status = (
        "reference_candidate"
        if known_reference and validation_mode == "reference"
        else "adaptation_required"
        if validation_mode == "active_learning"
        else "screening_only"
    )
    reference_status = (
        "literature_comparison"
        if known_reference
        else "no_external_reference"
    )
    reason = (
        "The structure is technically compatible with Module C and has a "
        "registered literature or regression reference."
        if known_reference
        else "The structure is technically compatible with Module C. Results "
        "for an unreferenced material are screening predictions until DFT "
        "adaptation or an external reference is available."
    )
    return {
        "status": "warning" if warnings or potential_status == "screening_only" else "supported",
        "current_module_capability": "module_c_campaign",
        "can_attempt_simulation": True,
        "requires_user_confirmation": False,
        "recommended_module": "C",
        "checks": checks,
        "warnings": warnings,
        "validation_mode": validation_mode,
        "sampling_status": "not_started",
        "potential_status": potential_status,
        "reference_status": reference_status,
        "dft_configuration_status": dft_configuration_status,
        "reason": reason,
    }


def automatic_supercell(cif_path: str | Path, cutoff_A: float) -> list[int]:
    """Return conservative repetitions satisfying the minimum-image cutoff."""
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError("ASE is required for automatic Module C supercells.") from exc
    atoms = read(str(cif_path))
    widths = cell_perpendicular_widths(atoms.cell.array)
    if any(not isfinite(width) or width <= 0.0 for width in widths):
        raise ValueError("Cannot determine a valid supercell from the CIF cell.")
    return [max(1, int(ceil((2.0 * float(cutoff_A)) / width))) for width in widths]


def _model_specs(settings: dict[str, Any]) -> list[dict[str, Any]]:
    campaign = settings.get("campaign", {})
    models = campaign.get("models", settings.get("models", [])) if isinstance(campaign, dict) else []
    if not isinstance(models, list):
        return []
    return [item for item in models if isinstance(item, dict)]


def _minimum_periodic_distance(atoms: Any) -> float:
    if len(atoms) < 2:
        return float("inf")
    distances = atoms.get_all_distances(mic=True)
    distances[distances <= 1.0e-12] = np.inf
    return float(np.min(distances))


def _approximate_accessible_fraction(
    atoms: Any,
    *,
    vdw_radii: Any,
    covalent_radii: Any,
    grid_size: int = 8,
    co2_probe_radius_A: float = 1.65,
) -> float:
    scaled_atoms = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    numbers = atoms.get_atomic_numbers()
    radii = []
    for number in numbers:
        radius = float(vdw_radii[number]) if number < len(vdw_radii) else float("nan")
        if not isfinite(radius):
            radius = 1.2 * float(covalent_radii[number])
        radii.append(radius)
    exclusion = np.asarray(radii) + co2_probe_radius_A * 0.35
    accessible = 0
    total = grid_size**3
    offsets = (np.arange(grid_size, dtype=float) + 0.5) / grid_size
    for x in offsets:
        for y in offsets:
            for z in offsets:
                delta = np.asarray([x, y, z]) - scaled_atoms
                delta -= np.round(delta)
                cartesian = delta @ cell
                distances = np.linalg.norm(cartesian, axis=1)
                if np.all(distances >= exclusion):
                    accessible += 1
    return accessible / total


def _check(name: str, passed: bool, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expected": expected,
        "actual": actual,
        **({} if passed else {"severity": "unsupported"}),
    }


def _unsupported(*, checks: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "current_module_capability": "module_c_campaign",
        "can_attempt_simulation": False,
        "requires_user_confirmation": False,
        "recommended_module": "C",
        "checks": checks,
        "warnings": [],
        "validation_mode": None,
        "sampling_status": "not_started",
        "potential_status": "failed",
        "reference_status": "not_assessed",
        "reason": reason,
    }


def _normalize_material_id(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
