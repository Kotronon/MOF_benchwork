from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ADSORBATE_NAME_ALIASES = {
    "CO2": {"co2", "carbon dioxide"},
    "N2": {"n2", "nitrogen"},
    "CH4": {"ch4", "methane"},
    "H2": {"h2", "hydrogen"},
}

ADSORBATE_MOLAR_MASS_G_MOL = {
    "CO2": 44.0095,
    "N2": 28.0134,
    "CH4": 16.0425,
    "H2": 2.01588,
}


def find_nist_isotherm_candidates(
    library_root: str | Path,
    *,
    material_names: list[str],
    components: list[str],
    temperature_K: float,
    pressures_bar: list[float] | None = None,
    temperature_tolerance_K: float = 2.0,
    max_candidates: int = 5,
    deduplicate_by_doi: bool = True,
    require_known_article_source: bool = False,
    max_loading_mol_per_kg: float | None = None,
    reference_points: list[dict[str, Any]] | None = None,
    max_reference_ratio: float | None = None,
    allowed_reference_basis: list[str] | None = None,
    allow_unknown_reference_basis: bool = True,
    allow_excess_reference_basis: bool = True,
) -> list[dict[str, Any]]:
    """Find matching NIST ISODB JSON isotherms and rank them deterministically."""
    root = Path(library_root)
    search_root = root / "Library" if (root / "Library").exists() else root
    if not search_root.exists():
        return []

    candidates: list[dict[str, Any]] = []
    for path in search_root.glob("*/*.json"):
        try:
            metadata = inspect_nist_isotherm(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue

        if not _material_matches(metadata["material"], material_names):
            continue
        if not _adsorbates_match(metadata["adsorbates"], components):
            continue

        reference_temperature = metadata["temperature_K"]
        if reference_temperature is None:
            continue
        temperature_delta = abs(reference_temperature - temperature_K)
        if temperature_delta > temperature_tolerance_K:
            continue

        score = _candidate_score(metadata, temperature_delta, pressures_bar)
        exclusion_reasons = _metadata_exclusion_reasons(
            metadata,
            max_loading_mol_per_kg=max_loading_mol_per_kg,
            reference_points=reference_points,
            max_reference_ratio=max_reference_ratio,
            allowed_reference_basis=allowed_reference_basis,
            allow_unknown_reference_basis=allow_unknown_reference_basis,
            allow_excess_reference_basis=allow_excess_reference_basis,
            component=components[0] if components else "CO2",
            path=path,
        )
        candidates.append(
            {
                "component": components[0].upper() if len(components) == 1 else ",".join(components),
                "path": str(path),
                "matched_by": "nist_isodb_ranked",
                "source": "nist_isodb",
                "format": "nist_json",
                "selected": False,
                "score": score,
                "material": metadata["material"],
                "adsorbates": metadata["adsorbates"],
                "temperature_K": reference_temperature,
                "temperature_delta_K": temperature_delta,
                "pressure_units": metadata["pressure_units"],
                "adsorption_units": metadata["adsorption_units"],
                "pressure_min_bar": metadata["pressure_min_bar"],
                "pressure_max_bar": metadata["pressure_max_bar"],
                "point_count": metadata["point_count"],
                "doi": metadata["doi"],
                "category": metadata["category"],
                "article_source": metadata["article_source"],
                "adsorption_basis": metadata["adsorption_basis"],
                "excluded": bool(exclusion_reasons),
                "exclusion_reasons": exclusion_reasons,
            }
        )

    if require_known_article_source:
        for candidate in candidates:
            if _is_unknown_metadata(candidate["article_source"]) and not candidate["excluded"]:
                candidate["excluded"] = True
                candidate["exclusion_reasons"] = [
                    *candidate["exclusion_reasons"],
                    "unknown_article_source",
                ]

    candidates.sort(key=_candidate_sort_key)
    if deduplicate_by_doi:
        candidates = _deduplicate_candidates_by_doi(candidates)
    active_candidates = [candidate for candidate in candidates if not candidate.get("excluded")]
    excluded_candidates = [candidate for candidate in candidates if candidate.get("excluded")]
    return active_candidates[:max_candidates] + excluded_candidates[:max_candidates]


def inspect_nist_isotherm(path: str | Path) -> dict[str, Any]:
    """Read metadata from one NIST ISODB JSON file."""
    data = _load_json(path)
    pressures_bar = [
        _pressure_to_bar(point["pressure"], data.get("pressureUnits", ""))
        for point in data.get("isotherm_data", [])
        if "pressure" in point
    ]
    return {
        "path": str(path),
        "doi": data.get("DOI", ""),
        "material": (data.get("adsorbent") or {}).get("name", ""),
        "adsorbates": [adsorbate.get("name", "") for adsorbate in data.get("adsorbates", [])],
        "temperature_K": float(data["temperature"]) if data.get("temperature") is not None else None,
        "pressure_units": data.get("pressureUnits", ""),
        "adsorption_units": data.get("adsorptionUnits", ""),
        "category": data.get("category", ""),
        "article_source": data.get("articleSource", ""),
        "adsorption_basis": _infer_adsorption_basis(data),
        "pressure_min_bar": min(pressures_bar) if pressures_bar else None,
        "pressure_max_bar": max(pressures_bar) if pressures_bar else None,
        "point_count": len(data.get("isotherm_data", [])),
    }


def load_nist_isotherm(
    path: str | Path,
    *,
    component: str = "CO2",
    framework_mass_amu: float | None = None,
) -> list[dict[str, Any]]:
    """Load one NIST ISODB JSON isotherm as normalized pressure/loading points."""
    data = _load_json(path)
    adsorption_units = data.get("adsorptionUnits", "")
    pressure_units = data.get("pressureUnits", "")
    basis = _infer_adsorption_basis(data)

    points: list[dict[str, Any]] = []
    for raw_point in data.get("isotherm_data", []):
        pressure = raw_point.get("pressure")
        adsorption = _adsorption_value(raw_point, component)
        if pressure is None or adsorption is None:
            continue

        loading = _loading_to_mol_per_kg(
            adsorption,
            adsorption_units,
            component=component,
            framework_mass_amu=framework_mass_amu,
        )
        if loading is None:
            continue

        points.append(
            {
                "pressure_Pa": _pressure_to_pa(pressure, pressure_units),
                "loading_mol_per_kg": loading,
                "error_mol_per_kg": 0.0,
                "source": "nist_isodb",
                "doi": data.get("DOI", ""),
                "adsorption_basis": basis,
                "path": str(path),
            }
        )
    return sorted(points, key=lambda point: point["pressure_Pa"])


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _adsorption_value(raw_point: dict[str, Any], component: str) -> float | None:
    target = _normalized_adsorbate_names(component)
    species_data = raw_point.get("species_data", [])
    for species in species_data:
        name = str(species.get("name", "")).casefold()
        if name and name in target:
            return float(species["adsorption"])
    if len(species_data) == 1 and species_data[0].get("adsorption") is not None:
        return float(species_data[0]["adsorption"])
    if raw_point.get("total_adsorption") is not None:
        return float(raw_point["total_adsorption"])
    return None


def _pressure_to_pa(value: Any, units: str) -> float:
    pressure = float(value)
    normalized = units.strip().casefold()
    if normalized == "bar":
        return pressure * 100000.0
    if normalized == "pa":
        return pressure
    if normalized == "kpa":
        return pressure * 1000.0
    if normalized == "mpa":
        return pressure * 1000000.0
    if normalized == "atm":
        return pressure * 101325.0
    if normalized in {"torr", "mmhg"}:
        return pressure * 133.32236842105263
    if normalized == "mbar":
        return pressure * 100.0
    raise ValueError(f"Unsupported NIST pressure unit: {units!r}.")


def _pressure_to_bar(value: Any, units: str) -> float:
    return _pressure_to_pa(value, units) / 100000.0


def _loading_to_mol_per_kg(
    value: Any,
    units: str,
    *,
    component: str,
    framework_mass_amu: float | None,
) -> float | None:
    loading = float(value)
    normalized = units.strip().casefold().replace(" ", "")
    if normalized in {"mmol/g", "mol/kg"}:
        return loading
    if normalized == "mg/g":
        return loading / _molar_mass(component)
    if normalized in {"cm3(stp)/g", "cm^3(stp)/g", "ml/g", "cm3/g"}:
        return loading * 1000.0 / 22414.0
    if normalized in {"molecules/unitcell", "molecule/unitcell", "molecules/uc", "molecules/cell"}:
        if framework_mass_amu is None:
            return None
        return 1000.0 * loading / framework_mass_amu
    return None


def _molar_mass(component: str) -> float:
    key = component.upper()
    if key not in ADSORBATE_MOLAR_MASS_G_MOL:
        raise ValueError(f"No molar mass known for adsorbate {component!r}.")
    return ADSORBATE_MOLAR_MASS_G_MOL[key]


def _material_matches(reference_material: str, material_names: list[str]) -> bool:
    reference = _normalize_name(reference_material)
    return reference in {_normalize_name(name) for name in material_names if name}


def _adsorbates_match(reference_adsorbates: list[str], components: list[str]) -> bool:
    if len(reference_adsorbates) != len(components):
        return False
    unmatched = {_normalize_name(name) for name in reference_adsorbates}
    for component in components:
        names = {_normalize_name(name) for name in _normalized_adsorbate_names(component)}
        overlap = unmatched & names
        if not overlap:
            return False
        unmatched.remove(next(iter(overlap)))
    return not unmatched


def _normalized_adsorbate_names(component: str) -> set[str]:
    return ADSORBATE_NAME_ALIASES.get(component.upper(), {component.casefold()})


def _infer_adsorption_basis(data: dict[str, Any]) -> str:
    haystack = " ".join(
        str(data.get(key, ""))
        for key in ("adsorptionType", "adsorption_type", "isotherm_type", "category")
    ).casefold()
    if "absolute" in haystack:
        return "absolute"
    if "excess" in haystack:
        return "excess"
    return "unknown"


def _candidate_score(
    metadata: dict[str, Any],
    temperature_delta: float,
    pressures_bar: list[float] | None,
) -> dict[str, Any]:
    pressure_min = metadata["pressure_min_bar"]
    pressure_max = metadata["pressure_max_bar"]
    overlap_count = 0
    if pressure_min is not None and pressure_max is not None and pressures_bar:
        overlap_count = sum(1 for pressure in pressures_bar if pressure_min <= pressure <= pressure_max)
    return {
        "temperature_delta_K": temperature_delta,
        "pressure_overlap_count": overlap_count,
        "unit_rank": _adsorption_unit_rank(metadata["adsorption_units"]),
        "point_count": metadata["point_count"],
    }


def _adsorption_unit_rank(units: str) -> int:
    normalized = units.strip().casefold().replace(" ", "")
    if normalized in {"mmol/g", "mol/kg"}:
        return 0
    if normalized in {"mg/g", "cm3(stp)/g", "cm^3(stp)/g", "ml/g", "cm3/g"}:
        return 1
    if normalized in {"molecules/unitcell", "molecule/unitcell", "molecules/uc", "molecules/cell"}:
        return 2
    return 9


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    score = candidate["score"]
    return (
        bool(candidate.get("excluded")),
        score["temperature_delta_K"],
        score["unit_rank"],
        -score["pressure_overlap_count"],
        -score["point_count"],
        candidate["doi"],
        candidate["path"],
    )


def _metadata_exclusion_reasons(
    metadata: dict[str, Any],
    *,
    max_loading_mol_per_kg: float | None,
    reference_points: list[dict[str, Any]] | None,
    max_reference_ratio: float | None,
    allowed_reference_basis: list[str] | None,
    allow_unknown_reference_basis: bool,
    allow_excess_reference_basis: bool,
    component: str,
    path: Path,
) -> list[str]:
    reasons: list[str] = []
    basis = str(metadata.get("adsorption_basis") or "unknown")
    allowed_basis = {str(value) for value in allowed_reference_basis or []}
    if allowed_basis and basis not in allowed_basis and not (basis == "unknown" and allow_unknown_reference_basis):
        reasons.append(f"basis_not_allowed_{basis}")
    if basis == "unknown" and not allow_unknown_reference_basis:
        reasons.append("unknown_adsorption_basis")
    if basis == "excess" and not allow_excess_reference_basis:
        reasons.append("excess_reference_for_absolute_simulation")

    points: list[dict[str, Any]] = []
    if max_loading_mol_per_kg is not None:
        try:
            points = load_nist_isotherm(path, component=component)
        except (OSError, ValueError, json.JSONDecodeError):
            points = []
        if points:
            max_loading = max(float(point["loading_mol_per_kg"]) for point in points)
            if max_loading > max_loading_mol_per_kg:
                reasons.append(f"max_loading_gt_{max_loading_mol_per_kg:g}_mol_per_kg")
    if reference_points and max_reference_ratio is not None:
        if not points:
            try:
                points = load_nist_isotherm(path, component=component)
            except (OSError, ValueError, json.JSONDecodeError):
                points = []
        if _exceeds_reference_ratio(points, reference_points, max_reference_ratio):
            reasons.append(f"loading_ratio_gt_{max_reference_ratio:g}_vs_primary_reference")
    return reasons


def _exceeds_reference_ratio(
    points: list[dict[str, Any]],
    reference_points: list[dict[str, Any]],
    max_reference_ratio: float,
) -> bool:
    for point in points:
        pressure_pa = float(point["pressure_Pa"])
        reference = _reference_at_pressure(reference_points, pressure_pa)
        if reference is None:
            continue
        reference_loading = float(reference["loading_mol_per_kg"])
        candidate_loading = float(point["loading_mol_per_kg"])
        if reference_loading > 0 and candidate_loading / reference_loading > max_reference_ratio:
            return True
    return False


def _reference_at_pressure(reference_points: list[dict[str, Any]], pressure_pa: float) -> dict[str, Any] | None:
    if not reference_points:
        return None
    for point in reference_points:
        if abs(float(point["pressure_Pa"]) - pressure_pa) <= max(1e-9, pressure_pa * 1e-9):
            return point

    lower = None
    upper = None
    for point in reference_points:
        point_pressure = float(point["pressure_Pa"])
        if point_pressure < pressure_pa:
            lower = point
        elif point_pressure > pressure_pa:
            upper = point
            break
    if lower is None or upper is None:
        return None

    span = float(upper["pressure_Pa"]) - float(lower["pressure_Pa"])
    fraction = (pressure_pa - float(lower["pressure_Pa"])) / span
    return {
        "pressure_Pa": pressure_pa,
        "loading_mol_per_kg": float(lower["loading_mol_per_kg"])
        + fraction * (float(upper["loading_mol_per_kg"]) - float(lower["loading_mol_per_kg"])),
    }


def _deduplicate_candidates_by_doi(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        doi = str(candidate.get("doi") or candidate["path"]).casefold()
        if doi in seen:
            continue
        seen.add(doi)
        selected.append(candidate)
    return selected


def _is_unknown_metadata(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"", "unknown"}


def _normalize_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
