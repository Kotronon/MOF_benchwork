from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.nist_isodb_parser import find_nist_isotherm_candidates


class ReferenceResolver:
    def __init__(
        self,
        crafted_root: str | Path = "CRAFTED-2.0.0",
        nist_root: str | Path = "isodb-library",
    ) -> None:
        self.isotherm_root = Path(crafted_root) / "ISOTHERM_FILES"
        self.nist_root = Path(nist_root)

    def resolve(
        self,
        material_id: str,
        charge_scheme: str,
        forcefield: str,
        components: list[str],
        temperature_K: float,
        source: str = "crafted",
        material_aliases: list[str] | None = None,
        pressures_bar: list[float] | None = None,
        temperature_tolerance_K: float = 2.0,
        max_nist_candidates: int = 5,
        preferred_sources: list[str] | None = None,
        deduplicate_nist_by_doi: bool = True,
        exclude_nist_outliers: bool = True,
        nist_max_loading_mol_per_kg: float | None = None,
        nist_max_reference_ratio: float | None = None,
        allowed_reference_basis: list[str] | None = None,
        allow_unknown_reference_basis: bool = True,
        allow_excess_reference_basis: bool = True,
        require_known_nist_article_source: bool = False,
    ) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        primary_reference_points: list[dict[str, Any]] = []
        for source_name in _source_order(source, preferred_sources):
            if source_name == "crafted":
                crafted_references = self._resolve_crafted(
                    material_id=material_id,
                    charge_scheme=charge_scheme,
                    forcefield=forcefield,
                    components=components,
                    temperature_K=temperature_K,
                )
                references.extend(crafted_references)
                primary_reference_points = _load_crafted_reference_points(crafted_references)
            elif source_name == "nist_isodb":
                references.extend(
                    self._resolve_nist(
                        material_id=material_id,
                        material_aliases=material_aliases or [],
                        components=components,
                        temperature_K=temperature_K,
                        pressures_bar=pressures_bar,
                        temperature_tolerance_K=temperature_tolerance_K,
                        max_candidates=max_nist_candidates,
                        deduplicate_by_doi=deduplicate_nist_by_doi,
                        exclude_outliers=exclude_nist_outliers,
                        max_loading_mol_per_kg=nist_max_loading_mol_per_kg,
                        reference_points=primary_reference_points,
                        max_reference_ratio=nist_max_reference_ratio,
                        allowed_reference_basis=allowed_reference_basis,
                        allow_unknown_reference_basis=allow_unknown_reference_basis,
                        allow_excess_reference_basis=allow_excess_reference_basis,
                        require_known_article_source=require_known_nist_article_source,
                    )
                )

        _mark_primary_reference(references, source, preferred_sources)
        return references

    def _resolve_crafted(
        self,
        *,
        material_id: str,
        charge_scheme: str,
        forcefield: str,
        components: list[str],
        temperature_K: float,
    ) -> list[dict[str, Any]]:
        if not self.isotherm_root.exists():
            return []

        references = []
        temperature_tag = str(int(round(temperature_K)))
        for component in components:
            component_name = component.upper()
            exact = self.isotherm_root / (
                f"{charge_scheme}_{material_id}_{forcefield}_{component_name}_{temperature_tag}.csv"
            )
            if exact.exists():
                references.append(_reference_entry(exact, component_name, "exact", "crafted", "crafted_csv"))
                continue

            candidates = sorted(
                path
                for path in self.isotherm_root.glob(
                    f"*_{material_id}_{forcefield}_{component_name}_*.csv"
                )
                if not path.name.startswith("._")
            )
            references.extend(
                _reference_entry(path, component_name, "candidate", "crafted", "crafted_csv")
                for path in candidates[:5]
            )

        return references

    def _resolve_nist(
        self,
        *,
        material_id: str,
        material_aliases: list[str],
        components: list[str],
        temperature_K: float,
        pressures_bar: list[float] | None,
        temperature_tolerance_K: float,
        max_candidates: int,
        deduplicate_by_doi: bool,
        exclude_outliers: bool,
        max_loading_mol_per_kg: float | None,
        reference_points: list[dict[str, Any]] | None,
        max_reference_ratio: float | None,
        allowed_reference_basis: list[str] | None,
        allow_unknown_reference_basis: bool,
        allow_excess_reference_basis: bool,
        require_known_article_source: bool,
    ) -> list[dict[str, Any]]:
        material_names = _dedupe([material_id, *material_aliases])
        candidates = find_nist_isotherm_candidates(
            self.nist_root,
            material_names=material_names,
            components=components,
            temperature_K=temperature_K,
            pressures_bar=pressures_bar,
            temperature_tolerance_K=temperature_tolerance_K,
            max_candidates=max_candidates,
            deduplicate_by_doi=deduplicate_by_doi,
            require_known_article_source=require_known_article_source,
            max_loading_mol_per_kg=max_loading_mol_per_kg,
            reference_points=reference_points,
            max_reference_ratio=max_reference_ratio,
            allowed_reference_basis=allowed_reference_basis,
            allow_unknown_reference_basis=allow_unknown_reference_basis,
            allow_excess_reference_basis=allow_excess_reference_basis,
        )
        if not exclude_outliers:
            return candidates
        return [
            {
                **candidate,
                "use_in_evaluation": not candidate.get("excluded", False),
            }
            for candidate in candidates
        ]


def _reference_entry(path: Path, component: str, match_type: str, source: str, file_format: str) -> dict[str, Any]:
    return {
        "component": component,
        "path": str(path),
        "matched_by": match_type,
        "source": source,
        "format": file_format,
        "selected": False,
    }


def _load_crafted_reference_points(references: list[dict[str, Any]]) -> list[dict[str, float]]:
    for reference in references:
        if reference.get("source") != "crafted":
            continue
        path = Path(reference["path"])
        points: list[dict[str, float]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                pressure_pa, loading_mol_per_kg, error_mol_per_kg = [
                    float(value) for value in stripped.split(",")[:3]
                ]
                points.append(
                    {
                        "pressure_Pa": pressure_pa,
                        "loading_mol_per_kg": loading_mol_per_kg,
                        "error_mol_per_kg": error_mol_per_kg,
                    }
                )
        return sorted(points, key=lambda point: point["pressure_Pa"])
    return []


def _source_order(source: str, preferred_sources: list[str] | None = None) -> list[str]:
    normalized = str(source or "crafted").strip().casefold()
    if normalized == "auto":
        if preferred_sources:
            ordered_sources = [
                source_name
                for source_name in (_normalize_source_name(value) for value in preferred_sources)
                if source_name in {"crafted", "nist_isodb"}
            ]
            if ordered_sources:
                return ordered_sources
        return ["crafted", "nist_isodb"]
    if normalized in {"nist", "nist_isodb"}:
        return ["nist_isodb"]
    if normalized == "crafted":
        return ["crafted"]
    return []


def _mark_primary_reference(
    references: list[dict[str, Any]],
    source: str,
    preferred_sources: list[str] | None = None,
) -> None:
    if not references:
        return
    source_order = _source_order(source, preferred_sources)
    for source_name in source_order:
        for reference in references:
            if reference["source"] == source_name:
                reference["selected"] = True
                return
    references[0]["selected"] = True


def _normalize_source_name(source: str) -> str:
    normalized = str(source or "").strip().casefold()
    if normalized == "nist":
        return "nist_isodb"
    return normalized


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
