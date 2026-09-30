"""Model-specific atom labels that preserve physical guest properties."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def apply_adsorbate_species_aliases(
    adsorbate: Any,
    aliases: dict[str, str] | None,
) -> tuple[Any, dict[str, Any]]:
    """Return a copied guest with model labels and its physical metadata.

    Some interaction models use otherwise unrelated element symbols to
    distinguish chemically identical host and guest atoms. The Goeminne ZIF-8
    model, for example, represents the carbon and oxygen atoms of guest CO2 as
    Co and Os. Masses and the original symbols remain recorded explicitly.
    """
    guest = adsorbate.copy()
    physical_symbols = guest.get_chemical_symbols()
    physical_masses = guest.get_masses().copy()
    normalized = _validate_alias_mapping(aliases)
    model_symbols = [normalized.get(symbol, symbol) for symbol in physical_symbols]

    physical_species = np.asarray(physical_symbols, dtype="U8")
    if "physical_species" in guest.arrays:
        guest.set_array("physical_species", physical_species)
    else:
        guest.new_array("physical_species", physical_species)
    if normalized:
        try:
            guest.set_chemical_symbols(model_symbols)
        except Exception as exc:
            raise ValueError(
                "Adsorbate species aliases must map to valid ASE element symbols."
            ) from exc
        guest.set_masses(physical_masses)

    replacements = Counter(
        f"{physical}->{model}"
        for physical, model in zip(physical_symbols, model_symbols)
        if physical != model
    )
    return guest, {
        "aliases": normalized,
        "physical_symbols": physical_symbols,
        "model_symbols": model_symbols,
        "replacement_counts": dict(sorted(replacements.items())),
        "physical_masses_preserved": True,
    }


def validate_model_species_mapping(
    model_symbols: list[str],
    mapping: dict[str, str] | bool | None,
) -> None:
    """Reject explicit calculator mappings that omit generated symbols."""
    if not isinstance(mapping, dict):
        return
    missing = sorted(set(model_symbols) - set(mapping))
    if missing:
        raise ValueError(
            "Model species mapping does not cover generated symbols: "
            + ", ".join(missing)
        )


def _validate_alias_mapping(
    aliases: dict[str, str] | None,
) -> dict[str, str]:
    if aliases is None:
        return {}
    if not isinstance(aliases, dict) or not all(
        isinstance(source, str)
        and source.strip()
        and isinstance(target, str)
        and target.strip()
        for source, target in aliases.items()
    ):
        raise TypeError(
            "Adsorbate species aliases must be an object of non-empty string pairs."
        )
    return {
        source.strip(): target.strip()
        for source, target in aliases.items()
    }
