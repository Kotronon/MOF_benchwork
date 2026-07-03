from __future__ import annotations

from pathlib import Path
from typing import Any


class ReferenceResolver:
    def __init__(self, crafted_root: str | Path = "CRAFTED-2.0.0") -> None:
        self.isotherm_root = Path(crafted_root) / "ISOTHERM_FILES"

    def resolve(
        self,
        material_id: str,
        charge_scheme: str,
        forcefield: str,
        components: list[str],
        temperature_K: float,
        source: str = "crafted",
    ) -> list[dict[str, Any]]:
        if str(source).lower() != "crafted":
            return []
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
                references.append(_reference_entry(exact, component_name, "exact"))
                continue

            candidates = sorted(
                self.isotherm_root.glob(f"*_{material_id}_{forcefield}_{component_name}_*.csv")
            )
            references.extend(_reference_entry(path, component_name, "candidate") for path in candidates[:5])

        return references


def _reference_entry(path: Path, component: str, match_type: str) -> dict[str, str]:
    return {
        "component": component,
        "path": str(path),
        "matched_by": match_type,
    }
