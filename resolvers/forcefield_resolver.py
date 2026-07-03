from __future__ import annotations

from pathlib import Path
from typing import Any


class ForcefieldResolver:
    def __init__(self, crafted_root: str | Path = "CRAFTED-2.0.0") -> None:
        self.forcefield_root = Path(crafted_root) / "FORCEFIELDS"

    def resolve(self, forcefield_config: dict[str, Any], components: list[str]) -> dict[str, Any]:
        framework = _normalize_forcefield_name(forcefield_config.get("framework", "UFF"))
        forcefield_dir = self.forcefield_root / framework
        if not forcefield_dir.exists():
            raise LookupError(f"Force field {framework!r} not found below {self.forcefield_root}.")

        base_files = {
            "force_field": forcefield_dir / "force_field.def",
            "mixing_rules": forcefield_dir / "force_field_mixing_rules.def",
            "pseudo_atoms": forcefield_dir / "pseudo_atoms.def",
        }
        missing = [name for name, path in base_files.items() if not path.exists()]
        if missing:
            raise LookupError(f"Force field {framework!r} is missing files: {', '.join(missing)}.")

        adsorbate_files = {}
        for component in components:
            component_name = component.upper()
            component_path = forcefield_dir / f"{component_name}.def"
            if not component_path.exists():
                raise LookupError(
                    f"Adsorbate definition for {component_name!r} not found in force field {framework!r}."
                )
            adsorbate_files[component_name] = str(component_path)

        return {
            "framework": framework,
            "adsorbate": forcefield_config.get("adsorbate", "auto"),
            "cross_interactions": forcefield_config.get("cross_interactions", "auto"),
            "files": {name: str(path) for name, path in base_files.items()},
            "adsorbates": adsorbate_files,
        }


def _normalize_forcefield_name(value: Any) -> str:
    normalized = str(value or "UFF").upper()
    if normalized == "AUTO":
        return "UFF"
    return normalized
