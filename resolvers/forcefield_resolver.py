from __future__ import annotations

from pathlib import Path
from typing import Any


RASPA_COMPONENT_FILES = {
    "AR": "argon.def",
    "ARGON": "argon.def",
    "CH4": "methane.def",
    "METHANE": "methane.def",
    "HE": "helium.def",
    "HELIUM": "helium.def",
    "O2": "O2.def",
}


class ForcefieldResolver:
    def __init__(
        self,
        crafted_root: str | Path = "CRAFTED-2.0.0",
        raspa2_root: str | Path = "external/RASPA2",
    ) -> None:
        self.forcefield_root = Path(crafted_root) / "FORCEFIELDS"
        self.raspa2_root = Path(raspa2_root)

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
        adsorbate_sources = {}
        adsorbate_parameter_files = []
        adsorbate_parameter_keys = set()
        for component in components:
            component_name = component.upper()
            component_path = forcefield_dir / f"{component_name}.def"
            if component_path.exists():
                adsorbate_files[component_name] = str(component_path)
                adsorbate_sources[component_name] = "crafted"
                continue

            raspa_adsorbate = self._resolve_raspa2_adsorbate(component_name)
            if raspa_adsorbate is None:
                raise LookupError(
                    f"Adsorbate definition for {component_name!r} not found in force field {framework!r}."
                )
            adsorbate_files[component_name] = raspa_adsorbate["definition"]
            adsorbate_sources[component_name] = raspa_adsorbate["source"]
            parameter_key = raspa_adsorbate["source"]
            if parameter_key not in adsorbate_parameter_keys:
                adsorbate_parameter_files.append(
                    {
                        "source": raspa_adsorbate["source"],
                        "files": raspa_adsorbate["files"],
                    }
                )
                adsorbate_parameter_keys.add(parameter_key)

        return {
            "framework": framework,
            "source": "crafted",
            "adsorbate": forcefield_config.get("adsorbate", "auto"),
            "cross_interactions": forcefield_config.get("cross_interactions", "auto"),
            "files": {name: str(path) for name, path in base_files.items()},
            "adsorbates": adsorbate_files,
            "adsorbate_sources": adsorbate_sources,
            "adsorbate_parameter_files": adsorbate_parameter_files,
        }

    def _resolve_raspa2_adsorbate(self, component_name: str) -> dict[str, Any] | None:
        molecule_name = RASPA_COMPONENT_FILES.get(component_name)
        if molecule_name is None:
            return None

        molecule_path = self.raspa2_root / "molecules" / "ExampleDefinitions" / molecule_name
        parameter_dir = self.raspa2_root / "forcefield" / "ExampleMoleculeForceField"
        parameter_files = {
            "mixing_rules": parameter_dir / "force_field_mixing_rules.def",
            "pseudo_atoms": parameter_dir / "pseudo_atoms.def",
        }
        if not molecule_path.exists() or any(not path.exists() for path in parameter_files.values()):
            return None

        return {
            "definition": str(molecule_path),
            "source": "raspa2_example_molecule_forcefield",
            "files": {name: str(path) for name, path in parameter_files.items()},
        }


def _normalize_forcefield_name(value: Any) -> str:
    normalized = str(value or "UFF").upper()
    if normalized == "AUTO":
        return "UFF"
    return normalized
