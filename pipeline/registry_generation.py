from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pipeline.adsorbate_registry import infer_adsorbate_properties


def generate_crafted_material_registry(crafted_root: str | Path = "CRAFTED-2.0.0") -> dict[str, Any]:
    """Generate a material registry from CRAFTED CIF and geometry files."""
    root = Path(crafted_root)
    registry: dict[str, Any] = {}

    for cif_path in sorted((root / "CIF_FILES").glob("*/*.cif")):
        if cif_path.name.startswith("._") or cif_path.parent.name.startswith("._"):
            continue
        material_id = cif_path.stem
        entry = registry.setdefault(
            material_id,
            {
                "material_id": material_id,
                "charge_schemes": {},
                "geometry": None,
            },
        )
        entry["charge_schemes"][cif_path.parent.name] = str(cif_path)

    geometry_path = root / "RAC_DBSCAN" / "CRAFTED_MOF_geometric.csv"
    if geometry_path.exists():
        with geometry_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                material_id = row.get("FrameworkName")
                if not material_id:
                    continue
                entry = registry.setdefault(
                    material_id,
                    {
                        "material_id": material_id,
                        "charge_schemes": {},
                        "geometry": None,
                    },
                )
                entry["geometry"] = {
                    "source": str(geometry_path),
                    "largest_cavity_diameter_A": _optional_float(row.get("D_is")),
                    "pore_limiting_diameter_A": _optional_float(row.get("D_fs")),
                    "accessible_surface_area_m2_g": _optional_float(row.get("ASA_m^2/g")),
                    "accessible_volume_fraction": _optional_float(row.get("AV_Volume_fraction")),
                    "pore_volume_cm3_g": _optional_float(row.get("AV_cm^3/g")),
                    "density_g_cm3": _optional_float(row.get("Density")),
                }

    return dict(sorted(registry.items()))


RASPA_COMPONENT_ALIASES = {
    "argon": "AR",
    "helium": "HE",
    "methane": "CH4",
}


def generate_adsorbate_registry(
    crafted_root: str | Path = "CRAFTED-2.0.0",
    raspa2_root: str | Path = "external/RASPA2",
) -> dict[str, Any]:
    """Generate adsorbate properties from available CRAFTED and RASPA2 forcefield files."""
    registry = generate_crafted_adsorbate_registry(crafted_root)
    _merge_registry(registry, generate_raspa2_adsorbate_registry(raspa2_root))
    return dict(sorted(registry.items()))


def generate_forcefield_registry(
    crafted_root: str | Path = "CRAFTED-2.0.0",
    raspa2_root: str | Path = "external/RASPA2",
) -> dict[str, Any]:
    """Generate available framework and molecule force-field metadata."""
    crafted = generate_crafted_forcefield_registry(crafted_root)
    raspa2 = generate_raspa2_forcefield_registry(raspa2_root)
    registry = {**crafted, **raspa2}
    return dict(sorted(registry.items()))


def generate_molecule_definition_registry(
    crafted_root: str | Path = "CRAFTED-2.0.0",
    raspa2_root: str | Path = "external/RASPA2",
) -> dict[str, Any]:
    """Generate a registry of all discovered molecule definition files."""
    registry: dict[str, Any] = {}
    crafted_adsorbates = generate_crafted_adsorbate_registry(crafted_root)
    for forcefield_name, forcefield in generate_crafted_forcefield_registry(crafted_root).items():
        for component in forcefield["adsorbates"]:
            molecule_path = Path(crafted_root) / "FORCEFIELDS" / forcefield_name / f"{component}.def"
            _add_molecule_definition(
                registry,
                component=component,
                source="crafted",
                forcefield=forcefield_name,
                molecule_definition=molecule_path,
                properties_inferred=component in crafted_adsorbates
                and forcefield_name in crafted_adsorbates.get(component, {}),
            )

    raspa_registry = generate_raspa2_adsorbate_registry(raspa2_root)
    molecule_dir = Path(raspa2_root) / "molecules" / "ExampleDefinitions"
    if molecule_dir.exists():
        for molecule_path in sorted(molecule_dir.glob("*.def")):
            component = _canonical_component_name(molecule_path.stem)
            _add_molecule_definition(
                registry,
                component=component,
                source="raspa2",
                forcefield="RASPA2_ExampleMoleculeForceField",
                molecule_definition=molecule_path,
                properties_inferred=component in raspa_registry
                and "RASPA2_ExampleMoleculeForceField" in raspa_registry.get(component, {}),
            )

    return dict(sorted(registry.items()))


def generate_crafted_forcefield_registry(crafted_root: str | Path = "CRAFTED-2.0.0") -> dict[str, Any]:
    root = Path(crafted_root)
    registry: dict[str, Any] = {}
    for forcefield_dir in sorted((root / "FORCEFIELDS").glob("*")):
        if not forcefield_dir.is_dir() or forcefield_dir.name.startswith("._"):
            continue
        base_files = {
            "force_field": forcefield_dir / "force_field.def",
            "mixing_rules": forcefield_dir / "force_field_mixing_rules.def",
            "pseudo_atoms": forcefield_dir / "pseudo_atoms.def",
        }
        if any(not path.exists() for path in base_files.values()):
            continue
        adsorbates = sorted(
            molecule_def.stem.upper()
            for molecule_def in forcefield_dir.glob("*.def")
            if molecule_def.name not in {"force_field.def", "force_field_mixing_rules.def", "pseudo_atoms.def"}
        )
        registry[forcefield_dir.name] = {
            "name": forcefield_dir.name,
            "source": "crafted",
            "framework_forcefield": True,
            "molecule_forcefield": True,
            "files": {name: str(path) for name, path in base_files.items()},
            "adsorbates": adsorbates,
        }
    return dict(sorted(registry.items()))


def generate_raspa2_forcefield_registry(raspa2_root: str | Path = "external/RASPA2") -> dict[str, Any]:
    root = Path(raspa2_root)
    molecule_dir = root / "molecules" / "ExampleDefinitions"
    parameter_dir = root / "forcefield" / "ExampleMoleculeForceField"
    base_files = {
        "mixing_rules": parameter_dir / "force_field_mixing_rules.def",
        "pseudo_atoms": parameter_dir / "pseudo_atoms.def",
    }
    if not molecule_dir.exists() or any(not path.exists() for path in base_files.values()):
        return {}
    adsorbates = sorted(
        _canonical_component_name(path.stem)
        for path in molecule_dir.glob("*.def")
        if not path.name.startswith("._")
    )
    return {
        "RASPA2_ExampleMoleculeForceField": {
            "name": "RASPA2_ExampleMoleculeForceField",
            "source": "raspa2",
            "framework_forcefield": False,
            "molecule_forcefield": True,
            "files": {name: str(path) for name, path in base_files.items()},
            "molecule_definition_dir": str(molecule_dir),
            "adsorbates": adsorbates,
        }
    }


def generate_crafted_adsorbate_registry(crafted_root: str | Path = "CRAFTED-2.0.0") -> dict[str, Any]:
    """Generate adsorbate properties from all available CRAFTED forcefield molecule defs."""
    root = Path(crafted_root)
    registry: dict[str, Any] = {}
    for forcefield_dir in sorted((root / "FORCEFIELDS").glob("*")):
        if not forcefield_dir.is_dir() or forcefield_dir.name.startswith("._"):
            continue
        base_files = {
            "force_field": forcefield_dir / "force_field.def",
            "mixing_rules": forcefield_dir / "force_field_mixing_rules.def",
            "pseudo_atoms": forcefield_dir / "pseudo_atoms.def",
        }
        if any(not path.exists() for path in base_files.values()):
            continue
        forcefield = {
            "framework": forcefield_dir.name,
            "source": "generated_from_crafted_forcefield",
            "files": {name: str(path) for name, path in base_files.items()},
            "adsorbates": {},
        }
        for molecule_def in sorted(forcefield_dir.glob("*.def")):
            if molecule_def.name in {"force_field.def", "force_field_mixing_rules.def", "pseudo_atoms.def"}:
                continue
            component = molecule_def.stem.upper()
            forcefield["adsorbates"] = {component: str(molecule_def)}
            try:
                properties = infer_adsorbate_properties(component, forcefield)
            except (LookupError, ValueError, OSError, KeyError):
                continue
            registry.setdefault(component, {})[forcefield_dir.name] = properties

    return dict(sorted(registry.items()))


def generate_raspa2_adsorbate_registry(raspa2_root: str | Path = "external/RASPA2") -> dict[str, Any]:
    """Generate adsorbate properties from RASPA2 example molecule definitions."""
    root = Path(raspa2_root)
    molecule_dir = root / "molecules" / "ExampleDefinitions"
    parameter_dir = root / "forcefield" / "ExampleMoleculeForceField"
    base_files = {
        "mixing_rules": parameter_dir / "force_field_mixing_rules.def",
        "pseudo_atoms": parameter_dir / "pseudo_atoms.def",
    }
    if not molecule_dir.exists() or any(not path.exists() for path in base_files.values()):
        return {}

    registry: dict[str, Any] = {}
    for molecule_def in sorted(molecule_dir.glob("*.def")):
        component = _canonical_component_name(molecule_def.stem)
        forcefield = {
            "framework": "RASPA2_ExampleMoleculeForceField",
            "source": "generated_from_raspa2_example_molecule_forcefield",
            "files": {name: str(path) for name, path in base_files.items()},
            "adsorbates": {component: str(molecule_def)},
        }
        try:
            properties = infer_adsorbate_properties(component, forcefield)
        except (LookupError, ValueError, OSError, KeyError):
            continue
        registry.setdefault(component, {})["RASPA2_ExampleMoleculeForceField"] = properties

    return dict(sorted(registry.items()))


def generate_reference_registry(crafted_root: str | Path = "CRAFTED-2.0.0") -> dict[str, Any]:
    """Generate CRAFTED isotherm and enthalpy reference availability metadata."""
    root = Path(crafted_root)
    registry: dict[str, Any] = {"isotherm": [], "enthalpy": []}
    for kind, directory_name in [("isotherm", "ISOTHERM_FILES"), ("enthalpy", "ENTHALPY_FILES")]:
        directory = root / directory_name
        if not directory.exists():
            continue
        entries = []
        for path in sorted(directory.glob("*.csv")):
            parsed = _parse_crafted_reference_name(path)
            if parsed is None:
                continue
            entries.append({**parsed, "path": str(path)})
        registry[kind] = entries
    return registry


def generate_capability_database(
    crafted_root: str | Path = "CRAFTED-2.0.0",
    raspa2_root: str | Path = "external/RASPA2",
) -> dict[str, Any]:
    """Generate the complete local database of usable benchmark resources."""
    materials = generate_crafted_material_registry(crafted_root)
    adsorbates = generate_adsorbate_registry(crafted_root, raspa2_root)
    molecule_definitions = generate_molecule_definition_registry(crafted_root, raspa2_root)
    forcefields = generate_forcefield_registry(crafted_root, raspa2_root)
    references = generate_reference_registry(crafted_root)
    return {
        "schema_version": 1,
        "sources": {
            "crafted_root": str(Path(crafted_root)),
            "raspa2_root": str(Path(raspa2_root)),
        },
        "summary": {
            "material_count": len(materials),
            "adsorbate_count": len(adsorbates),
            "molecule_definition_count": sum(len(entries) for entries in molecule_definitions.values()),
            "forcefield_count": len(forcefields),
            "isotherm_reference_count": len(references.get("isotherm", [])),
            "enthalpy_reference_count": len(references.get("enthalpy", [])),
        },
        "materials": materials,
        "adsorbates": adsorbates,
        "molecule_definitions": molecule_definitions,
        "forcefields": forcefields,
        "references": references,
    }


def write_generated_registries(
    output_dir: str | Path = "data/generated",
    crafted_root: str | Path = "CRAFTED-2.0.0",
    raspa2_root: str | Path = "external/RASPA2",
) -> dict[str, str]:
    """Write generated registries to disk and return their paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    material_path = output / "crafted_material_registry.json"
    adsorbate_path = output / "adsorbate_registry.json"
    molecule_definition_path = output / "molecule_definition_registry.json"
    forcefield_path = output / "forcefield_registry.json"
    reference_path = output / "reference_registry.json"
    database_path = output / "capability_database.json"

    material_registry = generate_crafted_material_registry(crafted_root)
    adsorbate_registry = generate_adsorbate_registry(crafted_root, raspa2_root)
    molecule_definition_registry = generate_molecule_definition_registry(crafted_root, raspa2_root)
    forcefield_registry = generate_forcefield_registry(crafted_root, raspa2_root)
    reference_registry = generate_reference_registry(crafted_root)
    capability_database = {
        "schema_version": 1,
        "sources": {
            "crafted_root": str(Path(crafted_root)),
            "raspa2_root": str(Path(raspa2_root)),
        },
        "summary": {
            "material_count": len(material_registry),
            "adsorbate_count": len(adsorbate_registry),
            "molecule_definition_count": sum(len(entries) for entries in molecule_definition_registry.values()),
            "forcefield_count": len(forcefield_registry),
            "isotherm_reference_count": len(reference_registry.get("isotherm", [])),
            "enthalpy_reference_count": len(reference_registry.get("enthalpy", [])),
        },
        "materials": material_registry,
        "adsorbates": adsorbate_registry,
        "molecule_definitions": molecule_definition_registry,
        "forcefields": forcefield_registry,
        "references": reference_registry,
    }

    _write_json(material_path, material_registry)
    _write_json(adsorbate_path, adsorbate_registry)
    _write_json(molecule_definition_path, molecule_definition_registry)
    _write_json(forcefield_path, forcefield_registry)
    _write_json(reference_path, reference_registry)
    _write_json(database_path, capability_database)
    return {
        "material_registry": str(material_path),
        "adsorbate_registry": str(adsorbate_path),
        "molecule_definition_registry": str(molecule_definition_path),
        "forcefield_registry": str(forcefield_path),
        "reference_registry": str(reference_path),
        "capability_database": str(database_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate derived CRAFTED registry files.")
    parser.add_argument("--crafted-root", default="CRAFTED-2.0.0")
    parser.add_argument("--raspa2-root", default="external/RASPA2")
    parser.add_argument("--output-dir", default="data/generated")
    args = parser.parse_args(argv)
    print(json.dumps(write_generated_registries(args.output_dir, args.crafted_root, args.raspa2_root), indent=2))
    return 0


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _canonical_component_name(name: str) -> str:
    return RASPA_COMPONENT_ALIASES.get(name.casefold(), name.upper())


def _merge_registry(target: dict[str, Any], source: dict[str, Any]) -> None:
    for component, forcefields in source.items():
        target.setdefault(component, {}).update(forcefields)


def _add_molecule_definition(
    registry: dict[str, Any],
    *,
    component: str,
    source: str,
    forcefield: str,
    molecule_definition: Path,
    properties_inferred: bool,
) -> None:
    registry.setdefault(component, []).append(
        {
            "component": component,
            "source": source,
            "forcefield": forcefield,
            "molecule_definition": str(molecule_definition),
            "properties_inferred": properties_inferred,
            "usable_by_current_lammps_pipeline": properties_inferred,
        }
    )


def _parse_crafted_reference_name(path: Path) -> dict[str, Any] | None:
    parts = path.stem.split("_")
    if len(parts) < 5:
        return None
    temperature = _optional_float(parts[-1])
    if temperature is None:
        return None
    return {
        "charge_scheme": parts[0],
        "material_id": "_".join(parts[1:-3]),
        "forcefield": parts[-3],
        "component": parts[-2].upper(),
        "temperature_K": temperature,
        "source": "crafted",
    }


if __name__ == "__main__":
    raise SystemExit(main())
