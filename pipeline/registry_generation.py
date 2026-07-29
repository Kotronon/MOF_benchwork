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


def generate_adsorbate_registry(crafted_root: str | Path = "CRAFTED-2.0.0") -> dict[str, Any]:
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


def write_generated_registries(
    output_dir: str | Path = "data/generated",
    crafted_root: str | Path = "CRAFTED-2.0.0",
) -> dict[str, str]:
    """Write generated registries to disk and return their paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    material_path = output / "crafted_material_registry.json"
    adsorbate_path = output / "crafted_adsorbate_registry.json"

    _write_json(material_path, generate_crafted_material_registry(crafted_root))
    _write_json(adsorbate_path, generate_adsorbate_registry(crafted_root))
    return {
        "material_registry": str(material_path),
        "adsorbate_registry": str(adsorbate_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate derived CRAFTED registry files.")
    parser.add_argument("--crafted-root", default="CRAFTED-2.0.0")
    parser.add_argument("--output-dir", default="data/generated")
    args = parser.parse_args(argv)
    print(json.dumps(write_generated_registries(args.output_dir, args.crafted_root), indent=2))
    return 0


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
