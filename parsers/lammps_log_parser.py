from typing import Any
from pathlib import Path
import json


def get_log(path: str | Path) -> str:
    """Read a LAMMPS log file as text."""
    with Path(path).open("r", encoding="utf-8") as file:
        return file.read()


def parse_lammps_log(log_data: str) -> dict[str, Any]:
    """Parse numeric thermo tables from a LAMMPS log."""
    rows: list[dict[str, int | float]] = []
    header = None

    for line in log_data.splitlines():
        parts = line.split()
        if not parts:
            continue

        if parts[0] == "Step" and "Atoms" in parts:
            header = parts
            continue

        if header and _is_numeric_row(parts, len(header)):
            values = [_to_number(value) for value in parts[: len(header)]]
            rows.append(dict(zip(header, values)))

    return {"rows": rows}


def summarize_adsorption(
    rows: list[dict[str, Any]],
    framework_atoms: int,
    adsorbate_atoms_per_molecule: int,
    discard_fraction: float = 0.0,
) -> dict[str, Any]:
    """Summarize adsorbate loading from parsed LAMMPS thermo rows."""
    if adsorbate_atoms_per_molecule <= 0:
        raise ValueError("adsorbate_atoms_per_molecule must be positive.")
    if not 0 <= discard_fraction < 1:
        raise ValueError("discard_fraction must be in the range [0, 1).")

    if not rows:
        return {
            "sample_count": 0,
            "inserted": False,
            "max_atoms": framework_atoms,
            "max_adsorbates": 0,
            "mean_adsorbates": 0.0,
        }

    start = int(len(rows) * discard_fraction)
    sampled_rows = rows[start:]
    if not sampled_rows:
        sampled_rows = rows

    if any("Atoms" not in row for row in sampled_rows):
        raise ValueError("Parsed thermo rows must contain an 'Atoms' column.")

    adsorbate_counts = [
        (row["Atoms"] - framework_atoms) / adsorbate_atoms_per_molecule
        for row in sampled_rows
    ]

    return {
        "sample_count": len(sampled_rows),
        "inserted": max(adsorbate_counts) > 0,
        "max_atoms": max(row["Atoms"] for row in sampled_rows),
        "max_adsorbates": max(adsorbate_counts),
        "mean_adsorbates": sum(adsorbate_counts) / len(adsorbate_counts),
    }


def summarize_component_adsorption(
    rows: list[dict[str, Any]],
    component_atom_counts: dict[str, int],
    mixture: dict[str, float],
    discard_fraction: float = 0.0,
    framework_mass_amu: float | None = None,
) -> dict[str, Any]:
    """Summarize mixture loading from component count variables in a LAMMPS thermo table."""
    if not component_atom_counts:
        raise ValueError("component_atom_counts must not be empty.")
    if not 0 <= discard_fraction < 1:
        raise ValueError("discard_fraction must be in the range [0, 1).")

    start = int(len(rows) * discard_fraction)
    sampled_rows = rows[start:] if rows else []
    if rows and not sampled_rows:
        sampled_rows = rows

    components: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for component, atom_count in component_atom_counts.items():
        if atom_count <= 0:
            raise ValueError(f"Atom count for component {component!r} must be positive.")
        variable_key = f"v_n{_component_id(component)}"
        if sampled_rows and any(variable_key not in row for row in sampled_rows):
            raise ValueError(f"Parsed thermo rows must contain component column {variable_key!r}.")
        values = [float(row[variable_key]) for row in sampled_rows]
        mean_molecules = sum(values) / len(values) if values else 0.0
        loading = _loading_mol_per_kg(mean_molecules, framework_mass_amu)
        components[component] = {
            "sample_count": len(values),
            "max_molecules": max(values) if values else 0.0,
            "mean_molecules": mean_molecules,
            "loading_mol_per_kg": loading,
            "loading_mmol_per_g": loading,
            "atom_count_per_molecule": atom_count,
            "gas_phase_mole_fraction": mixture.get(component),
            "thermo_column": variable_key,
        }

    return {
        "sample_count": len(sampled_rows),
        "inserted": any(component["max_molecules"] > 0.0 for component in components.values()),
        "components": components,
        "selectivity": _selectivity(components, mixture),
        "warnings": warnings,
    }

def log_to_json(
    log_file: str | Path,
    output_path: str | Path,
    framework_atoms: int,
    adsorbate_atoms_per_molecule: int,
    discard_fraction: float = 0.0,
    component_atom_counts: dict[str, int] | None = None,
    mixture: dict[str, float] | None = None,
    framework_mass_amu: float | None = None,
) -> dict[str, Any]:
    """Write the adsorption summary to a JSON file."""
    log_data = get_log(log_file)
    parsed_data = parse_lammps_log(log_data)
    summary = summarize_adsorption(
        parsed_data["rows"],
        framework_atoms=framework_atoms,
        adsorbate_atoms_per_molecule=adsorbate_atoms_per_molecule,
        discard_fraction=discard_fraction,
    )
    if component_atom_counts and mixture:
        summary.update(
            summarize_component_adsorption(
                parsed_data["rows"],
                component_atom_counts=component_atom_counts,
                mixture=mixture,
                discard_fraction=discard_fraction,
                framework_mass_amu=framework_mass_amu,
            )
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    return summary


def _is_numeric_row(parts: list[str], expected_columns: int) -> bool:
    if len(parts) < expected_columns:
        return False
    try:
        [_to_number(value) for value in parts[:expected_columns]]
    except ValueError:
        return False
    return True


def _to_number(value: str) -> int | float:
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _loading_mol_per_kg(mean_molecules_per_cell: float, framework_mass_amu: float | None) -> float | None:
    if framework_mass_amu is None:
        return None
    if framework_mass_amu <= 0:
        raise ValueError("framework_mass_amu must be positive.")
    return 1000.0 * mean_molecules_per_cell / framework_mass_amu


def _selectivity(
    components: dict[str, dict[str, Any]],
    mixture: dict[str, float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    component_names = list(components)
    total_adsorbed = sum(float(component["mean_molecules"]) for component in components.values())
    for left in component_names:
        for right in component_names:
            if left == right:
                continue
            key = f"{left}/{right}"
            left_adsorbed = float(components[left]["mean_molecules"])
            right_adsorbed = float(components[right]["mean_molecules"])
            left_gas = float(mixture.get(left, 0.0) or 0.0)
            right_gas = float(mixture.get(right, 0.0) or 0.0)
            if total_adsorbed <= 0.0 or left_gas <= 0.0 or right_gas <= 0.0 or right_adsorbed <= 0.0:
                result[key] = {
                    "value": None,
                    "reason": "undefined_zero_adsorbed_or_gas_fraction",
                }
                continue
            x_left = left_adsorbed / total_adsorbed
            x_right = right_adsorbed / total_adsorbed
            result[key] = {
                "value": (x_left / x_right) / (left_gas / right_gas),
                "adsorbed_mole_fraction_left": x_left,
                "adsorbed_mole_fraction_right": x_right,
                "gas_mole_fraction_left": left_gas,
                "gas_mole_fraction_right": right_gas,
            }
    return result


def _component_id(component: str) -> str:
    return str(component).lower().replace("-", "_")
