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

def log_to_json(
    log_file: str | Path,
    output_path: str | Path,
    framework_atoms: int,
    adsorbate_atoms_per_molecule: int,
    discard_fraction: float = 0.0,
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

