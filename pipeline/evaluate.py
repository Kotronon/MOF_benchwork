from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pipeline.config import load_benchmark_data


ISOTHERM_CSV_FIELDS = [
    "pressure_bar",
    "pressure_Pa",
    "sample_count",
    "inserted",
    "max_adsorbates",
    "mean_adsorbates_per_cell",
    "max_atoms",
    "framework_mass_amu",
    "loading_mol_per_kg",
    "loading_mmol_per_g",
    "reference_mol_per_kg",
    "reference_error_mol_per_kg",
    "reference_match",
    "absolute_error_mol_per_kg",
    "relative_error_percent",
    "log_file",
    "summary_file",
]


def sim_results_to_csv(
    sim_results: str | Path,
    output_path: str | Path,
    framework_data: str | Path | None = None,
    reference_csv: str | Path | None = None,
) -> Path:
    """Write an isotherm summary JSON to a comparable adsorption CSV table."""
    sim_results_path = Path(sim_results)
    data = load_benchmark_data(sim_results_path)

    framework_data_path = Path(framework_data) if framework_data else _infer_framework_data(sim_results_path)
    reference_csv_path = Path(reference_csv) if reference_csv else _infer_reference_csv(sim_results_path)

    framework_mass_amu = framework_mass_from_lammps_data(framework_data_path) if framework_data_path else None
    reference_points = load_reference_isotherm(reference_csv_path) if reference_csv_path else []
    rows = [
        _result_to_row(
            result,
            framework_mass_amu=framework_mass_amu,
            reference_points=reference_points,
        )
        for result in data.get("results", [])
    ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ISOTHERM_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def framework_mass_from_lammps_data(data_path: str | Path) -> float:
    """Return the mass of atoms present in a framework-only LAMMPS data file in amu."""
    lines = Path(data_path).read_text(encoding="utf-8").splitlines()
    masses = _parse_masses(lines)
    atom_type_counts = _parse_atom_type_counts(lines)
    if not masses:
        raise ValueError(f"No Masses section found in {data_path}.")
    if not atom_type_counts:
        raise ValueError(f"No Atoms section found in {data_path}.")
    return sum(masses[type_id] * count for type_id, count in atom_type_counts.items())


def load_reference_isotherm(reference_path: str | Path) -> list[dict[str, float]]:
    """Load CRAFTED-style reference points with pressure[Pa], loading[mol/kg], error[mol/kg]."""
    points: list[dict[str, float]] = []
    with Path(reference_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pressure_pa, loading_mol_per_kg, error_mol_per_kg = [float(value) for value in stripped.split(",")[:3]]
            points.append(
                {
                    "pressure_Pa": pressure_pa,
                    "loading_mol_per_kg": loading_mol_per_kg,
                    "error_mol_per_kg": error_mol_per_kg,
                }
            )
    return sorted(points, key=lambda point: point["pressure_Pa"])


def _result_to_row(
    result: dict[str, Any],
    framework_mass_amu: float | None,
    reference_points: list[dict[str, float]],
) -> dict[str, Any]:
    summary = result.get("summary", {})
    pressure_bar = float(result.get("pressure_bar", 0.0))
    pressure_pa = pressure_bar * 100000.0
    mean_adsorbates = float(summary.get("mean_adsorbates", 0.0))
    loading_mol_per_kg = _loading_mol_per_kg(mean_adsorbates, framework_mass_amu)
    reference = _reference_at_pressure(reference_points, pressure_pa)
    reference_loading = reference["loading_mol_per_kg"] if reference else None
    absolute_error = (
        loading_mol_per_kg - reference_loading
        if loading_mol_per_kg is not None and reference_loading is not None
        else None
    )
    relative_error = (
        absolute_error / reference_loading * 100.0
        if absolute_error is not None and reference_loading not in (None, 0.0)
        else None
    )

    return {
        "pressure_bar": pressure_bar,
        "pressure_Pa": pressure_pa,
        "sample_count": summary.get("sample_count", ""),
        "inserted": summary.get("inserted", ""),
        "max_adsorbates": summary.get("max_adsorbates", ""),
        "mean_adsorbates_per_cell": mean_adsorbates,
        "max_atoms": summary.get("max_atoms", ""),
        "framework_mass_amu": framework_mass_amu if framework_mass_amu is not None else "",
        "loading_mol_per_kg": loading_mol_per_kg if loading_mol_per_kg is not None else "",
        "loading_mmol_per_g": loading_mol_per_kg if loading_mol_per_kg is not None else "",
        "reference_mol_per_kg": reference_loading if reference_loading is not None else "",
        "reference_error_mol_per_kg": reference["error_mol_per_kg"] if reference else "",
        "reference_match": reference["match"] if reference else "",
        "absolute_error_mol_per_kg": absolute_error if absolute_error is not None else "",
        "relative_error_percent": relative_error if relative_error is not None else "",
        "log_file": result.get("log_file", ""),
        "summary_file": result.get("summary_file", ""),
    }


def _loading_mol_per_kg(mean_adsorbates_per_cell: float, framework_mass_amu: float | None) -> float | None:
    if framework_mass_amu is None:
        return None
    if framework_mass_amu <= 0:
        raise ValueError("framework_mass_amu must be positive.")
    return 1000.0 * mean_adsorbates_per_cell / framework_mass_amu


def _reference_at_pressure(reference_points: list[dict[str, float]], pressure_pa: float) -> dict[str, float | str] | None:
    if not reference_points:
        return None

    for point in reference_points:
        if abs(point["pressure_Pa"] - pressure_pa) <= max(1e-9, pressure_pa * 1e-9):
            return {**point, "match": "exact"}

    lower = None
    upper = None
    for point in reference_points:
        if point["pressure_Pa"] < pressure_pa:
            lower = point
        elif point["pressure_Pa"] > pressure_pa:
            upper = point
            break

    if lower is None or upper is None:
        return None

    span = upper["pressure_Pa"] - lower["pressure_Pa"]
    fraction = (pressure_pa - lower["pressure_Pa"]) / span
    return {
        "pressure_Pa": pressure_pa,
        "loading_mol_per_kg": lower["loading_mol_per_kg"]
        + fraction * (upper["loading_mol_per_kg"] - lower["loading_mol_per_kg"]),
        "error_mol_per_kg": lower["error_mol_per_kg"]
        + fraction * (upper["error_mol_per_kg"] - lower["error_mol_per_kg"]),
        "match": "linear_interpolation",
    }


def _parse_masses(lines: list[str]) -> dict[int, float]:
    masses: dict[int, float] = {}
    for line in _section_lines(lines, "Masses"):
        parts = line.split("#", 1)[0].split()
        if len(parts) >= 2:
            masses[int(parts[0])] = float(parts[1])
    return masses


def _parse_atom_type_counts(lines: list[str]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for line in _section_lines(lines, "Atoms"):
        parts = line.split("#", 1)[0].split()
        if len(parts) >= 3:
            type_id = int(parts[2])
            counts[type_id] = counts.get(type_id, 0) + 1
    return counts


def _section_lines(lines: list[str], section_name: str) -> list[str]:
    section_start = None
    for index, line in enumerate(lines):
        if line.strip().startswith(section_name):
            section_start = index + 1
            break
    if section_start is None:
        return []

    section: list[str] = []
    started = False
    for line in lines[section_start:]:
        stripped = line.strip()
        if not stripped:
            if started:
                break
            continue
        if stripped[0].isalpha():
            break
        started = True
        section.append(stripped)
    return section


def _infer_framework_data(sim_results_path: Path) -> Path | None:
    data_dir = sim_results_path.parent / "data"
    matches = sorted(data_dir.glob("*.data"))
    return matches[0] if len(matches) == 1 else None


def _infer_reference_csv(sim_results_path: Path) -> Path | None:
    reference_dir = sim_results_path.parent / "source" / "references"
    matches = sorted(reference_dir.glob("*.csv"))
    return matches[0] if len(matches) == 1 else None
