from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from analysis.plotting import legend_outside_right
from pipeline.config import load_benchmark_data
from pipeline.eos import gas_molar_density
from pipeline.nist_isodb_parser import load_nist_isotherm


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
    "loading_absolute_mol_per_kg",
    "loading_absolute_mmol_per_g",
    "loading_excess_mol_per_kg",
    "loading_excess_mmol_per_g",
    "simulation_basis",
    "pore_volume_cm3_g",
    "pore_volume_source",
    "pore_volume_method",
    "gas_density_mol_per_m3",
    "reference_mol_per_kg",
    "reference_error_mol_per_kg",
    "reference_match",
    "reference_source",
    "reference_doi",
    "reference_basis",
    "reference_compared_against",
    "reference_path",
    "comparison_error_mol_per_kg",
    "comparison_relative_error_percent",
    "absolute_error_mol_per_kg",
    "relative_error_percent",
    "log_file",
    "summary_file",
]

BASIS_COMPARISON_CSV_FIELDS = [
    "pressure_bar",
    "pressure_Pa",
    "sample_count",
    "mean_adsorbates_per_cell",
    "simulation_absolute_mol_per_kg",
    "simulation_excess_mol_per_kg",
    "reference_key",
    "reference_source",
    "reference_doi",
    "reference_basis",
    "reference_compared_against",
    "selected_simulation_mol_per_kg",
    "reference_mol_per_kg",
    "comparison_error_mol_per_kg",
    "comparison_relative_error_percent",
    "reference_match",
    "basis_interpretation",
    "reference_path",
]


def sim_results_to_csv(
    sim_results: str | Path,
    output_path: str | Path,
    framework_data: str | Path | None = None,
    reference_csv: str | Path | None = None,
    evaluation_config: dict[str, Any] | None = None,
) -> Path:
    """Write an isotherm summary JSON to a comparable adsorption CSV table."""
    rows = build_evaluation_rows(
        sim_results,
        framework_data=framework_data,
        reference_csv=reference_csv,
        evaluation_config=evaluation_config,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ISOTHERM_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def evaluate_isotherm(
    sim_results: str | Path,
    output_dir: str | Path | None = None,
    framework_data: str | Path | None = None,
    reference_csv: str | Path | None = None,
    evaluation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create CSV, Markdown, and plot artifacts for an isotherm run."""
    sim_results_path = Path(sim_results)
    target_dir = Path(output_dir) if output_dir else sim_results_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    rows = build_evaluation_rows(
        sim_results_path,
        framework_data=framework_data,
        reference_csv=reference_csv,
        evaluation_config=evaluation_config,
    )
    csv_path = target_dir / "evaluated_isotherm.csv"
    report_path = target_dir / "evaluation_report.md"
    plot_path = target_dir / "isotherm_comparison.png"

    _write_evaluation_csv(rows, csv_path)
    _write_markdown_report(rows, report_path, sim_results_path)
    plot_result = _write_isotherm_plot(rows, plot_path)

    return {
        "evaluated_csv": str(csv_path),
        "evaluation_report": str(report_path),
        "isotherm_plot": str(plot_result) if plot_result else None,
        "point_count": len(rows),
        "mean_comparison_error_mol_per_kg": _mean_abs(rows, "comparison_error_mol_per_kg"),
        "mean_comparison_relative_error_percent": _mean_abs(rows, "comparison_relative_error_percent"),
        "mean_absolute_error_mol_per_kg": _mean_abs(rows, "absolute_error_mol_per_kg"),
        "mean_absolute_relative_error_percent": _mean_abs(rows, "relative_error_percent"),
    }


def evaluate_run_directory(
    run_directory: str | Path,
    evaluation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an already materialized run directory."""
    run_dir = Path(run_directory)
    return evaluate_isotherm(
        run_dir / "isotherm_summary.json",
        output_dir=run_dir,
        evaluation_config=evaluation_config,
    )


def evaluate_all_references(
    run_directory: str | Path,
    references: list[dict[str, Any]] | None = None,
    evaluation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an already materialized run directory against every available reference."""
    run_dir = Path(run_directory)
    sim_results = run_dir / "isotherm_summary.json"
    reference_entries = _reference_entries_for_run(run_dir, references=references)
    if not reference_entries:
        raise FileNotFoundError(f"No reference files found in {run_dir / 'source' / 'references'}.")
    active_references = [
        reference
        for reference in reference_entries
        if reference.get("use_in_evaluation", True) and not reference.get("excluded", False)
    ]
    if not active_references:
        raise ValueError(f"All reference files for {run_dir} were excluded from evaluation.")

    evaluations_dir = run_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    reference_rows: dict[str, list[dict[str, Any]]] = {}
    used_keys: set[str] = set()
    normalized_references: list[dict[str, Any]] = []

    for index, reference in enumerate(reference_entries, start=1):
        reference_path = Path(reference.get("copied_path") or reference["path"])
        key = _reference_key(reference, reference_path, index, used_keys)
        reference["key"] = key
        reference["path"] = str(reference_path)
        normalized_references.append(reference)

    active_by_key = {
        reference["key"]: reference
        for reference in normalized_references
        if reference.get("use_in_evaluation", True) and not reference.get("excluded", False)
    }
    for reference in active_by_key.values():
        reference_path = Path(reference.get("copied_path") or reference["path"])
        key = reference["key"]

        rows = build_evaluation_rows(
            sim_results,
            reference_csv=reference_path,
            evaluation_config=evaluation_config,
        )
        reference_rows[key] = rows
        results[key] = {
            **evaluate_isotherm(
                sim_results,
                output_dir=evaluations_dir / key,
                reference_csv=reference_path,
                evaluation_config=evaluation_config,
            ),
            "reference": reference,
        }

    candidates_path = evaluations_dir / "reference_candidates.json"
    combined_csv_path = evaluations_dir / "combined_reference_comparison.csv"
    combined_plot_path = evaluations_dir / "combined_reference_comparison.png"
    basis_csv_path = evaluations_dir / "basis_reference_comparison.csv"
    basis_report_path = evaluations_dir / "basis_reference_comparison.md"
    basis_absolute_plot_path = evaluations_dir / "basis_absolute_reference_comparison.png"
    basis_excess_plot_path = evaluations_dir / "basis_excess_reference_comparison.png"

    candidates_path.write_text(json.dumps(normalized_references, indent=2) + "\n", encoding="utf-8")
    active_reference_list = list(active_by_key.values())
    combined_rows = _write_combined_reference_csv(reference_rows, active_reference_list, combined_csv_path)
    combined_plot = _write_combined_reference_plot(reference_rows, active_reference_list, combined_plot_path)
    basis_rows = _write_basis_comparison_csv(reference_rows, active_reference_list, basis_csv_path)
    _write_basis_comparison_report(basis_rows, basis_report_path, run_dir)
    basis_plots = _write_basis_comparison_plots(
        basis_rows,
        absolute_output_path=basis_absolute_plot_path,
        excess_output_path=basis_excess_plot_path,
    )

    return {
        "status": "completed",
        "run_directory": str(run_dir),
        "reference_count": len(active_reference_list),
        "excluded_reference_count": len(normalized_references) - len(active_reference_list),
        "reference_candidates": str(candidates_path),
        "combined_csv": str(combined_csv_path),
        "combined_plot": str(combined_plot) if combined_plot else None,
        "combined_row_count": len(combined_rows),
        "basis_comparison_csv": str(basis_csv_path),
        "basis_comparison_report": str(basis_report_path),
        "basis_absolute_plot": str(basis_plots.get("absolute")) if basis_plots.get("absolute") else None,
        "basis_excess_plot": str(basis_plots.get("excess")) if basis_plots.get("excess") else None,
        "basis_comparison_row_count": len(basis_rows),
        "evaluations": results,
    }


def build_evaluation_rows(
    sim_results: str | Path,
    framework_data: str | Path | None = None,
    reference_csv: str | Path | None = None,
    evaluation_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    sim_results_path = Path(sim_results)
    data = load_benchmark_data(sim_results_path)

    framework_data_path = Path(framework_data) if framework_data else _infer_framework_data(sim_results_path)
    framework_mass_amu = framework_mass_from_lammps_data(framework_data_path) if framework_data_path else None
    reference_path = Path(reference_csv) if reference_csv else _infer_reference_file(sim_results_path)
    component = _infer_component(sim_results_path)
    normalized_evaluation = _normalize_evaluation_config(
        evaluation_config or _infer_evaluation_config(sim_results_path)
    )
    reference_points = (
        load_reference_isotherm(reference_path, framework_mass_amu=framework_mass_amu, component=component)
        if reference_path
        else []
    )
    return [
        _result_to_row(
            result,
            framework_mass_amu=framework_mass_amu,
            reference_points=reference_points,
            component=component,
            temperature_K=_infer_temperature_K(data, normalized_evaluation),
            evaluation_config=normalized_evaluation,
        )
        for result in data.get("results", [])
    ]


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


def load_reference_isotherm(
    reference_path: str | Path,
    *,
    framework_mass_amu: float | None = None,
    component: str = "CO2",
) -> list[dict[str, Any]]:
    """Load CRAFTED CSV or NIST ISODB JSON reference points in mol/kg."""
    path = Path(reference_path)
    if path.suffix.casefold() == ".json":
        return load_nist_isotherm(path, framework_mass_amu=framework_mass_amu, component=component)

    points: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
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
                    "source": "crafted",
                    "doi": "",
                    "adsorption_basis": "simulation_reference",
                    "path": str(path),
                }
            )
    return sorted(points, key=lambda point: point["pressure_Pa"])


def _result_to_row(
    result: dict[str, Any],
    framework_mass_amu: float | None,
    reference_points: list[dict[str, Any]],
    component: str,
    temperature_K: float,
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    summary = result.get("summary", {})
    pressure_bar = float(result.get("pressure_bar", 0.0))
    pressure_pa = pressure_bar * 100000.0
    mean_adsorbates = float(summary.get("mean_adsorbates", 0.0))
    loading_absolute_mol_per_kg = _loading_mol_per_kg(mean_adsorbates, framework_mass_amu)
    excess = _excess_loading_mol_per_kg(
        loading_absolute_mol_per_kg,
        pressure_bar=pressure_bar,
        component=component,
        temperature_K=temperature_K,
        evaluation_config=evaluation_config,
    )
    reference = _reference_at_pressure(reference_points, pressure_pa)
    reference_loading = reference["loading_mol_per_kg"] if reference else None
    compared_against = _comparison_basis(reference, evaluation_config)
    simulation_loading_for_reference = _loading_for_basis(
        compared_against,
        absolute_loading=loading_absolute_mol_per_kg,
        excess_loading=excess["loading_excess_mol_per_kg"],
    )
    comparison_error = (
        simulation_loading_for_reference - reference_loading
        if simulation_loading_for_reference is not None and reference_loading is not None
        else None
    )
    relative_error = (
        comparison_error / reference_loading * 100.0
        if comparison_error is not None and reference_loading not in (None, 0.0)
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
        "loading_mol_per_kg": loading_absolute_mol_per_kg if loading_absolute_mol_per_kg is not None else "",
        "loading_mmol_per_g": loading_absolute_mol_per_kg if loading_absolute_mol_per_kg is not None else "",
        "loading_absolute_mol_per_kg": loading_absolute_mol_per_kg if loading_absolute_mol_per_kg is not None else "",
        "loading_absolute_mmol_per_g": loading_absolute_mol_per_kg if loading_absolute_mol_per_kg is not None else "",
        "loading_excess_mol_per_kg": (
            excess["loading_excess_mol_per_kg"] if excess["loading_excess_mol_per_kg"] is not None else ""
        ),
        "loading_excess_mmol_per_g": (
            excess["loading_excess_mol_per_kg"] if excess["loading_excess_mol_per_kg"] is not None else ""
        ),
        "simulation_basis": evaluation_config["simulation_basis"],
        "pore_volume_cm3_g": (
            evaluation_config["pore_volume_cm3_g"] if evaluation_config["pore_volume_cm3_g"] is not None else ""
        ),
        "pore_volume_source": evaluation_config.get("pore_volume_source") or "",
        "pore_volume_method": evaluation_config.get("pore_volume_method") or "",
        "gas_density_mol_per_m3": (
            excess["gas_density_mol_per_m3"] if excess["gas_density_mol_per_m3"] is not None else ""
        ),
        "reference_mol_per_kg": reference_loading if reference_loading is not None else "",
        "reference_error_mol_per_kg": reference["error_mol_per_kg"] if reference else "",
        "reference_match": reference["match"] if reference else "",
        "reference_source": reference.get("source", "") if reference else "",
        "reference_doi": reference.get("doi", "") if reference else "",
        "reference_basis": reference.get("adsorption_basis", "") if reference else "",
        "reference_compared_against": compared_against,
        "reference_path": reference.get("path", "") if reference else "",
        "comparison_error_mol_per_kg": comparison_error if comparison_error is not None else "",
        "comparison_relative_error_percent": relative_error if relative_error is not None else "",
        "absolute_error_mol_per_kg": comparison_error if comparison_error is not None else "",
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


def _excess_loading_mol_per_kg(
    absolute_loading_mol_per_kg: float | None,
    *,
    pressure_bar: float,
    component: str,
    temperature_K: float,
    evaluation_config: dict[str, Any],
) -> dict[str, float | None]:
    pore_volume_cm3_g = evaluation_config.get("pore_volume_cm3_g")
    if not evaluation_config.get("report_excess") or absolute_loading_mol_per_kg is None or pore_volume_cm3_g is None:
        return {"loading_excess_mol_per_kg": None, "gas_density_mol_per_m3": None}

    gas_density = gas_molar_density(
        component=component,
        temperature_K=temperature_K,
        pressure_bar=pressure_bar,
        backend=str(evaluation_config.get("gas_density_backend", "HEOS")),
    )
    pore_volume_m3_per_kg = float(pore_volume_cm3_g) * 1.0e-3
    return {
        "loading_excess_mol_per_kg": absolute_loading_mol_per_kg - gas_density * pore_volume_m3_per_kg,
        "gas_density_mol_per_m3": gas_density,
    }


def _normalize_evaluation_config(evaluation_config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(evaluation_config or {})
    pore_volume = config.get("pore_volume_cm3_g")
    if pore_volume in ("", None, "auto"):
        pore_volume = None
    else:
        pore_volume = float(pore_volume)

    return {
        "simulation_basis": str(config.get("simulation_basis", "absolute")).casefold(),
        "report_excess": bool(config.get("report_excess", False)),
        "pore_volume_cm3_g": pore_volume,
        "pore_volume_source": config.get("pore_volume_source"),
        "pore_volume_method": config.get("pore_volume_method"),
        "gas_density_backend": config.get("gas_density_backend", "HEOS"),
        "reference_matching": str(config.get("reference_matching", "by_basis")).casefold(),
        "temperature_K": float(config.get("temperature_K", 298.15)),
    }


def _infer_temperature_K(data: dict[str, Any], evaluation_config: dict[str, Any]) -> float:
    return float(
        evaluation_config.get("temperature_K")
        or data.get("temperature_K")
        or data.get("conditions", {}).get("temperature_K", 298.15)
    )


def _infer_evaluation_config(sim_results_path: Path) -> dict[str, Any] | None:
    summary_path = sim_results_path.parent / "prepare_summary.json"
    if not summary_path.exists():
        return None

    try:
        summary = load_benchmark_data(summary_path)
    except (OSError, json.JSONDecodeError, TypeError):
        return None

    prepare_plan = summary.get("prepare_plan", {})
    if isinstance(prepare_plan, dict) and isinstance(prepare_plan.get("evaluation"), dict):
        return prepare_plan["evaluation"]

    result = summary.get("result", {})
    if isinstance(result, dict) and isinstance(result.get("evaluation"), dict):
        return result["evaluation"]
    return None


def _comparison_basis(reference: dict[str, Any] | None, evaluation_config: dict[str, Any]) -> str:
    if not reference:
        return ""

    if evaluation_config.get("reference_matching") != "by_basis":
        return evaluation_config["simulation_basis"]

    reference_basis = str(reference.get("adsorption_basis", "")).casefold()
    if reference_basis == "excess":
        return "excess"
    if reference_basis in ("absolute", "simulation_reference"):
        return "absolute"
    return "absolute_assumed_for_unknown_reference"


def _loading_for_basis(
    basis: str,
    *,
    absolute_loading: float | None,
    excess_loading: float | None,
) -> float | None:
    if basis == "excess":
        return excess_loading
    if basis in ("absolute", "simulation_reference", "absolute_assumed_for_unknown_reference"):
        return absolute_loading
    return None


def _write_evaluation_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ISOTHERM_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _write_markdown_report(rows: list[dict[str, Any]], output_path: str | Path, sim_results_path: Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Isotherm Evaluation",
        "",
        f"Source: `{sim_results_path}`",
        "",
        "## Summary",
        "",
        f"- Pressure points: {len(rows)}",
        f"- Mean comparison error: {_format_optional(_mean_abs(rows, 'comparison_error_mol_per_kg'))} mol/kg",
        f"- Mean comparison relative error: {_format_optional(_mean_abs(rows, 'comparison_relative_error_percent'))} %",
        "",
        "## Isotherm Table",
        "",
        "| pressure / bar | CO2 / cell | absolute / mol kg^-1 | excess / mol kg^-1 | reference / mol kg^-1 | compared as | rel. error / % |",
        "|---:|---:|---:|---:|---:|:---|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{_format_optional(row['pressure_bar'])} | "
            f"{_format_optional(row['mean_adsorbates_per_cell'])} | "
            f"{_format_optional(row['loading_absolute_mol_per_kg'])} | "
            f"{_format_optional(row['loading_excess_mol_per_kg'])} | "
            f"{_format_optional(row['reference_mol_per_kg'])} | "
            f"{row['reference_compared_against']} | "
            f"{_format_optional(row['comparison_relative_error_percent'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Loading is absolute adsorption from the GCMC molecule count.",
            "- Excess loading is calculated as absolute loading minus bulk gas density times pore volume.",
            "- mol/kg and mmol/g have the same numeric value.",
            "- Reference values are matched exactly where possible and otherwise linearly interpolated.",
            "- Excess references are compared against excess loading; unknown-basis references are reported and treated as absolute for comparison.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _write_isotherm_plot(rows: list[dict[str, Any]], output_path: str | Path) -> Path | None:
    if not rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    pressures = [_as_float(row["pressure_bar"]) for row in rows]
    simulated_absolute = [_as_float(row["loading_absolute_mol_per_kg"]) for row in rows]
    simulated_excess = [_as_float(row["loading_excess_mol_per_kg"]) for row in rows]
    reference = [_as_float(row["reference_mol_per_kg"]) for row in rows]

    valid_abs = [(p, y) for p, y in zip(pressures, simulated_absolute) if p is not None and y is not None]
    valid_excess = [(p, y) for p, y in zip(pressures, simulated_excess) if p is not None and y is not None]
    valid_ref = [(p, y) for p, y in zip(pressures, reference) if p is not None and y is not None]
    if not valid_abs:
        return None

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot([p for p, _y in valid_abs], [y for _p, y in valid_abs], marker="o", label="Simulation absolute")
    if valid_excess:
        ax.plot(
            [p for p, _y in valid_excess],
            [y for _p, y in valid_excess],
            marker="D",
            label="Simulation excess",
        )
    if valid_ref:
        ax.plot(
            [p for p, _y in valid_ref],
            [y for _p, y in valid_ref],
            marker="s",
            linestyle="--",
            label=_single_reference_label(rows),
        )
    ax.set_xlabel("Pressure / bar")
    ax.set_ylabel("Loading / mol kg$^{-1}$")
    ax.set_title("MOF-5 CO2 adsorption isotherm")
    ax.grid(True, alpha=0.3)
    legend_outside_right(ax)
    if all(p and p > 0 for p in pressures if p is not None):
        ax.set_xscale("log")
        _set_decimal_pressure_ticks(ax, pressures)
    fig.tight_layout(rect=(0, 0, 0.78, 1))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _write_combined_reference_csv(
    reference_rows: dict[str, list[dict[str, Any]]],
    references: list[dict[str, Any]],
    output_path: str | Path,
) -> list[dict[str, Any]]:
    fieldnames = [
        "pressure_bar",
        "pressure_Pa",
        "sample_count",
        "mean_adsorbates_per_cell",
        "simulation_mol_per_kg",
        "simulation_mmol_per_g",
        "simulation_absolute_mol_per_kg",
        "simulation_excess_mol_per_kg",
        "gas_density_mol_per_m3",
        "pore_volume_cm3_g",
    ]
    for reference in references:
        key = reference["key"]
        fieldnames.extend(
            [
                f"{key}_reference_mol_per_kg",
                f"{key}_compared_against",
                f"{key}_comparison_error_mol_per_kg",
                f"{key}_comparison_relative_error_percent",
                f"{key}_absolute_error_mol_per_kg",
                f"{key}_relative_error_percent",
                f"{key}_match",
                f"{key}_source",
                f"{key}_doi",
                f"{key}_basis",
                f"{key}_path",
            ]
        )

    rows_by_pressure: dict[float, dict[str, Any]] = {}
    for reference in references:
        key = reference["key"]
        for row in reference_rows.get(key, []):
            pressure_bar = float(row["pressure_bar"])
            combined = rows_by_pressure.setdefault(
                pressure_bar,
                {
                    "pressure_bar": row["pressure_bar"],
                    "pressure_Pa": row["pressure_Pa"],
                    "sample_count": row["sample_count"],
                    "mean_adsorbates_per_cell": row["mean_adsorbates_per_cell"],
                    "simulation_mol_per_kg": row["loading_mol_per_kg"],
                    "simulation_mmol_per_g": row["loading_mmol_per_g"],
                    "simulation_absolute_mol_per_kg": row["loading_absolute_mol_per_kg"],
                    "simulation_excess_mol_per_kg": row["loading_excess_mol_per_kg"],
                    "gas_density_mol_per_m3": row["gas_density_mol_per_m3"],
                    "pore_volume_cm3_g": row["pore_volume_cm3_g"],
                },
            )
            combined[f"{key}_reference_mol_per_kg"] = row["reference_mol_per_kg"]
            combined[f"{key}_compared_against"] = row["reference_compared_against"]
            combined[f"{key}_comparison_error_mol_per_kg"] = row["comparison_error_mol_per_kg"]
            combined[f"{key}_comparison_relative_error_percent"] = row["comparison_relative_error_percent"]
            combined[f"{key}_absolute_error_mol_per_kg"] = row["absolute_error_mol_per_kg"]
            combined[f"{key}_relative_error_percent"] = row["relative_error_percent"]
            combined[f"{key}_match"] = row["reference_match"]
            combined[f"{key}_source"] = row["reference_source"]
            combined[f"{key}_doi"] = row["reference_doi"]
            combined[f"{key}_basis"] = row["reference_basis"]
            combined[f"{key}_path"] = row["reference_path"]

    combined_rows = [rows_by_pressure[pressure] for pressure in sorted(rows_by_pressure)]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)
    return combined_rows


def _write_basis_comparison_csv(
    reference_rows: dict[str, list[dict[str, Any]]],
    references: list[dict[str, Any]],
    output_path: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference in references:
        key = reference["key"]
        for row in reference_rows.get(key, []):
            selected_loading = _selected_simulation_loading(row)
            rows.append(
                {
                    "pressure_bar": row["pressure_bar"],
                    "pressure_Pa": row["pressure_Pa"],
                    "sample_count": row["sample_count"],
                    "mean_adsorbates_per_cell": row["mean_adsorbates_per_cell"],
                    "simulation_absolute_mol_per_kg": row["loading_absolute_mol_per_kg"],
                    "simulation_excess_mol_per_kg": row["loading_excess_mol_per_kg"],
                    "reference_key": key,
                    "reference_source": row["reference_source"],
                    "reference_doi": row["reference_doi"],
                    "reference_basis": row["reference_basis"],
                    "reference_compared_against": row["reference_compared_against"],
                    "selected_simulation_mol_per_kg": selected_loading if selected_loading is not None else "",
                    "reference_mol_per_kg": row["reference_mol_per_kg"],
                    "comparison_error_mol_per_kg": row["comparison_error_mol_per_kg"],
                    "comparison_relative_error_percent": row["comparison_relative_error_percent"],
                    "reference_match": row["reference_match"],
                    "basis_interpretation": _basis_interpretation(row),
                    "reference_path": row["reference_path"],
                }
            )

    rows.sort(key=lambda item: (float(item["pressure_bar"]), str(item["reference_key"])))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASIS_COMPARISON_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_basis_comparison_report(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    run_dir: Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Basis-Separated Reference Comparison",
        "",
        f"Source: `{run_dir}`",
        "",
        "## Summary by reference basis",
        "",
        "| reference basis | rows | mean abs. rel. error / % | interpretation |",
        "|:---|---:|---:|:---|",
    ]
    for basis, basis_rows in _basis_groups(rows).items():
        lines.append(
            "| "
            f"{basis} | "
            f"{len(basis_rows)} | "
            f"{_format_optional(_mean_abs(basis_rows, 'comparison_relative_error_percent'))} | "
            f"{_basis_group_note(basis)} |"
        )

    lines.extend(
        [
            "",
            "## Comparison rows",
            "",
            "| pressure / bar | reference | basis | compared simulation | simulation / mol kg^-1 | reference / mol kg^-1 | rel. error / % | note |",
            "|---:|:---|:---|:---|---:|---:|---:|:---|",
        ]
    )
    for row in rows:
        label = row["reference_doi"] or row["reference_key"]
        lines.append(
            "| "
            f"{_format_optional(row['pressure_bar'])} | "
            f"{label} | "
            f"{row['reference_basis']} | "
            f"{row['reference_compared_against']} | "
            f"{_format_optional(row['selected_simulation_mol_per_kg'])} | "
            f"{_format_optional(row['reference_mol_per_kg'])} | "
            f"{_format_optional(row['comparison_relative_error_percent'])} | "
            f"{row['basis_interpretation']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Absolute or simulation-reference data are compared against the absolute GCMC loading.",
            "- Excess references are compared against the excess loading when excess reporting is available.",
            "- Unknown-basis references are retained, but the comparison assumes absolute loading and should be interpreted as orienting evidence.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _write_basis_comparison_plots(
    rows: list[dict[str, Any]],
    *,
    absolute_output_path: str | Path,
    excess_output_path: str | Path,
) -> dict[str, Path | None]:
    return {
        "absolute": _write_basis_comparison_plot(
            rows,
            output_path=absolute_output_path,
            mode="absolute",
            title="Absolute adsorption reference comparison",
            simulation_column="simulation_absolute_mol_per_kg",
            simulation_label="Simulation absolute",
        ),
        "excess": _write_basis_comparison_plot(
            rows,
            output_path=excess_output_path,
            mode="excess",
            title="Excess adsorption reference comparison",
            simulation_column="simulation_excess_mol_per_kg",
            simulation_label="Simulation excess",
        ),
    }


def _write_basis_comparison_plot(
    rows: list[dict[str, Any]],
    *,
    output_path: str | Path,
    mode: str,
    title: str,
    simulation_column: str,
    simulation_label: str,
) -> Path | None:
    plot_rows = [row for row in rows if _row_matches_basis_plot(row, mode)]
    if not plot_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    sim_points = _unique_pressure_series(plot_rows, simulation_column)
    if not sim_points:
        return None

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.plot(
        [pressure for pressure, _loading in sim_points],
        [loading for _pressure, loading in sim_points],
        marker="o",
        linewidth=2.2,
        label=simulation_label,
    )

    for key, reference_rows in _rows_by_reference(plot_rows).items():
        ref_points = [
            (_as_float(row["pressure_bar"]), _as_float(row["reference_mol_per_kg"]))
            for row in reference_rows
            if row.get("reference_mol_per_kg") not in ("", None)
        ]
        valid_ref_points = [
            (pressure, loading)
            for pressure, loading in ref_points
            if pressure is not None and loading is not None
        ]
        if not valid_ref_points:
            continue
        first = reference_rows[0]
        label = _basis_plot_reference_label(first, key)
        marker = "s" if first.get("reference_source") == "crafted" else "^"
        ax.plot(
            [pressure for pressure, _loading in valid_ref_points],
            [loading for _pressure, loading in valid_ref_points],
            marker=marker,
            linestyle="--",
            label=label,
        )

    ax.set_xlabel("Pressure / bar")
    ax.set_ylabel("Loading / mol kg$^{-1}$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if all(pressure > 0 for pressure, _loading in sim_points):
        ax.set_xscale("log")
        _set_decimal_pressure_ticks(ax, [pressure for pressure, _loading in sim_points])
    legend_outside_right(ax, fontsize="small")
    fig.tight_layout(rect=(0, 0, 0.76, 1))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _row_matches_basis_plot(row: dict[str, Any], mode: str) -> bool:
    if row.get("reference_mol_per_kg") in ("", None):
        return False
    compared_against = str(row.get("reference_compared_against", "")).casefold()
    if mode == "excess":
        return compared_against == "excess"
    return compared_against in ("absolute", "absolute_assumed_for_unknown_reference")


def _unique_pressure_series(rows: list[dict[str, Any]], column: str) -> list[tuple[float, float]]:
    series: dict[float, float] = {}
    for row in rows:
        pressure = _as_float(row.get("pressure_bar"))
        loading = _as_float(row.get(column))
        if pressure is not None and loading is not None:
            series.setdefault(pressure, loading)
    return sorted(series.items())


def _rows_by_reference(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["reference_key"]), []).append(row)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _basis_plot_reference_label(row: dict[str, Any], key: str) -> str:
    doi = str(row.get("reference_doi", "")).strip()
    basis = str(row.get("reference_basis", "") or "unknown").strip()
    if row.get("reference_source") == "crafted":
        return f"CRAFTED ({basis})"
    return f"{doi or key} ({basis})"


def _selected_simulation_loading(row: dict[str, Any]) -> float | None:
    return _loading_for_basis(
        str(row.get("reference_compared_against", "")),
        absolute_loading=_as_float(row.get("loading_absolute_mol_per_kg")),
        excess_loading=_as_float(row.get("loading_excess_mol_per_kg")),
    )


def _basis_interpretation(row: dict[str, Any]) -> str:
    basis = str(row.get("reference_basis", "")).casefold()
    compared_against = str(row.get("reference_compared_against", "")).casefold()
    selected_loading = _selected_simulation_loading(row)
    if row.get("reference_mol_per_kg") in ("", None):
        return "no_reference_at_pressure"
    if compared_against == "excess" and selected_loading is None:
        return "excess_reference_without_excess_loading"
    if basis == "excess":
        return "matched_excess_reference"
    if basis in ("absolute", "simulation_reference"):
        return "matched_absolute_reference"
    if basis == "unknown":
        return "unknown_basis_absolute_assumed"
    return "missing_basis_absolute_assumed"


def _basis_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        basis = str(row.get("reference_basis") or "missing")
        groups.setdefault(basis, []).append(row)
    return dict(sorted(groups.items(), key=lambda item: item[0]))


def _basis_group_note(basis: str) -> str:
    normalized = basis.casefold()
    if normalized in ("absolute", "simulation_reference"):
        return "direct comparison to absolute loading"
    if normalized == "excess":
        return "direct comparison to excess loading"
    if normalized == "unknown":
        return "absolute assumed; use as orienting comparison"
    return "basis missing; use as orienting comparison"


def _write_combined_reference_plot(
    reference_rows: dict[str, list[dict[str, Any]]],
    references: list[dict[str, Any]],
    output_path: str | Path,
) -> Path | None:
    if not reference_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    first_rows = next(iter(reference_rows.values()))
    simulated_absolute = [
        (_as_float(row["pressure_bar"]), _as_float(row["loading_absolute_mol_per_kg"]))
        for row in first_rows
    ]
    simulated_excess = [
        (_as_float(row["pressure_bar"]), _as_float(row["loading_excess_mol_per_kg"]))
        for row in first_rows
    ]
    valid_abs = [
        (pressure, loading)
        for pressure, loading in simulated_absolute
        if pressure is not None and loading is not None
    ]
    valid_excess = [
        (pressure, loading)
        for pressure, loading in simulated_excess
        if pressure is not None and loading is not None
    ]
    if not valid_abs:
        return None

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.plot(
        [pressure for pressure, _loading in valid_abs],
        [loading for _pressure, loading in valid_abs],
        marker="o",
        linewidth=2.0,
        label="Simulation absolute",
    )
    if valid_excess:
        ax.plot(
            [pressure for pressure, _loading in valid_excess],
            [loading for _pressure, loading in valid_excess],
            marker="D",
            linewidth=2.0,
            label="Simulation excess",
        )

    for reference in references:
        key = reference["key"]
        values = [
            (_as_float(row["pressure_bar"]), _as_float(row["reference_mol_per_kg"]))
            for row in reference_rows.get(key, [])
        ]
        valid_values = [(pressure, loading) for pressure, loading in values if pressure is not None and loading is not None]
        if not valid_values:
            continue
        ax.plot(
            [pressure for pressure, _loading in valid_values],
            [loading for _pressure, loading in valid_values],
            marker="s" if reference.get("source") == "crafted" else "^",
            linestyle="--",
            label=_reference_label(reference),
        )

    ax.set_xlabel("Pressure / bar")
    ax.set_ylabel("Loading / mol kg$^{-1}$")
    ax.set_title("MOF-5 CO2 adsorption references")
    ax.grid(True, alpha=0.3)
    if all(pressure > 0 for pressure, _loading in valid_abs):
        ax.set_xscale("log")
        _set_decimal_pressure_ticks(
            ax, [pressure for pressure, _loading in valid_abs]
        )
    legend_outside_right(ax, fontsize="small")
    fig.tight_layout(rect=(0, 0, 0.76, 1))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _set_decimal_pressure_ticks(axis: Any, pressures: list[float | None]) -> None:
    """Keep logarithmic spacing while showing pressure values as decimals."""
    ticks = sorted(
        {float(value) for value in pressures if value is not None and value > 0}
    )
    if not ticks:
        return
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{value:g}" for value in ticks])
    axis.minorticks_off()


def _single_reference_label(rows: list[dict[str, Any]]) -> str:
    reference_rows = [
        row
        for row in rows
        if row.get("reference_mol_per_kg") not in (None, "")
    ]
    if not reference_rows:
        return "Reference"
    row = reference_rows[0]
    source = str(row.get("reference_source", "")).casefold()
    reference_path = str(row.get("reference_path", "")).casefold()
    if source == "crafted":
        if "ddec" in reference_path and "uff" in reference_path and "298" in reference_path:
            return "CRAFTED DDEC/UFF (298 K)"
        return "CRAFTED reference"
    doi = str(row.get("reference_doi", "")).strip()
    if source == "nist_isodb":
        return f"NIST ISODB {doi}" if doi else "NIST ISODB reference"
    return "Reference"


def _mean_abs(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_as_float(row.get(key)) for row in rows]
    clean_values = [abs(value) for value in values if value is not None]
    if not clean_values:
        return None
    return sum(clean_values) / len(clean_values)


def _as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return number


def _format_optional(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{number:.6g}"


def _reference_at_pressure(reference_points: list[dict[str, Any]], pressure_pa: float) -> dict[str, Any] | None:
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
        "error_mol_per_kg": float(lower.get("error_mol_per_kg", 0.0))
        + fraction * (float(upper.get("error_mol_per_kg", 0.0)) - float(lower.get("error_mol_per_kg", 0.0))),
        "match": "linear_interpolation",
        "source": lower.get("source", ""),
        "doi": lower.get("doi", ""),
        "adsorption_basis": lower.get("adsorption_basis", ""),
        "path": lower.get("path", ""),
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


def _reference_entries_for_run(
    run_dir: Path,
    references: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    prepare_summary = run_dir / "prepare_summary.json"
    reference_entries = [dict(reference) for reference in references] if references else []
    if not reference_entries and prepare_summary.exists():
        data = load_benchmark_data(prepare_summary)
        reference_entries = [
            dict(reference)
            for reference in (
                data.get("result", {})
                .get("files", {})
                .get("source_files", {})
                .get("reference_files", [])
            )
        ]
        if not reference_entries:
            reference_entries = [
                dict(reference)
                for reference in data.get("prepare_plan", {})
                .get("inputs", {})
                .get("reference_files", [])
            ]

    if not reference_entries:
        reference_dir = run_dir / "source" / "references"
        reference_entries = [
            {
                "path": str(path),
                "copied_path": str(path),
                "source": "crafted" if path.suffix.casefold() == ".csv" else "nist_isodb",
                "format": "crafted_csv" if path.suffix.casefold() == ".csv" else "nist_json",
                "selected": False,
            }
            for path in sorted([*reference_dir.glob("*.csv"), *reference_dir.glob("*.json")])
        ]

    available = []
    for reference in reference_entries:
        path = _existing_reference_path(reference)
        if path:
            reference["path"] = str(path)
            reference["copied_path"] = str(path)
            available.append(reference)
    return sorted(available, key=_reference_sort_key)


def _reference_sort_key(reference: dict[str, Any]) -> tuple[Any, ...]:
    source_rank = {"crafted": 0, "nist_isodb": 1}.get(str(reference.get("source", "")), 9)
    return (
        not bool(reference.get("selected")),
        source_rank,
        reference.get("doi", ""),
        reference.get("path", ""),
    )


def _reference_key(
    reference: dict[str, Any],
    reference_path: Path,
    index: int,
    used_keys: set[str],
) -> str:
    source = str(reference.get("source") or ("crafted" if reference_path.suffix == ".csv" else "nist_isodb"))
    doi = str(reference.get("doi") or "").strip()
    base = f"{source}_{doi or reference_path.stem}"
    key = _safe_key(base)[:80].strip("_") or f"reference_{index}"
    candidate = key
    suffix = 2
    while candidate in used_keys:
        candidate = f"{key}_{suffix}"
        suffix += 1
    used_keys.add(candidate)
    return candidate


def _safe_key(value: str) -> str:
    key = []
    previous_was_separator = False
    for character in value.casefold():
        if character.isalnum():
            key.append(character)
            previous_was_separator = False
        elif not previous_was_separator:
            key.append("_")
            previous_was_separator = True
    return "".join(key).strip("_")


def _reference_label(reference: dict[str, Any]) -> str:
    source = str(reference.get("source", "reference"))
    if source == "crafted":
        return "CRAFTED"
    doi = str(reference.get("doi", "")).strip()
    if doi:
        return f"NIST {doi}"
    return reference.get("key", "NIST")


def _existing_reference_path(reference: dict[str, Any]) -> Path | None:
    for key in ("copied_path", "path"):
        value = reference.get(key)
        if value and Path(value).exists():
            return Path(value)
    return None


def _infer_reference_file(sim_results_path: Path) -> Path | None:
    prepare_summary = sim_results_path.parent / "prepare_summary.json"
    if prepare_summary.exists():
        data = load_benchmark_data(prepare_summary)
        references = (
            data.get("result", {})
            .get("files", {})
            .get("source_files", {})
            .get("reference_files", [])
        )
        for reference in references:
            path = _existing_reference_path(reference)
            if reference.get("selected") and path:
                return path
        for reference in references:
            path = _existing_reference_path(reference)
            if path:
                return path

    reference_dir = sim_results_path.parent / "source" / "references"
    matches = sorted([*reference_dir.glob("*.csv"), *reference_dir.glob("*.json")])
    return matches[0] if len(matches) == 1 else None


def _infer_reference_csv(sim_results_path: Path) -> Path | None:
    return _infer_reference_file(sim_results_path)


def _infer_component(sim_results_path: Path) -> str:
    prepare_summary = sim_results_path.parent / "prepare_summary.json"
    if not prepare_summary.exists():
        return "CO2"
    data = load_benchmark_data(prepare_summary)
    components = data.get("prepare_plan", {}).get("parameters", {}).get("components", [])
    return str(components[0]).upper() if components else "CO2"
