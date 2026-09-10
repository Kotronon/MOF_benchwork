from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.config import load_benchmark_data, normalize_config, save_benchmark_data


def get_directory(config: dict[str, Any]) -> Path:
    """Get the directory that contains one subdirectory per variant."""
    normalized = normalize_config(config)
    output = normalized["output"]
    directory = Path(output.get("directory") or "outputs")
    run_id = output.get("run_id")
    if run_id:
        return directory / "runs" / str(run_id)
    return directory


def get_variant_result_jsons(
    config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Load relevant result JSON files for each variant run directory."""
    working_dir = get_directory(config)
    if not working_dir.exists():
        raise FileNotFoundError(f"Variant run directory does not exist: {working_dir}")

    variants = []
    for variant_dir in sorted(path for path in working_dir.iterdir() if path.is_dir()):
        summary_path = variant_dir / "isotherm_summary.json"
        convergence_path = variant_dir / "convergence_report.json"
        evaluation_path = variant_dir / "evaluation_report.json"

        if not summary_path.exists():
            continue

        variant_result = {
            "variant": variant_dir.name,
            "working_directory": str(variant_dir),
            "isotherm_summary_path": str(summary_path),
            "isotherm_summary": load_benchmark_data(summary_path),
        }

        if convergence_path.exists():
            variant_result["convergence_report_path"] = str(convergence_path)
            variant_result["convergence_report"] = load_benchmark_data(convergence_path)

        if evaluation_path.exists():
            variant_result["evaluation_report_path"] = str(evaluation_path)
            variant_result["evaluation_report"] = load_benchmark_data(evaluation_path)

        variants.append(variant_result)

    return variants


def compare_variant_results(config: dict[str, Any], output_path: Path | str) -> dict[str, Any]:
    """Compare variant run summaries and write a compact JSON report."""
    variant_results = get_variant_result_jsons(config)
    rows = _comparison_rows(variant_results)
    output = Path(output_path)
    report = {
        "status": "completed",
        "variant_count": len(variant_results),
        "point_count": len(rows),
        "source_directory": str(get_directory(config)),
        "rows": rows,
    }
    save_benchmark_data(output, report)
    return {
        **report,
        "comparison_json": str(output),
    }


def _comparison_rows(variant_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for variant_result in variant_results:
        variant_name = variant_result["variant"]
        summary = variant_result["isotherm_summary"]
        convergence_by_pressure = _convergence_by_pressure(
            variant_result.get("convergence_report", {})
        )

        for result in summary.get("results", []):
            pressure_bar = result.get("pressure_bar")
            result_summary = result.get("summary", {})
            convergence = convergence_by_pressure.get(pressure_bar, {})
            rows.append(
                {
                    "variant": variant_name,
                    "pressure_bar": pressure_bar,
                    "mean_adsorbates": result_summary.get("mean_adsorbates"),
                    "standard_error_adsorbates": result_summary.get("standard_error_adsorbates"),
                    "replicate_count": result.get("replicate_count"),
                    "seeds": result.get("seeds", []),
                    "converged": result.get("converged", convergence.get("converged")),
                    "mean_wall_time_seconds": result.get("mean_wall_time_seconds"),
                    "total_wall_time_seconds": result.get("total_wall_time_seconds"),
                    "summary_file": result.get("summary_file"),
                    "log_file": result.get("log_file"),
                }
            )

    return sorted(rows, key=lambda row: (str(row["variant"]), _sortable_pressure(row["pressure_bar"])))


def _convergence_by_pressure(convergence_report: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    entries = convergence_report.get("pressure_points", [])
    if not isinstance(entries, list):
        return {}
    return {entry.get("pressure_bar"): entry for entry in entries}


def _sortable_pressure(value: Any) -> float:
    if value is None:
        return float("inf")
    return float(value)
