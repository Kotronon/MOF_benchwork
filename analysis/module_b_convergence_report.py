from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

from analysis.block_convergence import _find_key, _input_metadata
from parsers.lammps_log_parser import get_log, parse_lammps_log
from pipeline.mixture_statistics import mean_interval
from pipeline.runners import _t_critical_975


def _autocorrelation_time(values: list[float]) -> float:
    count = len(values)
    average = sum(values) / count
    centered = [value - average for value in values]
    variance_sum = sum(value * value for value in centered)
    if count < 2 or variance_sum == 0:
        return 1.0

    rho_values = []
    for lag in range(1, count // 2):
        covariance = sum(
            centered[index] * centered[index + lag]
            for index in range(count - lag)
        )
        rho_values.append(covariance / variance_sum)

    # Geyer's initial-positive sequence reduces noisy long-lag contributions.
    rho_sum = 0.0
    for index in range(0, len(rho_values), 2):
        pair_sum = rho_values[index] + (
            rho_values[index + 1] if index + 1 < len(rho_values) else 0.0
        )
        if pair_sum <= 0:
            break
        rho_sum += pair_sum
    return max(1.0, 1.0 + 2.0 * rho_sum)


def _batch_statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot analyze an empty time series.")
    count = len(values)
    tau = _autocorrelation_time(values)
    batch_size = min(
        max(math.ceil(math.sqrt(count)), math.ceil(2.0 * tau)),
        count // 3,
    )
    batch_count = count // batch_size if batch_size else 0
    batch_means = [
        sum(values[start : start + batch_size]) / batch_size
        for start in range(0, batch_count * batch_size, batch_size)
    ]
    average = sum(values) / count
    standard_error = None
    half_width = None
    if batch_count >= 2:
        batch_average = sum(batch_means) / batch_count
        batch_variance = sum(
            (value - batch_average) ** 2 for value in batch_means
        ) / (batch_count - 1)
        standard_error = math.sqrt(batch_variance / batch_count)
        critical = _t_critical_975(batch_count - 1)
        half_width = critical * standard_error
    return {
        "mean": average,
        "standard_error_autocorrelation": standard_error,
        "ci95_lower": average - half_width if half_width is not None else None,
        "ci95_upper": average + half_width if half_width is not None else None,
        "ci95_status": "bounded" if half_width is not None else "insufficient_batches",
        "sample_count": count,
        "integrated_autocorrelation_time_samples": tau,
        "effective_sample_size": count / tau,
        "batch_size_samples": batch_size,
        "batch_count": batch_count,
    }


def _combined_component_interval(
    means: list[float], seed_statistics: list[dict[str, Any]]
) -> dict[str, Any]:
    result = mean_interval(means, _t_critical_975(len(means) - 1))
    if len(means) < 2 or any(
        item["standard_error_autocorrelation"] is None for item in seed_statistics
    ):
        result.update(
            ci95_status="insufficient_seed_or_batch_uncertainty",
            ci95_half_width=None,
            ci95_lower=None,
            ci95_upper=None,
            relative_ci95_half_width=None,
        )
        return result
    between_variance = sum(
        (value - sum(means) / len(means)) ** 2 for value in means
    ) / (len(means) - 1)
    seed_count = len(means)
    within_variance_of_mean = sum(
        item["standard_error_autocorrelation"] ** 2 for item in seed_statistics
    ) / seed_count**2
    standard_error = math.sqrt(between_variance / seed_count + within_variance_of_mean)
    half_width = _t_critical_975(seed_count - 1) * standard_error
    average = sum(means) / seed_count
    result.update(
        ci95_method="student_t_between_seeds_plus_autocorrelation_batch_error",
        ci95_status="bounded",
        ci95_half_width=half_width,
        ci95_lower=average - half_width,
        ci95_upper=average + half_width,
        relative_ci95_half_width=half_width / abs(average) if average else None,
        standard_error_combined=standard_error,
    )
    return result


def _batch_mean_covariance(
    co2_values: list[float], n2_values: list[float], batch_size: int
) -> float:
    batch_count = min(len(co2_values), len(n2_values)) // batch_size
    if batch_count < 2:
        return 0.0
    co2_means = [
        sum(co2_values[index * batch_size : (index + 1) * batch_size]) / batch_size
        for index in range(batch_count)
    ]
    n2_means = [
        sum(n2_values[index * batch_size : (index + 1) * batch_size]) / batch_size
        for index in range(batch_count)
    ]
    co2_average = sum(co2_means) / batch_count
    n2_average = sum(n2_means) / batch_count
    return sum(
        (co2 - co2_average) * (n2 - n2_average)
        for co2, n2 in zip(co2_means, n2_means)
    ) / (batch_count * (batch_count - 1))


def _combined_fieller_interval(
    replicates: list[dict[str, Any]],
    component_means: dict[str, list[float]],
    scale: float,
) -> dict[str, Any]:
    result = {
        "ci95_method": "fieller_paired_seeds_plus_autocorrelation_batch_error",
        "ci95_status": "insufficient_replicates",
        "ci95_lower": None,
        "ci95_upper": None,
        "ci95_half_width": None,
        "relative_ci95_half_width": None,
    }
    count = len(replicates)
    if count < 2:
        return result
    co2_means, n2_means = component_means["CO2"], component_means["N2"]
    co2_average = sum(co2_means) / count
    n2_average = sum(n2_means) / count
    if n2_average <= 0 or scale <= 0:
        result["ci95_status"] = "undefined_zero_denominator_or_gas_fraction"
        return result
    between_co2 = sum((value - co2_average) ** 2 for value in co2_means) / (count * (count - 1))
    between_n2 = sum((value - n2_average) ** 2 for value in n2_means) / (count * (count - 1))
    between_covariance = sum(
        (co2 - co2_average) * (n2 - n2_average)
        for co2, n2 in zip(co2_means, n2_means)
    ) / (count * (count - 1))
    within_co2 = 0.0
    within_n2 = 0.0
    within_covariance = 0.0
    for replicate in replicates:
        co2_values = [float(row["v_nco2"]) for row in replicate["rows"]]
        n2_values = [float(row["v_nn2"]) for row in replicate["rows"]]
        co2_stats = _batch_statistics(co2_values)
        n2_stats = _batch_statistics(n2_values)
        within_co2 += co2_stats["standard_error_autocorrelation"] ** 2
        within_n2 += n2_stats["standard_error_autocorrelation"] ** 2
        within_covariance += _batch_mean_covariance(
            co2_values,
            n2_values,
            max(co2_stats["batch_size_samples"], n2_stats["batch_size_samples"]),
        )
    variance_co2 = between_co2 + within_co2 / count**2
    variance_n2 = between_n2 + within_n2 / count**2
    covariance = between_covariance + within_covariance / count**2
    critical_squared = _t_critical_975(count - 1) ** 2
    a = n2_average**2 - critical_squared * variance_n2
    b = co2_average * n2_average - critical_squared * covariance
    c = co2_average**2 - critical_squared * variance_co2
    if a <= 0:
        result["ci95_status"] = "unbounded"
        return result
    discriminant = b * b - a * c
    if discriminant < -1e-12 * max(abs(b * b), abs(a * c), 1e-300):
        result["ci95_status"] = "not_estimable"
        return result
    root = math.sqrt(max(0.0, discriminant))
    lower = scale * (b - root) / a
    upper = scale * (b + root) / a
    value = scale * co2_average / n2_average
    half_width = (upper - lower) / 2
    result.update(
        ci95_status="bounded",
        ci95_lower=lower,
        ci95_upper=upper,
        ci95_half_width=half_width,
        relative_ci95_half_width=half_width / abs(value) if value else None,
    )
    return result


def _load_runs(run_dir: Path) -> tuple[dict[float, list[dict[str, Any]]], dict[str, Any]]:
    prepare_path = run_dir / "prepare_summary.json"
    if not prepare_path.exists():
        raise FileNotFoundError(f"Missing prepare summary: {prepare_path}")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    production_steps = _find_key(prepare, "production_steps")
    equilibration_steps = _find_key(prepare, "equilibration_steps")
    if production_steps is None or equilibration_steps is None:
        raise ValueError("prepare_summary.json lacks production/equilibration steps.")
    production_steps = int(production_steps)
    equilibration_steps = int(equilibration_steps)

    grouped: dict[float, list[dict[str, Any]]] = {}
    mixtures = []
    log_files = sorted((run_dir / "logs").glob("gcmc_*bar*.log"))
    if not log_files:
        raise FileNotFoundError(f"No pressure-point logs found in {run_dir / 'logs'}.")
    for log_file in log_files:
        input_path = run_dir / "inputs" / f"{log_file.stem}.in"
        metadata = _input_metadata(input_path)
        mixture = _input_mixture(input_path)
        mixtures.append(mixture)
        pressure = float(metadata["pressure_bar"])
        seed = metadata["seed"]
        parsed = parse_lammps_log(get_log(log_file))
        rows = [
            row
            for row in parsed["rows"]
            if "Step" in row
            and "v_nco2" in row
            and "v_nn2" in row
            and equilibration_steps <= int(row["Step"])
            <= equilibration_steps + production_steps
        ]
        if not rows:
            raise ValueError(f"No component production samples found in {log_file}.")
        if seed is None:
            raise ValueError(f"Could not read the random seed from {log_file}.")
        grouped.setdefault(pressure, []).append(
            {
                "seed": int(seed),
                "mixture": mixture,
                "log_file": str(log_file),
                "rows": rows,
                "complete": max(int(row["Step"]) for row in rows)
                >= equilibration_steps + production_steps,
            }
        )
    for pressure, replicates in grouped.items():
        seeds = [replicate["seed"] for replicate in replicates]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds found at {pressure:g} bar.")
        replicates.sort(key=lambda replicate: replicate["seed"])
    metadata = {
        "production_steps": production_steps,
        "equilibration_steps": equilibration_steps,
        "planned_end_step": equilibration_steps + production_steps,
        "mixture": mixtures[0],
    }
    if any(mixture != metadata["mixture"] for mixture in mixtures[1:]):
        raise ValueError("Mixture composition differs across pressure-point inputs.")
    return grouped, metadata


def _input_mixture(input_path: Path) -> dict[str, float]:
    pattern = re.compile(
        r"^#\s*(CO2|N2)\s+mole_fraction\s+([0-9.eE+-]+)", re.MULTILINE
    )
    mixture = {
        component: float(value)
        for component, value in pattern.findall(input_path.read_text(encoding="utf-8"))
    }
    if set(mixture) != {"CO2", "N2"} or not math.isclose(
        sum(mixture.values()), 1.0, abs_tol=1e-8
    ):
        raise ValueError(f"Expected normalized CO2/N2 fractions in {input_path}.")
    return mixture


def _window_means(
    replicates: list[dict[str, Any]], start_step: int, end_step: int
) -> dict[str, list[float]]:
    means: dict[str, list[float]] = {"CO2": [], "N2": []}
    for replicate in replicates:
        rows = [
            row
            for row in replicate["rows"]
            if start_step <= int(row["Step"]) < end_step
        ]
        if not rows:
            raise ValueError(
                f"No samples in requested step window {start_step}:{end_step} "
                f"for seed {replicate['seed']}."
            )
        means["CO2"].append(sum(float(row["v_nco2"]) for row in rows) / len(rows))
        means["N2"].append(sum(float(row["v_nn2"]) for row in rows) / len(rows))
    return means


def _early_late(
    early: list[float], late: list[float], total: list[float], absolute_floor: float
) -> dict[str, Any]:
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    total_mean = sum(total) / len(total)
    absolute_difference = abs(early_mean - late_mean)
    relative_difference = (
        absolute_difference / abs(total_mean) if total_mean != 0 else None
    )
    return {
        "early_mean": early_mean,
        "late_mean": late_mean,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "small_loading_absolute_check": abs(total_mean) <= absolute_floor,
    }


def _component_result(
    name: str,
    replicates: list[dict[str, Any]],
    early_values: list[float],
    late_values: list[float],
    relative_tolerance: float,
    absolute_floor: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_stats = []
    means = []
    for replicate in replicates:
        values = [float(row[f"v_n{name.lower()}"]) for row in replicate["rows"]]
        statistics = _batch_statistics(values)
        statistics.update(seed=replicate["seed"], log_file=replicate["log_file"])
        seed_stats.append(statistics)
        means.append(statistics["mean"])
    aggregate = _combined_component_interval(means, seed_stats)
    mean_value = sum(means) / len(means)
    early_late = _early_late(early_values, late_values, means, absolute_floor)
    relative_ci = aggregate["relative_ci95_half_width"]
    uncertainty_ok = (
        len(means) >= 3
        and aggregate["ci95_status"] == "bounded"
        and relative_ci is not None
        and relative_ci <= relative_tolerance
    )
    drift_ok = (
        early_late["relative_difference"] is not None
        and early_late["relative_difference"] <= relative_tolerance
    ) or (
        early_late["small_loading_absolute_check"]
        and early_late["absolute_difference"] <= absolute_floor
    )
    return (
        {
            "quantity": name,
            "value": mean_value,
            "unit": "molecules/cell",
            **aggregate,
            "seed_count": len(means),
            "seed_statistics": seed_stats,
            **{f"early_late_{key}": value for key, value in early_late.items()},
            "status": "plausible" if uncertainty_ok and drift_ok else "check",
        },
        {"means": means, "seed_statistics": seed_stats},
    )


def _selectivity_result(
    replicates: list[dict[str, Any]],
    component_means: dict[str, list[float]],
    early_means: dict[str, list[float]],
    late_means: dict[str, list[float]],
    gas_fractions: tuple[float, float],
    relative_tolerance: float,
) -> dict[str, Any]:
    co2_fraction, n2_fraction = gas_fractions
    scale = n2_fraction / co2_fraction
    co2_means, n2_means = component_means["CO2"], component_means["N2"]
    value = (
        sum(co2_means) / len(co2_means)
        / (sum(n2_means) / len(n2_means))
        * scale
        if sum(n2_means) > 0
        else None
    )
    interval = _combined_fieller_interval(replicates, component_means, scale)
    early_ratios = [
        (co2 / n2 * scale) if n2 > 0 else math.nan
        for co2, n2 in zip(early_means["CO2"], early_means["N2"])
    ]
    late_ratios = [
        (co2 / n2 * scale) if n2 > 0 else math.nan
        for co2, n2 in zip(late_means["CO2"], late_means["N2"])
    ]
    if all(math.isfinite(item) for item in [*early_ratios, *late_ratios]):
        total_ratios = [
            co2 / n2 * scale
            for co2, n2 in zip(co2_means, n2_means)
            if n2 > 0
        ]
        early_late = _early_late(early_ratios, late_ratios, total_ratios, 0.0)
    else:
        early_late = {
            "early_mean": None,
            "late_mean": None,
            "absolute_difference": None,
            "relative_difference": None,
            "small_loading_absolute_check": False,
        }
    relative_ci = interval.get("relative_ci95_half_width")
    uncertainty_ok = (
        len(replicates) >= 3
        and interval["ci95_status"] == "bounded"
        and relative_ci is not None
        and relative_ci <= relative_tolerance
    )
    drift_ok = (
        early_late["relative_difference"] is not None
        and early_late["relative_difference"] <= relative_tolerance
    )
    return {
        "quantity": "CO2/N2",
        "value": value,
        "unit": "dimensionless",
        **interval,
        "seed_count": len(replicates),
        "seed_ratios": [
            co2 / n2 * scale if n2 > 0 else None
            for co2, n2 in zip(co2_means, n2_means)
        ],
        **{f"early_late_{key}": item for key, item in early_late.items()},
        "status": "plausible" if uncertainty_ok and drift_ok else "check",
    }


def analyze_module_b_convergence(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    early_steps: int = 20_000,
    relative_tolerance: float = 0.2,
    absolute_loading_floor: float = 0.05,
) -> dict[str, Any]:
    if early_steps <= 0 or relative_tolerance <= 0 or absolute_loading_floor <= 0:
        raise ValueError("Window and convergence tolerances must be positive.")
    run_path = Path(run_dir).resolve()
    grouped, metadata = _load_runs(run_path)
    production_start = metadata["equilibration_steps"]
    production_end = metadata["planned_end_step"]
    production_steps = metadata["production_steps"]
    if early_steps >= production_steps:
        raise ValueError("early_steps must be shorter than the production run.")
    early_end = production_start + early_steps
    late_start = production_end - (production_steps - early_steps)

    rows: list[dict[str, Any]] = []
    point_reports = []
    for pressure, replicates in sorted(grouped.items()):
        early = _window_means(replicates, production_start, early_end)
        late = _window_means(replicates, late_start, production_end + 1)
        all_means = {
            component: [
                sum(float(row[f"v_n{component.lower()}"]) for row in replicate["rows"])
                / len(replicate["rows"])
                for replicate in replicates
            ]
            for component in ("CO2", "N2")
        }
        components = {}
        for component in ("CO2", "N2"):
            result, detail = _component_result(
                component,
                replicates,
                early[component],
                late[component],
                relative_tolerance,
                absolute_loading_floor,
            )
            components[component] = detail
            result.update(
                pressure_bar=pressure,
                replicates_complete=all(r["complete"] for r in replicates),
            )
            if len(replicates) < 3 or not result["replicates_complete"]:
                result["status"] = "check"
            rows.append(result)
        mixture = _selectivity_result(
            replicates,
            all_means,
            early,
            late,
            (metadata["mixture"]["CO2"], metadata["mixture"]["N2"]),
            relative_tolerance,
        )
        mixture.update(
            pressure_bar=pressure,
            replicates_complete=all(r["complete"] for r in replicates),
        )
        if len(replicates) < 3 or not mixture["replicates_complete"]:
            mixture["status"] = "check"
        rows.append(mixture)
        point_reports.append(
            {
                "pressure_bar": pressure,
                "seed_count": len(replicates),
                "seeds": [replicate["seed"] for replicate in replicates],
                "replicates_complete": all(r["complete"] for r in replicates),
                "components": {
                    component: {
                        "seed_means": all_means[component],
                        "seed_statistics": components[component]["seed_statistics"],
                    }
                    for component in ("CO2", "N2")
                },
            }
        )

    report = {
        "status": "completed" if all(row["replicates_complete"] for row in rows) else "provisional",
        "run_directory": str(run_path),
        "production_steps": production_steps,
        "equilibration_steps": metadata["equilibration_steps"],
        "windows": {
            "early_start_step": production_start,
            "early_end_step_exclusive": early_end,
            "late_start_step": late_start,
            "late_end_step_inclusive": production_end,
        },
        "decision_thresholds": {
            "minimum_independent_seeds": 3,
            "relative_ci95_half_width": relative_tolerance,
            "early_late_relative_difference": relative_tolerance,
            "small_loading_floor_molecules_per_cell": absolute_loading_floor,
        },
        "interpretation": "Autocorrelation-adjusted within-seed batch estimates and independent-seed intervals measure statistical precision, not proof of equilibrium.",
        "results": rows,
        "pressure_points": point_reports,
    }
    target = Path(output_dir) if output_dir else run_path / "analysis" / "module_b_convergence"
    target.mkdir(parents=True, exist_ok=True)
    report["outputs"] = {
        "json": str(target / "convergence_report.json"),
        "csv": str(target / "convergence_report.csv"),
        "markdown": str(target / "convergence_report.md"),
    }
    (target / "convergence_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = [
        "pressure_bar",
        "quantity",
        "value",
        "unit",
        "ci95_lower",
        "ci95_upper",
        "ci95_status",
        "ci95_method",
        "relative_ci95_half_width",
        "early_late_early_mean",
        "early_late_late_mean",
        "early_late_absolute_difference",
        "early_late_relative_difference",
        "status",
        "seed_count",
        "replicates_complete",
    ]
    with (target / "convergence_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(target / "convergence_report.md", report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Module B convergence report",
        "",
        "| Pressure (bar) | Species | Mean uptake | 95% CI | Early/late absolute | Early/late relative | Status |",
        "|---:|:---|---:|:---|---:|---:|:---|",
    ]
    for row in report["results"]:
        interval = (
            f"[{row['ci95_lower']:.5g}, {row['ci95_upper']:.5g}]"
            if row.get("ci95_lower") is not None
            else row.get("ci95_status", "not available")
        )
        relative = row.get("early_late_relative_difference")
        lines.append(
            "| {pressure:g} | {quantity} | {value} | {interval} | {absolute} | {relative} | {status} |".format(
                pressure=row["pressure_bar"],
                quantity=row["quantity"],
                value=(f"{row['value']:.5g}" if row.get("value") is not None else "undefined"),
                interval=interval,
                absolute=(f"{row['early_late_absolute_difference']:.5g}" if row.get("early_late_absolute_difference") is not None else "n/a"),
                relative=(f"{relative:.1%}" if relative is not None else "n/a"),
                status=row["status"],
            )
        )
    lines.extend(
        [
            "",
            "Species intervals combine independent-seed Student-t variation with within-seed batch-mean uncertainty sized from an initial-positive autocorrelation estimate.",
            "Selectivity intervals use paired-seed Fieller inference with the same autocorrelation-aware uncertainty propagation. These precision checks do not establish equilibrium.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--early-steps", type=int, default=20_000)
    parser.add_argument("--relative-tolerance", type=float, default=0.2)
    parser.add_argument("--absolute-loading-floor", type=float, default=0.05)
    args = parser.parse_args(argv)
    report = analyze_module_b_convergence(
        args.run_dir,
        args.output_dir,
        early_steps=args.early_steps,
        relative_tolerance=args.relative_tolerance,
        absolute_loading_floor=args.absolute_loading_floor,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())