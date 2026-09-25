"""Thermodynamic and statistical analysis for MLIP-MC Widom insertions."""

from __future__ import annotations

import csv
from math import exp, isfinite, log, sqrt
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np

from pipeline.config import save_benchmark_data


BOLTZMANN_EV_PER_K = 8.617333262145e-5
BOLTZMANN_J_PER_K = 1.380649e-23
AVOGADRO_PER_MOL = 6.02214076e23
ATOMIC_MASS_KG = 1.66053906660e-27
ANGSTROM3_TO_M3 = 1.0e-30
EV_TO_KJ_PER_MOL = 96.48533212331002
CI95_NORMAL_FACTOR = 1.96


def summarize_widom_samples(
    valid_energies_ev: list[float],
    *,
    attempts: int,
    temperature_K: float,
) -> dict[str, float | int | None]:
    """Calculate stable Widom statistics with overlap-zero normalization."""
    if attempts <= 0:
        raise ValueError("attempts must be positive.")
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive.")
    if len(valid_energies_ev) > attempts:
        raise ValueError("Valid Widom energy count cannot exceed attempts.")
    if any(not isfinite(value) for value in valid_energies_ev):
        raise ValueError("Widom adsorption energies must be finite.")
    if not valid_energies_ev:
        return {
            "valid_energy_count": 0,
            "log_average_boltzmann_factor": None,
            "average_boltzmann_factor": 0.0,
            "weighted_adsorption_energy_ev": None,
            "arithmetic_adsorption_energy_ev": None,
        }

    beta_ev = 1.0 / (BOLTZMANN_EV_PER_K * float(temperature_K))
    energies = np.asarray(valid_energies_ev, dtype=float)
    log_weights = -beta_ev * energies
    maximum = float(np.max(log_weights))
    shifted = np.exp(log_weights - maximum)
    shifted_sum = float(np.sum(shifted))
    log_weight_sum = maximum + log(shifted_sum)
    log_average = log_weight_sum - log(attempts)
    average = exp(log_average) if log_average < 709.0 else None
    weighted_energy = float(np.dot(energies, shifted) / shifted_sum)
    return {
        "valid_energy_count": len(valid_energies_ev),
        "log_average_boltzmann_factor": log_average,
        "average_boltzmann_factor": average,
        "weighted_adsorption_energy_ev": weighted_energy,
        "arithmetic_adsorption_energy_ev": fmean(valid_energies_ev),
    }


def widom_thermodynamic_metrics(
    statistics: dict[str, float | int | None],
    *,
    temperature_K: float,
    framework_volume_A3: float,
    framework_mass_amu: float,
) -> dict[str, float | None]:
    """Convert Widom averages to mass-normalized Henry and zero-loading heat."""
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive.")
    if framework_volume_A3 <= 0:
        raise ValueError("framework_volume_A3 must be positive.")
    if framework_mass_amu <= 0:
        raise ValueError("framework_mass_amu must be positive.")

    average = statistics["average_boltzmann_factor"]
    weighted_energy = statistics["weighted_adsorption_energy_ev"]
    henry_mol_kg_pa = None
    if average is not None:
        volume_m3 = framework_volume_A3 * ANGSTROM3_TO_M3
        framework_mass_kg = framework_mass_amu * ATOMIC_MASS_KG
        henry_mol_kg_pa = (
            volume_m3
            * float(average)
            / (
                BOLTZMANN_J_PER_K
                * temperature_K
                * AVOGADRO_PER_MOL
                * framework_mass_kg
            )
        )

    qst_kj_mol = None
    if weighted_energy is not None:
        qst_kj_mol = (
            BOLTZMANN_EV_PER_K * temperature_K - float(weighted_energy)
        ) * EV_TO_KJ_PER_MOL

    return {
        "henry_coefficient_mol_kg_pa": henry_mol_kg_pa,
        "henry_coefficient_mmol_g_bar": (
            None if henry_mol_kg_pa is None else henry_mol_kg_pa * 1.0e5
        ),
        "isosteric_heat_zero_loading_kj_mol": qst_kj_mol,
    }


def analyze_widom_attempts(
    attempt_energies_ev: list[float | None],
    *,
    temperature_K: float,
    framework_volume_A3: float,
    framework_mass_amu: float,
    block_size: int,
    convergence_checkpoints: list[int] | None = None,
) -> dict[str, Any]:
    """Calculate full-run metrics, non-overlapping block SEM, and convergence."""
    if not attempt_energies_ev:
        raise ValueError("At least one Widom attempt is required.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if any(
        value is not None and not isfinite(float(value))
        for value in attempt_energies_ev
    ):
        raise ValueError("Widom attempt energies must be finite or None.")

    overall = _metrics_for_attempts(
        attempt_energies_ev,
        temperature_K=temperature_K,
        framework_volume_A3=framework_volume_A3,
        framework_mass_amu=framework_mass_amu,
    )
    uncertainty = _block_uncertainty(
        attempt_energies_ev,
        full_metrics=overall,
        temperature_K=temperature_K,
        framework_volume_A3=framework_volume_A3,
        framework_mass_amu=framework_mass_amu,
        block_size=block_size,
    )
    checkpoints = _normalize_checkpoints(
        convergence_checkpoints,
        attempts=len(attempt_energies_ev),
    )
    convergence = []
    for checkpoint in checkpoints:
        prefix = attempt_energies_ev[:checkpoint]
        metrics = _metrics_for_attempts(
            prefix,
            temperature_K=temperature_K,
            framework_volume_A3=framework_volume_A3,
            framework_mass_amu=framework_mass_amu,
        )
        prefix_uncertainty = _block_uncertainty(
            prefix,
            full_metrics=metrics,
            temperature_K=temperature_K,
            framework_volume_A3=framework_volume_A3,
            framework_mass_amu=framework_mass_amu,
            block_size=block_size,
        )
        convergence.append(
            {
                "attempts": checkpoint,
                "valid_insertions": metrics["valid_energy_count"],
                "valid_fraction": metrics["valid_energy_count"] / checkpoint,
                "average_boltzmann_factor": metrics[
                    "average_boltzmann_factor"
                ],
                "henry_coefficient_mmol_g_bar": metrics[
                    "henry_coefficient_mmol_g_bar"
                ],
                "henry_ci95_half_width_mmol_g_bar": _ci95(
                    prefix_uncertainty,
                    "henry_coefficient_mmol_g_bar",
                ),
                "isosteric_heat_zero_loading_kj_mol": metrics[
                    "isosteric_heat_zero_loading_kj_mol"
                ],
                "qst_ci95_half_width_kj_mol": _ci95(
                    prefix_uncertainty,
                    "isosteric_heat_zero_loading_kj_mol",
                ),
            }
        )

    return {
        "schema_version": 1,
        "method": "widom_insertion",
        "temperature_K": float(temperature_K),
        "framework_volume_A3": float(framework_volume_A3),
        "framework_mass_amu": float(framework_mass_amu),
        "attempt_count": len(attempt_energies_ev),
        "overlap_rejections": sum(
            value is None for value in attempt_energies_ev
        ),
        "overall": overall,
        "uncertainty": uncertainty,
        "convergence": convergence,
        "units": {
            "henry_coefficient_mol_kg_pa": "mol kg^-1 Pa^-1",
            "henry_coefficient_mmol_g_bar": "mmol g^-1 bar^-1",
            "isosteric_heat_zero_loading_kj_mol": "kJ mol^-1",
        },
        "definitions": {
            "overlap_handling": (
                "Geometric overlap rejections contribute zero Boltzmann weight."
            ),
            "henry": (
                "K_H = V <exp(-beta DeltaU)> / "
                "(k_B T N_A m_framework)."
            ),
            "qst_zero_loading": (
                "Q_st^0 = RT - <DeltaU exp(-beta DeltaU)> / "
                "<exp(-beta DeltaU)>."
            ),
            "uncertainty": (
                "Normal 95% interval from the SEM of non-overlapping complete "
                "blocks; it is unavailable when fewer than two complete blocks exist."
            ),
        },
    }


def write_widom_analysis(
    analysis: dict[str, Any],
    output_directory: str | Path,
    *,
    write_csv: bool = True,
    write_plot: bool = True,
) -> dict[str, str]:
    """Write machine-readable convergence data and an optional summary plot."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "widom_analysis.json"
    save_benchmark_data(json_path, analysis)
    files = {"analysis_json": str(json_path)}

    if write_csv:
        csv_path = output / "widom_convergence.csv"
        rows = analysis["convergence"]
        fieldnames = list(rows[0]) if rows else ["attempts"]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        files["convergence_csv"] = str(csv_path)

    if write_plot:
        plot_path = output / "widom_convergence.png"
        _write_convergence_plot(analysis["convergence"], plot_path)
        files["convergence_plot"] = str(plot_path)
    return files


def _metrics_for_attempts(
    attempts: list[float | None],
    *,
    temperature_K: float,
    framework_volume_A3: float,
    framework_mass_amu: float,
) -> dict[str, float | int | None]:
    valid = [float(value) for value in attempts if value is not None]
    statistics = summarize_widom_samples(
        valid,
        attempts=len(attempts),
        temperature_K=temperature_K,
    )
    return {
        **statistics,
        **widom_thermodynamic_metrics(
            statistics,
            temperature_K=temperature_K,
            framework_volume_A3=framework_volume_A3,
            framework_mass_amu=framework_mass_amu,
        ),
    }


def _block_uncertainty(
    attempts: list[float | None],
    *,
    full_metrics: dict[str, float | int | None],
    temperature_K: float,
    framework_volume_A3: float,
    framework_mass_amu: float,
    block_size: int,
) -> dict[str, Any]:
    complete_block_count = len(attempts) // block_size
    blocks = [
        attempts[index * block_size : (index + 1) * block_size]
        for index in range(complete_block_count)
    ]
    block_metrics = [
        _metrics_for_attempts(
            block,
            temperature_K=temperature_K,
            framework_volume_A3=framework_volume_A3,
            framework_mass_amu=framework_mass_amu,
        )
        for block in blocks
    ]
    metric_names = (
        "average_boltzmann_factor",
        "henry_coefficient_mol_kg_pa",
        "henry_coefficient_mmol_g_bar",
        "weighted_adsorption_energy_ev",
        "isosteric_heat_zero_loading_kj_mol",
    )
    metric_uncertainty = {}
    for name in metric_names:
        values = [
            float(block[name])
            for block in block_metrics
            if block[name] is not None
        ]
        sem = stdev(values) / sqrt(len(values)) if len(values) >= 2 else None
        estimate = full_metrics.get(name)
        ci95 = None if sem is None else CI95_NORMAL_FACTOR * sem
        metric_uncertainty[name] = {
            "estimate": estimate,
            "block_mean": fmean(values) if values else None,
            "standard_error": sem,
            "ci95_half_width": ci95,
            "relative_ci95_half_width": (
                None
                if ci95 is None or estimate in (None, 0)
                else abs(ci95 / float(estimate))
            ),
            "contributing_block_count": len(values),
        }
    return {
        "method": "non_overlapping_block_sem",
        "block_size_attempts": block_size,
        "complete_block_count": complete_block_count,
        "discarded_tail_attempts": len(attempts) - complete_block_count * block_size,
        "metrics": metric_uncertainty,
    }


def _ci95(uncertainty: dict[str, Any], metric: str) -> float | None:
    return uncertainty["metrics"][metric]["ci95_half_width"]


def _normalize_checkpoints(
    configured: list[int] | None,
    *,
    attempts: int,
) -> list[int]:
    if configured is None:
        checkpoints = {
            max(1, round(attempts * fraction / 10))
            for fraction in range(1, 11)
        }
        return sorted(checkpoints)
    if not isinstance(configured, list) or not configured:
        raise ValueError("convergence_checkpoints must be a non-empty list.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > attempts
        for value in configured
    ):
        raise ValueError(
            "convergence_checkpoints must contain positive integers no larger "
            "than the attempt count."
        )
    if len(set(configured)) != len(configured):
        raise ValueError("convergence_checkpoints must not contain duplicates.")
    return sorted(configured)


def _write_convergence_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("Cannot plot an empty Widom convergence series.")
    attempts = [row["attempts"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    _error_series(
        axes[0],
        attempts,
        rows,
        value_key="henry_coefficient_mmol_g_bar",
        error_key="henry_ci95_half_width_mmol_g_bar",
        title="Henry coefficient convergence",
        ylabel=r"$K_H$ / mmol g$^{-1}$ bar$^{-1}$",
    )
    _error_series(
        axes[1],
        attempts,
        rows,
        value_key="isosteric_heat_zero_loading_kj_mol",
        error_key="qst_ci95_half_width_kj_mol",
        title=r"Zero-loading $Q_{st}$ convergence",
        ylabel=r"$Q_{st}^{0}$ / kJ mol$^{-1}$",
    )
    for axis in axes:
        axis.set_xlabel("Widom attempts")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _error_series(
    axis: Any,
    attempts: list[int],
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    error_key: str,
    title: str,
    ylabel: str,
) -> None:
    values = [row[value_key] for row in rows]
    errors = [row[error_key] for row in rows]
    if all(value is None for value in values):
        axis.text(0.5, 0.5, "No finite estimate", ha="center", va="center")
    else:
        plot_values = [float("nan") if value is None else value for value in values]
        plot_errors = [0.0 if value is None else value for value in errors]
        axis.errorbar(
            attempts,
            plot_values,
            yerr=plot_errors,
            marker="o",
            linewidth=1.5,
            capsize=3,
            color="#2878B5",
        )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
