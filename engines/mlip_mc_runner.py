"""Direct MLIP-MC execution with an explicitly supplied ASE calculator."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import json
from math import isfinite
from pathlib import Path
import random
from statistics import fmean, pstdev
import struct
from time import perf_counter
from typing import Any, Callable

import numpy as np

from analysis.mlip_widom import (
    analyze_widom_attempts,
    summarize_widom_samples,
    widom_thermodynamic_metrics,
    write_widom_analysis,
)
from pipeline.config import save_benchmark_data


def run_mlip_mc_widom(
    calculator: Any,
    framework: Any,
    adsorbate: Any,
    *,
    temperature_K: float,
    trial_count: int,
    seed: int,
    device: str,
    output_directory: str | Path,
    block_size: int | None = None,
    convergence_checkpoints: list[int] | None = None,
    write_analysis_csv: bool = True,
    write_analysis_plot: bool = True,
    engine_class: Any | None = None,
) -> dict[str, Any]:
    """Run MLIP-MC Widom insertion and write a normalized result."""
    if trial_count <= 0:
        raise ValueError("MLIP-MC Widom trial_count must be positive.")
    if temperature_K <= 0:
        raise ValueError("MLIP-MC temperature_K must be positive.")
    if block_size is not None and block_size <= 0:
        raise ValueError("MLIP-MC Widom block_size must be positive.")
    _validate_seed(seed)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "mlip_mc_widom.log"
    _seed_random_generators(seed)
    vdw_radii = _vdw_radii_for(framework, adsorbate)
    widom_class = engine_class or _load_widom_class()

    started = perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            simulation = widom_class(
                model=calculator,
                atoms_frame=framework,
                atoms_ads=adsorbate,
                T=float(temperature_K),
                device=device,
                vdw_radii=vdw_radii,
                output_dir=str(output),
            )
            simulation.run(int(trial_count))
    runtime_seconds = perf_counter() - started

    raw_path = output / "widom_results.json"
    if not raw_path.is_file():
        raise RuntimeError(
            f"MLIP-MC completed without producing {raw_path}. See {log_path}."
        )
    with raw_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    attempts = int(raw.get("attempts", trial_count))
    valid_insertions = int(
        raw.get(
            "valid_insertions",
            getattr(simulation, "stats", {}).get("valid_insertions", 0),
        )
    )
    overlap_rejections = int(
        getattr(simulation, "stats", {}).get(
            "vdw_overlaps",
            max(0, attempts - valid_insertions),
        )
    )
    energies = [float(value) for value in raw.get("raw_adsorption_energies", [])]
    framework_volume_A3 = float(framework.get_volume())
    framework_mass_amu = float(np.sum(framework.get_masses()))
    corrected_statistics = summarize_widom_samples(
        energies,
        attempts=attempts,
        temperature_K=temperature_K,
    )
    corrected_statistics.update(
        widom_thermodynamic_metrics(
            corrected_statistics,
            temperature_K=temperature_K,
            framework_volume_A3=framework_volume_A3,
            framework_mass_amu=framework_mass_amu,
        )
    )

    trial_log_path = output / "log_widom.bin"
    trial_records = _read_widom_trial_log(trial_log_path)
    analysis = None
    analysis_files: dict[str, str] = {}
    trace_status = "unavailable"
    if trial_records:
        attempt_energies = _reconstruct_widom_attempts(
            trial_records,
            attempts=attempts,
        )
        trace_status = "ordered_binary_log"
        effective_block_size = block_size or max(1, attempts // 10)
        analysis = analyze_widom_attempts(
            attempt_energies,
            temperature_K=temperature_K,
            framework_volume_A3=framework_volume_A3,
            framework_mass_amu=framework_mass_amu,
            block_size=effective_block_size,
            convergence_checkpoints=convergence_checkpoints,
        )
        corrected_statistics = analysis["overall"]
        analysis_files = write_widom_analysis(
            analysis,
            output,
            write_csv=write_analysis_csv,
            write_plot=write_analysis_plot,
        )
    result = {
        "status": "completed",
        "engine": "mlip-mc",
        "method": "widom",
        "temperature_K": float(temperature_K),
        "seed": seed,
        "attempts": attempts,
        "valid_insertions": valid_insertions,
        "overlap_rejections": overlap_rejections,
        "valid_fraction": valid_insertions / attempts if attempts else 0.0,
        "raw_adsorption_energies_ev": energies,
        "framework_volume_A3": framework_volume_A3,
        "framework_mass_amu": framework_mass_amu,
        "corrected_statistics": corrected_statistics,
        "uncertainty": None if analysis is None else analysis["uncertainty"],
        "convergence": [] if analysis is None else analysis["convergence"],
        "attempt_trace_status": trace_status,
        "trial_log_path": str(trial_log_path),
        "analysis_files": analysis_files,
        "runtime_seconds": runtime_seconds,
        "raw_result_path": str(raw_path),
        "log_path": str(log_path),
        "normalization": (
            "Boltzmann factors are divided by all attempted insertions; "
            "overlap rejections contribute zero."
        ),
    }
    normalized_path = output / "widom_normalized.json"
    result["output_path"] = str(normalized_path)
    save_benchmark_data(normalized_path, result)
    return result


def run_mlip_mc_gcmc(
    calculator: Any,
    framework: Any,
    adsorbate: Any,
    *,
    component: str,
    temperature_K: float,
    pressures_bar: list[float],
    equilibration_steps: int,
    production_steps: int,
    seed: int,
    device: str,
    output_directory: str | Path,
    checkpoint_interval: int = 10_000,
    write_trajectory: bool = False,
    trajectory_interval: int = 100,
    overwrite_checkpoints: bool = False,
    allow_ideal_gas_fallback: bool = False,
    engine_class: Any | None = None,
    eos_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Run one direct MLIP-MC GCMC simulation per pressure point."""
    if equilibration_steps < 0 or production_steps <= 0:
        raise ValueError(
            "MLIP-MC GCMC requires non-negative equilibration_steps and "
            "positive production_steps."
        )
    if not pressures_bar or any(float(value) <= 0 for value in pressures_bar):
        raise ValueError("MLIP-MC GCMC pressures must be positive.")
    _validate_seed(seed)

    try:
        from ase.units import bar
    except ImportError as exc:
        raise ImportError("ASE is required for MLIP-MC GCMC.") from exc

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    gcmc_class = engine_class or _load_gcmc_class()
    eos_builder = eos_factory or _load_preos_factory()
    vdw_radii = _vdw_radii_for(framework, adsorbate)
    total_steps = int(equilibration_steps) + int(production_steps)
    point_results = []
    started = perf_counter()

    for point_index, pressure_value in enumerate(pressures_bar):
        pressure_bar = float(pressure_value)
        point_seed = seed + point_index
        _seed_random_generators(point_seed)
        point_directory = output / f"{_pressure_label(pressure_bar)}bar"
        point_directory.mkdir(parents=True, exist_ok=True)
        log_path = point_directory / "mlip_mc_gcmc.log"
        pressure_internal = pressure_bar * bar
        try:
            fugacity_internal = eos_builder(component).calculate_fugacity(
                float(temperature_K),
                pressure_internal,
            )
            fugacity_source = "peng-robinson"
        except Exception:
            if not allow_ideal_gas_fallback:
                raise
            fugacity_internal = pressure_internal
            fugacity_source = "ideal-gas-fallback"

        point_started = perf_counter()
        with log_path.open("w", encoding="utf-8") as log_handle:
            with redirect_stdout(log_handle), redirect_stderr(log_handle):
                simulation = gcmc_class(
                    model=calculator,
                    atoms_frame=framework,
                    atoms_ads=adsorbate,
                    T=float(temperature_K),
                    P=pressure_internal,
                    fugacity=fugacity_internal,
                    device=device,
                    vdw_radii=vdw_radii,
                    output_dir=str(point_directory),
                    n_equilibration_steps=int(equilibration_steps),
                    n_production_steps=int(production_steps),
                    checkpoint_interval=int(checkpoint_interval),
                    write_trajectory=bool(write_trajectory),
                    trajectory_interval=int(trajectory_interval),
                    overwrite_checkpoints=bool(overwrite_checkpoints),
                )
                simulation.run(total_steps)
        point_runtime = perf_counter() - point_started

        raw_path = point_directory / f"results_{pressure_bar:.4f}bar.json"
        if not raw_path.is_file():
            raise RuntimeError(
                f"MLIP-MC completed without producing {raw_path}. See {log_path}."
            )
        with raw_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        transition_log_path = point_directory / f"log_{pressure_bar:.4f}bar.bin"
        transitions = _read_gcmc_transition_log(transition_log_path)
        reconstructed = _reconstruct_gcmc_samples(
            transitions,
            total_steps=total_steps,
        )
        production_rows = reconstructed[int(equilibration_steps):]
        production_uptake = [float(row["uptake"]) for row in production_rows]
        production_energies = [
            float(row["interaction_energy_ev"])
            for row in production_rows
        ]
        if not production_uptake:
            raise RuntimeError(
                f"No GCMC production samples were found for {pressure_bar:g} bar."
            )
        nonzero_energies = [value for value in production_energies if value != 0.0]
        point_results.append(
            {
                "pressure_bar": pressure_bar,
                "fugacity_bar": float(fugacity_internal / bar),
                "fugacity_source": fugacity_source,
                "seed": point_seed,
                "equilibration_steps": int(equilibration_steps),
                "production_steps": int(production_steps),
                "production_sample_count": len(production_uptake),
                "accepted_transition_count": len(transitions),
                "sampling_basis": "all_mc_steps_reconstructed_from_transition_log",
                "mean_uptake_molecules_per_unit_cell": fmean(production_uptake),
                "uptake_std_molecules_per_unit_cell": (
                    pstdev(production_uptake)
                    if len(production_uptake) > 1
                    else 0.0
                ),
                "mean_interaction_energy_ev": (
                    fmean(nonzero_energies) if nonzero_energies else 0.0
                ),
                "runtime_seconds": point_runtime,
                "raw_result_path": str(raw_path),
                "transition_log_path": str(transition_log_path),
                "log_path": str(log_path),
                "move_statistics": getattr(simulation, "moves", {}),
            }
        )

    result = {
        "status": "completed",
        "engine": "mlip-mc",
        "method": "gcmc",
        "component": component,
        "temperature_K": float(temperature_K),
        "seed": seed,
        "runtime_seconds": perf_counter() - started,
        "pressure_points": point_results,
    }
    normalized_path = output / "isotherm_normalized.json"
    result["output_path"] = str(normalized_path)
    save_benchmark_data(normalized_path, result)
    return result


def _read_gcmc_transition_log(path: Path) -> list[dict[str, float | int]]:
    """Read MLIP-MC's accepted-transition binary log for pinned version 0.1.3."""
    if not path.exists():
        return []
    header_format = "iiddi"
    header_size = struct.calcsize(header_format)
    records = []
    with path.open("rb") as handle:
        while True:
            payload = handle.read(header_size)
            if not payload:
                break
            if len(payload) != header_size:
                raise ValueError(f"Incomplete MLIP-MC GCMC log record in {path}.")
            step, uptake, interaction_energy, total_energy, atom_count = struct.unpack(
                header_format,
                payload,
            )
            records.append(
                {
                    "step": int(step),
                    "uptake": int(uptake),
                    "interaction_energy_ev": float(interaction_energy),
                    "total_energy_ev": float(total_energy),
                    "atom_count": int(atom_count),
                }
            )
    return records


def _read_widom_trial_log(path: Path) -> list[dict[str, float | int]]:
    """Read valid Widom insertions and their original trial indices."""
    if not path.exists():
        return []
    header_format = "iddi"
    header_size = struct.calcsize(header_format)
    records = []
    with path.open("rb") as handle:
        while True:
            payload = handle.read(header_size)
            if not payload:
                break
            if len(payload) != header_size:
                raise ValueError(f"Incomplete MLIP-MC Widom log record in {path}.")
            trial, adsorption_energy, total_energy, atom_count = struct.unpack(
                header_format,
                payload,
            )
            if atom_count <= 0:
                raise ValueError(f"Invalid atom count in MLIP-MC Widom log {path}.")
            skip_bytes = atom_count * 4 + atom_count * 3 * 8 + 9 * 8
            skipped = handle.read(skip_bytes)
            if len(skipped) != skip_bytes:
                raise ValueError(f"Incomplete MLIP-MC Widom structure in {path}.")
            records.append(
                {
                    "trial": int(trial),
                    "adsorption_energy_ev": float(adsorption_energy),
                    "total_energy_ev": float(total_energy),
                    "atom_count": int(atom_count),
                }
            )
    return records


def _reconstruct_widom_attempts(
    records: list[dict[str, float | int]],
    *,
    attempts: int,
) -> list[float | None]:
    """Restore valid energies and zero-weight overlaps in trial order."""
    reconstructed: list[float | None] = [None] * attempts
    for record in records:
        trial = int(record["trial"])
        if trial <= 0 or trial > attempts:
            raise ValueError(
                f"MLIP-MC Widom trial {trial} is outside 1..{attempts}."
            )
        if reconstructed[trial - 1] is not None:
            raise ValueError(f"Duplicate MLIP-MC Widom trial {trial}.")
        reconstructed[trial - 1] = float(record["adsorption_energy_ev"])
    return reconstructed


def _reconstruct_gcmc_samples(
    transitions: list[dict[str, float | int]],
    *,
    total_steps: int,
) -> list[dict[str, float | int]]:
    """Expand accepted transitions into one Markov-chain state per MC step."""
    by_step: dict[int, dict[str, float | int]] = {}
    for transition in transitions:
        step = int(transition["step"])
        if step <= 0 or step > total_steps:
            raise ValueError(
                f"MLIP-MC transition step {step} is outside 1..{total_steps}."
            )
        if step in by_step:
            raise ValueError(f"Duplicate MLIP-MC transition at step {step}.")
        by_step[step] = transition

    current_uptake = 0
    current_interaction_energy = 0.0
    current_total_energy = 0.0
    samples = []
    for step in range(1, total_steps + 1):
        transition = by_step.get(step)
        if transition is not None:
            current_uptake = int(transition["uptake"])
            current_interaction_energy = float(
                transition["interaction_energy_ev"]
            )
            current_total_energy = float(transition["total_energy_ev"])
        samples.append(
            {
                "step": step,
                "uptake": current_uptake,
                "interaction_energy_ev": current_interaction_energy,
                "total_energy_ev": current_total_energy,
            }
        )
    return samples


def _seed_random_generators(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise ValueError("MLIP-MC seed must be a positive integer.")


def _vdw_radii_for(framework: Any, adsorbate: Any) -> np.ndarray:
    try:
        from ase.data import vdw_radii
    except ImportError as exc:
        raise ImportError("ASE is required for MLIP-MC van der Waals radii.") from exc
    radii = np.asarray(vdw_radii, dtype=float).copy()
    atomic_numbers = set(framework.get_atomic_numbers()) | set(
        adsorbate.get_atomic_numbers()
    )
    missing = sorted(
        int(number)
        for number in atomic_numbers
        if number >= len(radii) or not isfinite(float(radii[number]))
    )
    if missing:
        raise ValueError(
            "ASE has no finite van der Waals radius for atomic numbers: "
            + ", ".join(map(str, missing))
        )
    return radii


def _load_widom_class() -> Any:
    try:
        from mlip_mc.src.widom import MLP_Widom
    except ImportError as exc:
        raise ImportError(
            "MLIP-MC is required for Widom execution. Re-run with "
            "--install-missing."
        ) from exc
    return MLP_Widom


def _load_gcmc_class() -> Any:
    try:
        from mlip_mc.src.gcmc import MLP_GCMC
    except ImportError as exc:
        raise ImportError(
            "MLIP-MC is required for GCMC execution. Re-run with "
            "--install-missing."
        ) from exc
    return MLP_GCMC


def _load_preos_factory() -> Callable[[str], Any]:
    try:
        from mlip_mc.src.utilities import PREOS
    except ImportError as exc:
        raise ImportError(
            "MLIP-MC is required for Peng-Robinson fugacity calculations."
        ) from exc
    return PREOS.from_name


def _pressure_label(pressure_bar: float) -> str:
    return f"{pressure_bar:g}".replace(".", "p")
