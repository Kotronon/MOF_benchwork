"""Diagnose D3 sensitivity from paired MLIP-MC Widom binary traces."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from math import exp
from pathlib import Path
import struct
from typing import Any

import numpy as np

from analysis.mlip_widom import BOLTZMANN_EV_PER_K, EV_TO_KJ_PER_MOL
from pipeline.config import save_benchmark_data


@dataclass(frozen=True)
class WidomStructureRecord:
    """One valid Widom insertion stored by MLIP-MC."""

    trial: int
    interaction_energy_ev: float
    total_energy_ev: float
    atomic_numbers: np.ndarray
    positions_A: np.ndarray
    cell_A: np.ndarray


def read_widom_structure_trace(path: str | Path) -> list[WidomStructureRecord]:
    """Read interaction energies and structures from an MLIP-MC binary trace."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing MLIP-MC Widom trace: {source}")
    records = []
    header_format = "iddi"
    header_size = struct.calcsize(header_format)
    with source.open("rb") as handle:
        while True:
            payload = handle.read(header_size)
            if not payload:
                break
            if len(payload) != header_size:
                raise ValueError(f"Incomplete Widom record header in {source}.")
            trial, interaction, total, atom_count = struct.unpack(
                header_format, payload
            )
            if trial <= 0 or atom_count <= 0:
                raise ValueError(f"Invalid Widom record in {source}.")
            numbers = _read_values(handle, f"{atom_count}i", source)
            positions = _read_values(handle, f"{atom_count * 3}d", source)
            cell = _read_values(handle, "9d", source)
            records.append(
                WidomStructureRecord(
                    trial=int(trial),
                    interaction_energy_ev=float(interaction),
                    total_energy_ev=float(total),
                    atomic_numbers=np.asarray(numbers, dtype=int),
                    positions_A=np.asarray(positions, dtype=float).reshape(
                        atom_count, 3
                    ),
                    cell_A=np.asarray(cell, dtype=float).reshape(3, 3),
                )
            )
    trials = [record.trial for record in records]
    if len(set(trials)) != len(trials):
        raise ValueError(f"Duplicate trial identifiers in {source}.")
    return records


def diagnose_dispersion_runs(
    baseline_run_directories: list[str | Path],
    dispersion_run_directories: list[str | Path],
    output_directory: str | Path,
    *,
    top_structure_count: int = 20,
) -> dict[str, Any]:
    """Compare paired no-D3 and D3 traces and write diagnostic artifacts."""
    if not baseline_run_directories or not dispersion_run_directories:
        raise ValueError("Both baseline and dispersion run directories are required.")
    baseline_runs = {
        run["seed"]: run for run in map(_load_run, map(Path, baseline_run_directories))
    }
    dispersion_runs = {
        run["seed"]: run
        for run in map(_load_run, map(Path, dispersion_run_directories))
    }
    common_seeds = sorted(set(baseline_runs) & set(dispersion_runs))
    if not common_seeds:
        raise ValueError("No common seeds exist between no-D3 and D3 runs.")

    rows: list[dict[str, Any]] = []
    records_for_export: dict[tuple[int, int], WidomStructureRecord] = {}
    seed_summaries = []
    metadata = None
    for seed in common_seeds:
        baseline = baseline_runs[seed]
        dispersion = dispersion_runs[seed]
        _validate_run_pair(baseline, dispersion)
        metadata = metadata or {
            key: baseline[key]
            for key in ("material", "adsorbate", "temperature_K")
        }
        paired_rows, paired_records = _pair_trace_records(baseline, dispersion)
        rows.extend(paired_rows)
        records_for_export.update(
            {(seed, trial): record for trial, record in paired_records.items()}
        )
        seed_summaries.append(
            _summarize_rows(
                paired_rows,
                attempts=baseline["attempts"],
                label=f"seed_{seed}",
            )
        )

    aggregate = _summarize_rows(
        rows,
        attempts=sum(baseline_runs[seed]["attempts"] for seed in common_seeds),
        label="all_paired_seeds",
    )
    report = {
        "schema_version": 1,
        **(metadata or {}),
        "paired_seeds": common_seeds,
        "valid_paired_configuration_count": len(rows),
        "seed_summaries": seed_summaries,
        "aggregate": aggregate,
        "interpretation": _interpret(aggregate),
        "definitions": {
            "d3_correction": (
                "E_int(MACE-MP+D3) - E_int(MACE-MP) for an identical "
                "host-guest configuration."
            ),
            "effective_sample_size": (
                "(sum w)^2 / sum(w^2), with overlap attempts contributing "
                "zero weight."
            ),
            "top_weight_fraction": (
                "Fraction of total valid-insertion Boltzmann weight carried by "
                "the most influential one or five percent of configurations."
            ),
        },
    }

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "widom_d3_paired_configurations.csv"
    json_path = output / "widom_d3_energy_diagnostics.json"
    plot_path = output / "widom_d3_energy_diagnostics.png"
    structures_path = output / "widom_d3_top_weight_configurations.extxyz"
    _write_rows(rows, csv_path)
    _write_plot(report, rows, plot_path)
    _write_top_structures(
        rows,
        records_for_export,
        structures_path,
        count=top_structure_count,
    )
    report["outputs"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "top_weight_structures": str(structures_path),
    }
    save_benchmark_data(json_path, report)
    return report


def _read_values(handle: Any, value_format: str, source: Path) -> tuple[Any, ...]:
    size = struct.calcsize(value_format)
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError(f"Incomplete Widom structure payload in {source}.")
    return struct.unpack(value_format, payload)


def _load_run(directory: Path) -> dict[str, Any]:
    result_path = directory / "results" / "mlip_mc_benchmark.json"
    restart_path = (
        directory
        / "engine"
        / "mlip_mc"
        / "widom"
        / "restart"
        / "restart_widom.json"
    )
    trace_path = directory / "engine" / "mlip_mc" / "widom" / "log_widom.bin"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing completed result: {result_path}")
    if not restart_path.is_file():
        raise FileNotFoundError(f"Missing Widom restart metadata: {restart_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    restart = json.loads(restart_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("method") != "widom":
        raise ValueError(f"Run is not a completed Widom run: {directory}")
    return {
        "directory": directory,
        "seed": int(result["seed"]),
        "attempts": int(result["attempts"]),
        "material": str(result["material"]),
        "adsorbate": str(result["adsorbate"]),
        "temperature_K": float(result["temperature_K"]),
        "dispersion": bool(result.get("model", {}).get("dispersion", False)),
        "framework_atom_count": int(restart["n_frame"]),
        "adsorbate_atom_count": int(restart["n_ads"]),
        "trace_path": trace_path,
    }


def _validate_run_pair(
    baseline: dict[str, Any], dispersion: dict[str, Any]
) -> None:
    for field in (
        "seed",
        "attempts",
        "material",
        "adsorbate",
        "temperature_K",
        "framework_atom_count",
        "adsorbate_atom_count",
    ):
        if baseline[field] != dispersion[field]:
            raise ValueError(
                f"Paired runs differ in {field}: "
                f"{baseline[field]!r} != {dispersion[field]!r}."
            )
    if baseline["dispersion"]:
        raise ValueError("Baseline run must have dispersion=false.")
    if not dispersion["dispersion"]:
        raise ValueError("Dispersion run must have dispersion=true.")


def _pair_trace_records(
    baseline: dict[str, Any], dispersion: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[int, WidomStructureRecord]]:
    baseline_records = {
        record.trial: record
        for record in read_widom_structure_trace(baseline["trace_path"])
    }
    dispersion_records = {
        record.trial: record
        for record in read_widom_structure_trace(dispersion["trace_path"])
    }
    if set(baseline_records) != set(dispersion_records):
        raise ValueError(
            f"Valid trial identifiers differ for seed {baseline['seed']}; "
            "the traces cannot be compared configuration by configuration."
        )
    rows = []
    for trial in sorted(baseline_records):
        no_d3 = baseline_records[trial]
        with_d3 = dispersion_records[trial]
        _validate_same_configuration(no_d3, with_d3, seed=baseline["seed"])
        distance = _minimum_host_guest_distance(
            with_d3,
            framework_atom_count=baseline["framework_atom_count"],
        )
        temperature = baseline["temperature_K"]
        no_d3_weight = exp(
            -no_d3.interaction_energy_ev / (BOLTZMANN_EV_PER_K * temperature)
        )
        d3_weight = exp(
            -with_d3.interaction_energy_ev / (BOLTZMANN_EV_PER_K * temperature)
        )
        rows.append(
            {
                "seed": baseline["seed"],
                "trial": trial,
                "minimum_host_guest_distance_A": distance["distance_A"],
                "closest_framework_element": distance["framework_element"],
                "closest_adsorbate_element": distance["adsorbate_element"],
                "interaction_energy_no_d3_ev": no_d3.interaction_energy_ev,
                "interaction_energy_with_d3_ev": with_d3.interaction_energy_ev,
                "d3_correction_ev": (
                    with_d3.interaction_energy_ev - no_d3.interaction_energy_ev
                ),
                "d3_correction_kj_mol": (
                    with_d3.interaction_energy_ev - no_d3.interaction_energy_ev
                )
                * EV_TO_KJ_PER_MOL,
                "boltzmann_weight_no_d3": no_d3_weight,
                "boltzmann_weight_with_d3": d3_weight,
            }
        )
    return rows, dispersion_records


def _validate_same_configuration(
    baseline: WidomStructureRecord,
    dispersion: WidomStructureRecord,
    *,
    seed: int,
) -> None:
    if baseline.trial != dispersion.trial:
        raise ValueError("Paired trace records have different trial identifiers.")
    if not np.array_equal(baseline.atomic_numbers, dispersion.atomic_numbers):
        raise ValueError(
            f"Atomic numbers differ for seed {seed}, trial {baseline.trial}."
        )
    if not np.allclose(baseline.cell_A, dispersion.cell_A, atol=1.0e-10):
        raise ValueError(f"Cells differ for seed {seed}, trial {baseline.trial}.")
    if not np.allclose(
        baseline.positions_A, dispersion.positions_A, atol=1.0e-10
    ):
        raise ValueError(
            f"Positions differ for seed {seed}, trial {baseline.trial}."
        )


def _minimum_host_guest_distance(
    record: WidomStructureRecord,
    *,
    framework_atom_count: int,
) -> dict[str, Any]:
    try:
        from ase.data import chemical_symbols
        from ase.geometry import find_mic
    except ImportError as exc:
        raise ImportError("ASE is required for periodic distance diagnostics.") from exc
    host = record.positions_A[:framework_atom_count]
    guest = record.positions_A[framework_atom_count:]
    if not len(host) or not len(guest):
        raise ValueError("Widom record does not contain both host and guest atoms.")
    vectors = (guest[:, None, :] - host[None, :, :]).reshape(-1, 3)
    _mic_vectors, distances = find_mic(vectors, record.cell_A, pbc=True)
    closest = int(np.argmin(distances))
    host_count = len(host)
    guest_index = closest // host_count + framework_atom_count
    host_index = closest % host_count
    return {
        "distance_A": float(distances[closest]),
        "framework_element": chemical_symbols[
            int(record.atomic_numbers[host_index])
        ],
        "adsorbate_element": chemical_symbols[
            int(record.atomic_numbers[guest_index])
        ],
    }


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    attempts: int,
    label: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No paired valid configurations were found.")
    corrections = np.asarray([row["d3_correction_ev"] for row in rows])
    distances = np.asarray(
        [row["minimum_host_guest_distance_A"] for row in rows]
    )
    no_d3_weights = np.asarray([row["boltzmann_weight_no_d3"] for row in rows])
    d3_weights = np.asarray([row["boltzmann_weight_with_d3"] for row in rows])
    return {
        "label": label,
        "attempt_count": attempts,
        "valid_paired_configuration_count": len(rows),
        "valid_fraction": len(rows) / attempts,
        "d3_correction_ev": {
            "mean": float(np.mean(corrections)),
            "median": float(np.median(corrections)),
            "standard_deviation": float(np.std(corrections, ddof=1)),
            "quantile_05": float(np.quantile(corrections, 0.05)),
            "quantile_95": float(np.quantile(corrections, 0.95)),
        },
        "d3_correction_kj_mol_mean": float(
            np.mean(corrections) * EV_TO_KJ_PER_MOL
        ),
        "minimum_host_guest_distance_A": {
            "minimum": float(np.min(distances)),
            "median": float(np.median(distances)),
            "maximum": float(np.max(distances)),
        },
        "distance_correction_pearson_r": float(
            np.corrcoef(distances, corrections)[0, 1]
        ),
        "without_d3": _weight_diagnostics(no_d3_weights, attempts=attempts),
        "with_d3": _weight_diagnostics(d3_weights, attempts=attempts),
        "mean_boltzmann_weight_ratio_d3_to_no_d3": float(
            np.sum(d3_weights) / np.sum(no_d3_weights)
        ),
    }


def _weight_diagnostics(weights: np.ndarray, *, attempts: int) -> dict[str, Any]:
    total = float(np.sum(weights))
    squared_total = float(np.dot(weights, weights))
    ordered = np.sort(weights)[::-1]
    one_percent = max(1, round(len(ordered) * 0.01))
    five_percent = max(1, round(len(ordered) * 0.05))
    return {
        "average_over_all_attempts": total / attempts,
        "effective_sample_size": total**2 / squared_total,
        "effective_sample_fraction_of_attempts": (
            total**2 / squared_total / attempts
        ),
        "top_1_percent_valid_weight_fraction": float(
            np.sum(ordered[:one_percent]) / total
        ),
        "top_5_percent_valid_weight_fraction": float(
            np.sum(ordered[:five_percent]) / total
        ),
    }


def _interpret(aggregate: dict[str, Any]) -> list[str]:
    correction = aggregate["d3_correction_ev"]["mean"]
    ratio = aggregate["mean_boltzmann_weight_ratio_d3_to_no_d3"]
    top_fraction = aggregate["with_d3"]["top_1_percent_valid_weight_fraction"]
    effective = aggregate["with_d3"]["effective_sample_size"]
    return [
        (
            f"D3 changes the same valid host-guest configurations by "
            f"{correction:.4f} eV on average ({correction * EV_TO_KJ_PER_MOL:.2f} "
            "kJ/mol)."
        ),
        (
            f"The resulting mean Boltzmann weight is {ratio:.1f} times the "
            "no-D3 value, demonstrating exponential amplification rather than "
            "a change in sampled configurations."
        ),
        (
            f"The highest-weight 1% of valid D3 configurations carry "
            f"{top_fraction:.1%} of the D3 weight; the effective sample size is "
            f"{effective:.1f}."
        ),
        (
            "A longer Widom run can reduce rare-event uncertainty, but it cannot "
            "by itself remove a systematic D3 interaction-energy shift."
        ),
    ]


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(
    report: dict[str, Any], rows: list[dict[str, Any]], path: Path
) -> None:
    import matplotlib.pyplot as plt

    no_d3 = np.asarray([row["interaction_energy_no_d3_ev"] for row in rows])
    with_d3 = np.asarray([row["interaction_energy_with_d3_ev"] for row in rows])
    correction = np.asarray([row["d3_correction_ev"] for row in rows])
    distance = np.asarray(
        [row["minimum_host_guest_distance_A"] for row in rows]
    )
    weight_no_d3 = np.asarray([row["boltzmann_weight_no_d3"] for row in rows])
    weight_d3 = np.asarray([row["boltzmann_weight_with_d3"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.3))
    scatter = axes[0, 0].scatter(
        no_d3,
        with_d3,
        c=distance,
        cmap="viridis",
        s=12,
        alpha=0.55,
        rasterized=True,
    )
    lower = float(min(np.min(no_d3), np.min(with_d3)))
    upper = float(max(np.max(no_d3), np.max(with_d3)))
    axes[0, 0].plot([lower, upper], [lower, upper], "--", color="#555555")
    axes[0, 0].set_xlabel("Interaction energy without D3 / eV")
    axes[0, 0].set_ylabel("Interaction energy with D3 / eV")
    axes[0, 0].set_title("Paired interaction energies")
    fig.colorbar(scatter, ax=axes[0, 0], label="Minimum host-guest distance / Å")

    axes[0, 1].scatter(
        distance,
        correction * EV_TO_KJ_PER_MOL,
        c=np.log10(weight_d3),
        cmap="magma",
        s=12,
        alpha=0.55,
        rasterized=True,
    )
    axes[0, 1].axhline(0.0, color="#555555", linestyle="--")
    axes[0, 1].set_xlabel("Minimum host-guest distance / Å")
    axes[0, 1].set_ylabel("D3 interaction correction / kJ mol$^{-1}$")
    axes[0, 1].set_title("Distance dependence of D3")

    bins = np.linspace(
        min(np.min(no_d3), np.min(with_d3)),
        max(np.max(no_d3), np.max(with_d3)),
        55,
    )
    axes[1, 0].hist(
        no_d3,
        bins=bins,
        density=True,
        alpha=0.58,
        color="#2878B5",
        label="MACE-MP without D3",
    )
    axes[1, 0].hist(
        with_d3,
        bins=bins,
        density=True,
        alpha=0.58,
        color="#E07A2D",
        label="MACE-MP + D3",
    )
    axes[1, 0].set_xlabel("Interaction energy / eV")
    axes[1, 0].set_ylabel("Probability density")
    axes[1, 0].set_title("Valid-insertion energy distribution")
    axes[1, 0].legend(frameon=False)

    for weights, label, color in (
        (weight_no_d3, "MACE-MP without D3", "#2878B5"),
        (weight_d3, "MACE-MP + D3", "#E07A2D"),
    ):
        ordered = np.sort(weights)[::-1]
        fraction = np.arange(1, len(ordered) + 1) / len(ordered)
        cumulative = np.cumsum(ordered) / np.sum(ordered)
        axes[1, 1].plot(fraction, cumulative, color=color, label=label)
    axes[1, 1].plot([0, 1], [0, 1], "--", color="#777777", label="Uniform weight")
    axes[1, 1].set_xlabel("Fraction of valid configurations, ranked by weight")
    axes[1, 1].set_ylabel("Cumulative Boltzmann-weight fraction")
    axes[1, 1].set_title("Rare-event dominance")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.22)
    fig.suptitle(
        f"{report['material']}/{report['adsorbate']} at "
        f"{report['temperature_K']:.2f} K: D3 energy diagnostics"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_top_structures(
    rows: list[dict[str, Any]],
    records: dict[tuple[int, int], WidomStructureRecord],
    path: Path,
    *,
    count: int,
) -> None:
    try:
        from ase import Atoms
        from ase.io import write
    except ImportError as exc:
        raise ImportError("ASE is required to export diagnostic structures.") from exc
    selected = sorted(
        rows,
        key=lambda row: row["boltzmann_weight_with_d3"],
        reverse=True,
    )[: max(1, count)]
    frames = []
    for rank, row in enumerate(selected, start=1):
        record = records[(int(row["seed"]), int(row["trial"]))]
        atoms = Atoms(
            numbers=record.atomic_numbers,
            positions=record.positions_A,
            cell=record.cell_A,
            pbc=True,
        )
        atoms.info.update(
            {
                "weight_rank": rank,
                "seed": int(row["seed"]),
                "trial": int(row["trial"]),
                "minimum_host_guest_distance_A": row[
                    "minimum_host_guest_distance_A"
                ],
                "interaction_energy_no_d3_ev": row[
                    "interaction_energy_no_d3_ev"
                ],
                "interaction_energy_with_d3_ev": row[
                    "interaction_energy_with_d3_ev"
                ],
                "d3_correction_ev": row["d3_correction_ev"],
            }
        )
        frames.append(atoms)
    write(path, frames, format="extxyz")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose D3 effects from paired MLIP-MC Widom traces."
    )
    parser.add_argument("--baseline-run", action="append", required=True)
    parser.add_argument("--dispersion-run", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-structures", type=int, default=20)
    args = parser.parse_args(argv)
    report = diagnose_dispersion_runs(
        args.baseline_run,
        args.dispersion_run,
        args.output_dir,
        top_structure_count=args.top_structures,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
