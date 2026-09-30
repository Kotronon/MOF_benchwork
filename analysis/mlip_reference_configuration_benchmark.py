"""Evaluate selected Widom configurations with a fine-tuned NequIP model."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from analysis.mlip_widom import BOLTZMANN_EV_PER_K, EV_TO_KJ_PER_MOL
from modules.module_c_mlips.interaction import (
    build_framework_configuration,
    evaluate_interaction,
)
from modules.module_c_mlips.models import InteractionConfiguration
from modules.module_c_mlips.potential_backends.nequip import NequipBackend
from pipeline.config import save_benchmark_data


CANDIDATES = {
    "mace_mp_no_d3": "interaction_energy_no_d3_ev",
    "mace_mp_d3": "interaction_energy_with_d3_ev",
}


def benchmark_reference_configurations(
    structures_path: str | Path,
    model_path: str | Path,
    output_directory: str | Path,
    *,
    framework_atom_count: int,
    material: str = "ZIF-8",
    adsorbate: str = "CO2",
    temperature_K: float = 273.0,
    device: str = "cpu",
    loader: str = "legacy",
    species_to_type_name: dict[str, str] | bool | None = None,
    supercell: tuple[int, int, int] | None = None,
    supercell_count: int = 0,
    calculator: Any | None = None,
) -> dict[str, Any]:
    """Compare stored MACE energies with a fine-tuned reference potential."""
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive.")
    if framework_atom_count <= 0:
        raise ValueError("framework_atom_count must be positive.")
    model = Path(model_path).expanduser()
    if calculator is None and not model.is_file():
        raise FileNotFoundError(f"NequIP model file does not exist: {model}")
    configurations = _load_configurations(
        Path(structures_path),
        framework_atom_count=framework_atom_count,
        material=material,
        adsorbate=adsorbate,
    )
    if species_to_type_name is None:
        symbols = sorted(set(configurations[0].atoms.get_chemical_symbols()))
        species_to_type_name = {symbol: symbol for symbol in symbols}
    backend = NequipBackend(
        model=model,
        backend_name="goeminne_zif8_n1000",
        device=device,
        loader=loader,
        species_to_type_name=species_to_type_name,
        energy_mode="interaction_direct",
        calculator=calculator,
    )

    framework_result = backend.evaluate(
        build_framework_configuration(configurations[0])
    )
    rows = []
    for configuration in configurations:
        reference = evaluate_interaction(
            configuration,
            backend,
            framework_result=framework_result,
        )
        metadata = configuration.metadata
        row = {
            "configuration_id": configuration.configuration_id,
            "weight_rank": int(metadata["weight_rank"]),
            "seed": int(metadata["seed"]),
            "trial": int(metadata["trial"]),
            "minimum_host_guest_distance_A": float(
                metadata["minimum_host_guest_distance_A"]
            ),
            "reference_interaction_energy_ev": reference.interaction_energy_ev,
            "interaction_energy_no_d3_ev": float(
                metadata["interaction_energy_no_d3_ev"]
            ),
            "interaction_energy_with_d3_ev": float(
                metadata["interaction_energy_with_d3_ev"]
            ),
            "reference_runtime_seconds": reference.runtime_seconds,
        }
        for candidate, field in CANDIDATES.items():
            error = row[field] - reference.interaction_energy_ev
            row[f"{candidate}_error_ev"] = error
            row[f"{candidate}_error_kj_mol"] = error * EV_TO_KJ_PER_MOL
        rows.append(row)

    metrics = {
        candidate: _error_metrics(
            rows,
            candidate=candidate,
            candidate_field=field,
            temperature_K=temperature_K,
        )
        for candidate, field in CANDIDATES.items()
    }
    supercell_report = None
    if supercell is not None and supercell_count > 0:
        supercell_report = _evaluate_supercell_invariance(
            configurations[:supercell_count],
            rows[:supercell_count],
            backend,
            repetitions=supercell,
        )

    report = {
        "schema_version": 1,
        "status": "completed",
        "material": material,
        "adsorbate": adsorbate,
        "temperature_K": temperature_K,
        "selection": "highest MACE-MP+D3 Boltzmann-weight configurations",
        "configuration_count": len(rows),
        "model": {
            "name": backend.name,
            "path": str(model),
            "sha256": _sha256(model) if model.is_file() else None,
            "device": device,
            "loader": loader,
            "species_to_type_name": species_to_type_name,
            "dispersion": False,
        },
        "metrics": metrics,
        "preferred_candidate": min(
            metrics,
            key=lambda name: metrics[name]["boltzmann_weighted_mae_ev"],
        ),
        "supercell_invariance": supercell_report,
        "limitations": (
            "The configurations were intentionally selected from the high-weight "
            "D3 tail. This is a targeted failure-mode diagnostic, not an unbiased "
            "test-set error estimate."
        ),
    }

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "reference_configuration_energies.csv"
    json_path = output / "reference_configuration_benchmark.json"
    plot_path = output / "reference_configuration_benchmark.png"
    _write_csv(rows, csv_path)
    _write_plot(report, rows, plot_path)
    report["outputs"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    save_benchmark_data(json_path, report)
    return report


def _load_configurations(
    path: Path,
    *,
    framework_atom_count: int,
    material: str,
    adsorbate: str,
) -> list[InteractionConfiguration]:
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError("ASE is required for reference configuration loading.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing diagnostic structures: {path}")
    frames = read(path, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    configurations = []
    required_metadata = {
        "weight_rank",
        "seed",
        "trial",
        "minimum_host_guest_distance_A",
        "interaction_energy_no_d3_ev",
        "interaction_energy_with_d3_ev",
    }
    for index, atoms in enumerate(frames):
        if not framework_atom_count < len(atoms):
            raise ValueError(
                f"Frame {index} does not contain atoms beyond the framework."
            )
        missing = required_metadata - set(atoms.info)
        if missing:
            raise ValueError(
                f"Frame {index} lacks metadata: {', '.join(sorted(missing))}."
            )
        metadata = {key: atoms.info[key] for key in required_metadata}
        configurations.append(
            InteractionConfiguration(
                configuration_id=(
                    f"seed_{int(metadata['seed'])}_trial_{int(metadata['trial'])}"
                ),
                material=material,
                adsorbate=adsorbate,
                atoms=atoms,
                framework_indices=list(range(framework_atom_count)),
                adsorbate_indices=list(range(framework_atom_count, len(atoms))),
                region="d3_high_boltzmann_weight",
                source=str(path),
                metadata=metadata,
            )
        )
    if not configurations:
        raise ValueError(f"No configurations found in {path}.")
    return configurations


def _error_metrics(
    rows: list[dict[str, Any]],
    *,
    candidate: str,
    candidate_field: str,
    temperature_K: float,
) -> dict[str, Any]:
    reference = np.asarray(
        [row["reference_interaction_energy_ev"] for row in rows]
    )
    predicted = np.asarray([row[candidate_field] for row in rows])
    errors = predicted - reference
    log_weights = -reference / (BOLTZMANN_EV_PER_K * temperature_K)
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= np.sum(weights)
    correlation = None
    if (
        len(rows) >= 2
        and float(np.std(reference)) > 0.0
        and float(np.std(predicted)) > 0.0
    ):
        correlation = float(np.corrcoef(reference, predicted)[0, 1])
    return {
        "candidate": candidate,
        "mean_signed_error_ev": float(np.mean(errors)),
        "mae_ev": float(np.mean(np.abs(errors))),
        "rmse_ev": float(sqrt(np.mean(errors**2))),
        "mae_kj_mol": float(np.mean(np.abs(errors)) * EV_TO_KJ_PER_MOL),
        "rmse_kj_mol": float(sqrt(np.mean(errors**2)) * EV_TO_KJ_PER_MOL),
        "pearson_r": correlation,
        "boltzmann_weighted_mae_ev": float(np.dot(weights, np.abs(errors))),
        "boltzmann_weighted_mae_kj_mol": float(
            np.dot(weights, np.abs(errors)) * EV_TO_KJ_PER_MOL
        ),
    }


def _evaluate_supercell_invariance(
    configurations: list[InteractionConfiguration],
    original_rows: list[dict[str, Any]],
    backend: NequipBackend,
    *,
    repetitions: tuple[int, int, int],
) -> dict[str, Any]:
    if any(value <= 0 for value in repetitions):
        raise ValueError("Supercell repetitions must be positive integers.")
    supercell_configurations = [
        _build_supercell_configuration(configuration, repetitions)
        for configuration in configurations
    ]
    framework_result = backend.evaluate(
        build_framework_configuration(supercell_configurations[0])
    )
    rows = []
    for configuration, original in zip(supercell_configurations, original_rows):
        result = evaluate_interaction(
            configuration,
            backend,
            framework_result=framework_result,
        )
        difference = (
            result.interaction_energy_ev
            - float(original["reference_interaction_energy_ev"])
        )
        rows.append(
            {
                "configuration_id": configuration.configuration_id,
                "original_energy_ev": original["reference_interaction_energy_ev"],
                "supercell_energy_ev": result.interaction_energy_ev,
                "difference_ev": difference,
                "absolute_difference_kj_mol": abs(difference)
                * EV_TO_KJ_PER_MOL,
            }
        )
    return {
        "repetitions": list(repetitions),
        "configuration_count": len(rows),
        "mean_absolute_difference_ev": fmean(
            row["absolute_difference_kj_mol"] / EV_TO_KJ_PER_MOL
            for row in rows
        ),
        "mean_absolute_difference_kj_mol": fmean(
            row["absolute_difference_kj_mol"] for row in rows
        ),
        "maximum_absolute_difference_kj_mol": max(
            row["absolute_difference_kj_mol"] for row in rows
        ),
        "rows": rows,
    }


def _build_supercell_configuration(
    configuration: InteractionConfiguration,
    repetitions: tuple[int, int, int],
) -> InteractionConfiguration:
    framework = configuration.atoms[configuration.framework_indices].repeat(
        repetitions
    )
    adsorbate = configuration.atoms[configuration.adsorbate_indices].copy()
    adsorbate.set_cell(framework.cell)
    adsorbate.set_pbc(True)
    atoms = framework + adsorbate
    framework_count = len(framework)
    return InteractionConfiguration(
        configuration_id=configuration.configuration_id + "_supercell",
        material=configuration.material,
        adsorbate=configuration.adsorbate,
        atoms=atoms,
        framework_indices=list(range(framework_count)),
        adsorbate_indices=list(range(framework_count, len(atoms))),
        region=configuration.region,
        source=configuration.source,
        metadata=configuration.metadata,
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(
    report: dict[str, Any], rows: list[dict[str, Any]], path: Path
) -> None:
    import matplotlib.pyplot as plt

    reference = np.asarray(
        [row["reference_interaction_energy_ev"] for row in rows]
    )
    no_d3 = np.asarray([row["interaction_energy_no_d3_ev"] for row in rows])
    d3 = np.asarray([row["interaction_energy_with_d3_ev"] for row in rows])
    ranks = np.asarray([row["weight_rank"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    lower = float(min(np.min(reference), np.min(no_d3), np.min(d3)))
    upper = float(max(np.max(reference), np.max(no_d3), np.max(d3)))
    axes[0].scatter(reference, no_d3, color="#2878B5", label="MACE-MP without D3")
    axes[0].scatter(reference, d3, color="#E07A2D", label="MACE-MP + D3")
    axes[0].plot([lower, upper], [lower, upper], "--", color="#666666")
    axes[0].set_xlabel("Fine-tuned reference interaction energy / eV")
    axes[0].set_ylabel("Candidate interaction energy / eV")
    axes[0].set_title("Energy parity on the problematic D3 tail")
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, color="#666666", linestyle="--")
    axes[1].plot(ranks, (no_d3 - reference) * EV_TO_KJ_PER_MOL, "o-", color="#2878B5", label="Without D3")
    axes[1].plot(ranks, (d3 - reference) * EV_TO_KJ_PER_MOL, "o-", color="#E07A2D", label="With D3")
    axes[1].set_xlabel("D3 Boltzmann-weight rank")
    axes[1].set_ylabel("Candidate minus reference / kJ mol$^{-1}$")
    axes[1].set_title("Error of the highest-impact configurations")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle(
        f"{report['material']}/{report['adsorbate']}: fine-tuned NequIP diagnostic"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_species_map(value: str | None) -> dict[str, str] | bool | None:
    if value is None:
        return None
    if value.strip().casefold() == "identity":
        return True
    path = Path(value)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in data.items()
    ):
        raise ValueError("Species map must be a JSON object of string pairs.")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark high-weight Widom configurations against NequIP."
    )
    parser.add_argument("--structures", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--framework-atoms", type=int, default=276)
    parser.add_argument("--temperature-k", type=float, default=273.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--loader", choices=["auto", "legacy", "compiled"], default="legacy")
    parser.add_argument("--species-map", help="JSON file or 'identity'.")
    parser.add_argument("--supercell", nargs=3, type=int, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--supercell-count", type=int, default=0)
    args = parser.parse_args(argv)
    report = benchmark_reference_configurations(
        args.structures,
        args.model,
        args.output_dir,
        framework_atom_count=args.framework_atoms,
        temperature_K=args.temperature_k,
        device=args.device,
        loader=args.loader,
        species_to_type_name=_parse_species_map(args.species_map),
        supercell=None if args.supercell is None else tuple(args.supercell),
        supercell_count=args.supercell_count,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
