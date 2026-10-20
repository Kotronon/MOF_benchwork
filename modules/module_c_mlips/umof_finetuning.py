"""Prepare and run CPU fine-tuning of uMOF-MH on released adsorption data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np

from pipeline.config import save_benchmark_data


DEFAULT_DATA_DIRECTORY = Path("external/umof_data")
DEFAULT_CHECKPOINT = Path("external/umof_models/uMOF-MH.model")
DEFAULT_OUTPUT_DIRECTORY = Path("outputs/module_C_umof_finetuning")
DEFAULT_VALIDATION_MOFS = {
    "HKUST-1-adsorbates",
    "MIL-53-Al-adsorbates",
}
DEFAULT_TEST_MOFS = {
    "MOF-303-adsorbates",
    "MgMOF-74-adsorbates",
}
TARGET_GASES = {"CO2", "N2", "CH4"}
KNOWN_GAS_FORMULAS = {
    "CO2": {"C": 1, "O": 2},
    "N2": {"N": 2},
    "CH4": {"C": 1, "H": 4},
    "H2": {"H": 2},
    "H2O": {"H": 2, "O": 1},
    "NH3": {"H": 3, "N": 1},
}


def prepare_adsorption_datasets(
    data_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    train_fraction: float = 1.0,
    seed: int = 12345,
    validation_mofs: set[str] | None = None,
    test_mofs: set[str] | None = None,
) -> dict[str, Any]:
    """Build gas-filtered, MOF-disjoint uMOF training/evaluation datasets.

    The official per-MOF train, validation, and test files are assigned to
    distinct framework groups. Test frames are written separately and never
    passed to the training command.
    """
    if not math.isfinite(train_fraction) or not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in the interval (0, 1].")
    try:
        from ase.io import read, write
    except ImportError as exc:
        raise ImportError("ASE is required to prepare uMOF datasets.") from exc

    data_root = Path(data_directory).expanduser()
    output_root = Path(output_directory).expanduser()
    train_paths = sorted(data_root.glob("*-adsorbates-train.extxyz"))
    valid_paths = sorted(data_root.glob("*-adsorbates-valid.extxyz"))
    test_paths = sorted(data_root.glob("*-adsorbates-test.extxyz"))
    if not train_paths or not valid_paths or not test_paths:
        raise FileNotFoundError(
            f"uMOF adsorption train/valid/test files are incomplete in {data_root}."
        )
    train_by_mof = {
        path.name.removesuffix("-train.extxyz"): path for path in train_paths
    }
    valid_by_mof = {
        path.name.removesuffix("-valid.extxyz"): path for path in valid_paths
    }
    test_by_mof = {
        path.name.removesuffix("-test.extxyz"): path for path in test_paths
    }
    if set(train_by_mof) != set(valid_by_mof):
        raise ValueError(
            "uMOF adsorption train/valid MOF groups differ: "
            f"train-only-valid={sorted(set(train_by_mof) - set(valid_by_mof))}, "
        )
    all_mofs = set(train_by_mof)
    validation_mofs = set(validation_mofs or DEFAULT_VALIDATION_MOFS)
    test_mofs = set(test_mofs or DEFAULT_TEST_MOFS)
    if not validation_mofs or not test_mofs:
        raise ValueError("Validation and test MOF groups must be non-empty.")
    if validation_mofs & test_mofs:
        raise ValueError("Validation and test MOF groups must be disjoint.")
    unknown_mofs = (validation_mofs | test_mofs) - all_mofs
    if unknown_mofs:
        raise ValueError(f"Unknown uMOF adsorption MOFs: {sorted(unknown_mofs)}")
    missing_test_files = test_mofs - set(test_by_mof)
    if missing_test_files:
        raise ValueError(
            f"Held-out test MOFs have no official test files: {sorted(missing_test_files)}"
        )
    training_mofs = all_mofs - validation_mofs - test_mofs
    if not training_mofs:
        raise ValueError("At least one uMOF must remain in the training group.")

    output_root.mkdir(parents=True, exist_ok=True)
    train_output = output_root / "train.extxyz"
    valid_output = output_root / "valid.extxyz"
    test_output = output_root / "test.extxyz"
    rng = random.Random(seed)
    source_counts: dict[str, dict[str, Any]] = {}
    total_train = 0
    total_valid = 0
    total_test = 0

    with train_output.open("w", encoding="utf-8") as train_handle:
        for mof in sorted(training_mofs):
            path = train_by_mof[mof]
            frames = read(path, index=":", format="extxyz")
            if not frames:
                raise ValueError(f"uMOF file is empty: {path}")
            prepared = [
                _prepare_frame(frame, path, "train")
                for frame in frames
            ]
            prepared = [frame for frame in prepared if frame is not None]
            if not prepared:
                raise ValueError(f"No CO2/N2/CH4 frames found in {path}.")
            selected_count = max(1, math.ceil(len(prepared) * train_fraction))
            indices = sorted(rng.sample(range(len(prepared)), selected_count))
            selected = [prepared[index] for index in indices]
            write(train_handle, selected, format="extxyz")
            source_counts[path.name] = {
                "available": len(prepared),
                "selected": len(selected),
            }
            total_train += len(selected)

    with valid_output.open("w", encoding="utf-8") as valid_handle:
        for mof in sorted(validation_mofs):
            path = valid_by_mof[mof]
            frames = read(path, index=":", format="extxyz")
            if not frames:
                raise ValueError(f"uMOF file is empty: {path}")
            prepared = [
                _prepare_frame(frame, path, "valid")
                for frame in frames
            ]
            prepared = [frame for frame in prepared if frame is not None]
            if not prepared:
                raise ValueError(f"No CO2/N2/CH4 frames found in {path}.")
            write(valid_handle, prepared, format="extxyz")
            source_counts[path.name] = {
                "available": len(prepared),
                "selected": len(prepared),
            }
            total_valid += len(prepared)

    with test_output.open("w", encoding="utf-8") as test_handle:
        for mof in sorted(test_mofs):
            path = test_by_mof[mof]
            frames = read(path, index=":", format="extxyz")
            prepared = [
                _prepare_frame(frame, path, "test")
                for frame in frames
            ]
            prepared = [frame for frame in prepared if frame is not None]
            if not prepared:
                raise ValueError(f"No CO2/N2/CH4 frames found in {path}.")
            write(test_handle, prepared, format="extxyz")
            source_counts[path.name] = {
                "available": len(prepared),
                "selected": len(prepared),
            }
            total_test += len(prepared)

    manifest = {
        "status": "prepared",
        "dataset": "uMOF released adsorbate-loaded frameworks",
        "energy_reference": "r2SCAN-D4",
        "target_gases": sorted(TARGET_GASES),
        "training_data_fraction": train_fraction,
        "seed": seed,
        "train_file": str(train_output),
        "valid_file": str(valid_output),
        "test_file": str(test_output),
        "train_configuration_count": total_train,
        "validation_configuration_count": total_valid,
        "test_configuration_count": total_test,
        "training_mofs": sorted(training_mofs),
        "validation_mofs": sorted(validation_mofs),
        "test_mofs": sorted(test_mofs),
        "test_files_used_for_training": False,
        "source_file_counts": source_counts,
        "label_mapping": {
            "dft_energy": "REF_energy",
            "dft_forces": "REF_forces",
        },
    }
    save_benchmark_data(output_root / "dataset_manifest.json", manifest)
    return manifest


def build_umof_mh_training_command(
    manifest: dict[str, Any],
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    max_epochs: int = 1000,
) -> list[str]:
    """Build a single-model CPU fine-tuning command from the uMOF-MH head."""
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"uMOF-MH checkpoint not found: {checkpoint_path}")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    output = Path(output_directory).expanduser()
    return [
        sys.executable,
        "-m",
        "mace.cli.run_train",
        "--name",
        "umof_mh_adsorption_finetuned",
        "--train_file",
        str(manifest["train_file"]),
        "--valid_file",
        str(manifest["valid_file"]),
        "--foundation_model",
        str(checkpoint_path),
        "--foundation_head",
        "r2scan+d4",
        "--foundation_model_elements",
        "True",
        "--multiheads_finetuning",
        "False",
        "--lr",
        "0.0001",
        "--energy_key",
        "REF_energy",
        "--forces_key",
        "REF_forces",
        "--E0s",
        "average",
        "--device",
        "cpu",
        "--default_dtype",
        "float64",
        "--batch_size",
        "1",
        "--valid_batch_size",
        "1",
        "--max_num_epochs",
        str(max_epochs),
        "--model_dir",
        str(output / "models"),
        "--checkpoints_dir",
        str(output / "checkpoints"),
        "--results_dir",
        str(output / "results"),
        "--log_dir",
        str(output / "logs"),
    ]


def evaluate_umof_mh_models(
    test_file: str | Path,
    finetuned_checkpoint: str | Path,
    *,
    baseline_checkpoint: str | Path = DEFAULT_CHECKPOINT,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compare zero-shot and fine-tuned uMOF-MH on the held-out test MOFs."""
    try:
        from ase.io import read
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise ImportError("ASE and MACE are required for uMOF evaluation.") from exc

    test_path = Path(test_file).expanduser()
    baseline_path = Path(baseline_checkpoint).expanduser()
    finetuned_path = Path(finetuned_checkpoint).expanduser()
    for path in (test_path, baseline_path, finetuned_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required uMOF evaluation file not found: {path}")
    frames = read(test_path, index=":", format="extxyz")
    baseline = MACECalculator(
        model_paths=str(baseline_path),
        device=device,
        default_dtype="float64",
        head="r2scan+d4",
    )
    finetuned = MACECalculator(
        model_paths=str(finetuned_path),
        device=device,
        default_dtype="float64",
        head="Default",
    )
    grouped: dict[str, dict[str, list[float]]] = {}
    gas_counts: dict[str, dict[str, int]] = {}
    for atoms in frames:
        gas_composition = _target_gas_composition(atoms)
        if not gas_composition:
            continue
        mof = str(atoms.info.get("umof_source_file", "unknown")).split(
            "-adsorbates-", maxsplit=1
        )[0]
        if mof not in grouped:
            grouped[mof] = {"baseline_energy": [], "baseline_forces": [], "finetuned_energy": [], "finetuned_forces": []}
            gas_counts[mof] = {gas: 0 for gas in sorted(TARGET_GASES)}
        for gas, count in gas_composition.items():
            gas_counts[mof][gas] += count
        reference_energy = float(atoms.info["dft_energy"])
        reference_forces = np.asarray(atoms.arrays["dft_forces"], dtype=float)
        for label, calculator in (("baseline", baseline), ("finetuned", finetuned)):
            current = atoms.copy()
            current.calc = calculator
            predicted_energy = float(current.get_potential_energy())
            predicted_forces = np.asarray(current.get_forces(), dtype=float)
            grouped[mof][f"{label}_energy"].append(
                (predicted_energy - reference_energy) / len(atoms)
            )
            grouped[mof][f"{label}_forces"].extend(
                (predicted_forces - reference_forces).reshape(-1).tolist()
            )

    results = {
        mof: {
            "configuration_count": len(metrics["baseline_energy"]),
            "gas_molecule_counts": gas_counts[mof],
            "baseline": _error_metrics(
                metrics["baseline_energy"], metrics["baseline_forces"]
            ),
            "finetuned": _error_metrics(
                metrics["finetuned_energy"], metrics["finetuned_forces"]
            ),
        }
        for mof, metrics in sorted(grouped.items())
    }
    if not results:
        raise ValueError(f"No CO2/N2/CH4 configurations found in {test_path}.")
    aggregate = {
        "baseline": _error_metrics(
            [value for metrics in grouped.values() for value in metrics["baseline_energy"]],
            [value for metrics in grouped.values() for value in metrics["baseline_forces"]],
        ),
        "finetuned": _error_metrics(
            [value for metrics in grouped.values() for value in metrics["finetuned_energy"]],
            [value for metrics in grouped.values() for value in metrics["finetuned_forces"]],
        ),
    }
    return {
        "status": "completed",
        "test_file": str(test_path),
        "baseline_checkpoint": str(baseline_path),
        "finetuned_checkpoint": str(finetuned_path),
        "test_configuration_count": sum(
            int(value["configuration_count"]) for value in results.values()
        ),
        "test_mof_count": len(results),
        "energy_metric": "MAE/RMSE of total-energy error divided by atom count (eV/atom)",
        "force_metric": "MAE/RMSE over Cartesian force-component errors (eV/A)",
        "aggregate": aggregate,
        "by_mof": results,
    }


def _target_gas_composition(atoms: Any) -> dict[str, int]:
    counts = _guest_molecule_counts(atoms)
    if not counts or set(counts) - TARGET_GASES:
        return {}
    return dict(counts)


def _error_metrics(energy_errors: Sequence[float], force_errors: Sequence[float]) -> dict[str, float]:
    energy = np.asarray(energy_errors, dtype=float)
    forces = np.asarray(force_errors, dtype=float)
    return {
        "energy_mae_ev_per_atom": float(np.mean(np.abs(energy))),
        "energy_rmse_ev_per_atom": float(np.sqrt(np.mean(np.square(energy)))),
        "force_mae_ev_per_A": float(np.mean(np.abs(forces))),
        "force_rmse_ev_per_A": float(np.sqrt(np.mean(np.square(forces)))),
    }


def _prepare_frame(atoms: Any, source: Path, split: str) -> Any | None:
    if "dft_energy" not in atoms.info or "dft_forces" not in atoms.arrays:
        raise ValueError(
            f"uMOF frame in {source} lacks dft_energy or dft_forces labels."
        )
    energy = float(atoms.info["dft_energy"])
    forces = np.asarray(atoms.arrays["dft_forces"], dtype=float)
    if not math.isfinite(energy) or forces.shape != (len(atoms), 3):
        raise ValueError(f"Invalid uMOF energy/force labels in {source}.")
    if not np.isfinite(forces).all():
        raise ValueError(f"Non-finite uMOF force labels in {source}.")
    guest_counts = _target_gas_composition(atoms)
    if not guest_counts:
        return None
    prepared = atoms.copy()
    prepared.info["REF_energy"] = energy
    prepared.new_array("REF_forces", forces.copy())
    prepared.info["umof_source_file"] = source.name
    prepared.info["umof_source_split"] = split
    prepared.info["umof_gas_composition"] = json.dumps(
        {
            gas: count / sum(guest_counts.values())
            for gas, count in sorted(guest_counts.items())
        },
        sort_keys=True,
    )
    return prepared


def _guest_molecule_counts(atoms: Any) -> dict[str, int]:
    from collections import Counter

    from ase.neighborlist import natural_cutoffs, neighbor_list
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    atom_count = len(atoms)
    neighbor_i, neighbor_j = neighbor_list(
        "ij",
        atoms,
        natural_cutoffs(atoms, mult=1.15),
        self_interaction=False,
    )
    adjacency = coo_matrix(
        (np.ones(len(neighbor_i), dtype=np.int8), (neighbor_i, neighbor_j)),
        shape=(atom_count, atom_count),
    ).tocsr()
    component_count, component_ids = connected_components(
        adjacency,
        directed=False,
    )
    species = np.asarray(atoms.get_chemical_symbols())
    gas_counts: Counter[str] = Counter()
    for component_id in range(component_count):
        formula = Counter(species[component_ids == component_id].tolist())
        for gas, expected_formula in KNOWN_GAS_FORMULAS.items():
            if formula == expected_formula:
                gas_counts[gas] += 1
                break
    return dict(gas_counts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument(
        "--validation-mofs",
        default=",".join(sorted(DEFAULT_VALIDATION_MOFS)),
    )
    parser.add_argument("--test-mofs", default=",".join(sorted(DEFAULT_TEST_MOFS)))
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args(argv)

    manifest = prepare_adsorption_datasets(
        args.data_directory,
        args.output_directory / "datasets",
        train_fraction=args.train_fraction,
        seed=args.seed,
        validation_mofs=set(args.validation_mofs.split(",")),
        test_mofs=set(args.test_mofs.split(",")),
    )
    command = build_umof_mh_training_command(
        manifest,
        args.checkpoint,
        args.output_directory,
        max_epochs=args.max_epochs,
    )
    save_benchmark_data(
        args.output_directory / "training_command.json",
        {"command": command, "device": "cpu", "foundation_head": "r2scan+d4"},
    )
    print(json.dumps(manifest, indent=2))
    print("Training command:", json.dumps(command))
    if args.train:
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
        model_path = args.output_directory / "models" / "umof_mh_adsorption_finetuned_compiled.model"
        if not model_path.is_file():
            model_path = args.output_directory / "models" / "umof_mh_adsorption_finetuned.model"
        report = evaluate_umof_mh_models(
            manifest["test_file"],
            model_path,
            baseline_checkpoint=args.checkpoint,
        )
        report_path = args.output_directory / "results" / "held_out_test_comparison.json"
        save_benchmark_data(report_path, report)
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())