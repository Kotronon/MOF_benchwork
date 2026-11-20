#!/usr/bin/env python3
"""Run a small paired FLAMES hybrid GCMC/MD pilot from a repository JSON input."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "input_json_files/benchmark_module_d_mil53_co2_flames_umof_hybrid.json"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "module_d.flames_umof_hybrid.v1":
        raise ValueError(f"Unsupported input schema in {path}")
    return config


def repository_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required repository asset is missing: {path}")
    return path


def read_structures(config: dict):
    from ase.io import read

    framework_spec = config["system"]["framework"]
    framework = read(
        str(repository_path(framework_spec["path"])),
        format=framework_spec["format"],
        atom_style=framework_spec["atom_style"],
        Z_of_type={int(k): int(v) for k, v in framework_spec["atomic_number_by_lammps_type"].items()},
    )
    co2 = read(str(repository_path(config["system"]["adsorbate"]["path"])))
    if len(co2) != 3 or co2.get_chemical_formula() != "CO2":
        raise ValueError(f"Expected one CO2 molecule, got {co2.get_chemical_formula()}")
    if len(framework) != 1216:
        raise ValueError(f"Expected the existing 1216-atom MIL-53 framework, got {len(framework)} atoms")
    framework.pbc = True
    co2.pbc = True
    co2.set_cell(framework.cell)
    return framework, co2


def check_inputs(config: dict, *, load_ase: bool) -> None:
    for model in config["models"]:
        repository_path(model["checkpoint"])
    framework_path = repository_path(config["system"]["framework"]["path"])
    co2_path = repository_path(config["system"]["adsorbate"]["path"])
    if load_ase:
        framework, co2 = read_structures(config)
        print(f"Framework: {len(framework)} atoms, formula {framework.get_chemical_formula()}, {framework_path}")
        print(f"Adsorbate: {co2.get_chemical_formula()}, {co2_path}")
    print(f"Cases: {len(config['models'])} models x {len(config['conditions']['reservoir_pressures_bar'])} pressures x {len(config['pilot']['seeds'])} seed(s)")
    print(f"Output: {ROOT / config['execution']['output_directory']}")


def peng_robinson_phi(pressure_pa: float, temperature_k: float, eos: dict) -> float:
    """Vapor-root Peng-Robinson fugacity coefficient for the legacy FLAMES API."""
    import numpy as np

    gas_constant = 8.31446261815324
    tc = float(eos["criticalTemperature_K"])
    pc = float(eos["criticalPressure_Pa"])
    omega = float(eos["acentricFactor"])
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega
    alpha = (1.0 + kappa * (1.0 - math.sqrt(temperature_k / tc))) ** 2
    a = 0.45724 * gas_constant**2 * tc**2 / pc * alpha
    b = 0.07780 * gas_constant * tc / pc
    aa = a * pressure_pa / (gas_constant**2 * temperature_k**2)
    bb = b * pressure_pa / (gas_constant * temperature_k)
    roots = np.roots([1.0, -(1.0 - bb), aa - 3.0 * bb**2 - 2.0 * bb, -(aa * bb - bb**2 - bb**3)])
    real_roots = [float(root.real) for root in roots if abs(float(root.imag)) < 1e-8 and root.real > bb]
    if not real_roots:
        raise ValueError(f"Peng-Robinson EOS has no physical vapor root at {pressure_pa:g} Pa")
    z = max(real_roots)
    ratio = (z + (1.0 + math.sqrt(2.0)) * bb) / (z + (1.0 - math.sqrt(2.0)) * bb)
    ln_phi = z - 1.0 - math.log(z - bb) - aa / (2.0 * math.sqrt(2.0) * bb) * math.log(ratio)
    return math.exp(ln_phi)


def construct_simulation(config: dict, model_config: dict, pressure_bar: float, seed: int, device: str, output: Path):
    import torch
    from ase.data import vdw_radii
    from modules.module_c_mlips.umof_profiles import calculator, polar_conditioning

    try:
        from flames.gcmc import GCMC
    except ImportError as exc:
        raise ImportError(
            "FLAMES is not installed in this Python environment. Activate the existing cluster environment "
            "that contains the repository's pinned FLAMES dependency; this runner does not install or bundle packages."
        ) from exc

    framework, co2 = read_structures(config)
    profile = model_config["profile"]
    if profile == "polar":
        polar_conditioning(framework)
        polar_conditioning(co2)
    calc = calculator(
        repository_path(model_config["checkpoint"]),
        model=profile,
        head=model_config["head"],
        device=device,
        dtype=model_config["dtype"],
    )
    signature = inspect.signature(GCMC.__init__)
    parameters = signature.parameters
    common = {
        "model": calc,
        "framework_atoms": framework,
        "temperature": float(config["conditions"]["temperature_K"]),
        "pressure": pressure_bar * 100_000.0,
        "device": device,
        "vdw_radii": vdw_radii,
        "vdw_factor": float(config["pilot"]["vdw_factor"]),
        "save_frequency": int(config["pilot"]["gcmc_save_frequency"]),
        "debug": False,
        "output_to_file": True,
        "output_folder": str(output),
        "random_seed": int(seed),
        "cutoff_radius": float(config["pilot"]["cutoff_A"]),
        "automatic_supercell": bool(config["system"]["framework"]["automatic_supercell"]),
    }
    if "adsorbates" in parameters:
        try:
            from flames.adsorbate import Adsorbate
        except ImportError as exc:
            raise ImportError("This FLAMES GCMC API needs flames.adsorbate.Adsorbate") from exc
        eos = config["system"]["adsorbate"]["eos"]
        adsorbate = Adsorbate(
            name="CO2",
            structure=str(repository_path(config["system"]["adsorbate"]["path"])),
            eos={
                "criticalTemperature": float(eos["criticalTemperature_K"]),
                "criticalPressure": float(eos["criticalPressure_Pa"]),
                "acentricFactor": float(eos["acentricFactor"]),
            },
        )
        common["adsorbates"] = adsorbate
    elif "adsorbate_atoms" in parameters:
        common["adsorbate_atoms"] = co2
        eos = config["system"]["adsorbate"]["eos"]
        common["fugacity_coeff"] = peng_robinson_phi(
            pressure_bar * 100_000.0,
            float(config["conditions"]["temperature_K"]),
            eos,
        )
    else:
        raise RuntimeError(f"Unsupported FLAMES GCMC constructor: {signature}")

    simulation = GCMC(**common)
    if not callable(getattr(simulation, "md", None)):
        raise RuntimeError("The installed FLAMES GCMC class has no hybrid MD method; use a FLAMES release with GCMC.md().")
    return simulation


def run_case(config: dict, model_config: dict, pressure_bar: float, seed: int, device: str) -> None:
    from ase.io import write

    label = f"{model_config['profile']}_p{pressure_bar:g}bar_seed{seed}"
    out_root = ROOT / config["execution"]["output_directory"]
    case_dir = out_root / label
    if case_dir.exists() and config["execution"].get("refuse_existing_case_directory", True):
        raise FileExistsError(f"Refusing to overwrite existing output: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=False)

    simulation = construct_simulation(config, model_config, pressure_bar, seed, device, case_dir)
    hybrid = config["pilot"]
    external_pressure_pa = float(config["conditions"]["external_npt_pressure_bar"]) * 100_000.0
    rows = []
    for block in range(1, int(hybrid["hybrid_blocks"]) + 1):
        simulation.run(int(hybrid["gcmc_steps_per_block"]))
        reservoir_pressure_pa = float(pressure_bar) * 100_000.0
        # FLAMES uses simulation.P for both the GCMC reservoir pressure and the MD barostat.
        # Switch only during the NPT segment so the solid is held at the requested external pressure.
        simulation.P = external_pressure_pa
        try:
            simulation.md(
                nsteps=int(hybrid["md_steps_per_block"]),
                time_step=float(hybrid["md_timestep_fs"]),
                ensemble=hybrid["md_ensemble"],
                thermostat=hybrid["md_thermostat"],
                output_interval=int(hybrid["md_output_interval"]),
                movie_interval=int(hybrid["md_movie_interval"]),
            )
        finally:
            simulation.P = reservoir_pressure_pa
        counts = simulation.get_number_of_adsorbates(simulation.current_system)
        n_co2 = int(counts.get("CO2", 0))
        mass_amu = float(sum(simulation.framework.get_masses()))
        volume_a3 = float(simulation.current_system.get_volume())
        rows.append({
            "model": model_config["name"],
            "pressure_bar": float(pressure_bar),
            "external_npt_pressure_bar": float(config["conditions"]["external_npt_pressure_bar"]),
            "seed": int(seed),
            "block": block,
            "gcmc_steps": int(hybrid["gcmc_steps_per_block"]),
            "md_steps": int(hybrid["md_steps_per_block"]),
            "co2_molecules_at_block_end": n_co2,
            "loading_mol_kg_at_block_end": n_co2 * 1000.0 / mass_amu,
            "cell_volume_A3": volume_a3,
        })
        print(f"[{label}] block {block}/{hybrid['hybrid_blocks']}: {n_co2} CO2, {rows[-1]['loading_mol_kg_at_block_end']:.4f} mol/kg")

    simulation.run(int(hybrid["final_gcmc_steps"]))
    simulation.save_results()
    final_counts = simulation.get_number_of_adsorbates(simulation.current_system)
    n_co2 = int(final_counts.get("CO2", 0))
    mass_amu = float(sum(simulation.framework.get_masses()))
    rows.append({
        "model": model_config["name"],
        "pressure_bar": float(pressure_bar),
        "external_npt_pressure_bar": float(config["conditions"]["external_npt_pressure_bar"]),
        "seed": int(seed),
        "block": "final_gcmc",
        "gcmc_steps": int(hybrid["final_gcmc_steps"]),
        "md_steps": 0,
        "co2_molecules_at_block_end": n_co2,
        "loading_mol_kg_at_block_end": n_co2 * 1000.0 / mass_amu,
        "cell_volume_A3": float(simulation.current_system.get_volume()),
    })
    with (case_dir / "hybrid_cycle_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(case_dir / "final_structure.extxyz", simulation.current_system)
    (case_dir / "run_metadata.json").write_text(
        json.dumps({
            "model": model_config,
            "reservoir_pressure_bar": float(pressure_bar),
            "external_npt_pressure_bar": float(config["conditions"]["external_npt_pressure_bar"]),
            "temperature_K": float(config["conditions"]["temperature_K"]),
            "seed": int(seed),
            "device": device,
            "status": "pilot_completed_not_convergence_claim",
            "final_co2_molecules": n_co2,
            "final_loading_mol_kg": n_co2 * 1000.0 / mass_amu,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Completed {label}: {case_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Check repository assets and the input structure; do not import FLAMES or load model checkpoints.")
    parser.add_argument("--model", choices=("mh", "polar"), help="Run only this model profile.")
    parser.add_argument("--pressure-bar", type=float, help="Run only this configured reservoir pressure.")
    parser.add_argument("--seed", type=int, help="Run only this configured seed.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override execution.device.")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    if args.dry_run:
        check_inputs(config, load_ase=True)
        print("Dry run passed; no model, FLAMES package, or simulation was started.")
        return 0

    check_inputs(config, load_ase=False)
    models = [model for model in config["models"] if args.model is None or model["profile"] == args.model]
    pressures = config["conditions"]["reservoir_pressures_bar"]
    if args.pressure_bar is not None:
        if not any(math.isclose(args.pressure_bar, float(p), abs_tol=1e-9) for p in pressures):
            raise ValueError(f"Pressure {args.pressure_bar:g} bar is not listed in the JSON input")
        pressures = [args.pressure_bar]
    seeds = config["pilot"]["seeds"]
    if args.seed is not None:
        if args.seed not in seeds:
            raise ValueError(f"Seed {args.seed} is not listed in the JSON input")
        seeds = [args.seed]
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch/MACE are required by the selected uMOF checkpoints in the active cluster environment") from exc
    requested_device = args.device or config["execution"]["device"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but torch.cuda.is_available() is false")

    for model_config in models:
        for pressure in pressures:
            for seed in seeds:
                run_case(config, model_config, float(pressure), int(seed), device)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
