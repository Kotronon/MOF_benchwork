"""Create explicitly selected ASE calculators for Module C."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from modules.module_c_mlips.dependencies import normalize_mlip_backend


def build_ase_calculator(specification: dict[str, Any]) -> Any:
    """Build an ASE calculator without MLIP-MC backend auto-detection."""
    if not isinstance(specification, dict):
        raise TypeError("MLIP model specification must be an object.")
    backend = normalize_mlip_backend(
        str(specification.get("backend") or specification.get("type") or "")
    )
    if backend == "mace-torch":
        return _build_mace_calculator(specification)
    if backend == "orb-models":
        return _build_orb_calculator(specification)
    if backend == "fairchem":
        return _build_fairchem_calculator(specification)
    if backend == "nequip":
        return _build_nequip_calculator(specification)
    raise AssertionError(f"Unhandled normalized MLIP backend {backend!r}.")


def build_interaction_direct_calculator(
    calculator: Any,
    *,
    framework_atom_count: int,
    adsorbate_atom_count: int,
) -> Any:
    """Adapt an interaction-energy model to MLIP-MC's total-energy API.

    The Goeminne NequIP models were trained to return the host-guest
    interaction energy directly. MLIP-MC normally subtracts isolated host and
    guest energies. Returning zero for those two baseline calls preserves the
    direct interaction energy while forwarding combined configurations to the
    model unchanged.
    """
    if framework_atom_count <= 0 or adsorbate_atom_count <= 0:
        raise ValueError("Interaction-direct atom counts must be positive.")
    try:
        from ase.calculators.calculator import Calculator, all_changes
    except ImportError as exc:
        raise ImportError("ASE is required for interaction-direct models.") from exc

    class InteractionDirectCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self,
            atoms=None,
            properties=("energy",),
            system_changes=all_changes,
        ) -> None:
            super().calculate(atoms, properties, system_changes)
            atom_count = len(atoms)
            is_baseline = atom_count in {
                framework_atom_count,
                adsorbate_atom_count,
            }
            if is_baseline:
                self.results = {
                    "energy": 0.0,
                    "forces": np.zeros((atom_count, 3), dtype=float),
                }
                return
            if atom_count < framework_atom_count + adsorbate_atom_count:
                raise ValueError(
                    "Interaction-direct calculator received an unexpected "
                    f"atom count: {atom_count}."
                )
            energy = float(calculator.get_potential_energy(atoms))
            forces = np.asarray(calculator.get_forces(atoms), dtype=float)
            self.results = {"energy": energy, "forces": forces}

    return InteractionDirectCalculator()


def _build_mace_calculator(specification: dict[str, Any]) -> Any:
    try:
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise ImportError(
            "MACE is required for the selected MLIP backend. Re-run with "
            "--install-missing or install mlip-mc[mace-torch]."
        ) from exc

    dtype = str(specification.get("default_dtype", "float64"))
    if dtype not in {"float32", "float64"}:
        raise ValueError("MACE default_dtype must be 'float32' or 'float64'.")
    return mace_mp(
        model=specification.get("model", "medium"),
        device=str(specification.get("device", "cpu")),
        dispersion=bool(specification.get("dispersion", False)),
        default_dtype=dtype,
    )


def _build_orb_calculator(specification: dict[str, Any]) -> Any:
    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
    except ImportError as exc:
        raise ImportError(
            "ORB is required for the selected MLIP backend. Re-run with "
            "--install-missing or install mlip-mc[orb-models]."
        ) from exc

    device = str(specification.get("device", "cpu"))
    arguments = {
        "device": device,
        "precision": str(specification.get("precision", "float32-high")),
    }
    model = specification.get("model")
    if model not in (None, "", "default"):
        model_path = Path(str(model)).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"ORB model file does not exist: {model_path}")
        arguments["weights_path"] = str(model_path)
    orb_model = pretrained.orb_v3_conservative_inf_omat(**arguments)
    return ORBCalculator(orb_model, device=device)


def _build_fairchem_calculator(specification: dict[str, Any]) -> Any:
    try:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
    except ImportError as exc:
        raise ImportError(
            "FAIRChem is required for the selected MLIP backend. Re-run with "
            "--install-missing or install mlip-mc[fairchem]."
        ) from exc

    model = specification.get("model")
    if not model:
        raise ValueError("FAIRChem requires a model checkpoint path.")
    predictor = load_predict_unit(
        str(model),
        device=str(specification.get("device", "cpu")),
    )
    return FAIRChemCalculator(
        predictor,
        task_name=str(specification.get("task_name", "odac")),
    )


def _build_nequip_calculator(specification: dict[str, Any]) -> Any:
    """Load either a legacy deployed or a current compiled NequIP model."""
    model = specification.get("model")
    if not model:
        raise ValueError("NequIP requires a local model path.")
    model_path = Path(str(model)).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"NequIP model file does not exist: {model_path}")
    if bool(specification.get("dispersion", False)):
        raise ValueError(
            "Do not add D3 to the fine-tuned NequIP reference model; its target "
            "energies already define the reference Hamiltonian."
        )

    loader = str(specification.get("loader", "auto")).strip().casefold()
    if loader not in {"auto", "legacy", "compiled"}:
        raise ValueError("NequIP loader must be 'auto', 'legacy', or 'compiled'.")
    device = str(specification.get("device", "cpu"))
    mapping = specification.get("species_to_type_name")
    energy_units = float(specification.get("energy_units_to_eV", 1.0))
    length_units = float(specification.get("length_units_to_A", 1.0))
    if energy_units <= 0 or length_units <= 0:
        raise ValueError("NequIP unit conversion factors must be positive.")

    modern_suffix = model_path.name.endswith((".nequip.pth", ".nequip.pt2"))
    attempts = (
        ["compiled", "legacy"]
        if loader == "auto" and modern_suffix
        else ["legacy", "compiled"]
        if loader == "auto"
        else [loader]
    )
    failures = []
    for candidate in attempts:
        try:
            if candidate == "legacy":
                module = importlib.import_module("nequip.ase")
                calculator_class = module.NequIPCalculator
                arguments = {
                    "model_path": str(model_path),
                    "device": device,
                    "energy_units_to_eV": energy_units,
                    "length_units_to_A": length_units,
                    "set_global_options": True,
                }
                if mapping is not None:
                    arguments["species_to_type_name"] = mapping
                return calculator_class.from_deployed_model(**arguments)

            module = importlib.import_module("nequip.integrations.ase")
            calculator_class = module.NequIPCalculator
            arguments = {
                "compile_path": str(model_path),
                "device": device,
                "energy_units_to_eV": energy_units,
                "length_units_to_A": length_units,
            }
            if mapping is not None:
                arguments["chemical_species_to_atom_type_map"] = mapping
            return calculator_class.from_compiled_model(**arguments)
        except Exception as exc:
            failures.append(f"{candidate}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"Could not load NequIP model {model_path}. Tried "
        + "; ".join(failures)
        + ". The Goeminne .pth model normally requires a compatible legacy "
        "NequIP release and loader='legacy'."
    )
