"""Create explicitly selected ASE calculators for Module C."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    raise AssertionError(f"Unhandled normalized MLIP backend {backend!r}.")


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
