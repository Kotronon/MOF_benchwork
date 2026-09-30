"""Runtime validation for model-specific structure assumptions."""

from __future__ import annotations

from typing import Any


def validate_model_input_contract(
    framework: Any,
    adsorbate: Any,
    contract: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate the generated host and guest against a model input contract."""
    if contract is None:
        return {"configured": False, "status": "not_requested"}
    if not isinstance(contract, dict):
        raise TypeError("Model input_contract must be an object.")

    tolerance = float(contract.get("cell_tolerance_A", 1.0e-3))
    if tolerance < 0.0:
        raise ValueError("Model input_contract cell_tolerance_A must be non-negative.")

    actual_framework_atoms = len(framework)
    expected_framework_atoms = contract.get("framework_atom_count")
    if (
        expected_framework_atoms is not None
        and actual_framework_atoms != int(expected_framework_atoms)
    ):
        raise ValueError(
            "Model input contract failed: expected "
            f"{int(expected_framework_atoms)} framework atoms, found "
            f"{actual_framework_atoms}."
        )

    actual_lengths = [float(value) for value in framework.cell.lengths()]
    expected_lengths = contract.get("cell_lengths_A")
    if expected_lengths is not None:
        if not isinstance(expected_lengths, list) or len(expected_lengths) != 3:
            raise ValueError("Model input_contract cell_lengths_A must contain 3 values.")
        if any(
            abs(actual - float(expected)) > tolerance
            for actual, expected in zip(actual_lengths, expected_lengths)
        ):
            raise ValueError(
                "Model input contract failed: framework cell lengths "
                f"{actual_lengths} A do not match {expected_lengths} A within "
                f"{tolerance:g} A."
            )

    angle_tolerance = float(contract.get("cell_angle_tolerance_deg", 1.0e-3))
    if angle_tolerance < 0.0:
        raise ValueError(
            "Model input_contract cell_angle_tolerance_deg must be non-negative."
        )
    actual_angles = [float(value) for value in framework.cell.angles()]
    expected_angles = contract.get("cell_angles_deg")
    if expected_angles is not None:
        if not isinstance(expected_angles, list) or len(expected_angles) != 3:
            raise ValueError("Model input_contract cell_angles_deg must contain 3 values.")
        if any(
            abs(actual - float(expected)) > angle_tolerance
            for actual, expected in zip(actual_angles, expected_angles)
        ):
            raise ValueError(
                "Model input contract failed: framework cell angles "
                f"{actual_angles} degrees do not match {expected_angles} degrees within "
                f"{angle_tolerance:g} degrees."
            )

    actual_model_symbols = adsorbate.get_chemical_symbols()
    expected_model_symbols = contract.get("adsorbate_model_symbols")
    if (
        expected_model_symbols is not None
        and actual_model_symbols != list(expected_model_symbols)
    ):
        raise ValueError(
            "Model input contract failed: adsorbate model symbols "
            f"{actual_model_symbols} do not match {expected_model_symbols}."
        )

    physical_symbols = (
        [str(value) for value in adsorbate.arrays["physical_species"]]
        if "physical_species" in adsorbate.arrays
        else actual_model_symbols
    )
    expected_physical_symbols = contract.get("adsorbate_physical_symbols")
    if (
        expected_physical_symbols is not None
        and physical_symbols != list(expected_physical_symbols)
    ):
        raise ValueError(
            "Model input contract failed: physical adsorbate symbols "
            f"{physical_symbols} do not match {expected_physical_symbols}."
        )

    return {
        "configured": True,
        "status": "passed",
        "framework_atom_count": actual_framework_atoms,
        "cell_lengths_A": actual_lengths,
        "cell_tolerance_A": tolerance,
        "cell_angles_deg": actual_angles,
        "cell_angle_tolerance_deg": angle_tolerance,
        "adsorbate_physical_symbols": physical_symbols,
        "adsorbate_model_symbols": actual_model_symbols,
    }
