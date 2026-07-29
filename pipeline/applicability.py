from __future__ import annotations

from typing import Any


MODULE_A_WARNING_MATERIALS = {
    "ZIF-7": {
        "status": "warning",
        "current_module_capability": "outside_validated_scope",
        "can_attempt_simulation": True,
        "requires_user_confirmation": True,
        "recommended_module": "D",
        "reason": (
            "ZIF-7 is a gate-opening/flexibility warning case. The current Module A "
            "implementation can run a rigid-framework GCMC calculation, but it cannot "
            "validate gate-opening or framework-response physics."
        ),
    },
    "ZnMOF-74": {
        "status": "warning",
        "current_module_capability": "outside_validated_scope",
        "can_attempt_simulation": True,
        "requires_user_confirmation": True,
        "recommended_module": "C",
        "reason": (
            "ZnMOF-74 is an open-metal-site forcefield warning case. The current "
            "Module A implementation can run a generic-UFF rigid GCMC calculation, "
            "but it cannot validate site-specific CO2 binding or forcefield transferability."
        ),
    },
    "Zn-DOBDC": {
        "status": "warning",
        "current_module_capability": "outside_validated_scope",
        "can_attempt_simulation": True,
        "requires_user_confirmation": True,
        "recommended_module": "C",
        "reason": (
            "Zn-DOBDC is an MOF-74-family forcefield warning case. The current "
            "Module A implementation can run a generic-UFF rigid GCMC calculation, "
            "but it cannot validate site-specific CO2 binding or forcefield transferability."
        ),
    },
}


def assess_applicability(
    config: dict[str, Any],
    module: dict[str, str],
    resolved: dict[str, Any],
) -> dict[str, Any]:
    """Return a JSON-serializable capability assessment for the selected run."""
    inferred = _infer_applicability(config, module, resolved)
    configured = config.get("benchmark", {}).get("applicability", {})
    if not isinstance(configured, dict):
        raise TypeError("'benchmark.applicability' must be an object when provided.")
    return _merge_applicability(inferred, configured)


def _infer_applicability(
    config: dict[str, Any],
    module: dict[str, str],
    resolved: dict[str, Any],
) -> dict[str, Any]:
    module_id = module["id"]
    if module_id != "A":
        return {
            "status": "unsupported",
            "current_module_capability": "not_implemented",
            "can_attempt_simulation": False,
            "requires_user_confirmation": False,
            "recommended_module": module_id,
            "reason": (
                f"Module {module_id} is selected, but the current executable pipeline "
                "only materializes and runs Module A."
            ),
        }

    material_id = resolved["material"]["material_id"]
    if material_id in MODULE_A_WARNING_MATERIALS:
        return dict(MODULE_A_WARNING_MATERIALS[material_id])

    if len(config.get("adsorbates", {}).get("components", [])) != 1:
        return {
            "status": "unsupported",
            "current_module_capability": "not_module_a_scope",
            "can_attempt_simulation": False,
            "requires_user_confirmation": False,
            "recommended_module": "B",
            "reason": "Module A supports only single-component adsorption.",
        }

    return {
        "status": "supported",
        "current_module_capability": "validated_scope",
        "can_attempt_simulation": True,
        "requires_user_confirmation": False,
        "reason": "The selected case is within the current Module A implementation scope.",
    }


def _merge_applicability(
    inferred: dict[str, Any],
    configured: dict[str, Any],
) -> dict[str, Any]:
    if not configured:
        return inferred

    merged = {**inferred, **configured}
    if inferred.get("status") == "unsupported":
        merged["status"] = "unsupported"
        merged["can_attempt_simulation"] = False
        merged["requires_user_confirmation"] = False
    return merged
