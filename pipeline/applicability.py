from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pipeline.adsorbate_registry import infer_adsorbate_properties


DEFAULT_RULES_PATH = Path("data/module_applicability_rules.json")


def assess_applicability(
    config: dict[str, Any],
    module: dict[str, str],
    resolved: dict[str, Any],
    rules_path: str | Path = DEFAULT_RULES_PATH,
) -> dict[str, Any]:
    """Return a JSON-serializable capability assessment for the selected run."""
    rules = load_applicability_rules(rules_path)
    inferred = _infer_applicability(config, module, resolved, rules)
    configured = config.get("benchmark", {}).get("applicability", {})
    if not isinstance(configured, dict):
        raise TypeError("'benchmark.applicability' must be an object when provided.")
    return _merge_applicability(inferred, configured)


def load_applicability_rules(rules_path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    """Load curated module applicability rules from JSON."""
    path = Path(rules_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        rules = json.load(handle)
    if not isinstance(rules, dict):
        raise TypeError("Applicability rules must be a JSON object.")
    return rules


def _infer_applicability(
    config: dict[str, Any],
    module: dict[str, str],
    resolved: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    module_id = module["id"]
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    reasons: list[str] = []

    if module_id != "A":
        return {
            "status": "unsupported",
            "current_module_capability": "not_implemented",
            "can_attempt_simulation": False,
            "requires_user_confirmation": False,
            "recommended_module": module_id,
            "checks": [
                {
                    "name": "implemented_module",
                    "status": "failed",
                    "expected": "A",
                    "actual": module_id,
                }
            ],
            "reason": (
                f"Module {module_id} is selected, but the current executable pipeline "
                "only materializes and runs Module A."
            ),
        }

    material_id = resolved["material"]["material_id"]
    module_rules = _module_rules(rules, module_id)
    material_rule = _material_rule(rules, module_id, material_id)

    requirements = module_rules.get("required", {})
    if not isinstance(requirements, dict):
        raise TypeError(f"'module_{module_id}.required' must be an object.")

    _append_exact_check(
        checks,
        "engine",
        requirements.get("engine"),
        config.get("simulation", {}).get("engine"),
        "Unsupported simulation engine for the current Module A implementation.",
        reasons,
        hard_failure=True,
    )
    _append_exact_check(
        checks,
        "method",
        requirements.get("method"),
        config.get("simulation", {}).get("method"),
        "Module A currently supports GCMC adsorption runs.",
        reasons,
        hard_failure=True,
    )
    _append_exact_check(
        checks,
        "framework",
        requirements.get("framework"),
        config.get("simulation", {}).get("framework"),
        "Flexible-framework calculations are outside the current Module A implementation.",
        reasons,
        hard_failure=True,
    )

    components = config.get("adsorbates", {}).get("components", [])
    if requirements.get("single_component") is True and len(components) != 1:
        checks.append(
            {
                "name": "single_component",
                "status": "failed",
                "expected": True,
                "actual": len(components),
                "severity": "unsupported",
            }
        )
        reasons.append("Module A supports only single-component adsorption.")
    else:
        checks.append(
            {
                "name": "single_component",
                "status": "passed",
                "expected": requirements.get("single_component", True),
                "actual": len(components),
            }
        )

    geometry = _crafted_geometry(material_id, module_rules.get("geometry_source"))
    adsorbate_properties: dict[str, Any] | None = None
    if geometry is None:
        checks.append(
            {
                "name": "crafted_geometry",
                "status": "warning",
                "expected": "geometry metadata available",
                "actual": None,
                "severity": "warning",
            }
        )
        warnings.append(f"No CRAFTED geometry metadata was found for {material_id}.")
    else:
        checks.append(
            {
                "name": "crafted_geometry",
                "status": "passed",
                "expected": "geometry metadata available",
                "actual": geometry["source"],
            }
        )
        _append_minimum_check(
            checks,
            name="pore_volume_cm3_g",
            expected=requirements.get("min_pore_volume_cm3_g"),
            actual=geometry.get("pore_volume_cm3_g"),
            failure_reason=(
                f"{material_id} has no accessible pore volume in the CRAFTED geometry table "
                "for the probe used there, so a rigid Module A adsorption run is outside the "
                "validated benchmark scope."
            ),
                warnings=warnings,
            )
        if components:
            try:
                adsorbate_properties = infer_adsorbate_properties(
                    components[0],
                    resolved.get("forcefield", {}),
                    module_rules.get("adsorbate_diameter_policy"),
                )
            except (LookupError, ValueError, OSError, KeyError) as exc:
                checks.append(
                    {
                        "name": "adsorbate_access_diameter_A",
                        "status": "warning",
                        "expected": "generated adsorbate diameter",
                        "actual": None,
                        "severity": "warning",
                    }
                )
                warnings.append(
                    f"No generated access diameter is available for {components[0]}: {exc}"
                )
            else:
                checks.append(
                    {
                        "name": "adsorbate_access_diameter_A",
                        "status": "passed",
                        "expected": "generated adsorbate diameter",
                        "actual": adsorbate_properties["access_diameter_A"],
                        "source": adsorbate_properties["diameter_source"],
                    }
                )
            _append_minimum_check(
                checks,
                name="pld_A",
                expected=(
                    None
                    if adsorbate_properties is None
                    else adsorbate_properties["access_diameter_A"]
                ),
                actual=geometry.get("pld_A"),
                failure_reason=(
                    f"{material_id} has a pore-limiting diameter below the generated "
                    f"access diameter for {components[0]}, so adsorption may be diffusion-limited "
                    "or physically inaccessible in a rigid-framework model."
                ),
                warnings=warnings,
            )

    if material_rule is not None:
        checks.append(
            {
                "name": "curated_material_rule",
                "status": "warning" if material_rule.get("status") == "warning" else "failed",
                "expected": "validated Module A material",
                "actual": material_id,
                "matched_rule": material_rule["matched_rule"],
                "severity": material_rule.get("status", "warning"),
            }
        )
        warnings.append(str(material_rule.get("reason", "Curated material rule matched.")))

    hard_failures = [check for check in checks if check.get("severity") == "unsupported"]
    warning_checks = [check for check in checks if check.get("status") == "warning"]
    if hard_failures:
        return {
            "status": "unsupported",
            "current_module_capability": "not_module_a_scope",
            "can_attempt_simulation": False,
            "requires_user_confirmation": False,
            "recommended_module": "B",
            "reason": " ".join(reasons) if reasons else "This case is outside Module A scope.",
            "reasons": reasons,
            "warnings": warnings,
            "checks": checks,
            "geometry": geometry,
            "adsorbate_properties": adsorbate_properties,
        }

    if material_rule is not None or warning_checks:
        recommendation = "A"
        if material_rule is not None:
            recommendation = material_rule.get("recommended_module", recommendation)
        return {
            "status": "warning",
            "current_module_capability": "outside_validated_scope",
            "can_attempt_simulation": True,
            "requires_user_confirmation": True,
            "recommended_module": recommendation,
            "reason": " ".join(warnings),
            "reasons": warnings,
            "warnings": warnings,
            "checks": checks,
            "geometry": geometry,
            "adsorbate_properties": adsorbate_properties,
            **({} if material_rule is None else {"matched_rule": material_rule["matched_rule"]}),
            **({} if material_rule is None else {"tags": material_rule.get("tags", [])}),
        }

    return {
        "status": "supported",
        "current_module_capability": "validated_scope",
        "can_attempt_simulation": True,
        "requires_user_confirmation": False,
        "recommended_module": "A",
        "checks": checks,
        "geometry": geometry,
        "adsorbate_properties": adsorbate_properties,
        "reason": "The selected case is within the current Module A implementation scope.",
    }


def _module_rules(rules: dict[str, Any], module_id: str) -> dict[str, Any]:
    module_rules = rules.get(f"module_{module_id}", {})
    if not isinstance(module_rules, dict):
        raise TypeError(f"Applicability rules for module {module_id!r} must be an object.")
    return module_rules


def _material_rule(
    rules: dict[str, Any],
    module_id: str,
    material_id: str,
) -> dict[str, Any] | None:
    module_rules = _module_rules(rules, module_id)
    warning_materials = module_rules.get("warning_materials", {})
    if not isinstance(warning_materials, dict):
        raise TypeError(f"'module_{module_id}.warning_materials' must be an object.")

    rule = warning_materials.get(material_id)
    if rule is None:
        return None
    if not isinstance(rule, dict):
        raise TypeError(f"Applicability rule for material {material_id!r} must be an object.")

    return {
        "matched_rule": f"module_{module_id}.warning_materials.{material_id}",
        **rule,
    }


def _crafted_geometry(
    material_id: str,
    geometry_source: Any,
) -> dict[str, Any] | None:
    path = Path(str(geometry_source or "CRAFTED-2.0.0/RAC_DBSCAN/CRAFTED_MOF_geometric.csv"))
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("FrameworkName") != material_id:
                continue
            return {
                "source": str(path),
                "material_id": material_id,
                "largest_cavity_diameter_A": _optional_float(row.get("D_is")),
                "pore_limiting_diameter_A": _optional_float(row.get("D_fs")),
                "pld_A": _optional_float(row.get("D_fs")),
                "accessible_surface_area_m2_g": _optional_float(row.get("ASA_m^2/g")),
                "non_accessible_surface_area_m2_g": _optional_float(row.get("NASA_m^2/g")),
                "accessible_volume_fraction": _optional_float(row.get("AV_Volume_fraction")),
                "pore_volume_cm3_g": _optional_float(row.get("AV_cm^3/g")),
                "non_accessible_volume_cm3_g": _optional_float(row.get("NAV_cm^3/g")),
                "density_g_cm3": _optional_float(row.get("Density")),
            }
    return None


def _append_exact_check(
    checks: list[dict[str, Any]],
    name: str,
    expected: Any,
    actual: Any,
    failure_reason: str,
    reasons: list[str],
    *,
    hard_failure: bool,
) -> None:
    if expected is None:
        return
    passed = str(actual).strip().casefold() == str(expected).strip().casefold()
    checks.append(
        {
            "name": name,
            "status": "passed" if passed else "failed",
            "expected": expected,
            "actual": actual,
            **({} if passed else {"severity": "unsupported" if hard_failure else "warning"}),
        }
    )
    if not passed:
        reasons.append(failure_reason)


def _append_minimum_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    expected: Any,
    actual: Any,
    failure_reason: str,
    warnings: list[str],
) -> None:
    if expected is None:
        return
    expected_float = float(expected)
    passed = actual is not None and float(actual) >= expected_float
    checks.append(
        {
            "name": name,
            "status": "passed" if passed else "warning",
            "expected_min": expected_float,
            "actual": actual,
            **({} if passed else {"severity": "warning"}),
        }
    )
    if not passed:
        warnings.append(failure_reason)


def _optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


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
