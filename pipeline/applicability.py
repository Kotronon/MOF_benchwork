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

    if module_id == "C" and _is_potential_benchmark(config):
        workflow = _potential_workflow(config)
        if workflow in {"mlip_mc_widom", "mlip_mc_gcmc"}:
            return _assess_mlip_mc_benchmark(config, workflow)
        return _assess_static_potential_benchmark(config)

    uses_module_a_execution = module_id == "A" or _is_module_c_gcmc_variant_benchmark(config)
    if not uses_module_a_execution:
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
    rule_module_id = "A" if uses_module_a_execution else module_id
    module_rules = _module_rules(rules, rule_module_id)
    material_rule = _material_rule(rules, rule_module_id, material_id)

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
        "current_module_capability": (
            "module_a_compatible_variant_benchmark"
            if module_id == "C"
            else "validated_scope"
        ),
        "can_attempt_simulation": True,
        "requires_user_confirmation": False,
        "recommended_module": module_id,
        "checks": checks,
        "geometry": geometry,
        "adsorbate_properties": adsorbate_properties,
        "reason": (
            "The selected Module C V1 case can be executed through the current "
            "rigid GCMC pipeline."
            if module_id == "C"
            else "The selected case is within the current Module A implementation scope."
        ),
    }


def _is_potential_benchmark(config: dict[str, Any]) -> bool:
    benchmark = config.get("benchmark", {})
    task = str(benchmark.get("task", "")).strip().casefold().replace("-", "_")
    return task == "potential_benchmark" and "potential_benchmark" in benchmark


def _potential_workflow(config: dict[str, Any]) -> str:
    settings = config.get("benchmark", {}).get("potential_benchmark", {})
    if not isinstance(settings, dict):
        return "static"
    return str(settings.get("workflow", "static")).strip().casefold().replace("-", "_")


def _assess_mlip_mc_benchmark(
    config: dict[str, Any],
    workflow: str,
) -> dict[str, Any]:
    settings = config.get("benchmark", {}).get("potential_benchmark", {})
    mlip_mc = settings.get("mlip_mc", {}) if isinstance(settings, dict) else {}
    model = mlip_mc.get("model", {}) if isinstance(mlip_mc, dict) else {}
    components = config.get("adsorbates", {}).get("components", [])
    simulation = config.get("simulation", {})
    framework = str(simulation.get("framework", "rigid")).casefold()
    engine = str(simulation.get("engine", "")).upper()
    method = str(simulation.get("method", "")).upper()
    backend = str(model.get("backend", "")) if isinstance(model, dict) else ""
    supported_backends = {
        "mace",
        "mace-torch",
        "mace_mp",
        "mace-mp",
        "orb",
        "orb-models",
        "fairchem",
        "odac",
    }
    expected_method = "WIDOM" if workflow == "mlip_mc_widom" else "GCMC"
    checks = [
        {
            "name": "workflow",
            "status": "passed",
            "expected": workflow,
            "actual": workflow,
        },
        {
            "name": "single_component",
            "status": "passed" if len(components) == 1 else "failed",
            "expected": 1,
            "actual": len(components),
        },
        {
            "name": "rigid_framework",
            "status": "passed" if framework == "rigid" else "failed",
            "expected": "rigid",
            "actual": framework,
        },
        {
            "name": "engine",
            "status": "passed" if engine == "MLIP_MC" else "failed",
            "expected": "MLIP_MC",
            "actual": engine,
        },
        {
            "name": "method",
            "status": "passed" if method == expected_method else "failed",
            "expected": expected_method,
            "actual": method,
        },
        {
            "name": "mlip_backend",
            "status": (
                "passed"
                if backend.strip().casefold().replace("_", "-") in {
                    value.replace("_", "-") for value in supported_backends
                }
                else "failed"
            ),
            "expected": "mace-torch, orb-models, or fairchem",
            "actual": backend,
        },
    ]
    if workflow == "mlip_mc_widom":
        trials = mlip_mc.get("trials") if isinstance(mlip_mc, dict) else None
        checks.append(
            {
                "name": "widom_trials",
                "status": (
                    "passed"
                    if isinstance(trials, int)
                    and not isinstance(trials, bool)
                    and trials > 0
                    else "failed"
                ),
                "expected": "positive integer",
                "actual": trials,
            }
        )
        block_size = (
            mlip_mc.get("block_size") if isinstance(mlip_mc, dict) else None
        )
        checkpoints = (
            mlip_mc.get("convergence_checkpoints")
            if isinstance(mlip_mc, dict)
            else None
        )
        analysis_valid = (
            (
                block_size is None
                or (
                    isinstance(block_size, int)
                    and not isinstance(block_size, bool)
                    and block_size > 0
                    and isinstance(trials, int)
                    and block_size <= trials
                )
            )
            and (
                checkpoints is None
                or (
                    isinstance(checkpoints, list)
                    and bool(checkpoints)
                    and len(set(checkpoints)) == len(checkpoints)
                    and all(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value > 0
                        and isinstance(trials, int)
                        and value <= trials
                        for value in checkpoints
                    )
                )
            )
        )
        checks.append(
            {
                "name": "widom_analysis",
                "status": "passed" if analysis_valid else "failed",
                "expected": (
                    "positive block_size and unique convergence checkpoints "
                    "within the trial count"
                ),
                "actual": {
                    "block_size": block_size,
                    "convergence_checkpoints": checkpoints,
                },
            }
        )
    else:
        equilibration = (
            mlip_mc.get("equilibration_steps")
            if isinstance(mlip_mc, dict)
            else None
        )
        production = (
            mlip_mc.get("production_steps")
            if isinstance(mlip_mc, dict)
            else None
        )
        checks.append(
            {
                "name": "gcmc_steps",
                "status": (
                    "passed"
                    if isinstance(equilibration, int)
                    and not isinstance(equilibration, bool)
                    and equilibration >= 0
                    and isinstance(production, int)
                    and not isinstance(production, bool)
                    and production > 0
                    else "failed"
                ),
                "expected": "non-negative equilibration and positive production",
                "actual": {
                    "equilibration_steps": equilibration,
                    "production_steps": production,
                },
            }
        )

    failed = [check for check in checks if check["status"] == "failed"]
    if failed:
        return {
            "status": "unsupported",
            "current_module_capability": "invalid_mlip_mc_configuration",
            "can_attempt_simulation": False,
            "requires_user_confirmation": False,
            "recommended_module": "C",
            "checks": checks,
            "reason": "The Module C MLIP-MC configuration is invalid.",
        }
    return {
        "status": "supported",
        "current_module_capability": workflow,
        "can_attempt_simulation": True,
        "requires_user_confirmation": False,
        "recommended_module": "C",
        "checks": checks,
        "reason": f"The Module C {workflow} workflow is executable.",
    }


def _assess_static_potential_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = config.get("benchmark", {}).get("potential_benchmark", {})
    components = config.get("adsorbates", {}).get("components", [])
    framework = str(config.get("simulation", {}).get("framework", "rigid")).casefold()
    configuration_set = str(
        benchmark.get("configuration_set", "smoke")
    ).strip().casefold()
    dataset = benchmark.get("dataset", {})
    backends = benchmark.get("backends", [])
    backend_types = (
        [str(item.get("type", "")).casefold() for item in backends]
        if isinstance(backends, list) and all(isinstance(item, dict) for item in backends)
        else []
    )
    backend_names = (
        [str(item.get("name", "")).strip() for item in backends]
        if isinstance(backends, list) and all(isinstance(item, dict) for item in backends)
        else []
    )
    baseline = str(benchmark.get("baseline_backend", "")).strip()

    checks = [
        {
            "name": "single_component",
            "status": "passed" if len(components) == 1 else "failed",
            "expected": 1,
            "actual": len(components),
        },
        {
            "name": "rigid_framework",
            "status": "passed" if framework == "rigid" else "failed",
            "expected": "rigid",
            "actual": framework,
        },
        {
            "name": "configuration_set",
            "status": (
                "passed"
                if configuration_set in {"smoke", "widom"}
                else "failed"
            ),
            "expected": "smoke or widom",
            "actual": configuration_set,
        },
        {
            "name": "potential_backends",
            "status": (
                "passed"
                if len(backends) >= 2
                and set(backend_types) <= {"classical_lammps", "mace_mp"}
                and len(set(backend_names)) == len(backend_names)
                and all(backend_names)
                else "failed"
            ),
            "expected": "at least two uniquely named classical_lammps/mace_mp backends",
            "actual": backend_names,
        },
        {
            "name": "baseline_backend",
            "status": "passed" if baseline in backend_names else "failed",
            "expected": "one configured backend name",
            "actual": baseline,
        },
    ]
    if configuration_set == "widom":
        sample_count = dataset.get("sample_count") if isinstance(dataset, dict) else None
        seed = dataset.get("seed") if isinstance(dataset, dict) else None
        minimum_distance = (
            dataset.get("minimum_distance_A", 0.0)
            if isinstance(dataset, dict)
            else None
        )
        random_orientations = (
            dataset.get("random_orientations", True)
            if isinstance(dataset, dict)
            else None
        )
        maximum_attempts = (
            dataset.get("maximum_attempts_per_sample", 10_000)
            if isinstance(dataset, dict)
            else None
        )
        checks.append(
            {
                "name": "widom_dataset",
                "status": (
                    "passed"
                    if isinstance(dataset, dict)
                    and isinstance(sample_count, int)
                    and not isinstance(sample_count, bool)
                    and sample_count > 0
                    and isinstance(seed, int)
                    and not isinstance(seed, bool)
                    and seed > 0
                    and isinstance(minimum_distance, (int, float))
                    and not isinstance(minimum_distance, bool)
                    and minimum_distance >= 0.0
                    and isinstance(random_orientations, bool)
                    and isinstance(maximum_attempts, int)
                    and not isinstance(maximum_attempts, bool)
                    and maximum_attempts > 0
                    else "failed"
                ),
                "expected": (
                    "positive sample_count and seed, non-negative "
                    "minimum_distance_A, positive maximum_attempts_per_sample, "
                    "boolean random_orientations"
                ),
                "actual": dataset,
            }
        )
    failed = [check for check in checks if check["status"] == "failed"]
    if failed:
        return {
            "status": "unsupported",
            "current_module_capability": "invalid_potential_benchmark_configuration",
            "can_attempt_simulation": False,
            "requires_user_confirmation": False,
            "recommended_module": "C",
            "checks": checks,
            "reason": "The static Module C potential benchmark configuration is invalid.",
        }
    return {
        "status": "supported",
        "current_module_capability": "module_c_static_potential_comparison",
        "can_attempt_simulation": True,
        "requires_user_confirmation": False,
        "recommended_module": "C",
        "checks": checks,
        "reason": "The static Module C potential comparison is executable.",
    }


def _is_module_c_gcmc_variant_benchmark(config: dict[str, Any]) -> bool:
    benchmark = config.get("benchmark", {})
    simulation = config.get("simulation", {})
    task = str(benchmark.get("task", "")).strip().casefold().replace("-", "_")
    method = str(simulation.get("method", "")).strip().casefold()
    framework = str(simulation.get("framework", "")).strip().casefold()
    variants = benchmark.get("variants", [])
    return (
        task == "potential_benchmark"
        and method == "gcmc"
        and framework == "rigid"
        and isinstance(variants, list)
        and len(variants) > 0
    )


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
