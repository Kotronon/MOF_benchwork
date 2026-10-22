"""Noise-aware, fixed-temperature CO2/N2 active learning over pressure/composition.

This is an adaptation of the GP/GCMC idea in doi:10.1039/D3DD00106G,
not an exact reproduction of its kernels or stopping rule. Simulation execution
is explicit: propose writes a normal benchmark config; ingest closes the loop.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

from modules.module_b_selectivity import build_candidate_pool, mixture_results_from_isotherm_summary
from pipeline.mixture_evaluation import corrected_summary, _prepare

DEFAULTS = {"relative_uncertainty_target": 0.2, "absolute_loading_floor": 0.05,
            "max_training_points": 24, "max_precision_retries": 1, "random_seed": 2026}
CONTEXT_KEYS = ("material_id", "charge_scheme", "temperature_K", "unit_cells", "cell_representation",
                "cutoff_A", "forcefield", "pair_style", "pair_modify_shift", "kspace_style", "kspace_accuracy")


def _save(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _context(summary_path: str | Path) -> dict[str, Any]:
    prep = _prepare(Path(summary_path))
    parameters = prep.get("parameters", {})
    if not all(parameters.get(k) is not None for k in ("material_id", "temperature_K", "forcefield")):
        raise ValueError("Campaign imports require prepare_summary.json with material, temperature and force-field metadata.")
    context = {k: parameters.get(k) for k in CONTEXT_KEYS}
    # Capture exact input assets when available, otherwise retain recorded paths.
    inputs = prep.get("inputs", {})
    assets = {"framework_cif": inputs.get("framework_cif"), **inputs.get("forcefield_files", {}),
              **inputs.get("adsorbate_definitions", {})}
    context["assets"] = {key: {"path": str(value), "sha256": hashlib.sha256(Path(value).read_bytes()).hexdigest() if value and Path(value).is_file() else None}
                         for key, value in assets.items()}
    return context


def _validate_observation(row: dict[str, Any]) -> bool:
    if int(row.get("replicate_count") or 0) < 3:
        return False
    for c in ("CO2", "N2"):
        for key in (f"mean_{c}", f"standard_error_{c}"):
            value = row.get(key)
            if value is None or not math.isfinite(float(value)) or value < 0:
                return False
    return True


def initialize_campaign(config: dict[str, Any], summary_path: str | Path, state_path: str | Path,
                        pressures: list[float] | None = None, fractions: list[float] | None = None,
                        options: dict[str, Any] | None = None) -> dict[str, Any]:
    if Path(state_path).exists():
        raise FileExistsError(f"Campaign already exists: {state_path}. Use ingest/propose to resume.")
    if set(config.get("adsorbates", {}).get("components", [])) != {"CO2", "N2"}:
        raise ValueError("Module B active learning currently supports CO2/N2 only.")
    pressure_values = pressures or config["conditions"]["pressures_bar"]
    fraction_values = fractions or [0.05, 0.15, 0.5, 0.85]
    if len(set(pressure_values)) < 2 or any(not math.isfinite(p) or p <= 0 for p in pressure_values):
        raise ValueError("Provide at least two distinct positive pressures.")
    if len(set(fraction_values)) < 2 or any(not math.isfinite(y) or not 0 < y < 1 for y in fraction_values):
        raise ValueError("Provide at least two distinct interior mole fractions.")
    context = _context(summary_path)
    if not math.isclose(float(config["conditions"]["temperature_K"]), context["temperature_K"], abs_tol=1e-8):
        raise ValueError("Base config and imported run must have the same temperature.")
    # Preparing is side-effect free and verifies that exported requests reproduce the imported setup.
    from pipeline.planning import build_run_plan
    from pipeline.prepare import prepare_benchmark
    planned = prepare_benchmark(build_run_plan(config))["parameters"]
    if any(planned.get(k) != context.get(k) for k in CONTEXT_KEYS):
        raise ValueError("Base config does not reproduce the imported run's material/simulation setup.")
    settings = {**DEFAULTS, **(options or {})}
    if settings["relative_uncertainty_target"] <= 0 or settings["absolute_loading_floor"] <= 0 or settings["max_training_points"] < 3 or settings["max_precision_retries"] < 0:
        raise ValueError("Invalid AL uncertainty/budget settings.")
    state = {"schema_version": 1, "status": "initialized", "strategy": "noise_aware_component_gp",
             "base_config": deepcopy(config), "context": context, "options": settings,
             "candidates": build_candidate_pool(sorted(set(pressure_values)), sorted(set(fraction_values))),
             "results": [], "imports": [], "history": [], "pending_request": None,
             "physical_validation": "open"}
    _ingest(state, summary_path)
    _save(state_path, state)
    return state


def _ingest(state: dict[str, Any], summary_path: str | Path) -> None:
    summary_path = Path(summary_path)
    if json.loads(summary_path.read_text()).get("status") != "completed":
        raise ValueError("Only completed GCMC run summaries can be ingested.")
    fingerprint = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    if any(item["sha256"] == fingerprint and item["path"] == str(summary_path.resolve()) for item in state["imports"]):
        return
    if _context(summary_path) != state["context"]:
        raise ValueError("Run context differs: do not mix materials, force fields, cells or temperatures.")
    summary = corrected_summary(summary_path)
    rows = mixture_results_from_isotherm_summary(summary)
    pending = state.get("pending_request")
    if pending and {r["candidate_id"] for r in rows} != {pending["candidate_id"]}:
        raise ValueError("Imported run does not match the pending candidate.")
    for row in rows:
        if set(row["mixture"]) != {"CO2", "N2"} or not math.isclose(sum(row["mixture"].values()), 1.0, abs_tol=1e-8):
            raise ValueError("Expected a normalized binary CO2/N2 mixture.")
        if not any(c["candidate_id"] == row["candidate_id"] for c in state["candidates"]):
            raise ValueError(f"Observation outside candidate pool: {row['candidate_id']}")
        previous = next((r for r in state["results"] if r["candidate_id"] == row["candidate_id"]), None)
        row["precision_retries"] = (previous.get("precision_retries", 0) + 1) if previous else 0
        row["source_summary"] = str(summary_path.resolve())
        row["usable_for_gp"] = _validate_observation(row)
        if previous:
            state["history"].append({"replaced_observation": previous})
        state["results"] = [r for r in state["results"] if r["candidate_id"] != row["candidate_id"]] + [row]
    state["imports"].append({"path": str(summary_path.resolve()), "sha256": fingerprint})
    state["pending_request"] = None
    state["status"] = "ready"


def ingest_campaign(state_path: str | Path, summary_path: str | Path) -> dict[str, Any]:
    state = json.loads(Path(state_path).read_text())
    if state.get("schema_version") == 2:
        from modules.module_b_campaign import ingest
        return ingest(state_path, summary_path)
    _ingest(state, summary_path)
    _save(state_path, state)
    return state


def _features(points: list[dict[str, Any]], domain: list[dict[str, Any]]):
    import numpy as np
    bounds = np.array([[math.log10(p["pressure_bar"]), p["mixture"]["CO2"]] for p in domain])
    span = np.maximum(np.ptp(bounds, axis=0), 1e-12)
    values = np.array([[math.log10(p["pressure_bar"]), p["mixture"]["CO2"]] for p in points])
    return (values - bounds.min(axis=0)) / span


def predict_components(training: list[dict[str, Any]], points: list[dict[str, Any]],
                       domain: list[dict[str, Any]], seed: int = 2026) -> dict[str, Any]:
    import numpy as np
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    except ImportError as exc:
        raise ImportError("Module B AL requires scikit-learn: pip install -r requirements-module-b-al.txt") from exc
    if len(training) < 3 or not all(_validate_observation(r) for r in training):
        raise ValueError("GP fitting requires at least three points with >=3 seeds and finite component errors.")
    train_x, test_x = _features(training, domain), _features(points, domain)
    output = {"components": {}, "kernels": {}, "warnings": []}
    for c in ("CO2", "N2"):
        q = np.array([r[f"mean_{c}"] for r in training], dtype=float)
        se = np.array([r[f"standard_error_{c}"] for r in training], dtype=float)
        transformed = np.log1p(q)
        offset, scale = float(transformed.mean()), max(float(transformed.std()), 0.1)
        # Heteroscedastic variance of seed means, propagated through log1p and scaling.
        alpha = np.maximum((se / (1 + q) / scale) ** 2, 1e-8)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern([0.5, 0.5], (0.05, 5.0), nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=False,
                                      n_restarts_optimizer=1, random_state=seed)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            gp.fit(train_x, (transformed - offset) / scale)
        output["warnings"].extend(str(w.message) for w in captured)
        mu, sigma = gp.predict(test_x, return_std=True)
        mu, sigma = mu * scale + offset, sigma * scale
        # Back-transformed latent median, not an unbiased arithmetic mean.
        output["components"][c] = [{"value": max(0., float(np.expm1(np.clip(m, -700, 50)))) ,
                                     "lower": max(0., float(np.expm1(np.clip(m - 1.96*s, -700, 50)))),
                                     "upper": max(0., float(np.expm1(np.clip(m + 1.96*s, -700, 50)))),
                                     "log1p_mean": float(m), "log1p_std": float(s)} for m, s in zip(mu, sigma)]
        output["kernels"][c] = str(gp.kernel_)
    return output


def _maximin(pending, observed, domain):
    import numpy as np
    x = _features(pending, domain)
    if not observed:
        return pending[0]
    distances = np.linalg.norm(x[:, None, :] - _features(observed, domain)[None, :, :], axis=2).min(axis=1)
    return pending[int(np.argmax(distances))]


def recommend(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("schema_version") == 2:
        from modules.module_b_campaign import recommend as recommend_general
        return recommend_general(state)
    options, candidates, observations = state["options"], state["candidates"], state["results"]
    if state.get("pending_request"):
        return {"status": "awaiting_simulation", "request": state["pending_request"]}
    # Simulation noise and epistemic GP uncertainty are different decisions.
    retry = [r for r in observations if not r.get("converged") and r.get("precision_retries", 0) < options["max_precision_retries"]]
    if retry:
        row = min(retry, key=lambda r: (r["pressure_bar"], r["mixture"]["CO2"]))
        return {"status": "request", "action": "improve_precision", "candidate": row,
                "reason": "Component/selectivity seed precision failed; extend sampling before expanding the landscape."}
    completed = {r["candidate_id"] for r in observations}
    pending = [c for c in candidates if c["candidate_id"] not in completed]
    usable = [r for r in observations if _validate_observation(r)]
    if len(observations) >= options["max_training_points"] or not pending:
        return {"status": "budget_exhausted" if pending else "candidate_pool_exhausted", "physical_validation": "open"}
    if len(usable) < 3 or len({r["mixture"]["CO2"] for r in usable}) < 2:
        return {"status": "request", "action": "sample", "candidate": _maximin(pending, usable, candidates),
                "reason": "Initial coverage in scaled log-pressure/composition space is incomplete."}
    prediction = predict_components(usable, candidates, candidates, options["random_seed"])
    scores, surfaces = {}, []
    for i, candidate in enumerate(candidates):
        components = {c: prediction["components"][c][i] for c in ("CO2", "N2")}
        score = max((s["upper"] - s["lower"]) / 2 / max(s["value"], options["absolute_loading_floor"]) for s in components.values())
        a, b = components["CO2"], components["N2"]
        gas_ratio = candidate["mixture"]["N2"] / candidate["mixture"]["CO2"]
        # No independence assumption is used to claim a calibrated ratio CI.
        selectivity = a["value"] / b["value"] * gas_ratio if b["value"] > 0 else None
        scores[candidate["candidate_id"]] = score
        surfaces.append({**candidate, "components": components, "acquisition_score": score,
                         "selectivity_CO2_N2": selectivity,
                         "selectivity_interval": None, "selectivity_interval_reason": "cross_component_GP_covariance_not_modelled"})
    model_info = {"predictions": surfaces, "kernels": prediction["kernels"], "warnings": prediction["warnings"],
                  "prediction_interpretation": "Back-transformed log1p posterior median; intervals describe latent GP uncertainty, not GCMC convergence."}
    if max(scores.values()) <= options["relative_uncertainty_target"] and all(r.get("converged") for r in observations):
        return {"status": "surrogate_uncertainty_target_met", "physical_validation": "open", **model_info}
    selected = max(pending, key=lambda c: scores[c["candidate_id"]])
    return {"status": "request", "action": "sample", "candidate": selected,
            "reason": "Largest normalized component GP interval on the candidate pool.", **model_info}


def propose_campaign(state_path: str | Path) -> dict[str, Any]:
    state_path = Path(state_path)
    state = json.loads(state_path.read_text())
    if state.get("schema_version") == 2:
        from modules.module_b_campaign import propose
        return propose(state_path)
    result = recommend(state)
    if result["status"] == "request":
        candidate = result["candidate"]
        request_index = len([x for x in state["history"] if "request" in x]) + 1
        config = deepcopy(state["base_config"])
        config["conditions"]["pressures_bar"] = [candidate["pressure_bar"]]
        config["adsorbates"]["mixture"] = candidate["mixture"]
        if result["action"] == "improve_precision":
            multiplier = 4 ** (candidate.get("precision_retries", 0) + 1)
            config["simulation"]["cycles"] *= multiplier
            config["simulation"]["initialization_cycles"] *= multiplier
        run_id = f"module_b_al_{request_index:03d}_{candidate['candidate_id']}"
        config["output"].update(run_id=run_id, overwrite=False)
        config["output"].pop("resume", None)
        request_path = state_path.parent / "requests" / f"{run_id}.json"
        _save(request_path, config)
        request = {"candidate_id": candidate["candidate_id"], "action": result["action"],
                   "config_path": str(request_path), "run_id": run_id,
                   "reason": result["reason"]}
        state["pending_request"] = request
        state["history"].append({"request": request})
        result["request"] = request
    state["status"] = result["status"]
    _save(state_path.parent / "acquisition_report.json", result)
    _save(state_path, state)
    return result


def compare_sampling_strategies(state: dict[str, Any], observations: list[dict[str, Any]],
                                holdout_ids: list[str], budgets: list[int], repeats: int = 5) -> dict[str, Any]:
    """Offline replay: holdout labels never enter fitting, scaling or acquisition.

    Budget unit is one pressure/composition point with its replicate bundle.
    Reports actual summed simulation times when recorded; equal point counts
    do not imply equal wall time.
    """
    import numpy as np
    if len({r["candidate_id"] for r in observations}) != len(observations):
        raise ValueError("Validation observations must have unique candidate IDs.")
    if not all(_validate_observation(r) and r.get("converged") for r in observations):
        raise ValueError("Offline validation requires precise observations with seed uncertainties.")
    held = set(holdout_ids)
    test = [r for r in observations if r["candidate_id"] in held]
    pool = [r for r in observations if r["candidate_id"] not in held]
    if len(test) != len(held) or len(test) < 2 or len(pool) < 4:
        raise ValueError("Need >=2 identified holdout points and >=4 independent training-pool points.")
    if len({r["mixture"]["CO2"] for r in pool}) < 2:
        raise ValueError("A landscape validation requires multiple training compositions.")
    budgets = sorted(set(budgets))
    if not budgets or budgets[0] < 3 or budgets[-1] > len(pool) or repeats < 1:
        raise ValueError("Budgets must be 3..training-pool size; repeats must be positive.")
    domain = state["candidates"]
    initial = [pool[0]]
    while len(initial) < 3:
        initial.append(_maximin([r for r in pool if r not in initial], initial, domain))
    reports = []
    for strategy in ("active_learning", "maximin", "random"):
        for repeat in range(repeats if strategy == "random" else 1):
            rng = np.random.default_rng(state["options"]["random_seed"] + repeat)
            chosen = list(initial)
            for size in range(3, budgets[-1] + 1):
                if size in budgets:
                    prediction = predict_components(chosen, test, domain, state["options"]["random_seed"])
                    metrics = {}
                    for c in ("CO2", "N2"):
                        residual = np.array([p["value"] - r[f"mean_{c}"] for p, r in zip(prediction["components"][c], test)])
                        metrics[c] = {"mae_molecules_per_cell": float(np.mean(np.abs(residual))),
                                      "rmse_molecules_per_cell": float(np.sqrt(np.mean(residual**2))),
                                      "latent_interval_coverage": float(np.mean([p["lower"] <= r[f"mean_{c}"] <= p["upper"]
                                                                                 for p, r in zip(prediction["components"][c], test)]))}
                    ratio_errors = []
                    for i, row in enumerate(test):
                        a, b = prediction["components"]["CO2"][i], prediction["components"]["N2"][i]
                        if b["value"] > 0 and row["mean_N2"] > 0:
                            factor = row["mixture"]["N2"] / row["mixture"]["CO2"]
                            ratio_errors.append(factor * (a["value"] / b["value"] - row["mean_CO2"] / row["mean_N2"]))
                    metrics["selectivity_CO2_N2"] = {"mae": float(np.mean(np.abs(ratio_errors))) if ratio_errors else None,
                                                      "valid_point_count": len(ratio_errors),
                                                      "undefined_point_count": len(test) - len(ratio_errors)}
                    times = [r.get("total_wall_time_seconds") for r in chosen]
                    reports.append({"strategy": strategy, "repeat": repeat, "point_budget": size,
                                    "seed_budget": sum(r["replicate_count"] for r in chosen),
                                    "simulation_time_seconds": sum(times) if all(t is not None for t in times) else None,
                                    "training_ids": [r["candidate_id"] for r in chosen], "metrics": metrics})
                if size == budgets[-1]:
                    break
                pending = [r for r in pool if r not in chosen]
                if strategy == "random":
                    selected = pending[int(rng.integers(len(pending)))]
                elif strategy == "maximin":
                    selected = _maximin(pending, chosen, domain)
                else:
                    prediction = predict_components(chosen, pending, domain, state["options"]["random_seed"])
                    scores = [max((prediction["components"][c][i]["upper"] - prediction["components"][c][i]["lower"]) / 2 /
                                  max(prediction["components"][c][i]["value"], state["options"]["absolute_loading_floor"]) for c in ("CO2", "N2")) for i in range(len(pending))]
                    selected = pending[int(np.argmax(scores))]
                chosen.append(selected)
    return {"status": "completed", "holdout_ids": holdout_ids, "results": reports,
            "budget_definition": "Equal sampled-state counts, same initial design; seed counts and runtime reported separately.",
            "physical_validation": "not_assessed"}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Initialize from a base config and an existing mixture run.")
    init.add_argument("config")
    init.add_argument("summary")
    init.add_argument("state")
    init.add_argument("--pressures", nargs="+", type=float)
    init.add_argument("--fractions", nargs="+", type=float)
    propose = sub.add_parser("propose", help="Write the next simulation request, without running GCMC.")
    propose.add_argument("state")
    ingest = sub.add_parser("ingest", help="Import a completed requested run and update the GP training data.")
    ingest.add_argument("state")
    ingest.add_argument("summary")
    validate = sub.add_parser("validate", help="Offline AL/maximin/random comparison on existing observations.")
    validate.add_argument("state")
    validate.add_argument("--holdout", nargs="+", required=True)
    validate.add_argument("--budgets", nargs="+", type=int, required=True)
    validate.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "init":
        state = initialize_campaign(json.loads(Path(args.config).read_text()), args.summary, args.state,
                                    pressures=args.pressures, fractions=args.fractions)
        result = {"status": state["status"], "state": args.state, "observations": len(state["results"])}
    elif args.command == "propose":
        result = propose_campaign(args.state)
    elif args.command == "ingest":
        state = ingest_campaign(args.state, args.summary)
        result = {"status": state["status"], "observations": len(state["results"])}
    else:
        state = json.loads(Path(args.state).read_text())
        result = compare_sampling_strategies(state, state["results"], args.holdout, args.budgets, args.repeats)
        _save(Path(args.state).parent / "validation_report.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
