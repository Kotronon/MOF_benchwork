"""General multicomponent LAMMPS campaigns (schema 2); legacy states stay separate."""

from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import warnings

from modules.module_b_profiles import point, paper_profile, simplex_candidates
from modules.module_b_active_learning import _save


def identity(config):
    """Hash the normalized physical setup and bytes of all explicitly supplied assets."""
    from pipeline.planning import build_run_plan
    from pipeline.prepare import prepare_benchmark

    plan = prepare_benchmark(build_run_plan(config))
    setup = deepcopy(plan["parameters"])
    for key in (
        "temperature_K",
        "pressures_bar",
        "mixture",
        "seeds",
        "replicate_count",
        "production_steps",
        "equilibration_steps",
        "run_steps",
        "save_dumps",
        "save_restarts",
        "dump_every_steps",
        "restart_every_steps",
    ):
        setup.pop(key, None)

    def assets(value):
        if isinstance(value, dict):
            return {k: assets(v) for k, v in value.items()}
        if isinstance(value, list):
            return [assets(v) for v in value]
        if isinstance(value, str) and Path(value).is_file():
            return {"sha256": hashlib.sha256(Path(value).read_bytes()).hexdigest()}
        return value

    setup["inputs"] = assets(
        {k: v for k, v in plan["inputs"].items() if k != "reference_files"}
    )
    setup["guest_models"] = assets(config["adsorbates"].get("models", {}))
    return hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()


def initialize(
    config,
    path,
    *,
    case=None,
    space="P-X",
    mode="noise_aware",
    candidates=None,
    budget=None,
):
    from pipeline.config import normalize_config

    if Path(path).exists():
        raise FileExistsError(path)
    config = normalize_config(config)
    if config["simulation"]["framework"] != "rigid":
        raise ValueError("Flexible frameworks belong to Module D.")
    names = sorted(config["adsorbates"]["components"])
    if budget is not None and (isinstance(budget, bool) or budget < 1):
        raise ValueError("Training budget must be positive.")
    if len(names) < 2:
        raise ValueError("Module B requires at least two components.")
    if mode not in {"paper", "upstream_compat", "noise_aware"}:
        raise ValueError("Unknown algorithm profile.")
    profile = paper_profile(case, space, mode) if case else None
    if profile:
        material = config["material"]["name"].upper().replace("-", "").replace(" ", "")
        if material not in {"CUBTC", "HKUST1", "BASOLITEC300"}:
            raise ValueError(
                "Paper cases require Cu-BTC/HKUST-1; use a general campaign for other MOFs."
            )
        profile["uptake_unit"] = config.get("benchmark", {}).get("paper_uptake_unit")
    if profile and set(names) != set(profile["components"]):
        raise ValueError("Paper profile components do not match config.")
    if not profile and mode != "noise_aware":
        raise ValueError("Paper algorithms require a named paper profile.")
    if profile and candidates is not None:
        raise ValueError(
            "Paper grids are fixed; use noise_aware without a paper case for custom grids."
        )
    candidates = (
        candidates
        if candidates is not None
        else (
            profile["candidates"]
            if profile
            else simplex_candidates(
                names,
                config["conditions"]["pressures_bar"],
                config["conditions"].get(
                    "temperatures_K", [config["conditions"]["temperature_K"]]
                ),
            )
        )
    )
    candidates = [
        point(p["pressure_bar"], p["temperature_K"], p["mixture"]) for p in candidates
    ]
    if any(set(p["mixture"]) != set(names) for p in candidates) or len(
        {p["candidate_id"] for p in candidates}
    ) != len(candidates):
        raise ValueError("Candidate components must match and states must be unique.")
    initial = profile["initial_design"] if profile else []
    import numpy as np

    eligible = [
        p
        for p in candidates
        if p["candidate_id"] not in {q["candidate_id"] for q in initial}
    ]
    holdout_n = profile["holdout_count"] if profile else min(64, len(eligible) // 5)
    held = set(
        np.random.default_rng(2026)
        .choice(len(eligible), holdout_n, replace=False)
        .tolist()
    )
    holdout = [p for i, p in enumerate(eligible) if i in held]
    from importlib.metadata import version, PackageNotFoundError

    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "ase", "CoolProp", "lammps"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    state = {
        "schema_version": 2,
        "status": "ready",
        "base_config": config,
        "model_identity": identity(config),
        "components": names,
        "profile": (
            {
                k: v
                for k, v in profile.items()
                if k not in {"candidates", "initial_design"}
            }
            if profile
            else None
        ),
        "mode": mode,
        "candidates": candidates,
        "initial_design": initial,
        "holdout": holdout,
        "results": [],
        "holdout_results": [],
        "imports": [],
        "history": [],
        "pending_request": None,
        "options": {
            "random_seed": 2026,
            "max_training_points": budget,
            "max_precision_retries": 1,
            "absolute_loading_floor": 0.05,
            "relative_uncertainty_target": 0.2,
        },
        "environment": {"python": platform.python_version(), **versions},
        "physical_validation": "open",
    }
    _save(path, state)
    return state


def _value(row, component, field="mean_molecules"):
    return row["components"][component][field]


def features(points, domain, component=None, paper=False):
    import numpy as np

    names = sorted(domain[0]["mixture"])
    varying_t = len({p["temperature_K"] for p in domain}) > 1

    def values(p):
        return (
            [math.log10(p["pressure_bar"])]
            + ([math.log10(p["temperature_K"])] if varying_t else [])
            + ([p["mixture"][component]] if paper else [p["mixture"][c] for c in names])
        )

    d, x = np.array([values(p) for p in domain]), np.array([values(p) for p in points])
    if paper:
        for j in range(1 + int(varying_t)):
            axis = np.unique(d[:, j])
            x[:, j] = (x[:, j] - axis.mean()) / max(float(axis.std(ddof=1)), 1e-12)
        x[:, -1] = (x[:, -1] - 0.5) * 25 / 12
    else:
        x = (x - d.min(axis=0)) / np.maximum(np.ptp(d, axis=0), 1e-12)
    return x


def predict(state, points):
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        RationalQuadratic,
        Matern,
        ConstantKernel,
    )

    train, domain = state["results"], state["candidates"]
    paper = state["mode"] != "noise_aware"
    output = {"components": {}, "kernels": {}, "warnings": []}
    profile = state.get("profile") or {}
    for c in state["components"]:
        x, xp = features(train, domain, c, paper), features(points, domain, c, paper)
        q = np.array([_value(r, c) for r in train])
        if (q < 0).any() or not np.isfinite(q).all():
            raise ValueError("Nonnegative finite observed loadings required.")
        if paper:
            unit = profile.get("uptake_unit")
            if unit not in {"mol/kg", "molecules/cell"}:
                raise ValueError(
                    "Set benchmark.paper_uptake_unit from verified source metadata before using log-space PAC."
                )
            conversion = 1.0
            if unit == "mol/kg":
                masses = [r.get("framework_mass_amu") for r in train]
                if any(m is None or m <= 0 for m in masses) or not np.allclose(
                    masses, masses[0], rtol=1e-10
                ):
                    raise ValueError(
                        "Paper uptake conversion requires consistent framework masses."
                    )
                conversion = 1000.0 / masses[0]
            q = q * conversion
            y = np.log10(np.where(q == 0, 1e-4, q))
            rbf = lambda: RBF(1.0, (1e-15, 1e15))
            kernel = {
                "rbf": lambda: rbf(),
                "rq": lambda: RationalQuadratic(1.0, 0.5, (1e-15, 1e15), (1e-15, 1e15)),
                "sum3_rbf": lambda: rbf() + rbf() + rbf(),
            }[profile["kernel"]]()
            alpha, restarts, offset, scale = (
                profile["alpha"],
                profile["optimizer_restarts"],
                0.0,
                1.0,
            )
        else:
            if any(
                r.get("replicate_count", 0) < 3
                or _value(r, c, "standard_error_molecules") is None
                for r in train
            ):
                raise ValueError(
                    "Noise-aware GP needs >=3 independent seeds at every training point."
                )
            y = np.log1p(q)
            offset, scale = float(y.mean()), max(float(y.std()), 0.1)
            y = (y - offset) / scale
            se = np.array([_value(r, c, "standard_error_molecules") for r in train])
            alpha = np.maximum((se / (1 + q) / scale) ** 2, 1e-8)
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                [0.5] * x.shape[1], (0.05, 5), nu=2.5
            )
            restarts = 1
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=paper,
            n_restarts_optimizer=restarts,
            random_state=state["options"]["random_seed"],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gp.fit(x, y)
        mu, sd = gp.predict(xp, return_std=True)
        mu, sd = mu * scale + offset, sd * scale
        transform = (
            (lambda a: np.power(10.0, np.clip(a, -300, 300)))
            if paper
            else (lambda a: np.maximum(0, np.expm1(np.clip(a, -700, 50))))
        )
        med, lo, hi = (
            transform(mu),
            transform(mu - 1.96 * sd),
            transform(mu + 1.96 * sd),
        )
        if paper:
            med, lo, hi = med / conversion, lo / conversion, hi / conversion
        output["components"][c] = [
            {
                "value": float(a),
                "lower": float(b),
                "upper": float(d),
                "transformed_mean": float(m),
                "transformed_std": float(s),
                "paper_relative_uncertainty": float(s / abs(m)) if m else None,
            }
            for a, b, d, m, s in zip(med, lo, hi, mu, sd)
        ]
        for index, p in enumerate(points):
            if p["mixture"][c] == 0:
                output["components"][c][index].update(value=0.0, lower=0.0, upper=0.0)
        output["kernels"][c] = str(gp.kernel_)
        output["warnings"].extend(str(w.message) for w in caught)
    return output


def recommend(state):
    import numpy as np

    if state["pending_request"]:
        return {"status": "awaiting_simulation", "request": state["pending_request"]}
    results, options = state["results"], state["options"]
    seen = {r["candidate_id"] for r in results}
    held = {r["candidate_id"] for r in state["holdout"]}
    pending = [
        r
        for r in state["initial_design"] + state["candidates"]
        if r["candidate_id"] not in seen | held
    ]
    initial = [r for r in state["initial_design"] if r["candidate_id"] not in seen]
    retry = [
        r
        for r in results
        if not r["precision_met"]
        and r.get("precision_retries", 0) < options["max_precision_retries"]
    ]
    if retry:
        return {
            "status": "request",
            "action": "improve_precision",
            "candidate": retry[0],
        }
    budget = options["max_training_points"]
    if budget is not None and len(results) >= budget:
        return {"status": "budget_exhausted", "physical_validation": "open"}
    if not pending:
        return {"status": "candidate_pool_exhausted"}
    if initial or len(results) < 3:
        if initial:
            selected = initial[0]
        elif not results:
            selected = pending[0]
        else:
            x = features(pending, state["candidates"])
            observed = features(results, state["candidates"])
            selected = pending[
                int(
                    np.argmax(
                        np.linalg.norm(
                            x[:, None, :] - observed[None, :, :], axis=2
                        ).min(axis=1)
                    )
                )
            ]
        return {"status": "request", "action": "sample", "candidate": selected}
    prediction = predict(state, state["candidates"])
    components = prediction["components"]
    if state["mode"] == "noise_aware":
        scores = [
            max(
                (components[c][i]["upper"] - components[c][i]["lower"])
                / 2
                / max(components[c][i]["value"], options["absolute_loading_floor"])
                for c in state["components"]
                if p["mixture"][c] > 0
            )
            for i, p in enumerate(state["candidates"])
        ]
        met = max(scores) <= options["relative_uncertainty_target"]
        status = "surrogate_uncertainty_target_met"
    else:
        relative = {
            c: [
                (
                    p["paper_relative_uncertainty"]
                    if p["paper_relative_uncertainty"] is not None
                    else math.inf
                )
                for p in rows
            ]
            for c, rows in components.items()
        }
        pac = {
            c: sum(x <= state["profile"]["beta"] for x in values) / len(values)
            for c, values in relative.items()
        }
        prediction["pac"] = pac
        met = (
            all(x >= 0.9 for x in pac.values())
            if state["mode"] == "paper"
            else all(max(v) < 0.005 for v in relative.values())
        )
        # Upstream chooses the worst relative-uncertainty component, then max absolute sigma.
        chosen = max(
            reversed(state["profile"]["components"]), key=lambda c: max(relative[c])
        )
        scores = [p["transformed_std"] for p in components[chosen]]
        status = "pac_target_met" if state["mode"] == "paper" else "upstream_target_met"
    prediction["scores"] = scores
    if met:
        return {
            "status": (
                status
                if all(r["precision_met"] for r in results)
                else "simulation_precision_unmet"
            ),
            "model": prediction,
        }
    available = [
        (i, p)
        for i, p in enumerate(state["candidates"])
        if p["candidate_id"] not in seen | held
    ]
    if not available:
        return {"status": "candidate_pool_exhausted", "model": prediction}
    i, p = max(available, key=lambda ip: scores[ip[0]])
    return {
        "status": "request",
        "action": "sample",
        "candidate": p,
        "model": prediction,
    }


def propose(path, *, holdout=False):
    path = Path(path)
    state = json.loads(path.read_text())
    if state["schema_version"] != 2:
        from modules.module_b_active_learning import propose_campaign

        return propose_campaign(path)
    if state["pending_request"]:
        return {"status": "awaiting_simulation", "request": state["pending_request"]}
    if identity(state["base_config"]) != state["model_identity"]:
        raise ValueError("Simulation assets changed since campaign initialization.")
    if holdout:
        done = {r["candidate_id"] for r in state["holdout_results"]}
        pending = [r for r in state["holdout"] if r["candidate_id"] not in done]
        decision = (
            {"status": "request", "action": "holdout", "candidate": pending[0]}
            if pending
            else {"status": "holdout_completed"}
        )
    else:
        decision = recommend(state)
    if decision["status"] == "request":
        p = decision["candidate"]
        cfg = deepcopy(state["base_config"])
        cfg["conditions"].update(
            temperature_K=p["temperature_K"], pressures_bar=[p["pressure_bar"]]
        )
        cfg["adsorbates"]["mixture"] = p["mixture"]
        if decision["action"] == "improve_precision":
            factor = 4 ** (p.get("precision_retries", 0) + 1)
            cfg["simulation"]["cycles"] = (
                int(cfg["simulation"].get("cycles", 10000)) * factor
            )
            cfg["simulation"]["initialization_cycles"] = (
                int(cfg["simulation"].get("initialization_cycles", 0)) * factor
            )
        run_id = f"al_{len(state['history']):05d}_{p['candidate_id']}"
        cfg["output"].update(run_id=run_id, overwrite=False, resume=True)
        target = path.parent / "requests" / (run_id + ".json")
        _save(target, cfg)
        request = {
            "candidate": {
                k: p[k]
                for k in ("candidate_id", "pressure_bar", "temperature_K", "mixture")
            },
            "action": decision["action"],
            "config_path": str(target.resolve()),
        }
        state["pending_request"] = request
        state["history"].append({"request": request})
        decision["request"] = request
    state["status"] = decision["status"]
    if "model" in decision:
        state["history"].append(
            {
                "iteration": len(results) if (results := state.get("results")) else 0,
                "status": decision["status"],
                "pac": decision["model"].get("pac"),
                "kernels": decision["model"]["kernels"],
            }
        )
    _save(path, state)
    _save(path.parent / "acquisition_report.json", decision)
    return decision


def ingest(path, summary_path):
    from pipeline.mixture_evaluation import corrected_summary, _prepare

    path = Path(path)
    state = json.loads(path.read_text())
    summary_path = Path(summary_path)
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    if digest in state["imports"]:
        return state
    request = state["pending_request"]
    if not request:
        raise ValueError("No pending request to ingest.")
    if json.loads(summary_path.read_text()).get("status") != "completed":
        raise ValueError("Only completed simulations may be imported.")
    cfg = json.loads(Path(request["config_path"]).read_text())
    if identity(cfg) != state["model_identity"]:
        raise ValueError("Model identity mismatch.")
    prep = _prepare(summary_path)
    params = prep["parameters"]
    expected = request["candidate"]
    actual = point(expected["pressure_bar"], params["temperature_K"], params["mixture"])
    if actual["candidate_id"] != expected["candidate_id"]:
        raise ValueError("Imported state does not match pending request.")
    # Require the actual run's preparation to match, not only the request file.
    from pipeline.planning import build_run_plan
    from pipeline.prepare import prepare_benchmark

    planned = prepare_benchmark(build_run_plan(cfg))
    for key in (
        "material_id",
        "charge_scheme",
        "components",
        "unit_cells",
        "cell_representation",
        "forcefield",
        "cutoff_A",
        "pair_style",
        "kspace_style",
        "kspace_accuracy",
        "pair_modify_shift",
        "reservoir",
        "guest_models",
        "sampling",
        "framework",
    ):
        if params.get(key) != planned["parameters"].get(key):
            raise ValueError(f"Imported simulation context differs: {key}")
    if prep["inputs"] != planned["inputs"]:
        raise ValueError("Imported input assets differ from request.")
    provenance = (
        json.loads((summary_path.parent / "prepare_summary.json").read_text())
        .get("result", {})
        .get("files", {})
        .get("source_sha256", {})
    )
    if not provenance:
        raise ValueError("New campaign imports require archived input checksums.")
    for asset, digest_expected in provenance.items():
        if (
            not Path(asset).is_file()
            or hashlib.sha256(Path(asset).read_bytes()).hexdigest() != digest_expected
        ):
            raise ValueError("Archived simulation input changed or is missing.")
    rows = corrected_summary(summary_path)["results"]
    if len(rows) != 1 or not math.isclose(
        rows[0]["pressure_bar"], expected["pressure_bar"], rel_tol=1e-10
    ):
        raise ValueError("Expected one matching pressure point.")
    row = rows[0]
    field = "holdout_results" if request["action"] == "holdout" else "results"
    previous = next(
        (x for x in state[field] if x["candidate_id"] == expected["candidate_id"]), None
    )
    if previous:
        state["history"].append({"replaced_observation": previous})
    from pipeline.mixture_evaluation import framework_mass

    mass, mass_source = framework_mass(summary_path)
    record = {
        **expected,
        "framework_mass_amu": mass,
        "framework_mass_source": mass_source,
        "components": row["summary"]["components"],
        "selectivity": row["summary"]["selectivity"],
        "total": {
            key: row["summary"].get(source)
            for key, source in {
                "value": "mean_adsorbates",
                "ci95_lower": "ci95_lower_adsorbates",
                "ci95_upper": "ci95_upper_adsorbates",
            }.items()
        },
        "precision_met": row["converged"],
        "replicate_count": row["replicate_count"],
        "seeds": row["seeds"],
        "total_wall_time_seconds": row.get("total_wall_time_seconds"),
        "source_summary": str(summary_path.resolve()),
        "precision_retries": (
            previous.get("precision_retries", 0) + 1 if previous else 0
        ),
    }
    state[field] = [
        r for r in state[field] if r["candidate_id"] != expected["candidate_id"]
    ] + [record]
    state["imports"].append(digest)
    state["pending_request"] = None
    state["status"] = "ready"
    _save(path, state)
    return state


def execute(path, max_requests, *, holdout=False):
    from pipeline.config import normalize_config
    from pipeline.planning import build_run_plan
    from pipeline.prepare import prepare_benchmark
    from pipeline.materialization import materialize_benchmark
    from pipeline.runners import run_isotherm

    if max_requests <= 0:
        raise ValueError("An explicit positive execution budget is required.")
    state = json.loads(Path(path).read_text())
    if state["options"]["max_training_points"] is None:
        raise ValueError(
            "Set an explicit total training-point budget before production execution."
        )
    if state.get("profile") and not state["base_config"].get("benchmark", {}).get(
        "paper_model_verified", False
    ):
        raise ValueError(
            "Paper simulation model is not verified; complete its provenance manifest first."
        )
    for _ in range(max_requests):
        decision = propose(path, holdout=holdout)
        if decision["status"] not in {"request", "awaiting_simulation"}:
            return decision
        request = decision["request"]
        try:
            cfg = normalize_config(json.loads(Path(request["config_path"]).read_text()))
            materialized = materialize_benchmark(prepare_benchmark(build_run_plan(cfg)))
            result = run_isotherm(materialized)
            ingest(path, result["summary_file"])
        except Exception as exc:
            state = json.loads(Path(path).read_text())
            state["status"] = "technical_error"
            state["last_error"] = str(exc)
            _save(path, state)
            raise
    state = json.loads(Path(path).read_text())
    state["status"] = "execution_budget_exhausted"
    _save(path, state)
    return {"status": "execution_budget_exhausted"}


def report(path):
    import numpy as np
    from itertools import combinations
    from modules.module_b_reporting import ratio, write_artifacts

    state = json.loads(Path(path).read_text())
    test = state["holdout_results"]
    results = state["results"]
    times = [r.get("total_wall_time_seconds") for r in results]
    result = {
        "physical_validation": "open",
        "training_points": len(results),
        "holdout_points": len(test),
        "metrics": {},
        "seed_runs": sum(r["replicate_count"] for r in results),
        "training_wall_time_seconds": (
            sum(times) if times and all(t is not None for t in times) else None
        ),
        "interpretation": "GP validation against independent LAMMPS labels; external paper agreement is a separate assessment.",
    }
    pred = predict(state, state["candidates"]) if len(results) >= 3 else None
    if pred and test:
        index = {p["candidate_id"]: i for i, p in enumerate(state["candidates"])}
        indices = [index[r["candidate_id"]] for r in test]
        for c in state["components"]:
            y = np.array([_value(r, c) for r in test])
            estimates = [pred["components"][c][i] for i in indices]
            p = np.array([r["value"] for r in estimates])
            e = p - y
            denom = np.sum((y - y.mean()) ** 2)
            unit = (state.get("profile") or {}).get("uptake_unit") or "molecules/cell"
            scale = (
                np.array([1000 / r["framework_mass_amu"] for r in test])
                if unit == "mol/kg"
                else 1.0
            )
            result["metrics"][c] = {
                "mae_molecules_per_cell": float(np.mean(abs(e))),
                "rmse_molecules_per_cell": float(np.sqrt(np.mean(e**2))),
                "mre_percent": float(
                    100 * np.mean(abs(e * scale) / (y * scale + 1e-3))
                ),
                "mre_denominator_floor": 1e-3,
                "mre_floor_unit": unit,
                "r2": float(1 - np.sum(e**2) / denom) if denom > 0 else None,
                "latent_interval_coverage": float(
                    np.mean(
                        [r["lower"] <= v <= r["upper"] for r, v in zip(estimates, y)]
                    )
                ),
            }
        for a, b in combinations(state["components"], 2):
            errors = []
            for r, i in zip(test, indices):
                actual = ratio(
                    _value(r, a), _value(r, b), r["mixture"][a], r["mixture"][b]
                )
                estimate = ratio(
                    pred["components"][a][i]["value"],
                    pred["components"][b][i]["value"],
                    r["mixture"][a],
                    r["mixture"][b],
                )
                if actual is not None and estimate is not None:
                    errors.append(estimate - actual)
            result["metrics"][f"{a}/{b}"] = {
                "mae": float(np.mean(np.abs(errors))) if errors else None,
                "valid_points": len(errors),
                "undefined_points": len(test) - len(errors),
                "gp_ratio_interval": "cross_component_covariance_not_modelled",
            }
    target = Path(path).parent / "report"
    result["artifacts"] = write_artifacts(state, pred, target)
    _save(target / "validation_report.json", result)
    lines = [
        "# Module B campaign",
        "",
        f"Training states: {len(results)}; independent holdouts: {len(test)}.",
        "",
        "Quantitative external reference validation remains open.",
        "",
        "Seed intervals measure statistical precision, not proof of equilibrium. GP bands describe latent surrogate uncertainty.",
        "",
        "Higher-dimensional figures are projections, not complete adsorption surfaces.",
        "",
        "See observations.csv, surrogate_predictions.csv and validation_report.json for reproducible values.",
    ]
    (target / "report.md").write_text("\n".join(lines) + "\n")
    # Learning curves use holdouts for reporting only; acquisition never reads these metrics.
    curve = target / "learning_curve.json"
    previous = json.loads(curve.read_text()) if curve.exists() else []
    previous = [r for r in previous if r["training_points"] != len(results)] + [result]
    _save(curve, previous)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("config")
    init.add_argument("state")
    init.add_argument("--case", choices=["CO2-CH4", "Xe-Kr", "H2S-CO2"])
    init.add_argument("--space", default="P-X", choices=["P-X", "P-X-T"])
    init.add_argument(
        "--mode",
        default="noise_aware",
        choices=["paper", "upstream_compat", "noise_aware"],
    )
    init.add_argument("--budget", type=int)
    for command in ("propose", "run", "ingest", "report"):
        p = sub.add_parser(command)
        p.add_argument("state")
        if command in {"propose", "run"}:
            p.add_argument("--holdout", action="store_true")
        if command == "run":
            p.add_argument("--max-requests", type=int, required=True)
        if command == "ingest":
            p.add_argument("summary")
    args = parser.parse_args(argv)
    if args.command == "init":
        result = initialize(
            json.loads(Path(args.config).read_text()),
            args.state,
            case=args.case,
            space=args.space,
            mode=args.mode,
            budget=args.budget,
        )
        result = {"status": result["status"], "state": args.state}
    elif args.command == "propose":
        result = propose(args.state, holdout=args.holdout)
    elif args.command == "run":
        result = execute(args.state, args.max_requests, holdout=args.holdout)
    elif args.command == "ingest":
        result = ingest(args.state, args.summary)
        result = {"status": result["status"]}
    else:
        result = report(args.state)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
