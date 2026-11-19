"""Pinned algorithm presets; simulation physics must be supplied independently."""

from __future__ import annotations
import hashlib
import itertools
import json
import math

UPSTREAM_COMMIT = "0415e03f2a67657cf88d8964ad665177c54bbde8"
CASES = {
    "CO2-CH4": ("CO2", "CH4", 300.0),
    "Xe-Kr": ("XE", "KR", 200.0),
    "H2S-CO2": ("H2S", "CO2", 100.0),
}


def point(pressure, temperature, mixture):
    values = {
        "pressure_bar": float(pressure),
        "temperature_K": float(temperature),
        "mixture": {c: float(y) for c, y in sorted(mixture.items())},
    }
    if not all(
        math.isfinite(x) and x > 0
        for x in (values["pressure_bar"], values["temperature_K"])
    ):
        raise ValueError("Invalid state pressure/temperature.")
    if any(not math.isfinite(y) or y < 0 for y in mixture.values()) or not math.isclose(
        sum(mixture.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("Invalid composition.")
    # Canonical numerical serialization prevents order-dependent IDs.
    signature = [
        format(values["pressure_bar"], ".12g"),
        format(values["temperature_K"], ".12g"),
        [(c, format(y, ".12g")) for c, y in values["mixture"].items()],
    ]
    return {
        "candidate_id": hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:20],
        **values,
    }


def paper_profile(case, space="P-X", mode="paper"):
    import numpy as np

    if mode not in {"paper", "upstream_compat", "noise_aware"} or space not in {
        "P-X",
        "P-X-T",
    }:
        raise ValueError("Unknown paper space or algorithm.")
    a, b, pmax = CASES[case]
    pxt = space == "P-X-T"
    pressures = np.linspace(1e-6, pmax, 21 if pxt else 51)
    temperatures = np.linspace(200, 400, 11) if pxt else [300.0]
    fractions = np.linspace(0.02, 0.98, 11 if pxt else 49)
    candidates = [
        point(p, t, {a: x, b: 1 - x})
        for p, t, x in itertools.product(pressures, temperatures, fractions)
    ]
    initial = [
        point(p, t, {a: x, b: 1 - x})
        for p, t, x in itertools.product(
            [1e-6, 1e-4, 0.01, 1.0, 100.0] if pxt else [10.0**i for i in range(-6, 3)],
            [200.0, 300.0, 400.0] if pxt else [300.0],
            [0.02, 0.2, 0.4, 0.6, 0.8, 0.98],
        )
    ]
    kernel = "sum3_rbf" if pxt else ("rbf" if case == "CO2-CH4" else "rq")
    if case == "H2S-CO2" and pxt and mode != "upstream_compat":
        kernel = "rq"
    return {
        "case": case,
        "space": space,
        "mode": mode,
        "components": [a, b],
        "material": "Cu-BTC",
        "upstream_commit": UPSTREAM_COMMIT,
        "candidates": candidates,
        "initial_design": initial,
        "kernel": kernel,
        "alpha": 1e-5 if case == "Xe-Kr" else 1e-4,
        "optimizer_restarts": 1000 if pxt else 100,
        "beta": 0.05 if pxt else 0.02,
        "pac_target": 0.90,
        "holdout_count": 96 if pxt else 64,
        "uptake_unit": "requires_verified_original_unit",
        "physics_status": "requires_verified_structure_charges_forcefield_and_reservoir",
        "deviations": [
            "LAMMPS replaces RASPA",
            "holdouts withheld from acquisition",
            "fixed optimizer random seed",
            "duplicate acquisition prevented",
        ],
    }


def simplex_candidates(components, pressures, temperatures, count=4096, seed=2026):
    import numpy as np
    from scipy.stats import qmc

    names = sorted(components)
    if len(names) < 2 or len(set(names)) != len(names) or count < 1:
        raise ValueError("Need distinct components and a positive candidate count.")
    pmin, pmax = min(pressures), max(pressures)
    tmin, tmax = min(temperatures), max(temperatures)
    if pmin <= 0 or tmin <= 0:
        raise ValueError("Positive pressure/temperature bounds required.")
    u = qmc.Sobol(len(names) + 2, scramble=True, seed=seed).random_base2(
        math.ceil(math.log2(count))
    )[:count]
    exponential = -np.log(np.clip(u[:, 2:], 1e-15, 1 - 1e-15))
    fractions = exponential / exponential.sum(axis=1, keepdims=True)
    return [
        point(
            10 ** (math.log10(pmin) + row[0] * math.log10(pmax / pmin)),
            tmin + row[1] * (tmax - tmin),
            dict(zip(names, y)),
        )
        for row, y in zip(u, fractions)
    ]
