from __future__ import annotations

R_J_MOL_K = 8.31446261815324


def mixture_fugacities(
    temperature_K,
    pressure_bar,
    mixture,
    critical,
    *,
    binary_interactions=None,
    root="vapor",
    fugacities_bar=None,
):
    """Multicomponent PR reservoir. All pressures in bar, critical Pc in Pa.

    Explicit roots are intentional: this evaluates a homogeneous reservoir,
    not a phase-stability calculation or a flash. No pure-partial-P fallback.
    """
    import math
    import numpy as np

    names = sorted(mixture)
    y = np.array([float(mixture[c]) for c in names])
    T, P = float(temperature_K), float(pressure_bar) * 1e5
    if not math.isfinite(T) or T <= 0 or not math.isfinite(P) or P <= 0:
        raise ValueError("Temperature and pressure must be finite and positive.")
    if (
        not np.isfinite(y).all()
        or (y < 0).any()
        or not np.isclose(y.sum(), 1, atol=1e-9, rtol=0)
    ):
        raise ValueError(
            "Mixture fractions must be finite, nonnegative and sum to one."
        )
    if fugacities_bar is not None:
        if set(fugacities_bar) != set(names):
            raise ValueError("Explicit fugacities must cover every component.")
        f = {c: float(fugacities_bar[c]) for c in names}
        if any(
            not math.isfinite(f[c]) or f[c] < 0 or (mixture[c] == 0) != (f[c] == 0)
            for c in names
        ):
            raise ValueError("Fugacity must be positive exactly for active components.")
        return {
            "fugacities_bar": f,
            "coefficients": {
                c: f[c] / (mixture[c] * pressure_bar) if mixture[c] else 1.0
                for c in names
            },
            "model": "explicit",
            "root": None,
            "compressibility": None,
        }
    if root not in {"vapor", "liquid"}:
        raise ValueError("PR root must explicitly be vapor or liquid.")
    tc, pc, omega = [], [], []
    for c in names:
        data = critical[c]
        tc.append(float(data["critical_temperature_K"]))
        pc.append(float(data["critical_pressure_Pa"]))
        omega.append(float(data["acentric_factor"]))
    tc, pc, omega = map(np.asarray, (tc, pc, omega))
    if (
        not all(np.isfinite(v).all() for v in (tc, pc, omega))
        or (tc <= 0).any()
        or (pc <= 0).any()
    ):
        raise ValueError(
            "PR requires finite critical data and positive Tc/Pc for every species."
        )
    kij = np.zeros((len(names), len(names)))
    for pair, value in (binary_interactions or {}).items():
        left, right = pair.split("/")
        i, j = names.index(left), names.index(right)
        if i == j or not math.isfinite(float(value)):
            raise ValueError("Invalid binary interaction parameter.")
        reverse = (binary_interactions or {}).get(f"{right}/{left}", value)
        if float(reverse) != float(value):
            raise ValueError("Binary interaction parameters must be symmetric.")
        kij[i, j] = kij[j, i] = value
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    a = 0.45724 * R_J_MOL_K**2 * tc**2 / pc * (1 + kappa * (1 - np.sqrt(T / tc))) ** 2
    b = 0.07780 * R_J_MOL_K * tc / pc
    aij = np.sqrt(a[:, None] * a[None, :]) * (1 - kij)
    am, bm = float(y @ aij @ y), float(y @ b)
    A, B = am * P / (R_J_MOL_K * T) ** 2, bm * P / (R_J_MOL_K * T)
    roots = sorted(
        float(z.real)
        for z in np.roots([1, -(1 - B), A - 3 * B * B - 2 * B, -(A * B - B * B - B**3)])
        if abs(z.imag) < 1e-9 and z.real > B
    )
    if not roots:
        raise ValueError("No admissible PR compressibility root.")
    z = roots[-1] if root == "vapor" else roots[0]
    s2 = math.sqrt(2)
    # log1p retains the ideal-gas limit at very small pressure.
    logarithm = math.log1p(2 * s2 * B / (z + (1 - s2) * B))
    lnphi = (
        b / bm * (z - 1)
        - math.log(z - B)
        - A / (2 * s2 * B) * (2 * (aij @ y) / am - b / bm) * logarithm
    )
    phi = np.exp(lnphi)
    if not np.isfinite(phi).all() or (phi <= 0).any():
        raise ValueError("Invalid PR fugacity coefficients.")
    return {
        "fugacities_bar": {
            c: float(y[i] * pressure_bar * phi[i]) for i, c in enumerate(names)
        },
        "coefficients": dict(zip(names, map(float, phi))),
        "model": "Peng-Robinson",
        "root": root,
        "compressibility": z,
        "admissible_roots": roots,
        "phase_stability": "not_assessed",
        "binary_interactions": binary_interactions or {},
    }


def fugacity_coeff_coolprop(
    component: str,
    temperature_K: float,
    pressure_bar: float,
    backend: str = "PR",
) -> float:
    try:
        from CoolProp import CoolProp as CP
    except ImportError as exc:
        raise ImportError(
            "CoolProp is required for EOS fugacity coefficients. "
            "Install it with: conda install -n MOF_sim -c conda-forge coolprop"
        ) from exc

    fluid_map = {
        "CO2": "CarbonDioxide",
        "N2": "Nitrogen",
    }

    fluid = fluid_map.get(component.upper(), component)
    pressure_pa = pressure_bar * 100000.0

    state = CP.AbstractState(backend, fluid)
    state.update(CP.PT_INPUTS, pressure_pa, temperature_K)
    return float(state.fugacity_coefficient(0))


def gas_molar_density(
    component: str,
    temperature_K: float,
    pressure_bar: float,
    backend: str = "HEOS",
) -> float:
    """Return bulk gas molar density in mol/m^3 for excess adsorption correction."""
    if backend.casefold() == "ideal":
        pressure_pa = pressure_bar * 100000.0
        return pressure_pa / (R_J_MOL_K * temperature_K)

    try:
        from CoolProp import CoolProp as CP
    except ImportError as exc:
        raise ImportError(
            "CoolProp is required for real-gas density. "
            "Install it with: conda install -n MOF_sim -c conda-forge coolprop"
        ) from exc

    fluid_map = {
        "CO2": "CarbonDioxide",
        "N2": "Nitrogen",
    }

    fluid = fluid_map.get(component.upper(), component)
    pressure_pa = pressure_bar * 100000.0

    state = CP.AbstractState(backend, fluid)
    state.update(CP.PT_INPUTS, pressure_pa, temperature_K)
    return float(state.rhomolar())
