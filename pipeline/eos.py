from __future__ import annotations


R_J_MOL_K = 8.31446261815324


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
