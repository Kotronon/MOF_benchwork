from __future__ import annotations


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

