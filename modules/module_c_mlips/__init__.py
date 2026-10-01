"""Potential benchmarking for Module C."""

from .classical_parity import (
    ClassicalParityReport,
    compare_classical_results,
    evaluate_module_a_static_files,
)
from .comparison import compare_interaction_results
from .interaction import InteractionResult, evaluate_interaction
from .models import InteractionBond, InteractionConfiguration
from .runner import run_potential_comparison
from .mlip_mc_workflow import run_mlip_mc_benchmark
from .workflow import run_potential_benchmark
from .adsorption_engines import (
    AdsorptionEngine,
    FLAMESAdsorptionEngine,
    LAMMPSClassicalEngine,
    MLIPMCAdsorptionEngine,
    build_adsorption_engine,
    compare_engine_results,
)
from .applicability import assess_campaign_structure, automatic_supercell

__all__ = [
    "ClassicalParityReport",
    "InteractionBond",
    "InteractionConfiguration",
    "InteractionResult",
    "compare_classical_results",
    "compare_interaction_results",
    "evaluate_module_a_static_files",
    "evaluate_interaction",
    "run_potential_comparison",
    "run_potential_benchmark",
    "run_mlip_mc_benchmark",
    "AdsorptionEngine",
    "FLAMESAdsorptionEngine",
    "LAMMPSClassicalEngine",
    "MLIPMCAdsorptionEngine",
    "build_adsorption_engine",
    "compare_engine_results",
    "assess_campaign_structure",
    "automatic_supercell",
]
