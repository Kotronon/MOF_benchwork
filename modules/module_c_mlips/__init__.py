"""Potential benchmarking for Module C."""

from .classical_parity import (
    ClassicalParityReport,
    compare_classical_results,
    evaluate_module_a_static_files,
)
from .interaction import InteractionResult, evaluate_interaction
from .models import InteractionBond, InteractionConfiguration

__all__ = [
    "ClassicalParityReport",
    "InteractionBond",
    "InteractionConfiguration",
    "InteractionResult",
    "compare_classical_results",
    "evaluate_module_a_static_files",
    "evaluate_interaction",
]
