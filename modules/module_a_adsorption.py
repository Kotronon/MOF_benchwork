"""Module A metadata.

Module A is implemented through the shared benchmark pipeline instead of a
separate module-local orchestration layer. This avoids duplicating behavior
that later modules also need, such as planning, materialization, LAMMPS
execution, log parsing, evaluation, and report generation.
"""

from __future__ import annotations

MODULE_ID = "A"
MODULE_NAME = "Module A"
TASK = "single_component_adsorption"
DESCRIPTION = "Single-component adsorption in a rigid framework."

REUSES_SHARED_PIPELINE = True

SHARED_PIPELINE_STAGES = (
    "configuration_normalization",
    "material_resolution",
    "forcefield_resolution",
    "reference_resolution",
    "prepare_plan",
    "materialization",
    "lammps_execution",
    "log_parsing",
    "evaluation",
    "reporting",
)

MODULE_SPECIFIC_SCOPE = (
    "single_adsorbate",
    "rigid_framework",
    "gcmc_isotherm",
    "absolute_and_excess_adsorption",
)


def metadata() -> dict[str, object]:
    """Return JSON-serializable module metadata."""
    return {
        "module_id": MODULE_ID,
        "module_name": MODULE_NAME,
        "task": TASK,
        "description": DESCRIPTION,
        "reuses_shared_pipeline": REUSES_SHARED_PIPELINE,
        "shared_pipeline_stages": list(SHARED_PIPELINE_STAGES),
        "module_specific_scope": list(MODULE_SPECIFIC_SCOPE),
    }
