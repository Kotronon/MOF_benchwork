"""Module D placeholder for flexible-framework benchmarks.

The module is registered through :mod:`modules.registry` so selection and
capability reporting already work. Execution intentionally remains disabled
until the flexible MD/GCMC workflow is implemented.
"""

from __future__ import annotations


MODULE_ID = "D"
MODULE_NAME = "Module D"
TASK = "framework_flexibility"
DESCRIPTION = "Flexible-framework adsorption and structural response benchmark."


def metadata() -> dict[str, object]:
    """Return JSON-serializable module metadata."""
    return {
        "module_id": MODULE_ID,
        "module_name": MODULE_NAME,
        "task": TASK,
        "description": DESCRIPTION,
        "status": "registered_not_implemented",
    }
