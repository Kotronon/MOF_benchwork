from __future__ import annotations

from typing import Any

from modules.registry import (
    module_by_id,
    module_for_task,
    registered_modules,
    select_registered_module,
)

MODULES = {module.id: module.metadata() for module in registered_modules()}
DEFAULT_METRICS = {
    module.id: list(module.default_metrics)
    for module in registered_modules()
}
EXPLICIT_TASKS = {
    alias: module.id
    for module in registered_modules()
    for alias in module.task_aliases
}


def select_module(config: dict[str, Any]) -> dict[str, str]:
    task = str(config.get("benchmark", {}).get("task", "auto")).strip().lower().replace("-", "_")
    if task and task != "auto":
        return module_for_task(task).metadata()

    return select_registered_module(config).metadata()


def default_metrics_for_module(module_id: str) -> list[str]:
    return list(module_by_id(module_id).default_metrics)


def _auto_select_module(config: dict[str, Any]) -> str:
    """Backward-compatible helper returning only the selected module id."""
    return select_registered_module(config).id
