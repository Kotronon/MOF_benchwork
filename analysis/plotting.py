from __future__ import annotations

from typing import Any


def legend_outside_right(axis: Any, *, fontsize: str = "small", ncol: int = 1) -> None:
    """Place a legend outside the plotting area when labelled artists exist."""
    handles, labels = axis.get_legend_handles_labels()
    labelled = [(handle, label) for handle, label in zip(handles, labels) if label and not label.startswith("_")]
    if not labelled:
        return
    axis.legend(
        [handle for handle, _label in labelled],
        [label for _handle, label in labelled],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=fontsize,
        ncol=ncol,
        frameon=False,
    )


def legend_above(axis: Any, *, fontsize: str = "small", ncol: int = 1) -> None:
    """Place a compact legend above one axis without covering data."""
    handles, labels = axis.get_legend_handles_labels()
    labelled = [(handle, label) for handle, label in zip(handles, labels) if label and not label.startswith("_")]
    if not labelled:
        return
    axis.legend(
        [handle for handle, _label in labelled],
        [label for _handle, label in labelled],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0.0,
        fontsize=fontsize,
        ncol=ncol,
        frameon=False,
    )


def repeated_seed_labels_needed(groups: dict[tuple[float, Any], list[dict[str, Any]]]) -> bool:
    """Return true only when seed labels disambiguate multiple curves per pressure."""
    seeds_by_pressure: dict[float, set[Any]] = {}
    for pressure, seed in groups:
        seeds_by_pressure.setdefault(float(pressure), set()).add(seed)
    return any(len(seeds) > 1 for seeds in seeds_by_pressure.values())
