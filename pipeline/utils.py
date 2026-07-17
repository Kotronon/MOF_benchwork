from __future__ import annotations

from typing import Any


def pressure_token(pressure_bar: int | float) -> str:
    return str(pressure_bar).replace(".", "p").replace("-", "m")


def unique(values: Any) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

