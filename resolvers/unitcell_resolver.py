from __future__ import annotations

from typing import Any


AUTO_UNIT_CELLS = {
    "IRMOF-1": [1, 1, 1],
    "MOF-5": [1, 1, 1],
    "ZIF-8": [2, 2, 2],
    "ZIF-4": [2, 2, 2],
    "Mg-MOF-74": [2, 2, 4],
    "MG-MOF-74": [2, 2, 4],
}


class UnitcellResolver:
    def resolve(self, requested: Any, material_id: str) -> list[int]:
        if requested is None or str(requested).lower() == "auto":
            return list(AUTO_UNIT_CELLS.get(material_id, [1, 1, 1]))

        if (
            isinstance(requested, list)
            and len(requested) == 3
            and all(isinstance(value, int) and value > 0 for value in requested)
        ):
            return list(requested)

        raise ValueError("'simulation.unit_cells' must be 'auto' or a list of three positive integers.")
