from __future__ import annotations

from pathlib import Path
from typing import Any

from converter.cif_to_lammps_data import plan_cif_to_lammps_data
from converter.molecule_to_lammps_template import plan_molecule_template
from engines.lammps_runner import build_lammps_command





def _pressure_token(pressure_bar: int | float) -> str:
    return str(pressure_bar).replace(".", "p").replace("-", "m")
