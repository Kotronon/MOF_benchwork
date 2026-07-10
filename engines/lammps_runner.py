from __future__ import annotations

from pathlib import Path
from typing import Any


def build_lammps_command(
    input_script: str | Path,
    lammps_executable: str = "lmp",
    log_file: str | Path | None = None,
) -> list[str]:
    """Build the LAMMPS command for a planned input script."""
    command = [lammps_executable, "-in", str(input_script)]
    if log_file is not None:
        command.extend(["-log", str(log_file)])
    return command


def run_lammps(command: list[str], cwd: str | Path | None = None) -> dict[str, Any]:
    """Placeholder for the real LAMMPS subprocess runner."""
    raise NotImplementedError(
        "LAMMPS execution is not implemented yet. "
        "Use build_lammps_command() for planning; add subprocess execution after input generation is reliable."
    )
