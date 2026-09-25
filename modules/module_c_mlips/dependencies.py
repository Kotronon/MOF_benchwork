"""Optional dependency management for the MLIP-MC execution path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from importlib import metadata, util
import subprocess
import sys
from typing import Any


MLIP_MC_VERSION = "0.1.3"
MLIP_MC_DISTRIBUTION = "mlip-mc"

_BACKEND_MODULES = {
    "mace-torch": "mace",
    "orb-models": "orb_models",
    "fairchem": "fairchem.core",
}


@dataclass(frozen=True)
class DependencyStatus:
    """Installed-state information for one MLIP-MC backend."""

    backend: str
    python_executable: str
    mlip_mc_required_version: str
    mlip_mc_installed_version: str | None
    backend_module: str
    mlip_mc_available: bool
    backend_available: bool
    ready: bool
    install_specification: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_mlip_backend(value: str) -> str:
    """Return the canonical optional-dependency name for a backend."""
    normalized = str(value).strip().casefold().replace("_", "-")
    aliases = {
        "mace": "mace-torch",
        "mace-mp": "mace-torch",
        "mace-torch": "mace-torch",
        "orb": "orb-models",
        "orb-v3": "orb-models",
        "orb-models": "orb-models",
        "fairchem": "fairchem",
        "odac": "fairchem",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_BACKEND_MODULES))
        raise ValueError(
            f"Unsupported MLIP-MC backend {value!r}; expected one of {supported}."
        ) from exc


def inspect_mlip_mc_dependencies(backend: str) -> DependencyStatus:
    """Inspect MLIP-MC and its selected model backend without importing them."""
    canonical_backend = normalize_mlip_backend(backend)
    backend_module = _BACKEND_MODULES[canonical_backend]
    installed_version = _distribution_version(MLIP_MC_DISTRIBUTION)
    mlip_available = _module_available("mlip_mc")
    backend_available = _module_available(backend_module)
    correct_version = installed_version == MLIP_MC_VERSION
    install_specification = _install_specification(
        canonical_backend,
        include_backend=not backend_available,
    )
    return DependencyStatus(
        backend=canonical_backend,
        python_executable=sys.executable,
        mlip_mc_required_version=MLIP_MC_VERSION,
        mlip_mc_installed_version=installed_version,
        backend_module=backend_module,
        mlip_mc_available=mlip_available,
        backend_available=backend_available,
        ready=mlip_available and backend_available and correct_version,
        install_specification=install_specification,
    )


def ensure_mlip_mc_dependencies(
    backend: str,
    *,
    install_missing: bool = False,
) -> DependencyStatus:
    """Ensure MLIP-MC and the explicitly selected backend are available.

    Installation is deliberately opt-in. When enabled, pip is invoked through
    ``sys.executable`` so packages are added to the active local or cluster
    environment rather than to a second hidden environment.
    """
    status = inspect_mlip_mc_dependencies(backend)
    if status.ready:
        return status

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        status.install_specification,
    ]
    if not install_missing:
        raise ImportError(
            _missing_dependency_message(status, command)
        )

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Automatic MLIP-MC dependency installation failed using the "
            f"active interpreter {sys.executable}.\n"
            f"Command: {' '.join(command)}\n{details}"
        )

    importlib.invalidate_caches()
    refreshed = inspect_mlip_mc_dependencies(backend)
    if not refreshed.ready:
        raise RuntimeError(
            "pip completed, but MLIP-MC is still unavailable in the active "
            f"interpreter {sys.executable}. Restart the Python process and run "
            "the benchmark again."
        )
    return refreshed


def _install_specification(backend: str, *, include_backend: bool) -> str:
    if include_backend:
        return f"{MLIP_MC_DISTRIBUTION}[{backend}]=={MLIP_MC_VERSION}"
    return f"{MLIP_MC_DISTRIBUTION}=={MLIP_MC_VERSION}"


def _distribution_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _module_available(module: str) -> bool:
    try:
        return util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _missing_dependency_message(
    status: DependencyStatus,
    command: list[str],
) -> str:
    problems = []
    if not status.mlip_mc_available:
        problems.append("mlip-mc is not installed")
    elif status.mlip_mc_installed_version != MLIP_MC_VERSION:
        problems.append(
            "mlip-mc version "
            f"{status.mlip_mc_installed_version!r} is installed; "
            f"version {MLIP_MC_VERSION!r} is required"
        )
    if not status.backend_available:
        problems.append(
            f"backend module {status.backend_module!r} is not installed"
        )
    return (
        "MLIP-MC dependencies are not ready: "
        + "; ".join(problems)
        + ". Re-run with --install-missing or install them explicitly with:\n"
        + " ".join(command)
    )
