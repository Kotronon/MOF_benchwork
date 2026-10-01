"""Optional dependency management for the MLIP-MC execution path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from importlib import metadata, util
import shlex
import subprocess
import sys
from typing import Any


MLIP_MC_VERSION = "0.1.3"
MLIP_MC_DISTRIBUTION = "mlip-mc"
LEGACY_NEQUIP_VERSION = "0.6.2"
FLAMES_VERSION = "0.4.8"
FLAMES_COMMIT = "c7eaae1"
FLAMES_INSTALL_SPECIFICATION = (
    f"git+https://github.com/lipelopesoliveira/flames.git@{FLAMES_COMMIT}"
)

_BACKEND_MODULES = {
    "mace-torch": "mace",
    "orb-models": "orb_models",
    "fairchem": "fairchem.core",
    "nequip": "nequip",
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
    backend_required_version: str | None = None
    backend_installed_version: str | None = None

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
        "nequip": "nequip",
        "nequip-legacy": "nequip",
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
    backend_distribution = (
        "nequip" if canonical_backend == "nequip" else None
    )
    backend_required_version = (
        LEGACY_NEQUIP_VERSION if canonical_backend == "nequip" else None
    )
    backend_installed_version = (
        _distribution_version(backend_distribution)
        if backend_distribution
        else None
    )
    installed_version = _distribution_version(MLIP_MC_DISTRIBUTION)
    mlip_available = _module_available("mlip_mc")
    backend_available = _module_available(backend_module) and (
        backend_required_version is None
        or backend_installed_version == backend_required_version
    )
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
        backend_required_version=backend_required_version,
        backend_installed_version=backend_installed_version,
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
        *shlex.split(status.install_specification),
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


def ensure_torch_dftd(*, install_missing: bool = False) -> dict[str, Any]:
    """Ensure the D3 calculator used by dispersion-enabled MACE runs exists."""
    module = "torch_dftd"
    specification = "torch-dftd"
    available = _module_available(module)
    if available:
        return {
            "module": module,
            "available": True,
            "install_specification": specification,
            "python_executable": sys.executable,
        }
    command = [sys.executable, "-m", "pip", "install", specification]
    if not install_missing:
        raise ImportError(
            "D3 dispersion requires torch-dftd. Install it in the active "
            "environment with:\n" + " ".join(command)
        )
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Automatic torch-dftd installation failed using the active "
            f"interpreter {sys.executable}.\n{details}"
        )
    importlib.invalidate_caches()
    if not _module_available(module):
        raise RuntimeError(
            "pip completed, but torch_dftd is still unavailable in the "
            f"active interpreter {sys.executable}."
        )
    return {
        "module": module,
        "available": True,
        "install_specification": specification,
        "python_executable": sys.executable,
    }


def ensure_flames_dependencies(*, install_missing: bool = False) -> dict[str, Any]:
    """Ensure the pinned optional FLAMES runner is importable."""
    available = _module_available("flames")
    installed_version = _distribution_version("flames")
    if available and installed_version in {None, FLAMES_VERSION}:
        return {
            "module": "flames",
            "required_version": FLAMES_VERSION,
            "installed_version": installed_version,
            "available": True,
            "install_specification": FLAMES_INSTALL_SPECIFICATION,
            "python_executable": sys.executable,
        }
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        FLAMES_INSTALL_SPECIFICATION,
    ]
    if not install_missing:
        raise ImportError(
            "FLAMES 0.4.8 is required for the selected cross-check engine. "
            "Install it with:\n" + " ".join(command)
        )
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Automatic FLAMES installation failed.\n{details}")
    importlib.invalidate_caches()
    if not _module_available("flames"):
        raise RuntimeError("FLAMES installation completed but the module is unavailable.")
    return {
        "module": "flames",
        "required_version": FLAMES_VERSION,
        "installed_version": _distribution_version("flames"),
        "available": True,
        "install_specification": FLAMES_INSTALL_SPECIFICATION,
        "python_executable": sys.executable,
    }


def _install_specification(backend: str, *, include_backend: bool) -> str:
    if backend == "nequip" and include_backend:
        return (
            f"{MLIP_MC_DISTRIBUTION}=={MLIP_MC_VERSION} "
            f"nequip=={LEGACY_NEQUIP_VERSION}"
        )
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
        if status.backend_required_version and status.backend_installed_version:
            problems.append(
                f"backend {status.backend_module!r} version "
                f"{status.backend_installed_version!r} is installed; version "
                f"{status.backend_required_version!r} is required"
            )
        else:
            problems.append(
                f"backend module {status.backend_module!r} is not installed"
            )
    return (
        "MLIP-MC dependencies are not ready: "
        + "; ".join(problems)
        + ". Re-run with --install-missing or install them explicitly with:\n"
        + " ".join(command)
    )
