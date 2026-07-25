from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any


PACKAGE_NAMES = ["ase", "CoolProp", "matplotlib", "numpy", "pandas"]
RUN_INPUT_DIRECTORIES = ["data", "forcefield", "inputs", "molecules", "source"]


def create_reproducibility_manifest(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    config_file: str | Path | None = None,
    launch_command: str | None = None,
) -> dict[str, Any]:
    """Archive software versions, Git state, and hashes of immutable run inputs."""
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else _find_project_root(Path.cwd())
    )
    target = (
        Path(output_dir)
        if output_dir is not None
        else run_path / "reproducibility"
    )
    target.mkdir(parents=True, exist_ok=True)

    files = _input_files(run_path)
    config_path = Path(config_file).resolve() if config_file else None
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
        files.append(config_path)
    files = sorted(set(files))
    checksums = {
        _display_path(path, run_path, root): _sha256(path) for path in files
    }

    lammps_path = shutil.which("lmp")
    conda_path = _conda_executable()
    conda_explicit = (
        _command_output(
            [conda_path, "list", "--prefix", sys.prefix, "--explicit"]
        )
        if conda_path is not None
        else None
    )
    git_commit = _command_output(["git", "rev-parse", "HEAD"], cwd=root)
    git_branch = _command_output(
        ["git", "branch", "--show-current"], cwd=root
    )
    git_status = _command_output(["git", "status", "--short"], cwd=root)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_path),
        "launch_command": launch_command,
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "packages": {name: _package_version(name) for name in PACKAGE_NAMES},
        "conda": {
            "executable": conda_path,
            "prefix": sys.prefix,
            "explicit_environment_available": conda_explicit is not None,
        },
        "lammps": {
            "executable": lammps_path,
            "version_line": _lammps_version(lammps_path),
        },
        "git": {
            "project_root": str(root),
            "commit": git_commit,
            "branch": git_branch,
            "dirty": bool(git_status),
            "status_short": git_status.splitlines() if git_status else [],
        },
        "configuration_file": str(config_path) if config_path else None,
        "checksums_sha256": checksums,
    }

    json_path = target / "reproducibility_manifest.json"
    checksum_path = target / "checksums.sha256"
    conda_path_output = target / "conda-explicit.txt"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_path.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in checksums.items()),
        encoding="utf-8",
    )
    if conda_explicit is not None:
        conda_path_output.write_text(conda_explicit + "\n", encoding="utf-8")
    manifest["outputs"] = {
        "json": str(json_path),
        "checksums": str(checksum_path),
        "conda_explicit": (
            str(conda_path_output) if conda_explicit is not None else None
        ),
    }
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _input_files(run_dir: Path) -> list[Path]:
    files = []
    prepare_summary = run_dir / "prepare_summary.json"
    if prepare_summary.exists():
        files.append(prepare_summary.resolve())
    for directory in RUN_INPUT_DIRECTORIES:
        root = run_dir / directory
        if root.exists():
            files.extend(
                path.resolve() for path in root.rglob("*") if path.is_file()
            )
    return files


def _display_path(path: Path, run_dir: Path, project_root: Path) -> str:
    for base, prefix in ((run_dir, "run"), (project_root, "project")):
        try:
            return f"{prefix}/{path.relative_to(base)}"
        except ValueError:
            continue
    return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _lammps_version(executable: str | None) -> str | None:
    if executable is None:
        return None
    output = _command_output([executable, "-h"])
    if output is None:
        return None
    return next(
        (line.strip() for line in output.splitlines() if line.strip()),
        None,
    )


def _conda_executable() -> str | None:
    environment_value = os.environ.get("CONDA_EXE")
    candidates = [
        Path(environment_value) if environment_value else None,
        Path(sys.prefix).parent.parent / "bin" / "conda",
    ]
    discovered = shutil.which("conda")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return str(candidate)
    return None


def _command_output(
    command: list[str],
    *,
    cwd: Path | None = None,
) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a reproducibility manifest for a materialized run."
    )
    parser.add_argument("run_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--project-root")
    parser.add_argument("--config")
    parser.add_argument("--command", dest="launch_command")
    args = parser.parse_args(argv)
    manifest = create_reproducibility_manifest(
        args.run_dir,
        args.output_dir,
        project_root=args.project_root,
        config_file=args.config,
        launch_command=args.launch_command,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
