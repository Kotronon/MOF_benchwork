"""DFT labeler interface and CP2K/Quickstep implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Sequence

from pipeline.config import save_benchmark_data


HARTREE_TO_EV = 27.211386245988
HARTREE_PER_BOHR_TO_EV_PER_A = 51.4220674763


class DFTLabeler(ABC):
    @abstractmethod
    def prepare(
        self,
        candidates: Sequence[dict[str, Any]],
        output_directory: str | Path,
    ) -> list[dict[str, Any]]:
        """Create label jobs without submitting them."""

    @abstractmethod
    def submit(self, jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Submit or execute prepared jobs."""

    @abstractmethod
    def collect(self, jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Parse all completed labels."""

    @abstractmethod
    def status(
        self,
        jobs: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return scheduler/output status without modifying the jobs."""


class CP2KLabeler(DFTLabeler):
    """Generate, execute, and parse periodic CP2K single-point labels."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = dict(settings)
        self.executable = str(self.settings.get("executable", "cp2k"))
        self.scheduler = str(self.settings.get("scheduler", "local")).casefold()
        if self.scheduler not in {"local", "slurm"}:
            raise ValueError("CP2K scheduler must be 'local' or 'slurm'.")
        if self.scheduler == "local":
            _local_execution_settings(self.settings)

    def prepare(
        self,
        candidates: Sequence[dict[str, Any]],
        output_directory: str | Path,
    ) -> list[dict[str, Any]]:
        try:
            from ase.io import read
        except ImportError as exc:
            raise ImportError("ASE is required to prepare CP2K labels.") from exc
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        jobs = []
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            job_dir = root / candidate_id
            job_dir.mkdir(parents=True, exist_ok=True)
            atoms = read(str(candidate["structure_path"]))
            input_path = job_dir / "cp2k.inp"
            output_path = job_dir / "cp2k.out"
            input_path.write_text(
                render_cp2k_input(atoms, project=candidate_id, settings=self.settings),
                encoding="utf-8",
            )
            command = [self.executable, "-i", input_path.name, "-o", output_path.name]
            script_path = job_dir / "submit.slurm"
            if self.scheduler == "slurm":
                script_path.write_text(
                    render_slurm_script(candidate_id, command, self.settings),
                    encoding="utf-8",
                )
            jobs.append(
                {
                    "candidate_id": candidate_id,
                    "structure_path": str(candidate["structure_path"]),
                    "job_directory": str(job_dir),
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "script_path": str(script_path) if self.scheduler == "slurm" else None,
                    "command": command,
                    "scheduler": self.scheduler,
                    "dft_profile": str(
                        self.settings.get("profile", "pbe_d3_bj")
                    ),
                    "cutoff_Ry": int(
                        self.settings.get("cutoff_Ry", 600)
                    ),
                    "relative_cutoff_Ry": int(
                        self.settings.get("relative_cutoff_Ry", 60)
                    ),
                    "status": "prepared",
                }
            )
        save_benchmark_data(root / "cp2k_jobs.json", {"jobs": jobs})
        save_benchmark_data(
            root / "dft_manifest.json",
            {
                "labeler": "CP2K/Quickstep",
                "executable": self.executable,
                "scheduler": self.scheduler,
                "functional": "PBE",
                "dispersion": str(
                    self.settings.get("profile", "pbe_d3_bj")
                ),
                "basis": "DZVP-MOLOPT-SR-GTH",
                "pseudopotential": "GTH-PBE",
                "gamma_point_only": True,
                "settings": self.settings,
                "input_files": [
                    str(job["input_path"]) for job in jobs
                ],
            },
        )
        return jobs

    def prepare_cutoff_test(
        self,
        candidate: dict[str, Any],
        output_directory: str | Path,
    ) -> list[dict[str, Any]]:
        """Prepare the configured 400/600/800 Ry convergence series."""
        cutoffs = [int(value) for value in self.settings.get("cutoff_test_Ry", [400, 600, 800])]
        if len(set(cutoffs)) < 2 or any(value <= 0 for value in cutoffs):
            raise ValueError("CP2K cutoff_test_Ry requires at least two positive cutoffs.")
        jobs = []
        root = Path(output_directory)
        for cutoff in sorted(set(cutoffs)):
            settings = {**self.settings, "cutoff_Ry": cutoff}
            labeler = CP2KLabeler(settings)
            cutoff_candidate = {
                **candidate,
                "candidate_id": f"{candidate['candidate_id']}__cutoff_{cutoff}Ry",
            }
            current = labeler.prepare([cutoff_candidate], root / f"{cutoff}Ry")[0]
            current["cutoff_Ry"] = cutoff
            jobs.append(current)
        save_benchmark_data(root / "cp2k_jobs.json", {"jobs": jobs})
        return jobs

    def submit(self, jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.scheduler == "local":
            local = _local_execution_settings(self.settings)
            with ThreadPoolExecutor(
                max_workers=local["max_parallel_jobs"]
            ) as executor:
                return list(executor.map(self._execute_local_job, jobs))

        submitted = []
        for job in jobs:
            current = dict(job)
            workdir = Path(current["job_directory"])
            completed = subprocess.run(
                ["sbatch", Path(current["script_path"]).name],
                cwd=workdir,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or completed.stdout).strip())
            match = re.search(r"Submitted batch job\s+(\d+)", completed.stdout)
            if match is None:
                raise RuntimeError(f"Could not parse sbatch output: {completed.stdout}")
            current["scheduler_job_id"] = match.group(1)
            current["status"] = "submitted"
            submitted.append(current)
        return submitted

    def _execute_local_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """Run one CP2K job directly on the current host."""
        current = dict(job)
        workdir = Path(current["job_directory"])
        local = _local_execution_settings(self.settings)
        command = build_local_cp2k_command(current["command"], self.settings)
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(local["omp_threads_per_process"])
        current["execution_command"] = command
        current["execution_resources"] = local
        try:
            _record_cp2k_version(self.executable, workdir)
            completed = subprocess.run(
                command,
                cwd=workdir,
                capture_output=True,
                text=True,
                env=environment,
            )
        except OSError as exc:
            current["returncode"] = None
            current["status"] = "failed"
            current["error"] = str(exc)
            return current
        (workdir / "launcher.stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (workdir / "launcher.stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        current["returncode"] = completed.returncode
        current["status"] = (
            "completed" if completed.returncode == 0 else "failed"
        )
        if completed.returncode != 0:
            current["error"] = (completed.stderr or completed.stdout).strip()
        return current

    def collect(self, jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        labels = []
        for job in jobs:
            output = Path(job["output_path"])
            if not output.is_file():
                continue
            try:
                parsed = parse_cp2k_output(output)
            except ValueError:
                continue
            label_path = Path(job["job_directory"]) / "label.json"
            label = {
                "candidate_id": job["candidate_id"],
                "structure_path": job["structure_path"],
                "label_path": str(label_path),
                **parsed,
            }
            save_benchmark_data(label_path, label)
            labels.append(label)
        return labels

    def status(
        self,
        jobs: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        statuses = []
        for job in jobs:
            output = Path(job["output_path"])
            status = str(job.get("status", "prepared"))
            detail = None
            if output.is_file():
                try:
                    parse_cp2k_output(output)
                except ValueError as exc:
                    text = output.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                    status = (
                        "failed"
                        if "ABORT" in text or "PROGRAM STOPPED" in text
                        else "running_or_incomplete"
                    )
                    detail = str(exc)
                else:
                    status = "completed"
            elif (
                self.scheduler == "slurm"
                and job.get("scheduler_job_id")
            ):
                try:
                    completed = subprocess.run(
                        [
                            "squeue",
                            "-h",
                            "-j",
                            str(job["scheduler_job_id"]),
                            "-o",
                            "%T",
                        ],
                        capture_output=True,
                        text=True,
                    )
                except FileNotFoundError:
                    status = "scheduler_status_unavailable"
                    detail = "squeue is not available on this host."
                else:
                    scheduler_state = completed.stdout.strip()
                    status = (
                        scheduler_state.casefold()
                        if scheduler_state
                        else "awaiting_output"
                    )
                    detail = (
                        completed.stderr.strip()
                        if completed.returncode != 0
                        else None
                    )
            statuses.append(
                {
                    "candidate_id": job["candidate_id"],
                    "status": status,
                    "scheduler_job_id": job.get("scheduler_job_id"),
                    "detail": detail,
                }
            )
        return statuses


def render_cp2k_input(atoms: Any, *, project: str, settings: dict[str, Any]) -> str:
    cutoff = float(settings.get("cutoff_Ry", 600))
    relative_cutoff = float(settings.get("relative_cutoff_Ry", 60))
    eps_scf = float(settings.get("scf_tolerance", 1.0e-6))
    charge = int(settings.get("charge", 0))
    multiplicity = int(settings.get("multiplicity", 1))
    dispersion = str(settings.get("profile", "pbe_d3_bj")).casefold()
    if dispersion not in {"pbe_d3_bj", "pbe_mbd"}:
        raise ValueError("CP2K profile must be 'pbe_d3_bj' or 'pbe_mbd'.")
    custom_template = settings.get("custom_input_template")
    if custom_template:
        template = Path(str(custom_template)).read_text(encoding="utf-8")
        return template.format(project=project, cutoff_Ry=cutoff, relative_cutoff_Ry=relative_cutoff)
    if dispersion == "pbe_mbd":
        raise ValueError(
            "The PBE-MBD profile requires a project-validated CP2K template; "
            "provide active_learning.cp2k.custom_input_template for Mg-MOF-74."
        )

    a, b, c = atoms.cell.array
    kinds = "\n".join(
        f"    &KIND {symbol}\n"
        "      BASIS_SET DZVP-MOLOPT-SR-GTH\n"
        "      POTENTIAL GTH-PBE\n"
        "    &END KIND"
        for symbol in sorted(set(atoms.get_chemical_symbols()))
    )
    coordinates = "\n".join(
        f"      {symbol:<3} {x: .12f} {y: .12f} {z: .12f}"
        for symbol, (x, y, z) in zip(
            atoms.get_chemical_symbols(), atoms.positions, strict=True
        )
    )
    return f"""&GLOBAL
  PROJECT {project}
  RUN_TYPE ENERGY_FORCE
  PRINT_LEVEL MEDIUM
&END GLOBAL
&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME GTH_POTENTIALS
    CHARGE {charge}
    MULTIPLICITY {multiplicity}
    &MGRID
      CUTOFF {cutoff:.1f}
      REL_CUTOFF {relative_cutoff:.1f}
    &END MGRID
    &SCF
      EPS_SCF {eps_scf:.3e}
      MAX_SCF 200
      &OT
        PRECONDITIONER FULL_SINGLE_INVERSE
        MINIMIZER DIIS
      &END OT
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
      &VDW_POTENTIAL
        POTENTIAL_TYPE PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3(BJ)
          PARAMETER_FILE_NAME dftd3.dat
          REFERENCE_FUNCTIONAL PBE
          R_CUTOFF 15.0
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
    &END XC
    &PRINT
      &FORCES ON
      &END FORCES
    &END PRINT
  &END DFT
  &SUBSYS
    &CELL
      A {a[0]:.12f} {a[1]:.12f} {a[2]:.12f}
      B {b[0]:.12f} {b[1]:.12f} {b[2]:.12f}
      C {c[0]:.12f} {c[1]:.12f} {c[2]:.12f}
      PERIODIC XYZ
    &END CELL
    &COORD
{coordinates}
    &END COORD
{kinds}
  &END SUBSYS
  &PRINT
    &STRESS_TENSOR ON
    &END STRESS_TENSOR
  &END PRINT
&END FORCE_EVAL
"""


def render_slurm_script(job_name: str, command: list[str], settings: dict[str, Any]) -> str:
    slurm = settings.get("slurm", {})
    if not isinstance(slurm, dict):
        raise TypeError("active_learning.cp2k.slurm must be an object.")
    directives = [
        f"#SBATCH --job-name={_safe_job_name(job_name)}",
        f"#SBATCH --nodes={int(slurm.get('nodes', 1))}",
        f"#SBATCH --ntasks={int(slurm.get('ntasks', 16))}",
        f"#SBATCH --time={slurm.get('time', '12:00:00')}",
        "#SBATCH --output=slurm-%j.out",
    ]
    if slurm.get("partition"):
        directives.append(f"#SBATCH --partition={slurm['partition']}")
    modules = "\n".join(f"module load {value}" for value in slurm.get("modules", []))
    launcher = str(slurm.get("launcher", "srun"))
    version_command = (
        f"{shlex.quote(str(command[0]))} --version "
        "> cp2k-version.txt 2>&1"
    )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            *directives,
            "set -euo pipefail",
            modules,
            version_command,
            f"{launcher} {shlex.join(command)}",
            "",
        ]
    )


def build_local_cp2k_command(
    command: Sequence[str],
    settings: dict[str, Any],
) -> list[str]:
    """Build the direct-host CP2K command, optionally using an MPI launcher."""
    local = _local_execution_settings(settings)
    launcher = local["launcher"]
    if launcher is None:
        return list(command)
    return [
        launcher,
        "-np",
        str(local["mpi_processes_per_job"]),
        *command,
    ]


def parse_cp2k_output(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if "PROGRAM ENDED AT" not in text:
        raise ValueError(f"CP2K output is incomplete: {path}")
    energy_matches = re.findall(
        r"ENERGY\|\s+Total FORCE_EVAL.*?([-+]?[0-9]*\.?[0-9]+(?:[Ee][-+]?\d+)?)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if not energy_matches:
        raise ValueError(f"No CP2K total energy found in {path}.")
    forces = _parse_cp2k_forces(text)
    return {
        "status": "completed",
        "energy_ev": float(energy_matches[-1]) * HARTREE_TO_EV,
        "forces_ev_per_A": forces,
        "stress_GPa": _parse_cp2k_stress(text),
        "source_output": str(path),
    }


def cutoff_converged(
    results: dict[int, dict[str, Any]],
    *,
    energy_tolerance_ev_per_atom: float = 0.001,
    force_tolerance_ev_per_A: float = 0.02,
    atom_count: int,
) -> dict[str, Any]:
    ordered = sorted(results)
    if len(ordered) < 2:
        raise ValueError("At least two CP2K cutoff results are required.")
    comparisons = []
    selected = None
    for low, high in zip(ordered, ordered[1:]):
        left = results[low]
        right = results[high]
        energy_delta = abs(float(right["energy_ev"]) - float(left["energy_ev"])) / atom_count
        force_delta = _maximum_force_difference(left["forces_ev_per_A"], right["forces_ev_per_A"])
        passed = energy_delta <= energy_tolerance_ev_per_atom and force_delta <= force_tolerance_ev_per_A
        comparisons.append(
            {
                "lower_cutoff_Ry": low,
                "higher_cutoff_Ry": high,
                "energy_delta_ev_per_atom": energy_delta,
                "maximum_force_delta_ev_per_A": force_delta,
                "passed": passed,
            }
        )
        if passed and selected is None:
            selected = high
    return {
        "status": "passed" if selected is not None else "failed",
        "selected_cutoff_Ry": selected,
        "comparisons": comparisons,
    }


def _parse_cp2k_forces(text: str) -> list[list[float]]:
    blocks = re.findall(
        r"ATOMIC FORCES in \[a\.u\.\](.*?)(?:SUM OF ATOMIC FORCES|\n\s*\n)",
        text,
        flags=re.DOTALL,
    )
    if not blocks:
        raise ValueError("No CP2K atomic-force block found.")
    forces = []
    for line in blocks[-1].splitlines():
        tokens = line.split()
        if len(tokens) < 6 or not tokens[0].isdigit() or not tokens[1].isdigit():
            continue
        try:
            xyz = [float(value) * HARTREE_PER_BOHR_TO_EV_PER_A for value in tokens[-3:]]
        except ValueError:
            continue
        forces.append(xyz)
    if not forces:
        raise ValueError("CP2K atomic-force block is empty.")
    return forces


def _parse_cp2k_stress(text: str) -> list[list[float]] | None:
    blocks = re.findall(
        r"STRESS TENSOR \[GPa\](.*?)(?:PRESSURE|\n\s*\n)",
        text,
        flags=re.DOTALL,
    )
    if not blocks:
        return None
    rows = []
    for line in blocks[-1].splitlines():
        tokens = line.split()
        if len(tokens) >= 4 and tokens[0] in {"X", "Y", "Z"}:
            try:
                rows.append([float(value) for value in tokens[-3:]])
            except ValueError:
                continue
    return rows if len(rows) == 3 else None


def _maximum_force_difference(left: list[list[float]], right: list[list[float]]) -> float:
    import numpy as np

    if len(left) != len(right):
        raise ValueError("CP2K cutoff force arrays have different lengths.")
    return float(np.max(np.linalg.norm(np.asarray(left) - np.asarray(right), axis=1)))


def _safe_job_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:100]


def _record_cp2k_version(executable: str, directory: Path) -> None:
    completed = subprocess.run(
        [executable, "--version"],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    (directory / "cp2k-version.txt").write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )


def _local_execution_settings(settings: dict[str, Any]) -> dict[str, Any]:
    raw = settings.get("local", {})
    if not isinstance(raw, dict):
        raise TypeError("active_learning.cp2k.local must be an object.")
    launcher_value = raw.get("launcher", "mpirun")
    launcher = (
        None
        if launcher_value is None or not str(launcher_value).strip()
        else str(launcher_value)
    )
    normalized = {
        "launcher": launcher,
        "mpi_processes_per_job": int(raw.get("mpi_processes_per_job", 1)),
        "omp_threads_per_process": int(raw.get("omp_threads_per_process", 1)),
        "max_parallel_jobs": int(raw.get("max_parallel_jobs", 1)),
    }
    for key in (
        "mpi_processes_per_job",
        "omp_threads_per_process",
        "max_parallel_jobs",
    ):
        if normalized[key] <= 0:
            raise ValueError(f"active_learning.cp2k.local.{key} must be positive.")
    if launcher is None and normalized["mpi_processes_per_job"] != 1:
        raise ValueError(
            "Local CP2K execution without an MPI launcher requires "
            "mpi_processes_per_job=1."
        )
    return normalized
