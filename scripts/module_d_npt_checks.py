"""Prepare and run isolated numerical checks from a completed MIL-53 v3 pilot.

Standard library only. Starts from the original dry structure, not final.data.
Default: one seed, four runs; this is a numerical screen, not equilibration.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


def transform(source, factor, dry):
    """Strictly transform the known v3 input; refuse changed protocols."""
    required = ["timestep 0.25", "run 2000", "run 20000", "thermo 100",
                "dump trajectory all custom 1000 trajectory.lammpstrj id mol type q x y z ix iy iz"]
    lines = source.splitlines()
    if any(lines.count(line) != 1 for line in required):
        raise ValueError("Expected the original v3 0.25 fs, 22000-step pilot input.")
    if not any(line.startswith("fix hostmd host npt ") for line in lines):
        raise ValueError("Expected an NPT input.")
    exchange = [line for line in lines if line.startswith("fix exchange ")]
    if len(exchange) != 1 or not exchange[0].startswith("fix exchange gas gcmc 20 5 0 0 "):
        raise ValueError("Unexpected GCMC protocol.")
    replacements = {
        "timestep 0.25": f"timestep {0.25 / factor}",
        "run 2000": f"run {2000 * factor}",
        "run 20000": f"run {20000 * factor}",
        "thermo 100": f"thermo {100 * factor}",
        required[-1]: f"dump trajectory all custom {1000 * factor} trajectory.lammpstrj id mol type q x y z ix iy iz",
    }
    result = []
    for line in lines:
        if dry and line.startswith(("molecule guestmol ", "fix guestmd ",
                                    "fix_modify guestmd ", "fix exchange ", "fix attempts ")):
            continue
        line = replacements.get(line, line)
        if line.startswith("fix exchange "):
            line = line.replace("gcmc 20 5", f"gcmc {20 * factor} 5", 1)
        if line.startswith(("fix cellstats ", "fix stats ")):
            if "ave/time 1 1000 1000 " not in line:
                raise ValueError("Unexpected averaging protocol.")
            # Same physical sampling interval and window at both timesteps.
            line = line.replace("ave/time 1 1000 1000 ", f"ave/time {factor} 1000 {1000 * factor} ", 1)
        if line.startswith("fix attempts "):
            line = line.replace("ave/time 1 1 1000 ", f"ave/time 1 1 {1000 * factor} ", 1)
        result.append(line)
    return "\n".join(result) + "\n"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_sampling(source, factor, warmup_ps, measurement_ps, dump_interval_ps, precise_dump):
    """Change duration/output only; retain the physical MD/MC coupling protocol."""
    def whole_units(value, unit, label):
        count = value / unit
        if not math.isfinite(count) or count <= 0 or not math.isclose(count, round(count), abs_tol=1e-8, rel_tol=0):
            raise ValueError(f"{label} must be positive and a multiple of {unit:g} ps.")
        return int(round(count))

    # Averaging windows remain 0.25 ps; start/end must align with their boundaries.
    warmup_blocks = whole_units(warmup_ps, .25, "warmup_ps")
    measurement_blocks = whole_units(measurement_ps, .25, "measurement_ps")
    dump_steps = whole_units(dump_interval_ps, .00025 / factor, "dump_interval_ps")
    replacements = {
        f"run {2000 * factor}": f"run {warmup_blocks * 1000 * factor}",
        f"run {20000 * factor}": f"run {measurement_blocks * 1000 * factor}",
        f"dump trajectory all custom {1000 * factor} trajectory.lammpstrj id mol type q x y z ix iy iz":
            f"dump trajectory all custom {dump_steps} trajectory.lammpstrj id mol type q x y z ix iy iz",
    }
    lines = source.splitlines()
    for expected in replacements:
        if lines.count(expected) != 1:
            raise ValueError(f"Expected exactly one line: {expected}")
    result = [replacements.get(line, line) for line in lines]
    if precise_dump:
        if result.count("dump_modify trajectory sort id") != 1:
            raise ValueError("Expected the original dump_modify line.")
        result[result.index("dump_modify trajectory sort id")] += " format float %.16g"
    return "\n".join(result) + "\n"


def prepare(args):
    pilot, output = args.pilot.resolve(), args.directory.resolve()
    assets = (args.assets or pilot / "assets").resolve()
    if not (assets / "initial_structure").is_file():
        raise FileNotFoundError(f"Missing archived assets: {assets}")
    inputs = {}
    sources = {}
    for seed in args.seeds:
        source = pilot / "results" / f"hybrid_npt__p000__seed{seed}" / "in.lammps"
        sources[str(source)] = digest(source)
        for dry in (False, True):
            for timestep in args.timesteps_fs:
                factor = 1 if timestep == .25 else 2
                name = f"{'dry' if dry else 'co2'}_dt{'025' if factor == 1 else '0125'}_seed{seed}"
                inputs[name] = configure_sampling(
                    transform(source.read_text(), factor, dry), factor,
                    args.warmup_ps, args.measurement_ps, args.dump_interval_ps, args.precise_dump)
    # Refuse reuse even of an empty directory.
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(assets, output / "assets")
    for name, content in inputs.items():
        case = output / name
        case.mkdir()
        (case / "assets").symlink_to("../assets", target_is_directory=True)
        (case / "in.lammps").write_text(content)
    manifest = {str(p.relative_to(output)): digest(p)
                for p in (output / "assets").rglob("*") if p.is_file()}
    manifest.update({f"{name}/in.lammps": digest(output / name / "in.lammps") for name in inputs})
    (output / "checks.json").write_text(json.dumps({
        "cases": list(inputs), "files": manifest, "source_inputs": sources,
        "duration_ps": args.warmup_ps + args.measurement_ps,
        "warmup_ps": args.warmup_ps, "measurement_ps": args.measurement_ps,
        "measurement_block_ps": .25, "dump_interval_ps": args.dump_interval_ps,
        "precise_dump": args.precise_dump, "timesteps_fs": args.timesteps_fs,
        "start": "original dry structure; independent new runs, not restart continuation",
        "purpose": "numerical screen and relaxation assessment; not equilibrium validation"
    }, indent=2) + "\n")
    print(f"Prepared {len(inputs)} cases in {output}; no simulations started.")


def run(args):
    root = args.directory.resolve()
    plan = json.loads((root / "checks.json").read_text())
    for name, expected in plan["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Changed input: {name}")
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(name, "1")
    for name in plan["cases"]:
        case = root / name
        if (case / "completed.json").is_file():
            if json.loads((case / "completed.json").read_text())["command"] != args.command:
                raise ValueError("Use the same LAMMPS command for all cases; otherwise prepare a new directory.")
            print(f"Already complete: {name}", flush=True)
            continue
        if (case / "console.log").exists():
            raise FileExistsError(f"Incomplete prior attempt: {case}. Prepare a new directory.")
        print(f"Running {name}", flush=True)
        with (case / "console.log").open("x") as log:
            result = subprocess.run([*args.command, "-in", "in.lammps"], cwd=case,
                                    env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or 'MODULE_D_COMPLETED' not in (case / "console.log").read_text():
            raise RuntimeError(f"LAMMPS failed; inspect {case / 'console.log'}")
        (case / "completed.json").write_text(json.dumps({"command": args.command, "returncode": result.returncode}) + "\n")
    print("All numerical checks finished. Inspect trajectories and cell histories before drawing conclusions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--pilot", type=Path, required=True, help="Completed campaign directory")
    prep.add_argument("--assets", type=Path, help="Optional assets location for result-only imports")
    prep.add_argument("--directory", type=Path, required=True)
    prep.add_argument("--seeds", type=int, nargs="+", default=[12345])
    prep.add_argument("--timesteps-fs", type=float, nargs="+", choices=[.25, .125], default=[.25, .125])
    prep.add_argument("--warmup-ps", type=float, default=.5, help="Positive multiple of 0.25 ps")
    prep.add_argument("--measurement-ps", type=float, default=5., help="Positive multiple of 0.25 ps; not guaranteed equilibrium")
    prep.add_argument("--dump-interval-ps", type=float, default=.25, help="Physical trajectory interval after warmup")
    prep.add_argument("--precise-dump", action="store_true", help="Write floating-point dump values with 16 significant digits")
    start = sub.add_parser("run")
    start.add_argument("--directory", type=Path, required=True)
    start.add_argument("--command", nargs="+", default=["lmp"])
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else run)(args)


if __name__ == "__main__":
    main()
