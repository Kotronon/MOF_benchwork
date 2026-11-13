"""Standalone MIL-53/CO2 technical pilot (run from the distributed package)."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess


def check_capabilities(help_text):
    required = {
        "Atom": {"full"}, "Pair": {"hybrid/overlay", "coul/long", "table"},
        "Bond": {"hybrid", "mm3", "table", "zero"},
        "Angle": {"hybrid", "mm3", "cosine/periodic", "cross"},
        "Dihedral": {"fourier"}, "Improper": {"distharm"},
        "KSpace": {"ewald"}, "Fix": {"gcmc", "rigid/nvt/small", "npt", "nvt"},
    }
    sections = re.split(r"\* ([A-Za-z]+) styles:?[ \t]*\n", help_text)
    available = {sections[i].lower(): set(sections[i+1].split())
                 for i in range(1, len(sections)-1, 2)}
    missing = [f"{category}:{style}" for category, styles in required.items()
               for style in sorted(styles - available.get(category.lower(), set()))]
    if missing:
        raise ValueError("LAMMPS lacks required styles: " + ", ".join(missing) +
                         ". Build with MOLECULE KSPACE RIGID MC YAFF EXTRA-MOLECULE; see README.md.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "smoke", "pilot"], default="check", nargs="?")
    parser.add_argument("--command", nargs="+", default=["lmp"])
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    os.chdir(root)
    for name, expected in json.loads((root / "package_manifest.json").read_text()).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid manifest path")
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Package checksum mismatch: " + name)
    from modules.module_d_general import verify_bundle, execute_bundle
    from analysis.module_d_flexibility import evaluate_flexibility
    plan = verify_bundle(root / "campaign")
    if plan["benchmark"]["applicability"]["blockers"]:
        raise ValueError(plan["benchmark"]["applicability"]["blockers"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    result = subprocess.run([*args.command, "-help"], capture_output=True,
                            text=True, check=True, timeout=60)
    check_capabilities(result.stdout + result.stderr)
    print("PASS: package, inputs and required LAMMPS styles verified.", flush=True)
    if args.stage == "check":
        return
    execute_bundle(root / "campaign", smoke=args.stage == "smoke", command=args.command)
    if args.stage == "pilot":
        report = evaluate_flexibility(root / "campaign")
        if report["status"] != "complete":
            raise RuntimeError("Incomplete pilot evaluation")
        print("Results: campaign/evaluation/module_d_report.json", flush=True)
    print(f"PASS: {args.stage}. Technical execution only; potential and sampling remain unvalidated.", flush=True)


if __name__ == "__main__":
    main()
