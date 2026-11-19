"""Build a portable, source-attributed MIL-53/CO2 pilot; never copy local results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from benchmark import build_run_plan
from modules.module_d.flexibility import prepare_flexibility, materialize_flexibility
from modules.module_d.mil53 import verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="reports/module_d_mil53_co2_pilot_v3")
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    archive = root.with_name(root.name + ".tar.gz")
    if root.exists() or archive.exists() or Path(str(archive)+".sha256").exists():
        raise FileExistsError("Package already exists; choose a new --directory.")
    verify_sources()
    config = json.loads((REPO / "input_json_files/benchmark_module_d_mil53_co2_pilot.json").read_text())
    p = config["benchmark"]["flexibility"]
    for key, value in p["assets"].items():
        p["assets"][key] = str(REPO / value)
    p["potential"]["files"] = {k: str(REPO / v) for k,v in p["potential"]["files"].items()}
    plan = build_run_plan(config)
    prepared = prepare_flexibility(plan)
    prepared["working_directory"] = str(root / "campaign")
    materialize_flexibility(prepared)
    files = ["modules/base.py", "modules/module_d/__init__.py", "modules/module_d/flexibility.py", "modules/module_d/general.py",
             "analysis/module_d_flexibility.py"]
    for name in files:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
    for package in ("modules", "analysis"):
        (root / package / "__init__.py").write_text('"""Standalone Module D runtime."""\n')
    shutil.copyfile(REPO / "scripts/run_module_d_mil53.py", root / "run_mil53.py")
    shutil.copyfile(REPO / "docs/modules/module_d/mil53_cluster.md", root / "README.md")
    shutil.copytree(REPO / "data/module_d/mil53_al_reference", root / "source_reference")
    shutil.copyfile(REPO / "docs/references/mil53_al_reference_source.md", root / "source_reference/README.md")
    shutil.copyfile(REPO / "data/module_d/mil53_co2_pilot/ATTRIBUTION.txt", root / "ATTRIBUTION.txt")
    manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.rglob("*")) if path.is_file()}
    (root / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(root, arcname=root.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    Path(str(archive)+".sha256").write_text(f"{digest}  {archive.name}\n")
    print(archive)


if __name__ == "__main__":
    main()
