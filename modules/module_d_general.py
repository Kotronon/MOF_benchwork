"""Material-independent Module D profiles and native LAMMPS MD/GCMC.

Classical NVT and hydrostatic anisotropic-cell NPT execution are implemented.
MLIP profiles are explicitly gated until their adapters have validation. No force
field, charges or topology are inferred from a material name or CIF.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

from modules.module_d_flexibility import CYCLE_COLUMNS, asset, finite, integer


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _commands(values, allowed, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} requires a non-empty list of LAMMPS commands.")
    for value in values:
        if not isinstance(value, str) or any(c in value for c in "\n\r$&"):
            raise ValueError(f"{name}: only literal, single-line commands are supported.")
        tokens = shlex.split(value)
        if not tokens or tokens[0] not in allowed:
            raise ValueError(f"Unsupported {name} command: {value}")
        if any(token.startswith(("/", "~")) or ".." in Path(token).parts for token in tokens):
            raise ValueError("Potential files must use bundle-relative assets/potential/ paths.")
        # Native TIP4P requires a separate, tested constraint adapter.
        if any("tip4p" in token.lower() for token in tokens):
            raise ValueError("Implicit TIP4P is not supported by the general rigid-molecule adapter.")
    return values


def _guest_template(path, guest):
    """Validate a single native molecule with explicit neutral charges and IDs."""
    sections = {}
    section = None
    count = None
    for raw in Path(path).read_text().splitlines()[1:]:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) == 2 and tokens[1] == "atoms":
            count = integer(tokens[0], "template atom count")
        elif line[0].isalpha():
            section = line
            sections[section] = []
        elif section:
            sections[section].append(tokens)
    if count != guest["atoms_per_molecule"]:
        raise ValueError("guest.atoms_per_molecule does not match the molecule template.")
    for name, width in (("Coords", 4), ("Types", 2), ("Charges", 2)):
        records = sections.get(name, [])
        if len(records) != count or any(len(row) != width for row in records):
            raise ValueError(f"Guest template requires explicit {name} for each atom.")
        if sorted(integer(row[0], "template atom ID") for row in records) != list(range(1, count + 1)):
            raise ValueError("Guest template IDs must be unique and contiguous.")
    for row in sections["Coords"]:
        for coordinate in row[1:]:
            finite(coordinate, "guest coordinate", minimum=-math.inf)
    charge = sum(finite(row[1], "guest charge", minimum=-math.inf) for row in sections["Charges"])
    if abs(charge) > 1e-6:
        raise ValueError("General Module D requires a neutral guest molecule.")
    return sorted({integer(row[1], "guest atom type") for row in sections["Types"]})


def build_general_plan(config, module):
    config = deepcopy(config)
    p = config["benchmark"]["flexibility"]
    simulation = config["simulation"]
    if simulation["engine"] != "LAMMPS" or simulation.get("framework") != "flexible":
        raise ValueError("General Module D requires flexible framework and LAMMPS.")
    if len(config["adsorbates"]["components"]) != 1:
        raise ValueError("The general adapter currently supports one guest component.")
    for key in ("potential", "assets", "guest", "md", "hybrid", "reservoir"):
        if not isinstance(p.get(key), dict):
            raise ValueError(f"flexibility.{key} must be an object.")
    potential = p["potential"]
    if potential.get("kind") not in {"classical", "mlip"}:
        raise ValueError("potential.kind must be classical or mlip.")
    if not potential.get("name") or not potential.get("source"):
        raise ValueError("potential.name and potential.source are required.")
    if potential.get("units") not in {"real", "metal"}:
        raise ValueError("potential.units must be real or metal.")
    _commands(potential.get("styles"), {"pair_style", "bond_style", "angle_style",
              "dihedral_style", "improper_style"}, "potential.styles")
    if not any(shlex.split(v)[0] == "pair_style" for v in potential["styles"]):
        raise ValueError("potential.styles must specify pair_style.")
    _commands(potential.get("coefficients"), {"pair_coeff", "bond_coeff", "angle_coeff",
              "dihedral_coeff", "improper_coeff", "pair_modify", "special_bonds",
              "kspace_style", "kspace_modify"}, "potential.coefficients")
    files = potential.setdefault("files", {})
    if not isinstance(files, dict) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", k) for k in files):
        raise ValueError("potential.files maps plain filenames to source paths.")
    guest = p["guest"]
    if guest.get("kind") not in {"atomic", "rigid_molecule"}:
        raise ValueError("guest.kind must be atomic or rigid_molecule; flexible guests need a separate adapter.")
    guest["atoms_per_molecule"] = integer(guest.get("atoms_per_molecule", 1), "atoms_per_molecule")
    if guest["kind"] == "atomic":
        guest["type"] = integer(guest.get("type"), "guest.type")
        if guest["atoms_per_molecule"] != 1:
            raise ValueError("Atomic guests require atoms_per_molecule=1.")
        guest["type_ids"] = [guest["type"]]
    elif guest["atoms_per_molecule"] < 2:
        raise ValueError("Rigid molecules require at least two atoms.")
    if guest["kind"] == "rigid_molecule" and ("intra_energy" not in guest or not guest.get("intra_energy_source")):
        raise ValueError("Rigid guests require explicit intra_energy (potential units) and intra_energy_source.")
    guest["intra_energy"] = finite(guest.get("intra_energy", 0), "guest.intra_energy", minimum=-math.inf)
    # Native fix gcmc has no flexible intramolecular insertion sampler.
    for key in ("extra_bond_per_atom", "extra_angle_per_atom", "extra_dihedral_per_atom",
                "extra_improper_per_atom", "extra_special_per_atom"):
        guest[key] = integer(guest.get(key, 0), f"guest.{key}", 0)
    md = p["md"]
    if md.get("ensemble") not in {"NVT", "NPT"}:
        raise ValueError("md.ensemble must be NVT or NPT.")
    md["timestep_fs"] = finite(md.get("timestep_fs", 0.5), "md.timestep_fs", positive=True)
    md["thermostat_damping_ps"] = finite(md.get("thermostat_damping_ps", .1), "md.thermostat_damping_ps", positive=True)
    if md["ensemble"] == "NPT":
        md["external_pressure_bar"] = finite(md.get("external_pressure_bar"), "md.external_pressure_bar", minimum=-math.inf)
        if md.get("cell_mode") not in {"aniso", "tri"}:
            raise ValueError("NPT requires explicit md.cell_mode = aniso or tri.")
        md["barostat_damping_ps"] = finite(md.get("barostat_damping_ps"), "md.barostat_damping_ps", positive=True)
    p["sr_cycles"] = integer(p.get("sr_cycles", 100), "sr_cycles")
    p["discard_cycles"] = integer(p.get("discard_cycles", 10), "discard_cycles", 0)
    p["convergence_window"] = integer(p.get("convergence_window", 10), "convergence_window", 2)
    if p["sr_cycles"] - p["discard_cycles"] < 2 * p["convergence_window"]:
        raise ValueError("Retained cycles must allow two convergence windows.")
    for key, default in (("uptake_drift_tolerance_mol_kg", .05), ("rmsd_drift_tolerance_A", .02),
                         ("volume_relative_drift_tolerance", .01)):
        p[key] = finite(p.get(key, default), key, positive=True)
    hybrid = p["hybrid"]
    for key, default in (("equilibration_steps", 10000), ("steps_per_cycle", 1000),
                         ("exchange_interval", 10), ("exchange_attempts", 10), ("move_attempts", 10)):
        hybrid[key] = integer(hybrid.get(key, default), f"hybrid.{key}", 0 if key == "move_attempts" else 1)
    if guest["kind"] == "rigid_molecule" and hybrid["move_attempts"] != 0:
        raise ValueError("Native LAMMPS gcmc rigid requires move_attempts=0; rigid-body MD samples translations/rotations.")
    if hybrid["equilibration_steps"] % hybrid["steps_per_cycle"] or hybrid["steps_per_cycle"] % hybrid["exchange_interval"]:
        raise ValueError("equilibration_steps must be a multiple of steps_per_cycle, itself a multiple of exchange_interval.")
    hybrid["max_displacement_A"] = finite(hybrid.get("max_displacement_A", 1), "max_displacement_A", positive=True)
    hybrid["overlap_cutoff_A"] = finite(hybrid.get("overlap_cutoff_A", 0), "overlap_cutoff_A")
    conditions = config["conditions"]
    temperature = finite(conditions["temperature_K"], "temperature_K", positive=True)
    pressures = [finite(v, "pressure_bar", positive=True) for v in conditions["pressures_bar"]]
    if not pressures or len(set(pressures)) != len(pressures):
        raise ValueError("General profiles require unique, positive pressures_bar.")
    reservoir = p["reservoir"]
    fugacities = reservoir.get("fugacities_bar")
    if not isinstance(fugacities, list) or len(fugacities) != len(pressures) or not reservoir.get("source"):
        raise ValueError("reservoir requires one explicit fugacities_bar value per pressure and a source.")
    fugacities = [finite(v, "fugacity_bar", positive=True) for v in fugacities]
    seeds = simulation["seeds"]
    if any(seed >= 900000000 for seed in seeds):
        raise ValueError("LAMMPS seeds must be less than 900000000.")
    roles = {"initial_structure": p["assets"].get("initial_structure")}
    if guest["kind"] == "rigid_molecule":
        roles["guest_template"] = p["assets"].get("guest_template")
    roles.update({f"potential/{k}": v for k, v in files.items()})
    if potential["kind"] == "mlip":
        roles["mlip_model"] = p["assets"].get("mlip_model")
    if p["assets"].get("validation"):
        roles["validation"] = p["assets"]["validation"]
    assets = [asset(path, role) for role, path in roles.items()]
    if guest["kind"] == "rigid_molecule" and roles.get("guest_template") and Path(roles["guest_template"]).expanduser().is_file():
        guest["type_ids"] = _guest_template(Path(roles["guest_template"]).expanduser(), guest)
    blockers = [f"Missing asset: {a['role']}" for a in assets if a["status"] != "present"]
    if potential["kind"] == "mlip":
        blockers.append("MLIP execution adapter is not validated; planning and analysis only.")
    points = [dict(point_id=f"p{i:03d}", pressure_bar=v, fugacity_bar=fugacities[i], temperature_K=temperature)
              for i, v in enumerate(pressures)]
    cases = [dict(case_id=f"{variant}__{point['point_id']}__seed{seed}", variant=variant,
                  **point, seed=seed, expected_cycles=p["sr_cycles"])
             for variant in (("rigid", "hybrid_nvt", "hybrid_npt") if md["ensemble"] == "NPT"
                             else ("rigid", "hybrid_nvt")) for point in points for seed in seeds]
    output = Path(config["output"].get("directory") or "outputs/module_D_general")
    run_id = config["output"].get("run_id")
    if run_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(run_id)):
            raise ValueError("Invalid run_id.")
        output /= "runs/" + str(run_id)
    p["sampling_protocol"] = "native_same_potential_hybrid_md_gcmc"
    p["rmsd_convention"] = "block mean of sqrt(unwrapped framework MSD); center-of-mass drift removed; includes affine cell deformation in NPT"
    applicability = dict(status="blocked" if blockers else "technical_run_available",
        current_module_capability="classical_nvt_npt_execution_mlip_planning", can_attempt_simulation=not blockers,
        requires_user_confirmation=False, recommended_module="D", assets=assets, blockers=blockers,
        validation_status="not_validated", reason="Technical readiness is separate from material-specific physical validation.",
        notes=["No automatic force-field/topology/charge assignment. Input must be a dry, parameterized periodic framework.",
               "NVT fixes the cell; NPT permits anisotropic cell deformation. A changing volume is not proof of breathing. No RASPA dependency.",
               "Rigid and flexible cases use the same potential, reservoir, initial cell and sampling length.",
               "A short test is required before production. Neither a successful test nor a validation file proves physical accuracy."])
    return dict(status="planned", module=module, material=config["material"], adsorbates=config["adsorbates"],
        conditions=conditions, simulation=simulation, benchmark={**config["benchmark"], "flexibility": p,
        "applicability": applicability}, outputs={**config["output"], "directory": str(output)}, state_points=points, cases=cases)


def _destination(role):
    return Path("assets") / role


def verify_bundle(root):
    root = Path(root)
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    for name, expected in manifest.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid bundle path.")
        if asset(root / path, name).get("sha256") != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return json.loads((root / "module_d_plan.json").read_text())


def materialize_general(prepared):
    plan = prepared["run_plan"]
    root = Path(prepared["working_directory"])
    if root.exists():
        if plan["outputs"].get("resume"):
            old, new = verify_bundle(root), deepcopy(plan)
            for entry in (old, new):
                for flag in ("resume", "overwrite"):
                    entry["outputs"].pop(flag, None)
            if old == new:
                return dict(status="reused_plan", working_directory=str(root), simulation_started=False)
        raise FileExistsError(f"Existing Module D directory is preserved: {root}. Use an identical resume or a new run_id.")
    for entry in plan["benchmark"]["applicability"]["assets"]:
        if asset(entry["path"], entry["role"]) != entry:
            raise ValueError(f"Asset changed since planning: {entry['role']}")
    root.mkdir(parents=True)
    _json(root / "module_d_plan.json", plan)
    for entry in plan["benchmark"]["applicability"]["assets"]:
        if entry["status"] == "present":
            dest = root / _destination(entry["role"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(entry["path"]).expanduser(), dest)
    with (root / "cases.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan["cases"][0]))
        writer.writeheader()
        writer.writerows(plan["cases"])
    with (root / "cycle_results.template.csv").open("x", newline="") as handle:
        csv.writer(handle).writerow(CYCLE_COLUMNS)
    (root / "README.txt").write_text("Module D general profile. Preparation does not run LAMMPS.\n"
        "Run python -m modules.module_d_general check --directory THIS_DIRECTORY\n"
        "Then smoke, then run (same CLI); --command lmp selects the executable.\n"
        "Missing assets and MLIP execution remain blocked. No scientific validation is implied.\n")
    _json(root / "bundle_manifest.json", {str(path.relative_to(root)): asset(path, "file")["sha256"]
        for path in sorted(root.rglob("*")) if path.is_file()})
    return dict(status="prepared_general_bundle", working_directory=str(root), simulation_started=False,
                applicability=plan["benchmark"]["applicability"])


def render_input(plan, case):
    """All files are relative to a case directory containing an archived assets/.

    Native MD/GCMC retains one state and Hamiltonian throughout. Production
    blocks are measurements, not independent samples or paper SR cycles.
    """
    p = plan["benchmark"]["flexibility"]
    pot, guest, md, hybrid = (p[k] for k in ("potential", "guest", "md", "hybrid"))
    if pot["kind"] != "classical":
        raise NotImplementedError("Only classical execution is available.")
    real = pot["units"] == "real"
    timestep = md["timestep_fs"] * (1 if real else .001)
    damping = md["thermostat_damping_ps"] * (1000 if real else 1)
    pressure = case["pressure_bar"] / (1.01325 if real else 1)
    energy = .04336410424180094 if real else 1.0  # kcal/mol -> eV
    t, seed = case["temperature_K"], case["seed"]
    npt = case["variant"] == "hybrid_npt"
    extra = " ".join(f"extra/{key[6:].replace('_', '/')} {guest[key]}" for key in guest
                     if key.startswith("extra_") and guest[key])
    lines = ["clear", f"units {pot['units']}", "atom_style full", "boundary p p p", *pot["styles"],
             f"read_data assets/initial_structure {extra}", *pot["coefficients"],
             "neighbor 2.0 bin", "neigh_modify delay 0 every 1 check yes", "group host union all", "group gas empty",
             "variable hostmass equal mass(host)", "variable hostcount equal count(host)",
             'if "${hostcount} < 2" then "quit 1"',
             "compute q all property/atom q", "compute qsum all reduce sum c_q",
             "thermo_style custom step pe c_qsum", "run 0",
             'if "$(abs(c_qsum)) > 1.0e-5" then "quit 1"',
             "compute displacement host msd com yes", "variable rmsd equal sqrt(c_displacement[4])",
             f"variable uptake equal count(gas)/{guest['atoms_per_molecule']}*1000/v_hostmass",
             f"variable energy equal pe*{energy}", "variable volume equal vol", f"timestep {timestep}",
             "thermo 100", "thermo_style custom step atoms temp pe press vol v_uptake v_rmsd",
             "thermo_modify lost error flush yes", "compute_modify thermo_temp dynamic/dof yes"]
    if npt and md["cell_mode"] == "tri":
        # Already-triclinic data do not need this conversion. The LAMMPS
        # triclinic thermo keyword reports 0/1, including a zero-tilt box.
        lines += ['if "$(extract_setting(triclinic)) == 0" then "change_box all triclinic"']
    host_integration = []
    if case["variant"] != "rigid":
        velocity_seed = (seed + 104729) % 899999999 + 1
        lines += [f"velocity host create {t} {velocity_seed} mom yes rot no dist gaussian"]
        if npt:
            pext = md["external_pressure_bar"] / (1.01325 if real else 1)
            pdamp = md["barostat_damping_ps"] * (1000 if real else 1)
            host_integration += [f"fix hostmd host npt temp {t} {t} {damping} {md['cell_mode']} {pext} {pext} {pdamp} dilate all mtk yes flip no",
                      "compute globaltemp all temp", "compute_modify globaltemp dynamic/dof yes",
                      "compute globalpress all pressure globaltemp", "fix_modify hostmd press globalpress"]
        else:
            host_integration += [f"fix hostmd host nvt temp {t} {t} {damping}"]
    lines += ["group guesttypes type " + " ".join(str(v) for v in guest["type_ids"]),
              "variable existing equal count(guesttypes)",
              'if "${existing} != 0" then "quit 1"', "group guesttypes delete"]
    if guest["kind"] == "atomic":
        lines += ["compute gastemp gas temp", "compute_modify gastemp dynamic/dof yes extra/dof 0",
                  f"fix guestmd gas nvt temp {t} {t} {damping}", "fix_modify guestmd temp gastemp"]
        options = ""
        guest_type = guest["type"]
    else:
        lines += ["molecule guestmol assets/guest_template",
                  f"fix guestmd gas rigid/nvt/small molecule temp {t} {t} {damping} mol guestmol",
                  "fix_modify guestmd dynamic/dof yes"]
        guest_type = 0
        options = f"mol guestmol rigid guestmd intra_energy {guest['intra_energy']}"
    # LAMMPS requires rigid-body fixes before any box-changing fix so its
    # remap callback scales molecular centers, not internal geometry.
    lines += host_integration
    if hybrid.get("overlap_cutoff_A", 0):
        options += f" overlap_cutoff {hybrid['overlap_cutoff_A']}"
    lines += [f"fix exchange gas gcmc {hybrid['exchange_interval']} {hybrid['exchange_attempts']} "
              f"{hybrid['move_attempts']} {guest_type} {seed} {t} 0 {hybrid['max_displacement_A']} "
              f"pressure {pressure} fugacity_coeff {case['fugacity_bar']/case['pressure_bar']} full_energy {options}",
              f"run {hybrid['equilibration_steps']}"]
    n = hybrid["steps_per_cycle"]
    for name in ("lx", "ly", "lz", "xy", "xz", "yz", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz"):
        lines += [f"variable cell_{name} equal {name}"]
    lines += [f"fix cellstats all ave/time 1 {n} {n} " + " ".join(f"v_cell_{name}" for name in
              ("lx", "ly", "lz", "xy", "xz", "yz", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz")) + " file cell_history.txt"]
    lines += [f"fix stats all ave/time 1 {n} {n} v_uptake v_rmsd v_volume v_energy file measurements.txt",
              f"fix attempts all ave/time 1 1 {n} f_exchange[1] f_exchange[2] f_exchange[3] f_exchange[4] "
              "f_exchange[5] f_exchange[6] f_exchange[7] f_exchange[8] file mc_attempts.txt",
              f"dump trajectory all custom {n} trajectory.lammpstrj id mol type q x y z ix iy iz",
              "dump_modify trajectory sort id",
              f"run {n * case['expected_cycles']}", "write_restart final.restart", "write_data final.data",
              'print "MODULE_D_COMPLETED"']
    return "\n".join(lines) + "\n"


def _measurements(path, case, hybrid):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError("Unexpected LAMMPS measurement columns.")
        step, uptake, rmsd, volume, energy = map(float, values)
        expected_step = hybrid["equilibration_steps"] + (len(rows) + 1) * hybrid["steps_per_cycle"]
        if step != expected_step:
            raise ValueError("LAMMPS measurement timesteps do not match the planned blocks.")
        row = dict(case_id=case["case_id"], cycle=len(rows) + 1,
                   uptake_mol_kg=finite(uptake, "uptake"), framework_rmsd_A=finite(rmsd, "rmsd"),
                   cell_volume_A3=finite(volume, "volume", positive=True),
                   potential_energy_eV=finite(energy, "energy", minimum=-math.inf))
        rows.append(row)
    if len(rows) != case["expected_cycles"]:
        raise ValueError("LAMMPS returned an incomplete number of measurement blocks.")
    return rows


def execute_bundle(directory, *, smoke=False, command=None):
    root = Path(directory).resolve()
    plan = verify_bundle(root)
    blockers = plan["benchmark"]["applicability"]["blockers"]
    if blockers:
        raise ValueError("Module D execution blocked: " + "; ".join(blockers))
    command = command or ["lmp"]
    if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command):
        raise ValueError("command must be a non-empty argument list (no shell evaluation).")
    fingerprint = asset(root / "bundle_manifest.json", "manifest")["sha256"]
    implementation = hashlib.sha256(Path(__file__).read_bytes() +
        (Path(__file__).parents[1] / "analysis/module_d_flexibility.py").read_bytes()).hexdigest()
    version = subprocess.run([*command, "-help"], capture_output=True, text=True, check=True, timeout=60)
    help_text = version.stdout + version.stderr
    runtime_hash = hashlib.sha256(help_text.encode()).hexdigest()
    if not smoke:
        marker = root / "smoke/status.json"
        if not marker.is_file():
            raise ValueError("Run the smoke test on this bundle before production.")
        previous = json.loads(marker.read_text())
        if previous.get("status") != "passed" or previous.get("bundle_sha256") != fingerprint or previous.get("command") != command:
            raise ValueError("A passing smoke test for this exact bundle and LAMMPS command is required.")
        if previous.get("implementation_sha256") != implementation or previous.get("lammps_help_sha256") != runtime_hash:
            raise ValueError("Runner or LAMMPS build changed since the smoke test; use a fresh bundle and smoke test.")
        if (root / "cycle_results.csv").exists():
            raise FileExistsError("Existing cycle_results.csv is preserved.")
    target = root / ("smoke" if smoke else "results")
    target.mkdir()  # Never overwrite or silently restart a partial calculation.
    state = dict(status="running", kind="technical_smoke" if smoke else "production",
                 bundle_sha256=fingerprint, implementation_sha256=implementation,
                 lammps_help_sha256=runtime_hash, command=command, validation_status="not_validated", cases=[])
    _json(target / "status.json", state)
    working = deepcopy(plan)
    if smoke:
        p = working["benchmark"]["flexibility"]
        p.update(sr_cycles=4, discard_cycles=0, convergence_window=2)
        n = max(10, p["hybrid"]["exchange_interval"])
        n = math.ceil(n / p["hybrid"]["exchange_interval"]) * p["hybrid"]["exchange_interval"]
        p["hybrid"].update(equilibration_steps=n, steps_per_cycle=n)
        working["cases"] = [dict(c, expected_cycles=4) for c in working["cases"]
                            if c["point_id"] == plan["state_points"][0]["point_id"] and c["seed"] == plan["simulation"]["seeds"][0]]
    _json(target / "execution_plan.json", working)
    all_rows = []
    try:
        (target / "lammps_help.txt").write_text(help_text)
        for case in working["cases"]:
            case_dir = target / case["case_id"]
            case_dir.mkdir()
            shutil.copytree(root / "assets", case_dir / "assets")
            (case_dir / "in.lammps").write_text(render_input(working, case))
            start = time.monotonic()
            with (case_dir / "console.log").open("x") as handle:
                subprocess.run([*command, "-in", "in.lammps"], cwd=case_dir, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            if "MODULE_D_COMPLETED" not in (case_dir / "console.log").read_text():
                raise ValueError("LAMMPS did not reach the completion marker.")
            rows = _measurements(case_dir / "measurements.txt", case, working["benchmark"]["flexibility"]["hybrid"])
            all_rows.extend(rows)
            state["cases"].append(dict(case_id=case["case_id"], status="complete", runtime_s=time.monotonic() - start))
            _json(target / "status.json", state)
        from analysis.module_d_flexibility import summarize_cycles
        report = summarize_cycles(working, all_rows)
        _json(target / "report.json", report)
        destination = target / "cycle_results.csv" if smoke else root / "cycle_results.csv"
        with destination.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CYCLE_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
        state["status"] = "passed" if smoke else "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _json(target / "status.json", state)
    return state


def run_general(plan, *, smoke):
    from modules.module_d_flexibility import prepare_flexibility
    if plan["benchmark"]["applicability"]["blockers"]:
        raise ValueError("Module D execution blocked: " + "; ".join(plan["benchmark"]["applicability"]["blockers"]))
    materialize_general(prepare_flexibility(plan))
    return execute_bundle(plan["outputs"]["directory"], smoke=smoke)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "smoke", "run"])
    parser.add_argument("--directory", required=True)
    parser.add_argument("--command", nargs="+", default=["lmp"], help="LAMMPS executable/launcher argument list; no shell")
    args = parser.parse_args(argv)
    if args.action == "check":
        plan = verify_bundle(args.directory)
        result = plan["benchmark"]["applicability"]
    else:
        result = execute_bundle(args.directory, smoke=args.action == "smoke", command=args.command)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if args.action == "check" and result["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
