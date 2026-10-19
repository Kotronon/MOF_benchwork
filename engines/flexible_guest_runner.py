"""LAMMPS-backed CBMC for flexible guests in a fixed Module-A framework."""

from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from time import perf_counter
import numpy as np

from engines.configurational_bias import (
    ConfigurationalBias,
    TreeProposal,
    RigidProposal,
)


def load_models(paths, temperature):
    models = {}
    for c, path in paths.items():
        model = (
            json.loads(Path(path).read_text())
            if isinstance(path, str)
            else deepcopy(path)
        )
        if model.get("rigid", False):
            raise ValueError(
                "Rigid guests use the original molecule definition, not a flexible model."
            )
        if model.get("rings") or model.get("constraints"):
            raise ValueError(
                "Flexible rings/constraints require a dedicated proposal backend."
            )
        parents = model["growth_parents"]
        bonds = {tuple(sorted(item["atoms"])) for item in model.get("bonds", [])}
        tree = {tuple(sorted((i, p))) for i, p in enumerate(parents) if p is not None}
        if len(bonds) != len(model.get("bonds", [])) or bonds != tree:
            raise ValueError(
                "Physical bonds must form exactly the declared acyclic growth tree."
            )
        norms = model["ideal_log_rosenbluth"]
        key = next(
            (k for k in norms if math.isclose(float(k), temperature, abs_tol=1e-9)),
            None,
        )
        if key is None or not model.get("normalization_source"):
            raise ValueError(
                "A documented ideal Rosenbluth normalization at this temperature is required."
            )
        model["normalization"] = float(norms[key])
        TreeProposal(
            parents,
            model["reference_bond_k_kcal_mol_A2"],
            temperature,
            model["normalization"],
        )
        for field, style in (
            ("bonds", "harmonic"),
            ("angles", "harmonic"),
            ("dihedrals", "opls"),
            ("impropers", "harmonic"),
        ):
            if model.get(field) and model.get(field[:-1] + "_style", style) != style:
                raise ValueError(f"Unsupported {field} style.")
        for kind, arity, coefficient_count in (
            ("bonds", 2, 2),
            ("angles", 3, 2),
            ("dihedrals", 4, 4),
            ("impropers", 4, 2),
        ):
            for term in model.get(kind, []):
                ids, coeff = term["atoms"], term["coeff"]
                if (
                    len(ids) != arity
                    or len(set(ids)) != arity
                    or any(
                        not isinstance(i, int) or i < 0 or i >= len(parents)
                        for i in ids
                    )
                ):
                    raise ValueError(f"Invalid {kind} atom indices.")
                if len(coeff) != coefficient_count or any(
                    not math.isfinite(float(x)) for x in coeff
                ):
                    raise ValueError(f"Invalid {kind} coefficients.")
        special = model.get("special_bonds", [0.0, 0.0, 0.0])
        if len(special) != 3 or any(
            not math.isfinite(v) or v < 0 or v > 1 for v in special
        ):
            raise ValueError("special_bonds requires three weights in [0, 1].")
        models[c] = model
    return models


class LammpsGuestEnergy:
    """Rebuild topology for each trial. Correctness-first backend, no framework motion."""

    def __init__(self, run, models):
        from lammps import lammps
        from ase.io import read

        self.run = run
        self.models = models
        self.framework = read(
            run["framework_data"],
            format="lammps-data",
            atom_style="full",
            sort_by_id=True,
        )
        self.cell = np.asarray(self.framework.cell)
        self.types = self.framework.arrays["type"].tolist()
        self.charges = self.framework.arrays.get(
            "initial_charges", np.zeros(len(self.framework))
        ).tolist()
        self.lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
        self.tmp = tempfile.TemporaryDirectory(prefix="module_b_cbmc_")
        self.data = Path(self.tmp.name) / "trial.data"
        self.reference_coords = self.framework.positions.copy()
        self.topologies = {}
        self.coeffs = {k: [] for k in ("bonds", "angles", "dihedrals", "impropers")}
        for c, model in models.items():
            self.topologies[c] = {}
            for kind in self.coeffs:
                self.topologies[c][kind] = []
                for item in model.get(kind, []):
                    self.coeffs[kind].append(item["coeff"])
                    self.topologies[c][kind].append(
                        (len(self.coeffs[kind]), item["atoms"])
                    )
        for c, atoms in run["guest_atoms"].items():
            if c not in models:
                self.topologies[c] = {"bonds": []}
                if len(atoms) > 1:
                    self.coeffs["bonds"].append([0.0, 0.0])
                    typ = len(self.coeffs["bonds"])
                    self.topologies[c]["bonds"] = [
                        (typ, [i, j])
                        for i in range(len(atoms))
                        for j in range(i + 1, len(atoms))
                    ]
        # Parse all masses, including initially empty gas types, from the base data.
        text = Path(run["framework_data"]).read_text()
        section = text.split("Masses", 1)[1].split("Atoms", 1)[0]
        self.masses = {
            int(p[0]): float(p[1])
            for line in section.splitlines()
            if (p := line.split()) and p[0].isdigit() and len(p) >= 2
        }
        self.cache = {}

    def __call__(self, molecules):
        return self.energy_forces(molecules)[0]

    def energy_forces(self, molecules):
        key = json.dumps(molecules, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        atoms = [
            (i + 1, 0, t, q, *xyz)
            for i, (t, q, xyz) in enumerate(
                zip(self.types, self.charges, self.reference_coords)
            )
        ]
        topo = {k: [] for k in self.coeffs}
        for mid, m in enumerate(molecules, 1):
            c = m["component"]
            spec = self.run["guest_atoms"][c]
            offset = len(atoms)
            for atom, xyz in zip(spec, m["positions"]):
                atoms.append((len(atoms) + 1, mid, atom["type"], atom["charge"], *xyz))
            for kind, items in self.topologies.get(c, {}).items():
                first_id = len(topo[kind]) + 1
                topo[kind].extend(
                    (first_id + i, typ, *[offset + j + 1 for j in ids])
                    for i, (typ, ids) in enumerate(items)
                )

        lines = [
            "LAMMPS CBMC fixed framework",
            "",
            f"{len(atoms)} atoms",
            f"{len(self.masses)} atom types",
        ]
        for kind, items in topo.items():
            if self.coeffs[kind]:
                lines.extend(
                    [
                        f"{len(items)} {kind}",
                        f"{len(self.coeffs[kind])} {kind[:-1]} types",
                    ]
                )
        cell = self.cell
        lines += [
            f"0 {cell[0,0]:.16g} xlo xhi",
            f"0 {cell[1,1]:.16g} ylo yhi",
            f"0 {cell[2,2]:.16g} zlo zhi",
            f"{cell[1,0]:.16g} {cell[2,0]:.16g} {cell[2,1]:.16g} xy xz yz",
            "",
            "Masses",
            "",
        ]
        lines += [f"{t} {m:.16g}" for t, m in self.masses.items()]
        lines += ["", "Atoms # full", ""] + [
            " ".join(str(v) for v in row) for row in atoms
        ]
        for kind, items in topo.items():
            if items:
                lines += ["", kind.capitalize(), ""] + [
                    " ".join(map(str, row)) for row in items
                ]
        self.data.write_text("\n".join(lines) + "\n")
        commands = [
            "clear",
            "units real",
            "atom_style full",
            "boundary p p p",
            "atom_modify map array",
        ]
        for kind, coeffs in self.coeffs.items():
            if coeffs:
                commands.append(
                    f"{kind[:-1]}_style {'opls' if kind=='dihedrals' else 'harmonic'}"
                )
        commands += [
            f"read_data {self.data}",
            f"include {Path(self.run['forcefield_include']).resolve()}",
        ]
        for kind, coeffs in self.coeffs.items():
            for i, values in enumerate(coeffs, 1):
                commands.append(f"{kind[:-1]}_coeff {i} " + " ".join(map(str, values)))
        specials = {
            tuple(m.get("special_bonds", [0.0, 0.0, 0.0])) for m in self.models.values()
        }
        if len(specials) > 1:
            raise ValueError(
                "This backend requires the same special_bonds convention across flexible guests."
            )
        special = next(iter(specials), (0.0, 0.0, 0.0))
        if special[0] != 0 and any(
            c not in self.models and len(a) > 1
            for c, a in self.run["guest_atoms"].items()
        ):
            raise ValueError(
                "Multisite rigid guests require excluded 1-2 interactions in this backend."
            )
        commands.append("special_bonds lj/coul " + " ".join(map(str, special)))
        commands += ["thermo 1", "thermo_style custom step atoms pe", "run 0"]
        self.lmp.commands_list(commands)
        value = float(self.lmp.get_thermo("pe"))
        if not math.isfinite(value):
            raise ValueError("Nonfinite LAMMPS trial energy.")
        if len(self.cache) > 256:
            self.cache.clear()
        all_forces = (
            np.ctypeslib.as_array(
                self.lmp.gather_atoms("f", 1, 3), shape=(3 * len(atoms),)
            )
            .reshape(-1, 3)
            .copy()
        )
        forces = []
        offset = len(self.framework)
        for m in molecules:
            n = len(m["positions"])
            forces.append(all_forces[offset : offset + n])
            offset += n
        self.cache[key] = (value, forces)
        return value, forces

    def close(self):
        self.lmp.close()
        self.tmp.cleanup()


def run_pressure_point(run, index, total_runs, discard_fraction):
    from pipeline.config import save_benchmark_data
    from pipeline.runners import _pressure_point_result
    from parsers.lammps_log_parser import log_to_json

    models = load_models(run["guest_models"], run["temperature_K"])
    backend = LammpsGuestEnergy(run, models)
    start = perf_counter()
    checkpoint = Path(run["log"]).with_suffix(".cbmc.json")
    log = Path(run["log"])
    summary_file = log.with_name(log.stem + "_summary.json")
    setup = {k: v for k, v in run.items() if k not in {"command", "resume"}}
    signature = hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()
    proposals = {
        c: TreeProposal(
            m["growth_parents"],
            m["reference_bond_k_kcal_mol_A2"],
            run["temperature_K"],
            m["normalization"],
        )
        for c, m in models.items()
    }
    for c, spec in run["guest_atoms"].items():
        if c not in proposals:
            proposals[c] = RigidProposal([a["position"] for a in spec])
    sampler = ConfigurationalBias(
        backend,
        proposals,
        run["reservoir"]["fugacities_bar"],
        run["temperature_K"],
        backend.cell,
        run["seed"],
        run.get("cbmc_trials", 8),
    )
    samples = []
    step = 0
    try:
        if checkpoint.exists() and run.get("resume"):
            previous = json.loads(checkpoint.read_text())
            if previous["signature"] != signature:
                raise ValueError("CBMC checkpoint belongs to a different setup.")
            sampler.restore(previous["sampler"])
            samples = previous["samples"]
            step = previous["step"]
        for step in range(step + 1, run["run_steps"] + 1):
            sampler.step()
            hmc = run.get("hmc", {})
            if hmc.get("steps", 0) and step % int(hmc.get("every", 10)) == 0:
                sampler.hmc(
                    backend.energy_forces,
                    {
                        c: [backend.masses[a["type"]] for a in spec]
                        for c, spec in run["guest_atoms"].items()
                    },
                    steps=int(hmc["steps"]),
                    step_size=float(hmc.get("step_size", 0.01)),
                )
            count = {
                c: sum(m["component"] == c for m in sampler.molecules)
                for c in proposals
            }
            samples.append(
                [
                    step,
                    len(backend.framework)
                    + sum(count[c] * len(run["guest_atoms"][c]) for c in proposals),
                    *[count[c] for c in proposals],
                ]
            )
            if step % 10 == 0 or step == run["run_steps"]:
                from modules.module_b_active_learning import _save

                _save(
                    checkpoint,
                    {
                        "signature": signature,
                        "step": step,
                        "sampler": sampler.checkpoint(),
                        "samples": samples,
                    },
                )
        log.write_text(
            "Step Atoms "
            + " ".join("v_n" + c.lower() for c in proposals)
            + "\n"
            + "\n".join(" ".join(map(str, row)) for row in samples)
            + "\nLoop time of 0 on 1 procs for "
            + str(run["run_steps"])
            + " steps\n"
        )
        summary = log_to_json(
            log,
            summary_file,
            framework_atoms=len(backend.framework),
            adsorbate_atoms_per_molecule=1,
            discard_fraction=discard_fraction,
            component_atom_counts={c: len(v) for c, v in run["guest_atoms"].items()},
            mixture=run["mixture"],
        )
        summary["sampling"] = {
            "method": "whole_tree_CBMC",
            "counts": sampler.counts,
            "framework_rigid": True,
            "framework_coordinate_sha256": hashlib.sha256(
                backend.reference_coords.tobytes()
            ).hexdigest(),
            "normalizations": {c: m["normalization"] for c, m in models.items()},
        }
        save_benchmark_data(summary_file, summary)
        return _pressure_point_result(
            run,
            ["python", "CBMC/LAMMPS"],
            log,
            summary_file,
            discard_fraction,
            perf_counter() - start,
            summary,
        )
    finally:
        backend.close()
