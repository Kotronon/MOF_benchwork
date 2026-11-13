"""Reproducible MIL-53(Al)/CO2 pilot preparation, retaining published host terms.

The CO2 cross potential is a documented project model, not a published fitted
MIL-53/CO2 model. Source files and derived parameter data are CC BY-SA 4.0.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

REFERENCE = Path(__file__).parents[1] / "data/module_d/mil53_al_reference"
KB_KCAL = 0.00198720425864083
COULOMB_REAL = 332.06371  # LAMMPS real units, same constant as the source tables


def source_sections(text):
    """Read named sections without reordering or inferring any topology."""
    sections = defaultdict(list)
    current = "header"
    names = {"Masses", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs",
             "Pair Coeffs", "Atoms", "Bonds", "Angles", "Dihedrals", "Impropers", "Velocities"}
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip()
        if clean in names:
            current = clean
        else:
            sections[current].append(line)
    return dict(sections)


def nonempty(lines):
    return [line.split("#", 1)[0].split() for line in lines if line.split("#", 1)[0].strip()]


def pair_energy_force(r, sigma, epsilon, charge_product=0., gaussian_radius=0.):
    """LJ plus Gaussian/point Coulomb correction; base coul/long adds 1/r."""
    s6 = (sigma / r) ** 6
    energy = 4 * epsilon * (s6 * s6 - s6)
    force = 24 * epsilon * (2 * s6 * s6 - s6) / r
    if gaussian_radius:
        erfc = math.erfc(r / gaussian_radius)
        energy -= COULOMB_REAL * charge_product * erfc / r
        force -= COULOMB_REAL * charge_product * (erfc / r ** 2 +
            2 * math.exp(-(r / gaussian_radius) ** 2) / (math.sqrt(math.pi) * gaussian_radius * r))
    return energy, force


def verify_sources():
    manifest = json.loads((REFERENCE / "provenance.json").read_text())
    for entry in manifest["files"]:
        if hashlib.sha256((REFERENCE / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Modified MIL-53 reference: {entry['path']}")
    return manifest


def prepare_mil53(directory, supercell="2x4x2"):
    manifest = verify_sources()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    original = REFERENCE / ("lammps.data" if supercell == "1x2x1" else "lammps_2x4x2.data")
    sections = source_sections(original.read_text())
    atoms = nonempty(sections["Atoms"])
    charges = {}
    for row in atoms:
        atom_type, charge = int(row[2]), float(row[3])
        if atom_type in charges and abs(charge - charges[atom_type]) > 1e-9:
            raise ValueError("Cross tables require a single source charge per host atom type.")
        charges[atom_type] = charge
    if set(charges) != set(range(1, 9)) or abs(sum(float(row[3]) for row in atoms)) > 1e-5:
        raise ValueError("Unexpected host types or nonneutral source structure.")
    # Add only guest capacity; preserve every original host term and atom.
    text = original.read_text()
    text = re.sub(r"\b8 atom types", "10 atom types", text)
    text = re.sub(r"\b25 bond types", "26 bond types", text)
    text = re.sub(r"\n\s*Bond Coeffs\n", "\n9 12.0107 # TraPPE CO2 C\n10 15.9994 # TraPPE CO2 O\n\nBond Coeffs\n", text)
    text = re.sub(r"\n\s*Angle Coeffs\n", "\n26 zero 1.16 # exclusion edges only; rigid guest\n\nAngle Coeffs\n", text)
    text = text.replace("eibonded.table", "assets/potential/eibonded.table")
    (root / "dry.data").write_text(text)
    for filename in ("lammps.table", "eibonded.table"):
        shutil.copyfile(REFERENCE / filename, root / filename)
    (root / "co2.mol").write_text("""TraPPE CO2; 3 zero-energy edges exclude every intramolecular pair

3 atoms
3 bonds

Coords

1 0 0 0
2 0 0 1.16
3 0 0 -1.16

Types

1 9
2 10
3 10

Charges

1 0.7
2 -0.35
3 -0.35

Bonds

1 26 1 2
2 26 1 3
3 26 2 3
""")
    labels = ["AL", "C_CA", "C_PC", "C_PH", "H_HY", "H_PH", "O_CA", "O_HY"]
    radii, mm3 = {}, {}
    for line in (REFERENCE / "pars.txt").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "FIXQ:ATOM":
            radii[parts[1]] = float(parts[3])
        if parts and parts[0] == "MM3:PARS":
            mm3[parts[1]] = (float(parts[2]), float(parts[3]))
    guest = {9: dict(sigma=2.8, epsilon=27 * KB_KCAL, charge=.7),
             10: dict(sigma=3.05, epsilon=79 * KB_KCAL, charge=-.35)}
    pairs = []
    for i, label in enumerate(labels, 1):
        host_sigma, host_epsilon = mm3[label]
        # Same-location-of-minimum mapping used for host/guest LJ in the
        # MIL-53 water study (DOI 10.1039/D0FD00025F), here adapted to CO2.
        host_sigma *= 2 ** (5 / 6)
        for j, gas in guest.items():
            pairs.append(dict(i=i, j=j, sigma=(host_sigma + gas["sigma"]) / 2,
                epsilon=math.sqrt(host_epsilon * gas["epsilon"]),
                charge_product=charges[i] * gas["charge"], gaussian_radius=radii[label]))
    for i in guest:
        for j in guest:
            if i <= j:
                pairs.append(dict(i=i, j=j, sigma=(guest[i]["sigma"] + guest[j]["sigma"]) / 2,
                    epsilon=math.sqrt(guest[i]["epsilon"] * guest[j]["epsilon"]),
                    charge_product=0., gaussian_radius=0.))
    with (root / "co2_cross.table").open("x") as handle:
        handle.write("# Project MIL-53/CO2 pilot: real units (angstrom, kcal/mol, kcal/mol/angstrom)\n")
        for pair in pairs:
            handle.write(f"\nPAIR_{pair['i']}_{pair['j']}\nN 5000 R 0.5 15.0\n\n")
            params = {key: value for key, value in pair.items() if key not in {"i", "j"}}
            for n in range(5000):
                r = .5 + n * 14.5 / 4999
                e, f = pair_energy_force(r, **params)
                handle.write(f"{n+1} {r:.12g} {e:.16g} {f:.16g}\n")
    styles, coefficients = [], []
    for line in (REFERENCE / "lammps.in").read_text().splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        if parts[0] in {"pair_style", "bond_style", "angle_style", "dihedral_style", "improper_style"}:
            if parts[0] == "bond_style":
                parts.append("zero")
            styles.append(" ".join(parts))
        if parts[0] in {"pair_coeff", "special_bonds", "kspace_style", "pair_modify"}:
            coefficients.append(" ".join(parts).replace("lammps.table", "assets/potential/lammps.table"))
    coefficients += [f"pair_coeff {p['i']} {p['j']} table assets/potential/co2_cross.table PAIR_{p['i']}_{p['j']}" for p in pairs]
    provenance = dict(source=manifest, source_structure=original.name, supercell=supercell,
        license="CC-BY-SA-4.0", validation_status="not_validated", host_atoms=len(atoms),
        host_charge=sum(float(row[3]) for row in atoms), guest_source="Potoff and Siepmann, DOI 10.1002/aic.690470719",
        cross_model="Project adaptation: LJ minimum-matched MM3 host parameters, Lorentz-Berthelot, Gaussian-host/point-guest electrostatics.",
        cross_method_reference="MIL-53/water host-guest LJ conversion: DOI 10.1039/D0FD00025F; this is not a fitted MIL-53/CO2 force field.",
        cutoff_A=15., tail_corrections="Native table styles provide no analytical tail correction, despite upstream pair_modify tail yes.",
        overlap_rejection_A=.75, guest_pairs=pairs,
        modifications=["Host topology, charges and all host coefficients unchanged.",
            "Reserved guest types/masses and zero-energy exclusion bond type added.",
            "Three rigid CO2 exclusion edges make every intramolecular pair 1-2 under the host's special_bonds rules.",
            "Bond table paths rewritten to archived relative locations."])
    (root / "model_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (root / "ATTRIBUTION.txt").write_text((REFERENCE / "ATTRIBUTION.txt").read_text() +
        "Derived files in this directory retain CC BY-SA 4.0. Changes are listed in model_provenance.json.\n")
    config = dict(material=dict(name="MIL-53(Al)", charge_scheme="published_QuickFF_Gaussian"),
        adsorbates=dict(components=["CO2"]), conditions=dict(temperature_K=300., pressures_bar=[1.]),
        simulation=dict(engine="LAMMPS", framework="flexible", seeds=[12345, 23456, 34567]),
        output=dict(directory="outputs/module_D_mil53_co2", run_id="classical_pilot_v1"),
        benchmark=dict(task="module_d", flexibility=dict(profile="general", sr_cycles=20,
            discard_cycles=4, convergence_window=4,
            potential=dict(kind="classical", name="Published MIL-53 QuickFF + project TraPPE-CO2 cross model",
                source="Neirynck et al. DOI 10.1021/acsphyschemau.5c00141; adaptations in model_provenance.json",
                units="real", styles=styles, coefficients=coefficients,
                files={name: str(root / name) for name in ("lammps.table", "eibonded.table", "co2_cross.table")}),
            assets=dict(initial_structure=str(root / "dry.data"), guest_template=str(root / "co2.mol"),
                        validation=str(root / "model_provenance.json")),
            guest=dict(kind="rigid_molecule", atoms_per_molecule=3, intra_energy=0.,
                intra_energy_source="Zero bonded energy and every intramolecular nonbonded pair excluded through zero-energy edges.",
                extra_bond_per_atom=3, extra_special_per_atom=4),
            md=dict(ensemble="NPT", cell_mode="tri", external_pressure_bar=1.,
                barostat_damping_ps=1., thermostat_damping_ps=.1, timestep_fs=.25),
            hybrid=dict(equilibration_steps=2000, steps_per_cycle=1000, exchange_interval=20,
                exchange_attempts=5, move_attempts=0, max_displacement_A=1., overlap_cutoff_A=.75),
            reservoir=dict(fugacities_bar=[1.], source="Explicit ideal-gas approximation f=p at 300 K and 1 bar; pilot only, not a calibrated TraPPE reservoir."))))
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--supercell", choices=["1x2x1", "2x4x2"], default="2x4x2")
    args = parser.parse_args(argv)
    if Path(args.config).exists():
        raise FileExistsError(args.config)
    config = prepare_mil53(args.directory, args.supercell)
    Path(args.config).write_text(json.dumps(config, indent=2) + "\n")
    print(args.config)


if __name__ == "__main__":
    main()
