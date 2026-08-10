from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

KCAL_PER_MOL_PER_K = 0.0019872041
VIRTUAL_SITE_MASS_AMU = 1.0e-6
NON_INTERACTING_SIGMA_A = 1.0


@dataclass(frozen=True)
class LjParameter:
    name: str
    epsilon_K: float
    sigma_A: float


@dataclass(frozen=True)
class PseudoAtom:
    name: str
    element: str
    mass: float
    charge: float


@dataclass(frozen=True)
class ForcefieldParameters:
    lj_parameters: dict[str, LjParameter]
    pseudo_atoms: dict[str, PseudoAtom]
    mixing_rule: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtomTypeAssignment:
    type_id: int
    label: str
    lj_type: str
    source: str
    mass: float | None = None
    charge: float | None = None


@dataclass(frozen=True)
class PairCoefficient:
    type_i: int
    type_j: int
    label_i: str
    label_j: str
    epsilon_kcal_mol: float
    sigma_A: float


@dataclass(frozen=True)
class LammpsForcefield:
    atom_types: tuple[AtomTypeAssignment, ...]
    pair_coefficients: tuple[PairCoefficient, ...]
    pair_style: str
    kspace_style: str
    pair_modify_lines: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_forcefield_to_lammps(
    forcefield_config: dict[str, Any],
    components: list[str],
    output_path: str | Path,
) -> dict[str, Any]:
    """Return the planned forcefield-to-LAMMPS conversion without writing files."""
    return {
        "converter": "forcefield_to_lammps",
        "forcefield_config": forcefield_config,
        "components": components,
        "output_file": str(output_path),
        "status": "planned",
    }



def load_forcefield_parameters(forcefield_config: dict[str, Any]) -> ForcefieldParameters:
    pseudo_atoms = parse_pseudo_atoms(forcefield_config["files"]["pseudo_atoms"])
    lj_parameters, mixing_rule = parse_mixing_rules(forcefield_config["files"]["mixing_rules"])
    for source in forcefield_config.get("adsorbate_parameter_files", []):
        source_files = source.get("files", source)
        if "pseudo_atoms" in source_files:
            pseudo_atoms.update(parse_pseudo_atoms(source_files["pseudo_atoms"]))
        if "mixing_rules" in source_files:
            source_lj_parameters, source_mixing_rule = parse_mixing_rules(source_files["mixing_rules"])
            if source_mixing_rule != mixing_rule:
                raise ValueError(
                    f"Cannot merge adsorbate forcefield parameters with mixing rule "
                    f"{source_mixing_rule!r} into base mixing rule {mixing_rule!r}."
                )
            lj_parameters.update(source_lj_parameters)
    return ForcefieldParameters(lj_parameters=lj_parameters, pseudo_atoms=pseudo_atoms, mixing_rule=mixing_rule)


def build_atom_type_assignments(
    framework_symbols: list[str] | tuple[str, ...],
    adsorbate_atom_types: list[str] | tuple[str, ...],
    parameters: ForcefieldParameters,
) -> tuple[AtomTypeAssignment, ...]:
    """Assign global LAMMPS atom type IDs for framework atoms followed by adsorbate pseudoatoms."""
    assignments: list[AtomTypeAssignment] = []
    used_labels: set[str] = set()

    for symbol in _unique(framework_symbols):
        lj_type = f"{symbol}_"
        if lj_type not in parameters.lj_parameters:
            raise LookupError(f"No LJ parameter {lj_type!r} found for framework element {symbol!r}.")
        assignments.append(
            AtomTypeAssignment(
                type_id=len(assignments) + 1,
                label=symbol,
                lj_type=lj_type,
                source="framework",
            )
        )
        used_labels.add(symbol)

    for atom_type in _unique(adsorbate_atom_types):
        if atom_type in used_labels:
            raise ValueError(f"Duplicate atom type label {atom_type!r} between framework and adsorbate.")
        pseudo_atom = parameters.pseudo_atoms.get(atom_type)
        if pseudo_atom is None:
            raise LookupError(f"No pseudo atom {atom_type!r} found in pseudo_atoms.def.")
        if atom_type not in parameters.lj_parameters:
            raise LookupError(f"No LJ parameter {atom_type!r} found in force_field_mixing_rules.def.")
        assignments.append(
            AtomTypeAssignment(
                type_id=len(assignments) + 1,
                label=atom_type,
                lj_type=atom_type,
                source="adsorbate",
                mass=(
                    pseudo_atom.mass
                    if pseudo_atom.mass > 0.0
                    else VIRTUAL_SITE_MASS_AMU
                ),
                charge=pseudo_atom.charge,
            )
        )
        used_labels.add(atom_type)

    return tuple(assignments)


def build_pair_coefficients(
    atom_types: tuple[AtomTypeAssignment, ...],
    parameters: ForcefieldParameters,
) -> tuple[PairCoefficient, ...]:
    """Build explicit Lorentz-Berthelot pair coefficients for all type pairs."""
    if parameters.mixing_rule != "Lorentz-Berthelot":
        raise ValueError(f"Unsupported mixing rule {parameters.mixing_rule!r}.")

    pair_coefficients: list[PairCoefficient] = []
    for index, left in enumerate(atom_types):
        left_lj = parameters.lj_parameters[left.lj_type]
        for right in atom_types[index:]:
            right_lj = parameters.lj_parameters[right.lj_type]
            epsilon_K = math.sqrt(left_lj.epsilon_K * right_lj.epsilon_K)
            sigma_A = (left_lj.sigma_A + right_lj.sigma_A) / 2.0
            pair_coefficients.append(
                PairCoefficient(
                    type_i=left.type_id,
                    type_j=right.type_id,
                    label_i=left.label,
                    label_j=right.label,
                    epsilon_kcal_mol=epsilon_K * KCAL_PER_MOL_PER_K,
                    sigma_A=sigma_A,
                )
            )

    return tuple(pair_coefficients)


def build_lammps_forcefield(
    framework_symbols: list[str] | tuple[str, ...],
    adsorbate_atom_types: list[str] | tuple[str, ...],
    parameters: ForcefieldParameters,
    pair_style: str = "lj/cut/coul/long 12.8",
    kspace_style: str = "pppm 1.0e-4",
    pair_modify_shift: bool = False,
) -> LammpsForcefield:
    atom_types = build_atom_type_assignments(framework_symbols, adsorbate_atom_types, parameters)
    pair_coefficients = build_pair_coefficients(atom_types, parameters)
    pair_modify_lines = ("pair_modify shift yes",) if pair_modify_shift else ()
    return LammpsForcefield(
        atom_types=atom_types,
        pair_coefficients=pair_coefficients,
        pair_style=pair_style,
        kspace_style=kspace_style,
        pair_modify_lines=pair_modify_lines,
    )


def render_lammps_forcefield_include(forcefield: LammpsForcefield) -> str:
    """Render a LAMMPS include file with explicit pair coefficients."""
    lines = [
        "# LAMMPS forcefield include generated from CRAFTED parameters",
        f"pair_style {forcefield.pair_style}",
    ]
    lines.extend(forcefield.pair_modify_lines)
    lines.extend([f"kspace_style {forcefield.kspace_style}", "", "# Atom type map"])
    for atom_type in forcefield.atom_types:
        lines.append(f"# {atom_type.type_id}: {atom_type.label} -> {atom_type.lj_type} ({atom_type.source})")

    lines.extend(["", "# Pair coefficients: type_i type_j epsilon[kcal/mol] sigma[A]"])
    for coefficient in forcefield.pair_coefficients:
        lines.append(
            "pair_coeff "
            f"{coefficient.type_i} {coefficient.type_j} "
            f"{coefficient.epsilon_kcal_mol:.10f} {coefficient.sigma_A:.8f} "
            f"# {coefficient.label_i}-{coefficient.label_j}"
        )

    return "\n".join(lines) + "\n"


def write_lammps_forcefield_include(forcefield: LammpsForcefield, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_lammps_forcefield_include(forcefield), encoding="utf-8")
    return output


def parse_pseudo_atoms(pseudo_atoms_path: str | Path) -> dict[str, PseudoAtom]:
    """Parse the pseudo_atoms.def file for LAMMPS forcefield parameters."""
    lines = Path(pseudo_atoms_path).read_text(encoding="utf-8").splitlines()
    content = _content_lines(lines)

    pseudo_atoms: dict[str, PseudoAtom] = {}

    for line in content:
        line_parts = line.split()
        # Skip count/header-like lines.
        if len(line_parts) < 7:
            continue

        name = line_parts[0]
        element = line_parts[2]
        mass = float(line_parts[5])
        charge = float(line_parts[6])

        pseudo_atoms[name] = PseudoAtom(
            name=name,
            element=element,
            mass=mass,
            charge=charge,
        )

    return pseudo_atoms


def parse_mixing_rules(mixing_rules_path: str | Path) -> tuple[dict[str, LjParameter], str]:
    """Parse the mixing_rules.def file for LAMMPS forcefield parameters."""
    lines = Path(mixing_rules_path).read_text(encoding="utf-8").splitlines()
    content = _content_lines(lines)

    lj_parameters: dict[str, LjParameter] = {}
    mixing_rule = "Lorentz-Berthelot"

    for line in content:
        line_parts = _strip_inline_comment(line).split()
        if not line_parts:
            continue
        if len(line_parts) == 1 and line_parts[0] == "Lorentz-Berthelot":
            mixing_rule = "Lorentz-Berthelot"
            continue
        elif len(line_parts) == 2 and line_parts[1].casefold() == "none":
            name = line_parts[0]
            lj_parameters[name] = LjParameter(
                name=name,
                epsilon_K=0.0,
                sigma_A=NON_INTERACTING_SIGMA_A,
            )
            continue
        elif len(line_parts) != 4:
            continue

        name, interaction_type, epsilon_K_str, sigma_A_str = line_parts
        if interaction_type.casefold() != "lennard-jones":
            continue

        lj_parameters[name] = LjParameter(
            name=name,
            epsilon_K=float(epsilon_K_str),
            sigma_A=float(sigma_A_str),
        )

    return lj_parameters, mixing_rule


def _content_lines(lines: list[str]) -> list[str]:
    """Return the non-empty, non-comment lines from a file."""
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _strip_inline_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def _unique(values: Any) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
