"""Multiple-try configurational-bias GCMC with normalized proposal measures.

The independent whole-tree proposal is less efficient than sequential growth,
but supports branched trees without assuming separable physical potentials.
Reference bonds have U_ref = k |r_child-r_parent|^2; arbitrary physical bonded
energies are evaluated by the energy backend. The dimensionless ideal
Rosenbluth normalization E_ref[exp(-beta*(U_intra-U_ref))] is mandatory.
"""

from __future__ import annotations
from copy import deepcopy
import math
import numpy as np
from scipy.special import logsumexp

KB_KCAL = 0.00198720425864083
BAR_A3_PER_KBT = 1e-25 / 1.380649e-23


class TreeProposal:
    def __init__(self, parents, spring_constants, temperature, ideal_log_rosenbluth):
        self.parents = list(parents)
        self.k = np.asarray(spring_constants[1:], float)
        self.beta = 1 / (KB_KCAL * temperature)
        self.log_ideal = float(ideal_log_rosenbluth)
        if (
            not self.parents
            or self.parents[0] is not None
            or len(self.k) != len(parents) - 1
        ):
            raise ValueError(
                "A rooted tree with one reference spring per non-root atom is required."
            )
        if any(
            not isinstance(p, int) or p < 0 or p >= i
            for i, p in enumerate(parents[1:], 1)
        ):
            raise ValueError(
                "Growth parents must precede their children; rings are unsupported."
            )
        if (
            not np.isfinite(self.k).all()
            or (self.k <= 0).any()
            or not math.isfinite(self.log_ideal)
        ):
            raise ValueError(
                "Finite positive reference springs and an ideal normalization are required."
            )

    def draw(self, rng, cell):
        xyz = [rng.random(3) @ cell]
        for i, parent in enumerate(self.parents[1:]):
            xyz.append(
                xyz[parent]
                + rng.normal(0, math.sqrt(1 / (2 * self.beta * self.k[i])), 3)
            )
        return np.asarray(xyz)

    def reference_energy(self, xyz):
        return sum(
            self.k[i - 1] * float(np.sum((xyz[i] - xyz[parent]) ** 2))
            for i, parent in enumerate(self.parents[1:], 1)
        )


class RigidProposal:
    log_ideal = 0.0

    def __init__(self, positions):
        self.positions = np.asarray(positions, float) - np.asarray(positions, float)[0]

    def draw(self, rng, cell):
        from scipy.spatial.transform import Rotation

        return (
            self.positions @ Rotation.random(random_state=rng).as_matrix()
            + rng.random(3) @ cell
        )

    def reference_energy(self, xyz):
        return 0.0


class ConfigurationalBias:
    def __init__(
        self, energy, proposals, fugacities_bar, temperature, cell, seed=2026, trials=8
    ):
        self.energy = energy
        self.proposals = proposals
        self.fugacities = fugacities_bar
        self.temperature = float(temperature)
        self.beta = 1 / (KB_KCAL * temperature)
        self.cell = np.asarray(cell, float)
        self.volume = float(np.linalg.det(self.cell))
        self.trials = int(trials)
        if self.volume <= 0 or self.trials < 1 or set(proposals) != set(fugacities_bar):
            raise ValueError("Invalid CBMC setup.")
        if any(not math.isfinite(f) or f < 0 for f in fugacities_bar.values()):
            raise ValueError("Invalid fugacities.")
        self.rng = np.random.default_rng(seed)
        self.molecules = []
        self.counts = {
            k: {"attempts": 0, "accepted": 0} for k in ("insert", "delete", "regrow")
        }

    def _trials(self, base, c, existing=None):
        proposal = self.proposals[c]
        u0 = self.energy(base)
        xyz = [existing] if existing is not None else []
        xyz += [
            proposal.draw(self.rng, self.cell) for _ in range(self.trials - len(xyz))
        ]
        logw = np.array(
            [
                -self.beta
                * (
                    self.energy(base + [{"component": c, "positions": x.tolist()}])
                    - u0
                    - proposal.reference_energy(x)
                )
                for x in xyz
            ]
        )
        if np.isnan(logw).any() or np.isposinf(logw).any():
            raise ValueError("Invalid CBMC trial energy.")
        total = float(logsumexp(logw))
        logW = total - math.log(self.trials)
        chosen = (
            0
            if existing is not None or not math.isfinite(total)
            else int(self.rng.choice(self.trials, p=np.exp(logw - total)))
        )
        return {"component": c, "positions": np.asarray(xyz[chosen]).tolist()}, logW

    def step(self, component=None, move=None):
        active = [c for c in sorted(self.proposals) if self.fugacities[c] > 0]
        if not active:
            return False
        c = component or active[int(self.rng.integers(len(active)))]
        move = move or ("insert", "delete", "regrow")[int(self.rng.integers(3))]
        if c not in active:
            raise ValueError("Cannot exchange a zero-fugacity component.")
        self.counts[move]["attempts"] += 1
        indices = [i for i, m in enumerate(self.molecules) if m["component"] == c]
        n = len(indices)
        activity = self.fugacities[c] * BAR_A3_PER_KBT / self.temperature
        ideal = self.proposals[c].log_ideal
        if move == "insert":
            new, lnew = self._trials(self.molecules, c)
            log_accept = math.log(activity * self.volume / (n + 1)) + lnew - ideal
            proposed = self.molecules + [new]
        else:
            if not n:
                return False
            index = indices[int(self.rng.integers(n))]
            old = self.molecules[index]
            base = self.molecules[:index] + self.molecules[index + 1 :]
            _, lold = self._trials(base, c, np.asarray(old["positions"]))
            if move == "delete":
                log_accept = math.log(n / (activity * self.volume)) + ideal - lold
                proposed = base
            else:
                new, lnew = self._trials(base, c)
                log_accept = lnew - lold
                proposed = base + [new]
        accepted = math.log(max(float(self.rng.random()), 1e-300)) < min(
            0.0, log_accept
        )
        if accepted:
            self.molecules = proposed
            self.counts[move]["accepted"] += 1
        return accepted

    def hmc(self, energy_forces, masses, steps=5, step_size=0.01):
        """Metropolis-corrected leapfrog on flexible guest atoms only.

        step_size is a fictitious HMC integration parameter, not a time in fs.
        The force callback uses kcal/mol and Angstrom, as in LAMMPS real units.
        """
        if steps < 1 or step_size <= 0:
            raise ValueError("Positive HMC steps and step size required.")
        indices = [
            i
            for i, m in enumerate(self.molecules)
            if isinstance(self.proposals[m["component"]], TreeProposal)
        ]
        if not indices:
            return False
        old = deepcopy(self.molecules)
        mass = np.array(
            [v for i in indices for v in masses[self.molecules[i]["component"]]], float
        )[:, None]
        if not np.isfinite(mass).all() or (mass <= 0).any():
            raise ValueError("HMC requires positive finite masses.")
        momenta = self.rng.normal(size=(len(mass), 3)) * np.sqrt(mass / self.beta)
        initial_u, forces = energy_forces(self.molecules)
        initial_h = initial_u + float(np.sum(momenta**2 / (2 * mass)))

        def selected(values):
            return np.concatenate([values[i] for i in indices], axis=0)

        momenta += 0.5 * step_size * selected(forces)
        for step in range(steps):
            offset = 0
            for i in indices:
                n = len(self.molecules[i]["positions"])
                xyz = (
                    np.asarray(self.molecules[i]["positions"])
                    + step_size
                    * momenta[offset : offset + n]
                    / mass[offset : offset + n]
                )
                self.molecules[i]["positions"] = xyz.tolist()
                offset += n
            energy, forces = energy_forces(self.molecules)
            momenta += (
                (0.5 if step == steps - 1 else 1.0) * step_size * selected(forces)
            )
        final_h = energy + float(np.sum(momenta**2 / (2 * mass)))
        accepted = math.isfinite(final_h) and math.log(
            max(float(self.rng.random()), 1e-300)
        ) < min(0.0, -self.beta * (final_h - initial_h))
        if not accepted:
            self.molecules = old
        else:
            for i in indices:
                xyz = np.asarray(self.molecules[i]["positions"])
                shift = np.floor(xyz[0] @ np.linalg.inv(self.cell)) @ self.cell
                self.molecules[i]["positions"] = (xyz - shift).tolist()
        counter = self.counts.setdefault("hmc", {"attempts": 0, "accepted": 0})
        counter["attempts"] += 1
        counter["accepted"] += int(accepted)
        return accepted

    def checkpoint(self):
        return {
            "molecules": deepcopy(self.molecules),
            "random_state": deepcopy(self.rng.bit_generator.state),
            "counts": deepcopy(self.counts),
        }

    def restore(self, state):
        self.molecules = deepcopy(state["molecules"])
        self.rng.bit_generator.state = state["random_state"]
        self.counts = deepcopy(state["counts"])
