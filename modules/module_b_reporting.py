"""Tables and explicitly labelled projections for multicomponent AL campaigns."""

from __future__ import annotations
import csv
from itertools import combinations
from pathlib import Path


def ratio(a, b, ya, yb):
    return a / b * yb / ya if b > 0 and ya > 0 and yb > 0 else None


def write_artifacts(state, prediction, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    names = state["components"]
    quantities = []
    for role in ("results", "holdout_results"):
        for p in state[role]:
            common = {
                "role": "training" if role == "results" else "holdout",
                "candidate_id": p["candidate_id"],
                "pressure_bar": p["pressure_bar"],
                "temperature_K": p["temperature_K"],
                **{f"y_{c}": p["mixture"][c] for c in names},
                "replicate_count": p["replicate_count"],
            }
            for c, s in p["components"].items():
                quantities.append(
                    {
                        **common,
                        "quantity": c,
                        "value": s["mean_molecules"],
                        "unit": "molecules/cell",
                        **{
                            key: s.get(key)
                            for key in (
                                "ci95_lower",
                                "ci95_upper",
                                "ci95_status",
                                "precision_met",
                            )
                        },
                    }
                )
            quantities.append(
                {
                    **common,
                    "quantity": "total",
                    "value": sum(s["mean_molecules"] for s in p["components"].values()),
                    "unit": "molecules/cell",
                    **p.get("total", {}),
                }
            )
            for a, b in combinations(names, 2):
                s = p.get("selectivity", {}).get(f"{a}/{b}", {})
                quantities.append(
                    {
                        **common,
                        "quantity": f"{a}/{b}",
                        "value": s.get("value"),
                        "unit": "dimensionless",
                        **{
                            key: s.get(key)
                            for key in (
                                "ci95_lower",
                                "ci95_upper",
                                "ci95_status",
                                "precision_met",
                            )
                        },
                    }
                )
    fields = [
        "role",
        "candidate_id",
        "pressure_bar",
        "temperature_K",
        *[f"y_{c}" for c in names],
        "replicate_count",
        "quantity",
        "value",
        "unit",
        "ci95_lower",
        "ci95_upper",
        "ci95_status",
        "precision_met",
    ]
    with (target / "observations.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        writer.writerows(quantities)
    files = {"observations": str(target / "observations.csv")}
    if prediction:
        fields = [
            "candidate_id",
            "pressure_bar",
            "temperature_K",
            *[f"y_{c}" for c in names],
        ]
        fields += [f"{c}_{key}" for c in names for key in ("median", "lower", "upper")]
        fields += [f"selectivity_{a}_{b}" for a, b in combinations(names, 2)]
        with (target / "surrogate_predictions.csv").open("w", newline="") as h:
            writer = csv.DictWriter(h, fieldnames=fields)
            writer.writeheader()
            for i, p in enumerate(state["candidates"]):
                row = {
                    k: p[k] for k in ("candidate_id", "pressure_bar", "temperature_K")
                }
                row.update({f"y_{c}": p["mixture"][c] for c in names})
                for c in names:
                    s = prediction["components"][c][i]
                    row.update(
                        {
                            f"{c}_median": s["value"],
                            f"{c}_lower": s["lower"],
                            f"{c}_upper": s["upper"],
                        }
                    )
                for a, b in combinations(names, 2):
                    row[f"selectivity_{a}_{b}"] = ratio(
                        prediction["components"][a][i]["value"],
                        prediction["components"][b][i]["value"],
                        p["mixture"][a],
                        p["mixture"][b],
                    )
                writer.writerow(row)
        files["predictions"] = str(target / "surrogate_predictions.csv")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(
            1,
            len(names),
            figsize=(5 * len(names), 4.3),
            squeeze=False,
            layout="constrained",
        )
        temperatures = sorted({p["temperature_K"] for p in state["candidates"]})
        fixed_t = (
            temperatures[len(temperatures) // 2] if len(temperatures) <= 11 else None
        )
        indices = [
            i
            for i, p in enumerate(state["candidates"])
            if fixed_t is None or p["temperature_K"] == fixed_t
        ]
        for c, ax in zip(names, axes[0]):
            pts = [state["candidates"][i] for i in indices]
            artist = ax.scatter(
                [p["pressure_bar"] for p in pts],
                [p["mixture"][c] for p in pts],
                c=[prediction["components"][c][i]["value"] for i in indices],
                s=12,
            )
            ax.set_xscale("log")
            ax.set_xlabel("Total pressure / bar")
            ax.set_ylabel(f"Gas fraction {c}")
            ax.set_title(c)
            fig.colorbar(artist, ax=ax, label="GP median / molecules per cell")
        descriptor = (
            f"T = {fixed_t:g} K" if fixed_t is not None else "temperature projected"
        )
        fig.suptitle(
            f"Surrogate projections ({descriptor}; remaining fractions projected)"
        )
        fig.savefig(target / "surrogate_projections.png", dpi=140)
        fig.savefig(target / "surrogate_projections.svg")
        plt.close(fig)
        files["projections"] = str(target / "surrogate_projections.png")
    return files
