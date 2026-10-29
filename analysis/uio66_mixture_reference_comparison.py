"""Compare the UiO-66 CO2/CH4 pressure screen with the local NIST volume-basis curve.

Reference: Yang et al., J. Phys. Chem. C 2011, DOI 10.1021/jp202633t,
NIST ISODB Isotherm25 (experimental, absolute, 50/50, 303 K).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ase.io import read


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "uio66_co2_ch4_nist_303K_50_50_pressure_screen_seed12345"
REFERENCE = ROOT / "isodb-library/Library/10.1021jp202633t/10.1021jp202633t.Isotherm25.json"
AVOGADRO = 6.02214076e23
STP_MOLAR_VOLUME_CM3 = 22414.0


def compare(run_dir: Path, output_dir: Path | None = None) -> Path:
    reference = json.loads(REFERENCE.read_text())
    if (reference["adsorbent"]["name"], reference["temperature"], reference["isotherm_type"], reference["adsorptionUnits"]) != (
        "UiO-66(Zr)", 303, "absolute", "cm3(STP)/cm3"
    ):
        raise ValueError("Unexpected NIST reference metadata.")
    prep = json.loads((run_dir / "prepare_summary.json").read_text())["prepare_plan"]
    params = prep["parameters"]
    if params["material_id"] != "UiO-66(Zr)" or params["temperature_K"] != 303 or params["mixture"] != {"CO2": 0.5, "CH4": 0.5}:
        raise ValueError("Run does not match the UiO-66 50/50 reference.")
    framework_cif = Path(prep["inputs"]["framework_cif"])
    if not framework_cif.is_absolute():
        framework_cif = ROOT / framework_cif
    crystal = read(framework_cif)
    nx, ny, nz = params["unit_cells"]
    volume_a3 = crystal.get_volume() * nx * ny * nz
    if volume_a3 <= 0:
        raise ValueError("Invalid simulation cell volume.")
    # Number of molecules per simulation cell -> STP volume per crystal volume.
    factor = STP_MOLAR_VOLUME_CM3 * 1e24 / (AVOGADRO * volume_a3)
    names = {a["InChIKey"]: "CO2" if a["name"] == "Carbon Dioxide" else "CH4" for a in reference["adsorbates"]}
    reference_points = {}
    for point in reference["isotherm_data"]:
        species = {names[s["InChIKey"]]: s for s in point["species_data"]}
        if set(species) == {"CO2", "CH4"} and all(s["composition"] == 0.5 for s in species.values()):
            reference_points[float(point["pressure"])] = {c: float(s["adsorption"]) for c, s in species.items()}
    with (run_dir / "mixture_results.csv").open(newline="") as handle:
        simulated = list(csv.DictReader(handle))
    rows = []
    for point in simulated:
        pressure = float(point["pressure_bar"])
        if pressure not in reference_points:
            raise ValueError(f"No exact NIST pressure match for {pressure:g} bar.")
        if float(point["y_CO2"]) != 0.5 or float(point["y_CH4"]) != 0.5:
            raise ValueError("Simulated gas composition differs from the reference.")
        ref = reference_points[pressure]
        sim = {c: float(point[f"mean_{c}_molecules_per_cell"]) * factor for c in ("CO2", "CH4")}
        rows.append({
            "pressure_bar": pressure,
            "simulation_CO2_cm3STP_cm3": sim["CO2"],
            "reference_CO2_cm3STP_cm3": ref["CO2"],
            "simulation_CH4_cm3STP_cm3": sim["CH4"],
            "reference_CH4_cm3STP_cm3": ref["CH4"],
            "simulation_selectivity_CO2_CH4": sim["CO2"] / sim["CH4"] if sim["CH4"] > 0 else None,
            "reference_selectivity_CO2_CH4": ref["CO2"] / ref["CH4"] if ref["CH4"] > 0 else None,
        })
    rows.sort(key=lambda row: row["pressure_bar"])
    destination = output_dir or run_dir / "analysis" / "uio66_nist_reference"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "pressure_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    x = [r["pressure_bar"] for r in rows]
    for component, color in (("CO2", "tab:blue"), ("CH4", "tab:orange")):
        axes[0].plot(x, [r[f"simulation_{component}_cm3STP_cm3"] for r in rows], "-o", color=color, label=f"{component} simulation")
        axes[0].plot(x, [r[f"reference_{component}_cm3STP_cm3"] for r in rows], "--s", color=color, label=f"{component} NIST experiment")
    axes[0].set_ylabel("Absolute uptake [cm³(STP)/cm³ crystal]")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    for source, style in (("simulation", "-o"), ("reference", "--s")):
        axes[1].plot(x, [r[f"{source}_selectivity_CO2_CH4"] for r in rows], style, label=source.title())
    axes[1].set_xlabel("Total pressure [bar]")
    axes[1].set_ylabel("CO₂/CH₄ selectivity")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("UiO-66, 303 K, CO₂/CH₄ = 50/50 (single-seed screen)")
    fig.savefig(destination / "pressure_comparison.png", dpi=170)
    plt.close(fig)
    (destination / "comparison_metadata.json").write_text(json.dumps({
        "source": str(REFERENCE),
        "doi": reference["DOI"],
        "source_category": reference["category"],
        "reference_basis": reference["isotherm_type"],
        "simulation_framework_cif": str(framework_cif),
        "simulation_cell_volume_A3": volume_a3,
        "simulation_molecules_to_cm3STP_per_cm3_factor": factor,
        "replicate_count": len(params["seeds"]),
        "limitation": "Local rigid UiO-66 CIF has zero framework charges; paper uses a charged dehydroxylated, flexible framework. One seed does not establish statistical precision or quantitative validation."
    }, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=ROOT / "outputs/module_B_mixture_selectivity/runs" / RUN_ID)
    args = parser.parse_args()
    print(compare(args.run_dir))
