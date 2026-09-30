"""Compare converged MLIP-MC Widom results with layered references."""

from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
from typing import Any

from pipeline.config import save_benchmark_data


CRAFTED_REFERENCE_CLASS = "classical_simulation"
EXPERIMENTAL_REFERENCE_CLASS = "experiment"
DFT_REFERENCE_CLASSES = {"dft", "dft_finetuned", "finetuned_mlip"}
DEFAULT_EXPERIMENTAL_DOI = "10.1021/acs.jced.1c00900"
GOEMINNE_DATASET_DOI = "10.5281/zenodo.7904959"
MLIP_MC_DATASET_DOI = "10.5281/zenodo.18637262"


def create_widom_reference_comparison(
    summary_path: str | Path,
    crafted_root: str | Path | None,
    output_directory: str | Path,
    *,
    charge_scheme: str = "DDEC",
    forcefields: tuple[str, ...] = ("DREIDING", "UFF"),
    low_pressure_cutoff_Pa: float = 2000.0,
    isodb_root: str | Path | None = "isodb-library",
    experimental_doi: str = DEFAULT_EXPERIMENTAL_DOI,
    experimental_pressure_cutoff_bar: float = 0.75,
    curated_reference_paths: tuple[str | Path, ...] = (),
    temperature_tolerance_K: float = 0.5,
) -> dict[str, Any]:
    """Create a block-aware comparison against layered reference data.

    CRAFTED and NIST ISODB references are discovered locally. DFT-derived or
    finetuned-model values are accepted only through explicit curated files so
    that unavailable literature data cannot silently become a numeric result.
    """
    summary_file = Path(summary_path)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    pooled = summary.get("pooled_block_analysis", {})
    blocks = pooled.get("blocks", [])
    if not blocks:
        raise ValueError("The Widom summary does not contain pooled block data.")

    material = str(summary["material"])
    adsorbate = str(summary["adsorbate"])
    temperature = float(summary["temperature_K"])
    temperature_label = int(round(temperature))
    crafted_references: list[dict[str, Any]] = []
    if crafted_root is not None:
        root = Path(crafted_root)
        for forcefield in forcefields:
            prefix = (
                f"{charge_scheme}_{material}_{forcefield}_"
                f"{adsorbate}_{temperature_label}"
            )
            crafted_references.append(
                _load_crafted_reference(
                    root / "ISOTHERM_FILES" / f"{prefix}.csv",
                    root / "ENTHALPY_FILES" / f"{prefix}.csv",
                    forcefield=forcefield,
                    low_pressure_cutoff_Pa=low_pressure_cutoff_Pa,
                )
            )

    experimental_references: list[dict[str, Any]] = []
    if isodb_root is not None:
        experimental = _load_experimental_isotherm_reference(
            Path(isodb_root),
            doi=experimental_doi,
            material=material,
            adsorbate=adsorbate,
            temperature_K=temperature,
            pressure_cutoff_bar=experimental_pressure_cutoff_bar,
            temperature_tolerance_K=temperature_tolerance_K,
        )
        if experimental is not None:
            experimental_references.append(experimental)

    curated_references = [
        _load_curated_reference(
            Path(path),
            material=material,
            adsorbate=adsorbate,
            temperature_K=temperature,
            temperature_tolerance_K=temperature_tolerance_K,
        )
        for path in curated_reference_paths
    ]
    references = [
        *crafted_references,
        *experimental_references,
        *curated_references,
    ]
    if not references:
        raise ValueError("No compatible Widom references were found.")

    mlip_metrics = {
        metric: {
            **pooled["metrics"][metric],
            "block_min": min(float(block[metric]) for block in blocks),
            "block_max": max(float(block[metric]) for block in blocks),
        }
        for metric in (
            "henry_coefficient_mmol_g_bar",
            "isosteric_heat_zero_loading_kj_mol",
        )
    }
    comparisons = [
        _compare_reference(mlip_metrics, reference)
        for reference in references
    ]
    compared_flags = [
        value
        for comparison in comparisons
        for key, value in comparison.items()
        if key.endswith("_outside_mlip_block_range") and value is not None
    ]
    if compared_flags and all(compared_flags):
        assessment = (
            "Sampling uncertainty does not explain the reference mismatch: "
            "every available reference metric lies outside the full observed "
            "MLIP block range."
        )
    else:
        assessment = (
            "At least one reference metric overlaps the observed MLIP block "
            "range; inspect the per-reference ratios and uncertainty intervals."
        )

    dft_reference_available = any(
        reference["reference_class"] in DFT_REFERENCE_CLASSES
        for reference in curated_references
    )
    report = {
        "schema_version": 2,
        "material": material,
        "adsorbate": adsorbate,
        "temperature_K": temperature,
        "mlip_model": summary.get("model", {}),
        "mlip": mlip_metrics,
        "references": references,
        "crafted_references": crafted_references,
        "experimental_references": experimental_references,
        "curated_references": curated_references,
        "comparisons": comparisons,
        "assessment": assessment,
        "reference_hierarchy": [
            "DFT interaction energies or a DFT-finetuned adsorption model",
            "curated experimental adsorption data at matching conditions",
            "classical force-field simulation baselines",
        ],
        "goeminne_reference": {
            "available": dft_reference_available,
            "status": "loaded" if dft_reference_available else "not_provided",
            "dataset_doi": GOEMINNE_DATASET_DOI,
            "mlip_mc_dataset_doi": MLIP_MC_DATASET_DOI,
            "expected_model": "model_ZIF8_N=1000.pth",
            "instruction": (
                "Pass a curated reference JSON with --curated-reference when "
                "the Goeminne/MLIP-MC reference result is locally available."
            ),
        },
        "limitations": (
            "CRAFTED is a classical simulation baseline. Experimental Henry "
            "coefficients depend on sample and fitting protocol. Quantitative "
            "potential validation requires identical configurations evaluated "
            "against DFT interaction energies or the published finetuned model."
        ),
    }

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "widom_reference_comparison.json"
    csv_path = output / "widom_reference_comparison.csv"
    plot_path = output / "widom_reference_comparison.png"
    _write_csv(report, csv_path)
    _write_plot(report, blocks, plot_path)
    report["outputs"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    save_benchmark_data(json_path, report)
    return report


def _load_crafted_reference(
    isotherm_path: Path,
    enthalpy_path: Path,
    *,
    forcefield: str,
    low_pressure_cutoff_Pa: float,
) -> dict[str, Any]:
    if not isotherm_path.is_file():
        raise FileNotFoundError(f"Missing CRAFTED isotherm: {isotherm_path}")
    if not enthalpy_path.is_file():
        raise FileNotFoundError(f"Missing CRAFTED enthalpy: {enthalpy_path}")
    isotherm_rows = _numeric_rows(isotherm_path)
    selected = [row for row in isotherm_rows if row[0] <= low_pressure_cutoff_Pa]
    if len(selected) < 2:
        raise ValueError(
            f"Need at least two CRAFTED points below {low_pressure_cutoff_Pa:g} Pa."
        )
    denominator = sum(pressure**2 for pressure, *_rest in selected)
    slope_mol_kg_pa = sum(
        pressure * loading for pressure, loading, *_rest in selected
    ) / denominator
    first_enthalpy = min(_numeric_rows(enthalpy_path), key=lambda row: row[0])
    heat = abs(first_enthalpy[1])
    heat_uncertainty = abs(first_enthalpy[2]) if len(first_enthalpy) >= 3 else None
    charge_scheme = isotherm_path.name.split("_", 1)[0]
    return {
        "source_id": f"crafted_{forcefield.casefold()}_{charge_scheme.casefold()}",
        "label": f"CRAFTED {forcefield}+{charge_scheme}",
        "reference_class": CRAFTED_REFERENCE_CLASS,
        "forcefield": forcefield,
        "charge_scheme": charge_scheme,
        "henry_coefficient_mol_kg_pa": slope_mol_kg_pa,
        "henry_coefficient_mmol_g_bar": slope_mol_kg_pa * 1.0e5,
        "henry_uncertainty_mmol_g_bar": None,
        "henry_fit_max_pressure_Pa": low_pressure_cutoff_Pa,
        "henry_fit_point_count": len(selected),
        "isosteric_heat_zero_loading_kj_mol": heat,
        "isosteric_heat_low_pressure_kj_mol": heat,
        "isosteric_heat_uncertainty_kj_mol": heat_uncertainty,
        "enthalpy_pressure_Pa": first_enthalpy[0],
        "provenance": {
            "isotherm_path": str(isotherm_path),
            "enthalpy_path": str(enthalpy_path),
        },
        "isotherm_path": str(isotherm_path),
        "enthalpy_path": str(enthalpy_path),
    }


def _load_experimental_isotherm_reference(
    isodb_root: Path,
    *,
    doi: str,
    material: str,
    adsorbate: str,
    temperature_K: float,
    pressure_cutoff_bar: float,
    temperature_tolerance_K: float,
) -> dict[str, Any] | None:
    article_directory = isodb_root / "Library" / doi.replace("/", "")
    if not article_directory.is_dir():
        return None
    candidates = []
    for path in sorted(article_directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("category") != "exp":
            continue
        adsorbent_name = data.get("adsorbent", {}).get("name")
        if _normalized_name(adsorbent_name) != _normalized_name(material):
            continue
        names = {
            _normalized_name(item.get("name"))
            for item in data.get("adsorbates", [])
        }
        if _normalized_adsorbate(adsorbate) not in names:
            continue
        reference_temperature = float(data.get("temperature", -1.0))
        if abs(reference_temperature - temperature_K) > temperature_tolerance_K:
            continue
        if data.get("pressureUnits") != "bar":
            continue
        if data.get("adsorptionUnits") != "mmol/g":
            continue
        candidates.append((path, data))
    if not candidates:
        return None
    if len(candidates) > 1:
        filenames = ", ".join(path.name for path, _data in candidates)
        raise ValueError(
            f"Ambiguous experimental reference for {doi}: {filenames}"
        )

    path, data = candidates[0]
    points = []
    for record in data.get("isotherm_data", []):
        pressure = float(record["pressure"])
        species = record.get("species_data", [])
        if not species or pressure <= 0.0 or pressure > pressure_cutoff_bar:
            continue
        loading = species[0].get("adsorption")
        if loading is not None:
            points.append((pressure, float(loading)))
    if len(points) < 2:
        raise ValueError(
            f"Need at least two experimental points below {pressure_cutoff_bar:g} bar."
        )
    slope, slope_standard_error = _origin_slope(points)
    return {
        "source_id": "experiment_hwang_2022",
        "label": "Experiment Hwang et al. (298 K)",
        "reference_class": EXPERIMENTAL_REFERENCE_CLASS,
        "doi": doi,
        "henry_coefficient_mmol_g_bar": slope,
        "henry_uncertainty_mmol_g_bar": slope_standard_error,
        "henry_fit_max_pressure_bar": pressure_cutoff_bar,
        "henry_fit_point_count": len(points),
        "henry_fit_method": "least_squares_through_origin",
        "isosteric_heat_zero_loading_kj_mol": None,
        "isosteric_heat_uncertainty_kj_mol": None,
        "provenance": {
            "isotherm_path": str(path),
            "article_source": data.get("articleSource"),
            "category": data.get("category"),
        },
    }


def _load_curated_reference(
    path: Path,
    *,
    material: str,
    adsorbate: str,
    temperature_K: float,
    temperature_tolerance_K: float,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing curated Widom reference: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in (("material", material), ("adsorbate", adsorbate)):
        if _normalized_name(data.get(key)) != _normalized_name(expected):
            raise ValueError(
                f"Curated reference {path} has incompatible {key}: "
                f"{data.get(key)!r}; expected {expected!r}."
            )
    reference_temperature = float(data["temperature_K"])
    if abs(reference_temperature - temperature_K) > temperature_tolerance_K:
        raise ValueError(
            f"Curated reference {path} is at {reference_temperature:g} K; "
            f"the MLIP result is at {temperature_K:g} K."
        )
    reference_class = str(data.get("reference_class", "dft_finetuned"))
    metrics = data.get("metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError("Curated reference 'metrics' must be an object.")
    henry, henry_uncertainty = _curated_metric(
        metrics,
        "henry_coefficient_mmol_g_bar",
    )
    heat, heat_uncertainty = _curated_metric(
        metrics,
        "isosteric_heat_zero_loading_kj_mol",
    )
    if henry is None and heat is None:
        raise ValueError(
            f"Curated reference {path} contains neither Henry nor Qst data."
        )
    return {
        "source_id": str(data.get("reference_id", path.stem)),
        "label": str(data.get("label", path.stem)),
        "reference_class": reference_class,
        "henry_coefficient_mmol_g_bar": henry,
        "henry_uncertainty_mmol_g_bar": henry_uncertainty,
        "isosteric_heat_zero_loading_kj_mol": heat,
        "isosteric_heat_uncertainty_kj_mol": heat_uncertainty,
        "provenance": {
            **data.get("provenance", {}),
            "curated_reference_path": str(path),
        },
    }


def _curated_metric(
    metrics: dict[str, Any],
    name: str,
) -> tuple[float | None, float | None]:
    value = metrics.get(name)
    if value is None:
        return None, None
    if isinstance(value, dict):
        estimate = value.get("value", value.get("estimate"))
        uncertainty = value.get("uncertainty", value.get("ci95_half_width"))
    else:
        estimate = value
        uncertainty = None
    if estimate is None:
        return None, None
    return float(estimate), None if uncertainty is None else abs(float(uncertainty))


def _compare_reference(
    mlip_metrics: dict[str, dict[str, Any]],
    reference: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reference": reference["label"],
        "reference_class": reference["reference_class"],
    }
    specifications = (
        (
            "henry",
            "henry_coefficient_mmol_g_bar",
            "henry_coefficient_mmol_g_bar",
        ),
        (
            "qst",
            "isosteric_heat_zero_loading_kj_mol",
            "isosteric_heat_zero_loading_kj_mol",
        ),
    )
    for prefix, mlip_key, reference_key in specifications:
        reference_value = reference.get(reference_key)
        if reference_value is None:
            result[f"{prefix}_ratio_mlip_to_reference"] = None
            result[f"{prefix}_reference_outside_mlip_block_range"] = None
            continue
        mlip = mlip_metrics[mlip_key]
        value = float(reference_value)
        result[f"{prefix}_ratio_mlip_to_reference"] = mlip["estimate"] / value
        result[f"{prefix}_reference_outside_mlip_block_range"] = (
            value > mlip["block_max"] or value < mlip["block_min"]
        )
    return result


def _origin_slope(points: list[tuple[float, float]]) -> tuple[float, float | None]:
    denominator = sum(x * x for x, _y in points)
    if denominator <= 0.0:
        raise ValueError("Reference pressures must contain positive values.")
    slope = sum(x * y for x, y in points) / denominator
    if len(points) < 2:
        return slope, None
    residual_sum = sum((y - slope * x) ** 2 for x, y in points)
    standard_error = sqrt((residual_sum / (len(points) - 1)) / denominator)
    return slope, standard_error


def _normalized_name(value: Any) -> str:
    return "".join(
        character
        for character in str(value or "").casefold()
        if character.isalnum()
    )


def _normalized_adsorbate(value: str) -> str:
    aliases = {
        "co2": "carbondioxide",
        "n2": "nitrogen",
        "ch4": "methane",
        "h2": "hydrogen",
    }
    normalized = _normalized_name(value)
    return aliases.get(normalized, normalized)


def _numeric_rows(path: Path) -> list[tuple[float, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            tuple(float(value) for value in row)
            for row in csv.reader(line for line in handle if not line.startswith("#"))
            if row
        ]


def _write_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "source",
        "reference_class",
        "henry_coefficient_mmol_g_bar",
        "henry_uncertainty_mmol_g_bar",
        "isosteric_heat_kj_mol",
        "isosteric_heat_uncertainty_kj_mol",
    ]
    model = report.get("mlip_model", {})
    rows = [
        {
            "source": _mlip_label(model).replace("\n", " "),
            "reference_class": "candidate_mlip",
            "henry_coefficient_mmol_g_bar": report["mlip"][
                "henry_coefficient_mmol_g_bar"
            ]["estimate"],
            "henry_uncertainty_mmol_g_bar": report["mlip"][
                "henry_coefficient_mmol_g_bar"
            ]["ci95_half_width"],
            "isosteric_heat_kj_mol": report["mlip"][
                "isosteric_heat_zero_loading_kj_mol"
            ]["estimate"],
            "isosteric_heat_uncertainty_kj_mol": report["mlip"][
                "isosteric_heat_zero_loading_kj_mol"
            ]["ci95_half_width"],
        },
        *[
            {
                "source": reference["label"],
                "reference_class": reference["reference_class"],
                "henry_coefficient_mmol_g_bar": reference.get(
                    "henry_coefficient_mmol_g_bar"
                ),
                "henry_uncertainty_mmol_g_bar": reference.get(
                    "henry_uncertainty_mmol_g_bar"
                ),
                "isosteric_heat_kj_mol": reference.get(
                    "isosteric_heat_zero_loading_kj_mol"
                ),
                "isosteric_heat_uncertainty_kj_mol": reference.get(
                    "isosteric_heat_uncertainty_kj_mol"
                ),
            }
            for reference in report["references"]
        ],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(
    report: dict[str, Any],
    blocks: list[dict[str, Any]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    seed_colors = {12345: "#2878B5", 23456: "#E07A2D", 34567: "#3A923A"}
    class_colors = {
        CRAFTED_REFERENCE_CLASS: "#6A4C93",
        EXPERIMENTAL_REFERENCE_CLASS: "#C14953",
        "dft": "#00876C",
        "dft_finetuned": "#00876C",
        "finetuned_mlip": "#00876C",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    specifications = (
        (
            "henry_coefficient_mmol_g_bar",
            "henry_coefficient_mmol_g_bar",
            "henry_uncertainty_mmol_g_bar",
            r"$K_H$ / mmol g$^{-1}$ bar$^{-1}$",
            True,
            "Henry coefficient (log scale)",
        ),
        (
            "isosteric_heat_zero_loading_kj_mol",
            "isosteric_heat_zero_loading_kj_mol",
            "isosteric_heat_uncertainty_kj_mol",
            r"$Q_{st}^{0}$ / kJ mol$^{-1}$",
            False,
            "Zero-loading adsorption heat",
        ),
    )
    for axis, specification in zip(axes, specifications):
        mlip_key, reference_key, uncertainty_key, ylabel, logarithmic, title = specification
        available = [
            reference
            for reference in report["references"]
            if reference.get(reference_key) is not None
        ]
        jitter = np.linspace(-0.16, 0.16, len(blocks))
        for offset, block in zip(jitter, blocks):
            axis.scatter(
                offset,
                block[mlip_key],
                color=seed_colors.get(int(block["seed"]), "#2878B5"),
                alpha=0.8,
                s=27,
                zorder=3,
            )
        mlip = report["mlip"][mlip_key]
        axis.errorbar(
            0,
            mlip["estimate"],
            yerr=mlip["ci95_half_width"],
            fmt="D",
            color="#111111",
            capsize=5,
            markersize=6,
            linewidth=1.6,
            label="Pooled mean and 95% CI",
            zorder=5,
        )
        for position, reference in enumerate(available, start=1):
            axis.errorbar(
                position,
                reference[reference_key],
                yerr=reference.get(uncertainty_key),
                fmt="s",
                color=class_colors.get(reference["reference_class"], "#555555"),
                capsize=5,
                markersize=8,
                linewidth=1.6,
                zorder=4,
            )
        labels = [_mlip_label(report.get("mlip_model", {}))] + [
            reference["label"] for reference in available
        ]
        axis.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(title)
        axis.tick_params(axis="x", labelsize=8)
        if logarithmic:
            axis.set_yscale("log")
    axes[0].legend(frameon=False, fontsize="small", loc="upper left")
    fig.suptitle(
        f"{report['material']}/{report['adsorbate']} at "
        f"{report['temperature_K']:.2f} K: precision and layered validation"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _mlip_label(model: dict[str, Any]) -> str:
    name = str(model.get("name") or model.get("model") or "MLIP")
    if model.get("dispersion") is True:
        name += "+D3"
    return name.replace("_", " ") + "\npooled"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a pooled MLIP-MC Widom result with layered references."
    )
    parser.add_argument("summary")
    parser.add_argument("--crafted-root", default="CRAFTED-2.0.0")
    parser.add_argument("--without-crafted", action="store_true")
    parser.add_argument("--isodb-root", default="isodb-library")
    parser.add_argument("--without-experiment", action="store_true")
    parser.add_argument("--experimental-doi", default=DEFAULT_EXPERIMENTAL_DOI)
    parser.add_argument(
        "--experimental-pressure-cutoff-bar",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--curated-reference",
        action="append",
        default=[],
        help="Curated DFT/finetuned Widom reference JSON; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--charge-scheme", default="DDEC")
    parser.add_argument("--low-pressure-cutoff-pa", type=float, default=2000.0)
    parser.add_argument("--temperature-tolerance-k", type=float, default=0.5)
    args = parser.parse_args(argv)
    report = create_widom_reference_comparison(
        args.summary,
        None if args.without_crafted else args.crafted_root,
        args.output_dir,
        charge_scheme=args.charge_scheme,
        low_pressure_cutoff_Pa=args.low_pressure_cutoff_pa,
        isodb_root=None if args.without_experiment else args.isodb_root,
        experimental_doi=args.experimental_doi,
        experimental_pressure_cutoff_bar=args.experimental_pressure_cutoff_bar,
        curated_reference_paths=tuple(args.curated_reference),
        temperature_tolerance_K=args.temperature_tolerance_k,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
