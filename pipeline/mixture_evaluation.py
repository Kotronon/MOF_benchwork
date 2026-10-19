"""Component-resolved evaluation; never compare mixture totals to pure gases."""
from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

FIELDS = ["pressure_bar", "quantity", "unit", "value", "ci95_lower", "ci95_upper",
          "ci95_status", "ci95_method", "relative_ci95_half_width", "precision_met",
          "point_precision_met", "replicate_count", "gas_phase_mole_fraction",
          "mean_molecules_per_cell", "framework_mass_amu", "framework_mass_source",
          "reference_value", "comparison_error", "comparison_relative_error_percent",
          "reference_path", "reference_status"]


def is_mixture(path: str | Path) -> bool:
    data = json.loads(Path(path).read_text())
    if any(len(r.get("summary", {}).get("components", {})) > 1
           for r in data.get("results", []) + data.get("replicate_results", [])):
        return True
    prep = _prepare(Path(path))
    return prep.get("module", {}).get("id") == "B" or len(prep.get("parameters", {}).get("mixture", {})) > 1


def _prepare(path: Path) -> dict[str, Any]:
    prep = path.parent / "prepare_summary.json"
    return json.loads(prep.read_text()).get("prepare_plan", {}) if prep.exists() else {}


def corrected_summary(path: str | Path) -> dict[str, Any]:
    # Local import avoids the runners -> evaluate -> mixture_evaluation cycle.
    from pipeline.runners import _aggregate_replicates, _convergence_report
    path = Path(path)
    data = deepcopy(json.loads(path.read_text()))
    config = _prepare(path).get("convergence", data.get("convergence", {}))
    replicates = data.get("replicate_results", [])
    if not replicates:
        # Legacy aggregate summaries preserve the paired order of seed means.
        for result in data.get("results", []):
            components = result["summary"].get("components", {})
            seeds = result.get("seeds", [])
            if not seeds or any(len(c.get("replicate_mean_molecules", [])) != len(seeds) for c in components.values()):
                raise ValueError("Mixture re-evaluation requires per-seed summaries or paired seed means.")
            for i, seed in enumerate(seeds):
                cs = {name: {"mean_molecules": c["replicate_mean_molecules"][i],
                             "gas_phase_mole_fraction": c["gas_phase_mole_fraction"]}
                      for name, c in components.items()}
                replicates.append({"pressure_bar": result["pressure_bar"], "seed": seed,
                                   "summary": {"components": cs, "mean_adsorbates": sum(c["mean_molecules"] for c in cs.values())}})
    if not replicates or any(len(r.get("summary", {}).get("components", {})) < 2 for r in replicates):
        raise ValueError("Mixture evaluation needs component-resolved seed results.")
    data["results"] = _aggregate_replicates(replicates, config)
    data["replicate_results"] = replicates
    data["convergence"] = _convergence_report(data["results"], config)
    return data


def framework_mass(path: Path, explicit: str | Path | None = None) -> tuple[float | None, str]:
    from pipeline.evaluate import _infer_framework_data, framework_mass_from_lammps_data
    source = Path(explicit) if explicit else _infer_framework_data(path)
    if source is not None and source.exists():
        value = framework_mass_from_lammps_data(source)
        return (value, str(source)) if value > 0 and math.isfinite(value) else (None, "invalid_framework_mass")
    # Archived conversion metadata, not the incorrect adsorption/reference columns.
    archive = path.parent / "evaluated_isotherm.csv"
    if archive.exists():
        try:
            with archive.open() as handle:
                rows = list(csv.DictReader(handle))
            masses = [float(r["framework_mass_amu"]) for r in rows]
            if masses and all(math.isfinite(m) and m > 0 and math.isclose(m, masses[0], rel_tol=1e-10) for m in masses):
                return masses[0], f"{archive}:framework_mass_amu (archived; framework data unavailable)"
        except (KeyError, ValueError, TypeError):
            pass
        return None, "missing_or_inconsistent_archived_mass"
    return None, "framework_mass_unavailable"


def _material(value: str) -> str:
    normalized = str(value).upper().replace("-", "").replace(" ", "")
    from resolvers.material_resolver import DEFAULT_ALIAS_MAP
    for canonical, aliases in DEFAULT_ALIAS_MAP.items():
        values = {str(x).upper().replace("-", "").replace(" ", "") for x in [canonical, *aliases]}
        if normalized in values:
            return "IRMOF1" if "MOF5" in values else canonical.upper().replace("-", "").replace(" ", "")
    return normalized


def assess_reference(reference: dict[str, Any], parameters: dict[str, Any],
                     mixture: dict[str, float], pressure: float, mass: float | None,
                     evaluation: dict[str, Any]) -> dict[str, Any]:
    """Admit only explicitly matching absolute NIST mixture data.

    Composition is matched per point. No interpolation across compositions or
    extrapolation in pressure, and no single-gas EOS conversion for mixtures.
    """
    from pipeline.nist_isodb_parser import ADSORBATE_NAME_ALIASES, _pressure_to_bar, _loading_to_mol_per_kg
    from pipeline.evaluate import _reference_at_pressure
    path = Path(reference.get("copied_path") or reference.get("path", ""))
    reasons = list(reference.get("exclusion_reasons", []))
    result = {"path": str(path), "pressure_bar": pressure, "accepted": False,
              "reasons": reasons, "component_loadings": {}}
    if reference.get("excluded") or not reference.get("use_in_evaluation", True):
        reasons.append("excluded_by_reference_metadata")
    if not path.is_file():
        reasons.append("reference_file_missing")
        return result
    if path.suffix.lower() != ".json":
        reasons.append("pure_component_or_missing_mixture_metadata")
        return result
    try:
        data = json.loads(path.read_text())
        material = data.get("adsorbent", {}).get("name")
        target_material = parameters.get("material_id")
        if not material or not target_material:
            reasons.append("material_metadata_missing")
        elif _material(material) != _material(target_material):
            reasons.append("material_mismatch")
        temperature, target_temperature = data.get("temperature"), parameters.get("temperature_K")
        if temperature is None or target_temperature is None:
            reasons.append("temperature_metadata_missing")
        elif not math.isfinite(float(temperature)) or abs(float(temperature) - float(target_temperature)) > float(evaluation.get("temperature_tolerance_K", 2.0)):
            reasons.append("temperature_mismatch")
        basis = str(data.get("isotherm_type", "")).lower()
        if basis not in ("absolute", "excess"):
            reasons.append("adsorption_basis_unknown")
        elif basis != "absolute":
            reasons.append("excess_reference_requires_mixture_conversion")
        if evaluation.get("simulation_basis", "absolute") != "absolute":
            reasons.append("unsupported_simulation_basis")
        names = {}
        for item in data.get("adsorbates", []):
            name = str(item.get("name", "")).lower()
            canonical = next((key for key, aliases in ADSORBATE_NAME_ALIASES.items() if name in aliases), name.upper())
            names[item.get("InChIKey", name)] = canonical
        if set(names.values()) != set(mixture) or len(names) != len(mixture):
            reasons.append("components_mismatch_or_missing")
        if data.get("compositionType") != "molefraction":
            reasons.append("gas_composition_metadata_missing")
        points = {c: [] for c in mixture}
        for raw in data.get("isotherm_data", []):
            species = {names.get(s.get("InChIKey"), str(s.get("name", "")).upper()): s for s in raw.get("species_data", [])}
            if set(species) != set(mixture):
                continue
            if any(s.get("composition") is None or not math.isclose(float(s["composition"]), mixture[c], abs_tol=1e-6, rel_tol=0)
                   for c, s in species.items()):
                continue
            p = _pressure_to_bar(raw["pressure"], data.get("pressureUnits", ""))
            values = {c: _loading_to_mol_per_kg(s["adsorption"], data.get("adsorptionUnits", ""), component=c, framework_mass_amu=mass)
                      for c, s in species.items()}
            if not math.isfinite(p) or p < 0 or any(v is None or not math.isfinite(v) or v < 0 for v in values.values()):
                continue
            if p == 0 and any(v > 0 for v in values.values()):
                continue
            for c, value in values.items():
                points[c].append({"pressure_Pa": p * 1e5, "loading_mol_per_kg": value})
        if not all(points.values()):
            reasons.append("composition_mismatch_or_missing_component_data")
        else:
            for c, series in points.items():
                if len({x["pressure_Pa"] for x in series}) != len(series):
                    reasons.append("duplicate_pressure_requires_reference_curation")
                    break
                match = _reference_at_pressure(sorted(series, key=lambda x: x["pressure_Pa"]), pressure * 1e5)
                if match is None:
                    reasons.append("pressure_outside_matching_composition_range")
                    break
                result["component_loadings"][c] = match["loading_mol_per_kg"]
    except (ValueError, KeyError, TypeError, AttributeError):
        reasons.append("invalid_or_incomplete_reference_metadata")
    result["reasons"] = sorted(set(reasons))
    result["accepted"] = not result["reasons"]
    if not result["accepted"]:
        result["component_loadings"] = {}
    return result


def build_mixture_evaluation(path: str | Path, framework_data=None, reference_csv=None,
                             evaluation_config=None, references=None) -> dict[str, Any]:
    from pipeline.evaluate import _reference_entries_for_run
    path = Path(path)
    data, prep = corrected_summary(path), _prepare(path)
    parameters = prep.get("parameters", {})
    evaluation = {**prep.get("evaluation", {}), **(evaluation_config or {})}
    mass, mass_source = framework_mass(path, framework_data)
    entries = references if references is not None else _reference_entries_for_run(path.parent)
    if reference_csv is not None:
        entries = [{"path": str(reference_csv)}]
    rows, assessments = [], []
    for point in data["results"]:
        summary = point["summary"]
        mixture = {c: s["gas_phase_mole_fraction"] for c, s in summary["components"].items()}
        checks = [assess_reference(ref, parameters, mixture, point["pressure_bar"], mass, evaluation) for ref in entries]
        assessments.extend(checks)
        accepted = [c for c in checks if c["accepted"]]
        quantities = [(c, s["mean_molecules"], s, mixture[c]) for c, s in summary["components"].items()]
        total_stat = {"ci95_lower": summary.get("ci95_lower_adsorbates"),
                      "ci95_upper": summary.get("ci95_upper_adsorbates"),
                      "ci95_status": "bounded" if summary.get("ci95_lower_adsorbates") is not None else "insufficient_replicates",
                      "ci95_method": "student_t_between_seeds", "relative_ci95_half_width": summary.get("relative_ci95_half_width")}
        quantities.append(("total", summary["mean_adsorbates"], total_stat, None))
        quantities.extend((pair, s["value"], s, None) for pair, s in summary["selectivity"].items())
        for quantity, value, statistic, fraction in quantities:
            ratio = "/" in quantity
            factor = 1.0 if ratio or mass is None else 1000 / mass
            row = {"pressure_bar": point["pressure_bar"], "quantity": quantity,
                   "unit": "dimensionless" if ratio else ("mol/kg" if mass else "molecules/cell"),
                   "value": value * factor if value is not None else None,
                   "ci95_lower": statistic.get("ci95_lower") * factor if statistic.get("ci95_lower") is not None else None,
                   "ci95_upper": statistic.get("ci95_upper") * factor if statistic.get("ci95_upper") is not None else None,
                   "ci95_status": statistic.get("ci95_status"), "ci95_method": statistic.get("ci95_method"),
                   "relative_ci95_half_width": statistic.get("relative_ci95_half_width"),
                   "precision_met": statistic.get("precision_met"), "point_precision_met": point["converged"],
                   "replicate_count": point["replicate_count"], "gas_phase_mole_fraction": fraction,
                   "mean_molecules_per_cell": None if ratio else value,
                   "framework_mass_amu": mass, "framework_mass_source": mass_source,
                   "reference_value": None, "comparison_error": None, "comparison_relative_error_percent": None,
                   "reference_path": None, "reference_status": "no_compatible_reference"}
            for match in accepted or [None]:
                compared = dict(row)
                if match is not None:
                    loads = match["component_loadings"]
                    if ratio:
                        left, right = quantity.split("/")
                        reference_value = loads[left] / loads[right] * mixture[right] / mixture[left] if loads[right] > 0 and mixture[left] > 0 else None
                    else:
                        reference_value = sum(loads.values()) if quantity == "total" else loads[quantity]
                    compared.update(reference_value=reference_value, reference_path=match["path"], reference_status="compatible")
                    if (ratio or mass is not None) and reference_value is not None and row["value"] is not None:
                        error = row["value"] - reference_value
                        compared.update(comparison_error=error, comparison_relative_error_percent=100 * error / reference_value if reference_value else None)
                    elif not ratio and mass is None:
                        compared.update(reference_value=None, reference_status="framework_mass_unavailable")
                rows.append(compared)
    return {"rows": rows, "reference_assessments": assessments, "summary": data,
            "framework_mass_amu": mass, "framework_mass_source": mass_source,
            "material": parameters.get("material_id", "Material unspecified"),
            "raw_logs_available": any((path.parent / "logs").glob("*.log"))}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_mixture(path: str | Path, output_dir=None, framework_data=None, reference_csv=None,
                     evaluation_config=None, references=None) -> dict[str, Any]:
    path = Path(path)
    target = Path(output_dir) if output_dir else path.parent / "evaluation_corrected"
    result = build_mixture_evaluation(path, framework_data, reference_csv, evaluation_config, references)
    target.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    write_csv(target / "evaluated_mixture.csv", rows)
    (target / "reference_assessments.json").write_text(json.dumps(result["reference_assessments"], indent=2) + "\n")
    (target / "isotherm_summary.json").write_text(json.dumps(result["summary"], indent=2) + "\n")
    (target / "convergence_report.json").write_text(json.dumps(result["summary"]["convergence"], indent=2) + "\n")
    accepted = any(r["comparison_error"] is not None for r in rows)
    status = "Mixture evaluated; compatible reference comparisons available" if accepted else "Pilot ausgewertet; quantitative Referenzvalidierung offen"
    fmt = lambda x: f"{x:.6g}" if isinstance(x, (int, float)) else (str(x) if x is not None else "—")
    lines = ["# Module B: korrigierte Gemisch-Auswertung", "", f"**{status}**", "",
             f"Quelle: `{path}`", "", f"Framework-Masse: {result['framework_mass_amu']} u. Herkunft: `{result['framework_mass_source']}`.", "",
             "Komponenten und Gesamtbeladung sind absolute Adsorption; mol/kg = mmol/g. Selektivität ist dimensionslos.",
             "95-%-Intervalle beschreiben die Streuung unabhängiger Seed-Mittelwerte. Komponenten: Student-t; Selektivität: Fieller mit gepaarten Seeds.",
             "Das Präzisionskriterium ist kein Nachweis der Equilibrierung. Unbeschränkte/undefinierte Intervalle erfüllen es nicht.", "",
             f"Präzisionsziel: relative Intervallhalbbreite ≤ {100*result['summary']['convergence']['relative_ci95_target']:g} %, mindestens {max(3, result['summary']['convergence']['minimum_replicates'])} Seeds.", "",
             "| Druck / bar | Größe | Wert | Einheit | 95-%-Intervall | relative Halbbreite | Intervallstatus | Präzisionsziel (Größe) | Präzisionsziel (Punkt) |",
             "|---:|:---|---:|:---|:---|---:|:---|:---|:---|"]
    unique = {(r["pressure_bar"], r["quantity"]): r for r in rows}
    for r in unique.values():
        interval = f"[{fmt(r['ci95_lower'])}, {fmt(r['ci95_upper'])}]" if r['ci95_lower'] is not None else "—"
        relative = f"{100*r['relative_ci95_half_width']:.2f} %" if r['relative_ci95_half_width'] is not None else "—"
        precision = "—" if r['precision_met'] is None else ("ja" if r['precision_met'] else "nein")
        lines.append(f"| {r['pressure_bar']:g} | {r['quantity']} | {fmt(r['value'])} | {r['unit']} | {interval} | {relative} | {r['ci95_status']} | {precision} | {'ja' if r['point_precision_met'] else 'nein'} |")
    lines += ["", "## Referenzprüfung", "", "Reingasisothermen werden nicht als Gemischreferenz verwendet. Unbekannte Adsorptionsbasis wird nicht als absolut interpretiert.", ""]
    for check in result["reference_assessments"]:
        lines.append(f"- {check['pressure_bar']:g} bar, `{Path(check['path']).name}`: " + ("kompatibel" if check['accepted'] else ", ".join(check['reasons'])))
    if accepted:
        lines += ["", "Komponentenweise Referenzwerte und Fehler stehen in `evaluated_mixture.csv`."]
    if not result["raw_logs_available"]:
        lines += ["", "**Rohlogs fehlen:** Zeitverlauf und Equilibrierung konnten nicht erneut geprüft werden."]
    lines += ["", "Ein bisheriger Vergleich von CO₂+N₂-Gesamtbeladung mit reinem CO₂ ist ungültig; daraus abgeleitete Fehlerprozente sind kein Validierungsergebnis."]
    report = target / "evaluation_report.md"
    report.write_text("\n".join(lines) + "\n")
    plot = _plot(list(unique.values()), target / "mixture_results.png", result["material"])
    # Preserve old artifacts, but put the correction directly in front of their reader.
    old = path.parent / "evaluation_report.md"
    if target != path.parent and old.exists() and "# Isotherm Evaluation" in old.read_text():
        marker = "> **ÜBERHOLT:** Gemischgesamtbeladung wurde mit einer Reingasreferenz verglichen."
        if marker not in old.read_text():
            old.write_text(marker + f" Korrigierte Auswertung: [{report.name}]({target.name}/{report.name}).\n\n" + old.read_text())
    return {"status": status, "evaluated_csv": str(target / "evaluated_mixture.csv"),
            "evaluation_report": str(report), "isotherm_plot": str(plot) if plot else None,
            "corrected_summary": str(target / "isotherm_summary.json"),
            "reference_candidates": str(target / "reference_assessments.json"),
            "point_count": len(result["summary"]["results"]),
            "mean_comparison_error_mol_per_kg": None, "mean_comparison_relative_error_percent": None}


def _plot(rows: list[dict[str, Any]], path: Path, material: str) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for quantity in dict.fromkeys(r["quantity"] for r in rows):
        if "/" in quantity and tuple(quantity.split("/")) != tuple(sorted(quantity.split("/"))):
            continue
        series = sorted((r for r in rows if r["quantity"] == quantity and r["value"] is not None), key=lambda r: r["pressure_bar"])
        if not series:
            continue
        ax = axes[1 if "/" in quantity else 0]
        line, = ax.plot([r["pressure_bar"] for r in series], [r["value"] for r in series], "o-", label=quantity)
        bounded = [r for r in series if r["ci95_lower"] is not None]
        if bounded:
            ax.vlines([r["pressure_bar"] for r in bounded], [r["ci95_lower"] for r in bounded], [r["ci95_upper"] for r in bounded], color=line.get_color())
        missing_interval = [r for r in series if r["ci95_lower"] is None]
        for r in missing_interval:
            ax.annotate(r["ci95_status"].replace("_", " "),
                        (r["pressure_bar"], r["value"]), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=line.get_color())
        if ax == axes[0]:
            ax.set_ylabel(f"Absolute loading / {series[0]['unit']}")
    for ax in axes:
        ax.set_xlabel("Total pressure / bar")
        if rows and all(r["pressure_bar"] > 0 for r in rows):
            ax.set_xscale("log")
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend()
    axes[1].set_ylabel("Adsorption selectivity / dimensionless")
    fig.suptitle(f"{material}: mixture results with 95% between-seed intervals")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
