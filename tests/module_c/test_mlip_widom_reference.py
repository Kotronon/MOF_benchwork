from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from analysis.mlip_widom_reference import create_widom_reference_comparison


class WidomReferenceComparisonTests(unittest.TestCase):
    def test_comparison_reports_reference_outside_block_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            crafted = root / "crafted"
            (crafted / "ISOTHERM_FILES").mkdir(parents=True)
            (crafted / "ENTHALPY_FILES").mkdir(parents=True)
            for forcefield, scale, heat in (
                ("DREIDING", 1.0e-5, -16.0),
                ("UFF", 1.5e-5, -18.0),
            ):
                prefix = f"DDEC_ZIF-8_{forcefield}_CO2_298"
                self._write_rows(
                    crafted / "ISOTHERM_FILES" / f"{prefix}.csv",
                    [(100.0, scale * 100.0, 0.0), (1000.0, scale * 1000.0, 0.0)],
                )
                self._write_rows(
                    crafted / "ENTHALPY_FILES" / f"{prefix}.csv",
                    [(100.0, heat, 0.2)],
                )
            summary = {
                "material": "ZIF-8",
                "adsorbate": "CO2",
                "temperature_K": 298.15,
                "model": {"name": "mace_mp_small"},
                "pooled_block_analysis": {
                    "metrics": {
                        "henry_coefficient_mmol_g_bar": {
                            "estimate": 0.04,
                            "ci95_half_width": 0.002,
                        },
                        "isosteric_heat_zero_loading_kj_mol": {
                            "estimate": 8.5,
                            "ci95_half_width": 0.2,
                        },
                    },
                    "blocks": [
                        {
                            "seed": 12345,
                            "henry_coefficient_mmol_g_bar": 0.039,
                            "isosteric_heat_zero_loading_kj_mol": 8.4,
                        },
                        {
                            "seed": 23456,
                            "henry_coefficient_mmol_g_bar": 0.041,
                            "isosteric_heat_zero_loading_kj_mol": 8.6,
                        },
                    ],
                },
            }
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            report = create_widom_reference_comparison(
                summary_path,
                crafted,
                root / "output",
                isodb_root=None,
            )

            self.assertEqual(len(report["crafted_references"]), 2)
            self.assertTrue(
                report["comparisons"][0][
                    "henry_reference_outside_mlip_block_range"
                ]
            )
            for path in report["outputs"].values():
                self.assertTrue(Path(path).is_file())

    def test_comparison_loads_matching_experimental_isotherm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = self._write_summary(root)
            isodb = root / "isodb"
            article = (
                isodb
                / "Library"
                / "10.1021acs.jced.1c00900"
            )
            article.mkdir(parents=True)
            isotherm = {
                "DOI": "10.1021/acs.jced.1c00900",
                "adsorbates": [{"name": "Carbon Dioxide"}],
                "adsorbent": {"name": "ZIF-8"},
                "adsorptionUnits": "mmol/g",
                "articleSource": "test",
                "category": "exp",
                "isotherm_data": [
                    self._isotherm_point(0.0, 0.0),
                    self._isotherm_point(0.2, 0.14),
                    self._isotherm_point(0.5, 0.35),
                    self._isotherm_point(1.0, 0.8),
                ],
                "pressureUnits": "bar",
                "temperature": 298.0,
            }
            path = article / "test.Isotherm1.json"
            path.write_text(json.dumps(isotherm), encoding="utf-8")

            report = create_widom_reference_comparison(
                summary_path,
                None,
                root / "output",
                isodb_root=isodb,
                experimental_pressure_cutoff_bar=0.75,
            )

            self.assertEqual(len(report["experimental_references"]), 1)
            reference = report["experimental_references"][0]
            self.assertAlmostEqual(
                reference["henry_coefficient_mmol_g_bar"],
                0.7,
            )
            self.assertEqual(reference["henry_fit_point_count"], 2)
            self.assertIsNone(
                reference["isosteric_heat_zero_loading_kj_mol"]
            )

    def test_curated_reference_requires_matching_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = self._write_summary(root)
            curated_path = root / "goeminne.json"
            curated_path.write_text(
                json.dumps(
                    {
                        "reference_id": "goeminne_zif8",
                        "label": "Goeminne finetuned NequIP",
                        "reference_class": "dft_finetuned",
                        "material": "ZIF-8",
                        "adsorbate": "CO2",
                        "temperature_K": 273.0,
                        "metrics": {
                            "isosteric_heat_zero_loading_kj_mol": 14.0
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "is at 273 K"):
                create_widom_reference_comparison(
                    summary_path,
                    None,
                    root / "output",
                    isodb_root=None,
                    curated_reference_paths=(curated_path,),
                )

    def test_curated_dft_reference_is_included(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = self._write_summary(root)
            curated_path = root / "goeminne.json"
            curated_path.write_text(
                json.dumps(
                    {
                        "reference_id": "goeminne_zif8",
                        "label": "Goeminne finetuned NequIP",
                        "reference_class": "dft_finetuned",
                        "material": "ZIF-8",
                        "adsorbate": "CO2",
                        "temperature_K": 298.15,
                        "metrics": {
                            "isosteric_heat_zero_loading_kj_mol": {
                                "value": 14.0,
                                "uncertainty": 0.3,
                            }
                        },
                        "provenance": {
                            "doi": "10.5281/zenodo.7904959"
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = create_widom_reference_comparison(
                summary_path,
                None,
                root / "output",
                isodb_root=None,
                curated_reference_paths=(curated_path,),
            )

            self.assertTrue(report["goeminne_reference"]["available"])
            self.assertEqual(len(report["curated_references"]), 1)
            self.assertEqual(
                report["curated_references"][0][
                    "isosteric_heat_zero_loading_kj_mol"
                ],
                14.0,
            )

    @staticmethod
    def _write_summary(root: Path) -> Path:
        summary = {
            "material": "ZIF-8",
            "adsorbate": "CO2",
            "temperature_K": 298.15,
            "model": {"name": "mace_mp_small", "dispersion": False},
            "pooled_block_analysis": {
                "metrics": {
                    "henry_coefficient_mmol_g_bar": {
                        "estimate": 0.04,
                        "ci95_half_width": 0.002,
                    },
                    "isosteric_heat_zero_loading_kj_mol": {
                        "estimate": 8.5,
                        "ci95_half_width": 0.2,
                    },
                },
                "blocks": [
                    {
                        "seed": 12345,
                        "henry_coefficient_mmol_g_bar": 0.039,
                        "isosteric_heat_zero_loading_kj_mol": 8.4,
                    },
                    {
                        "seed": 23456,
                        "henry_coefficient_mmol_g_bar": 0.041,
                        "isosteric_heat_zero_loading_kj_mol": 8.6,
                    },
                ],
            },
        }
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary_path

    @staticmethod
    def _isotherm_point(pressure: float, loading: float) -> dict:
        return {
            "pressure": pressure,
            "species_data": [
                {
                    "name": "Carbon Dioxide",
                    "adsorption": loading,
                    "composition": 1,
                }
            ],
        }

    @staticmethod
    def _write_rows(path: Path, rows: list[tuple[float, ...]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
