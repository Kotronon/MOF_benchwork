from pathlib import Path
import tempfile
import unittest

from analysis.module_c_isotherm_report import load_crafted_reference, compare_reference


class IsothermReferenceReportTests(unittest.TestCase):
    def test_units_metadata_and_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'DDEC_ZIF-8_UFF_CO2_298.csv'
            path.write_text('# pressure[Pa],mean_volume[mol/kg],mean_error[mol/kg]\n10000,1,.1\n100000,3,.2\n')
            ref = load_crafted_reference(path, 'ZIF-8', 'CO2', 298.)
            self.assertEqual(ref['points'][0]['pressure_bar'], .1)
            self.assertEqual(ref['points'][0]['loading_mol_per_kg'], 1.)
            self.assertEqual(ref['points'][0]['source_reported_error_mol_per_kg'], .1)
            self.assertEqual(ref['adsorption_basis'], 'absolute')
            for args in [('ZIF-4', 'CO2', 298), ('ZIF-8', 'N2', 298), ('ZIF-8', 'CO2', 273)]:
                with self.assertRaises(ValueError):
                    load_crafted_reference(path, *args)

    def test_exact_matching_interpolation_and_no_extrapolation(self):
        ref = {'points': [{'pressure_bar': 1., 'loading_mol_per_kg': 2.},
                          {'pressure_bar': 3., 'loading_mol_per_kg': 6.}]}
        rows = [{'model': 'model', 'pressure_bar': p, 'mean_loading_mol_per_kg': q}
                for p, q in [(1., 3.), (2., 3.), (4., 99.)]]
        compared, metrics = compare_reference(rows, ref)
        self.assertEqual(len(compared), 2)
        self.assertEqual(compared[0]['reference_match'], 'exact')
        self.assertEqual(compared[1]['reference_loading_mol_per_kg'], 4.)
        self.assertEqual(metrics[0]['rmse_mol_per_kg'], 1.)
        self.assertEqual(metrics[0]['mean_signed_deviation_mol_per_kg'], 0.)
        self.assertEqual(metrics[0]['unmatched_pressure_count'], 1)
        self.assertEqual(metrics[0]['mean_absolute_relative_deviation_percent'], 37.5)

    def test_zero_reference_does_not_produce_infinite_relative_error(self):
        ref = {'points': [{'pressure_bar': 1., 'loading_mol_per_kg': 0.},
                          {'pressure_bar': 2., 'loading_mol_per_kg': 1.}]}
        rows = [{'model': 'model', 'pressure_bar': 1., 'mean_loading_mol_per_kg': 1.}]
        compared, metrics = compare_reference(rows, ref)
        self.assertIsNone(compared[0]['relative_deviation_percent'])
        self.assertIsNone(metrics[0]['mean_absolute_relative_deviation_percent'])
        self.assertEqual(metrics[0]['mae_mol_per_kg'], 1.)
