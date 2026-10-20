import unittest
import numpy as np
import pandas as pd
from ase import Atoms
from modules.evaluate import error_row, summarize


class StaticEvaluationTests(unittest.TestCase):
    def structure(self, symbols, framework_count):
        atoms = Atoms(symbols)
        atoms.info.update(candidate_id='iter00_single_adsorbate_0010',
                          framework_atom_count=framework_count, REF_energy=-10.0)
        atoms.new_array('REF_forces', np.zeros((len(atoms), 3)))
        return atoms

    def test_atom_groups_and_weighted_summary(self):
        host = self.structure('H', 1)
        guest = self.structure('HCOO', 1)
        first = error_row(host, -9, [[1, 0, 0]], 'model', 'pretrained')
        second = error_row(guest, -12, [[2, 0, 0]]*4, 'model', 'pretrained')
        self.assertTrue(np.isnan(first['co2_force_mae']))
        self.assertEqual(second['n_co2'], 1)
        self.assertEqual(second['framework_force_mae'], 2)
        self.assertEqual(second['co2_force_mae'], 2)
        self.assertEqual(second['raw_total_energy_signed_error_ev'], -2)
        summary = summarize(pd.DataFrame([first, second]), ['model']).iloc[0]
        self.assertAlmostEqual(summary.force_mae_ev_A, 1.8)
        self.assertAlmostEqual(summary.configuration_weighted_force_mae_ev_A, 1.5)
        self.assertAlmostEqual(summary.co2_force_mae, 2)
        self.assertAlmostEqual(summary.framework_force_mae, 1.5)

    def test_reject_invalid_groups_and_forces(self):
        atoms = self.structure('HCOO', 2)
        with self.assertRaisesRegex(ValueError, 'CO2'):
            error_row(atoms, 0, np.zeros((4, 3)), 'x', 'pretrained')
        atoms.info['framework_atom_count'] = 1
        with self.assertRaisesRegex(ValueError, 'shape'):
            error_row(atoms, 0, np.zeros((3, 3)), 'x', 'pretrained')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            error_row(atoms, 0, np.full((4, 3), np.nan), 'x', 'pretrained')


if __name__ == '__main__':
    unittest.main()
