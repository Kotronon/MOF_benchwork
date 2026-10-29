"""Validate reference input conventions and refuse unconverged DFT energies."""
from pathlib import Path
import unittest
from ase import Atoms
from modules.module_c_mlips.umof_dft_reference import render_input, parse_energy, select_samples, HARTREE_TO_EV


class DFTReferenceTests(unittest.TestCase):
    def test_ghost_fragment_keeps_combined_basis_without_nuclei_or_double_d4(self):
        atoms=Atoms('ZnOCO',positions=[[0,0,0],[3,0,0],[4,0,0],[5,0,0]],cell=[20,20,20],pbc=True)
        text=render_input(atoms,project='fragment',cutoff=1200,assets=Path('/tmp/assets'),real_indices=[0])
        self.assertIn('&MGGA_X_R2SCAN',text)
        self.assertIn('&MGGA_C_R2SCAN',text)
        self.assertNotIn('VDW_POTENTIAL',text)
        self.assertIn('&KIND O_ghost\n      ELEMENT O',text)
        self.assertIn('GHOST T',text)
        self.assertIn('TZV2P-MOLOPT-SCAN-GTH-q12',text)
        self.assertIn('POTENTIAL GTH-SCAN-q12',text)
        self.assertNotIn('POTENTIAL GTH-SCAN-q6',text)
        self.assertIn('PERIODIC XYZ',text)

    def test_energy_requires_normal_end_and_final_scf_convergence(self):
        text='SCF run converged in 20 steps\nENERGY| Total FORCE_EVAL ( QS ) energy [hartree]: -1.25\nPROGRAM ENDED AT 2026-10-29\n'
        self.assertAlmostEqual(parse_energy(text),-1.25*HARTREE_TO_EV)
        for invalid in [text.replace('PROGRAM ENDED AT','unfinished'),
                        text.replace('SCF run converged','SCF run NOT converged'),
                        text+'[ABORT]',text.replace('-1.25','NaN')]:
            with self.assertRaises(ValueError): parse_energy(invalid)

    def test_selection_covers_both_source_trajectories_and_disagreement_scales(self):
        rows=[]
        for offset,name in [(0,'umof_mh_pretrained'),(10,'umof_mh_finetuned_025pct')]:
            for i,delta in enumerate([-.01,-.04,-.02]):
                rows.append({'sample_id':str(i+offset),'source_model':name,
                             'pretrained_adsorption_energy_ev':'-.1','finetuned_minus_pretrained_ev':str(delta)})
        self.assertEqual(select_samples(rows),[1,2,11,12])
