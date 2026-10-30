"""Check cancellation and model-head selection in the energy diagnostic."""
from types import SimpleNamespace
import unittest

import numpy as np
from ase import Atoms

from modules.module_c_mlips.umof_energy_diagnostic import interaction_energy, atomic_reference


class FakeTensor:
    def __init__(self, values): self.values = np.asarray(values)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.values


class EnergyDiagnosticTests(unittest.TestCase):
    def test_elemental_energy_zero_cancels_for_balanced_composition(self):
        host_e, gas_e, pair_e = -100., -25., -125.2
        host_shift = 8*2. + 48*(-.4) + 24*.1 + 26*(-.2)
        gas_shift = -.4 + 2*(-.2)
        original = interaction_energy(pair_e, host_e, gas_e)
        shifted = interaction_energy(pair_e+host_shift+gas_shift,
                                     host_e+host_shift, gas_e+gas_shift)
        self.assertAlmostEqual(original, shifted, places=12)

    def test_mixed_model_reference_would_introduce_spurious_binding(self):
        # Disconnected fragments must give zero when all terms use one model.
        self.assertAlmostEqual(interaction_energy(-124.9, -100., -24.9), 0.)
        self.assertAlmostEqual(interaction_energy(-124.9, -100., -25.), .1)

    def test_atomic_reference_uses_selected_head_not_first_head(self):
        model = SimpleNamespace(heads=['pt_head', 'Default'],
            atomic_numbers=FakeTensor([1,6,8]),
            atomic_energies_fn=SimpleNamespace(atomic_energies=FakeTensor([[1.,2.,3.], [4.,5.,6.]])))
        calc = SimpleNamespace(models=[model])
        self.assertEqual(atomic_reference(calc, {'head': 'Default'}, Atoms('CO2')), 17.)
        self.assertEqual(atomic_reference(calc, {'head': 'pt_head'}, Atoms('CO2')), 8.)
