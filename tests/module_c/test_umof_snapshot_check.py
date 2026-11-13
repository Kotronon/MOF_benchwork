"""Diagnostic geometry guards using synthetic fixtures, not scientific results."""
import tempfile
from pathlib import Path
import struct
import unittest

from ase import Atoms

from analysis.umof_snapshot_check import geometry, final_transition


class SnapshotChecks(unittest.TestCase):
    def setUp(self):
        self.host = Atoms('Zn', positions=[[0, 0, 0]], cell=[20, 20, 20], pbc=True)
        self.guest = Atoms('OCO', positions=[[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]])

    def test_intramolecular_bonds_are_excluded_from_contact_search(self):
        guest = self.guest.copy(); guest.translate([5, 5, 5])
        state = geometry(self.host + guest, self.host, self.guest)
        self.assertEqual(state['guest_count'], 1)
        self.assertGreater(state['minimum_contact_A'], 5)
        self.assertEqual(state['contact_kind'], 'host-guest')

    def test_guest_guest_contact_and_removal_identity(self):
        first = self.guest.copy(); first.translate([5, 5, 5])
        second = first.copy(); second.translate([1.2, 0, 0])
        state = geometry(self.host + first + second, self.host, self.guest)
        self.assertEqual(state['contact_kind'], 'guest-guest')
        self.assertAlmostEqual(state['minimum_contact_A'], 1.2)
        self.assertIn(state['remove_guest_index'], (0, 1))

    def test_host_movement_and_guest_distortion_are_rejected(self):
        atoms = self.host + self.guest
        atoms.positions[0, 0] += .1
        with self.assertRaisesRegex(ValueError, 'Framework moved'):
            geometry(atoms, self.host, self.guest)
        atoms = self.host + self.guest
        atoms.positions[-1, 2] += .1
        with self.assertRaisesRegex(ValueError, 'Nonrigid guest'):
            geometry(atoms, self.host, self.guest)

    def test_empty_host_has_no_guest_contact(self):
        state = geometry(self.host, self.host, self.guest)
        self.assertIsNone(state['minimum_contact_A'])
        self.assertIsNone(state['remove_guest_index'])

    def test_final_log_record_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'trace.bin'
            path.write_bytes(struct.pack('iiddi', 1, 1, -.2, -100., 4)
                             + struct.pack('iiddi', 7, 2, -.5, -125., 7))
            result = final_transition(path)
            self.assertEqual(result['step'], 7)
            self.assertEqual(result['guest_count'], 2)
            self.assertEqual(result['interaction_eV'], -.5)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                final_transition(path)


if __name__ == '__main__':
    unittest.main()
