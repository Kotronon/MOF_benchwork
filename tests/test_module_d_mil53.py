"""MIL-53 adaptation contracts, without requiring a LAMMPS installation."""
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

from modules.module_d_mil53 import REFERENCE, nonempty, pair_energy_force, prepare_mil53, source_sections
_spec = importlib.util.spec_from_file_location("mil53_cluster_runner", Path(__file__).parents[1] / "scripts/run_module_d_mil53.py")
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)
check_capabilities = _runner.check_capabilities


class Mil53Tests(unittest.TestCase):
    def test_cross_force_is_negative_energy_derivative(self):
        for r in (1.5, 2.5, 4., 7., 14.):
            for charge in (-.7, 0., .35):
                parameters = dict(sigma=3.2, epsilon=.08, charge_product=charge, gaussian_radius=1.6742)
                _, force = pair_energy_force(r, **parameters)
                h = 1e-5
                numeric = -(pair_energy_force(r+h, **parameters)[0] - pair_energy_force(r-h, **parameters)[0])/(2*h)
                self.assertAlmostEqual(force, numeric, delta=max(1e-7, abs(force)*1e-7))

    def test_source_host_is_preserved_and_guest_pairs_are_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            prepare_mil53(root)
            source = source_sections((REFERENCE / "lammps_2x4x2.data").read_text())
            derived = source_sections((root / "dry.data").read_text())
            for section in ("Atoms", "Bonds", "Angles", "Dihedrals", "Impropers", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs"):
                self.assertEqual(nonempty(source[section]), nonempty(derived[section]), section)
            self.assertEqual(len(nonempty(derived["Atoms"])), 1216)
            for name in ("lammps.table", "eibonded.table"):
                self.assertEqual((root/name).read_bytes(), (REFERENCE/name).read_bytes())
            provenance = json.loads((root / "model_provenance.json").read_text())
            pairs = {(p["i"], p["j"]) for p in provenance["guest_pairs"]}
            self.assertEqual(pairs, {(i,j) for i in range(1,9) for j in (9,10)} | {(9,9),(9,10),(10,10)})
            guest = source_sections((root / "co2.mol").read_text())
            self.assertEqual({tuple(map(int, row[2:])) for row in nonempty(guest["Bonds"])}, {(1,2),(1,3),(2,3)})
            with self.assertRaises(FileExistsError):
                prepare_mil53(root)

    def test_missing_yaff_fails_before_execution(self):
        with self.assertRaisesRegex(ValueError, "YAFF"):
            check_capabilities("* Bond styles:\nharmonic\n* Fix styles:\ngcmc npt nvt rigid/nvt/small\n")

    def test_native_help_accepts_fix_heading_without_colon(self):
        check_capabilities("""* Atom styles:
full
* Pair styles:
hybrid/overlay coul/long table
* Bond styles:
hybrid mm3 table zero
* Angle styles:
hybrid mm3 cosine/periodic cross
* Dihedral styles:
fourier
* Improper styles:
distharm
* KSpace styles:
ewald
* Fix styles
gcmc rigid/nvt/small npt nvt
* Compute styles:
temp pressure
""")


if __name__ == "__main__":
    unittest.main()
