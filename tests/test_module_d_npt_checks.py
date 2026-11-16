"""Physical-time and input-preservation contracts; no MD execution."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("npt_checks", Path(__file__).parents[1] / "scripts/module_d_npt_checks.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)

SOURCE = """timestep 0.25
run 2000
thermo 100
molecule guestmol assets/guest_template
fix guestmd gas rigid/nvt/small molecule temp 300 300 100 mol guestmol reinit no
fix_modify guestmd dynamic/dof yes
fix hostmd host npt temp 300 300 100 tri 1 1 1000
fix exchange gas gcmc 20 5 0 0 12345 300 0 1 pressure 1
fix cellstats all ave/time 1 1000 1000 v_cell_lx file cell_history.txt
fix stats all ave/time 1 1000 1000 v_uptake file measurements.txt
fix attempts all ave/time 1 1 1000 f_exchange[1] file mc_attempts.txt
dump trajectory all custom 1000 trajectory.lammpstrj id mol type q x y z ix iy iz
dump_modify trajectory sort id
run 20000
"""


class NptCheckTests(unittest.TestCase):
    def test_default_sampling_preserves_original_transformation(self):
        for factor in (1, 2):
            for dry in (True, False):
                original = checks.transform(SOURCE, factor, dry)
                self.assertEqual(checks.configure_sampling(original, factor, .5, 5, .25, False), original)

    def test_extended_runs_keep_physical_exchange_and_averaging_intervals(self):
        for factor in (1, 2):
            output = checks.configure_sampling(checks.transform(SOURCE, factor, False), factor, 5, 15, .05, True)
            self.assertIn(f"run {20000 * factor}\n", output)
            self.assertIn(f"run {60000 * factor}\n", output)
            self.assertIn(f"gcmc {20 * factor} 5 0 0", output)
            self.assertIn(f"ave/time {factor} 1000 {1000 * factor}", output)
            self.assertIn(f"dump trajectory all custom {200 * factor} ", output)
            self.assertIn("sort id format float %.16g", output)
            self.assertIn("temp 300 300 100 tri 1 1 1000", output)

    def test_dry_run_has_no_guest_integration_or_exchange(self):
        output = checks.configure_sampling(checks.transform(SOURCE, 1, True), 1, 5, 15, .05, True)
        for removed in ("molecule guestmol", "fix guestmd", "fix_modify guestmd", "fix exchange", "fix attempts"):
            self.assertNotIn(removed, output)
        self.assertIn("fix hostmd host npt", output)

    def test_invalid_sampling_is_rejected(self):
        source = checks.transform(SOURCE, 1, False)
        for settings in ((0, 15, .05), (5.1, 15, .05), (5, float("nan"), .05),
                         (5, 15, -.05), (5, 15, .0003), (5, float("inf"), .05)):
            with self.assertRaises(ValueError):
                checks.configure_sampling(source, 1, *settings, True)


if __name__ == "__main__":
    unittest.main()
