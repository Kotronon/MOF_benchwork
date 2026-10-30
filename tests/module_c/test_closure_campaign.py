"""Guard the scientific identities, units and finite execution budget."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from analysis.module_c_closure import error_metrics, seed_interval
from modules.module_c_mlips.closure_campaign import completed_result, job_list, validate
from modules.module_c_mlips.umof_finetuning import sha256_file
from modules.module_c_mlips import umof_adsorption_comparison as adapter


class ClosureTests(unittest.TestCase):
    def test_single_chain_does_not_get_seed_interval(self):
        self.assertIsNone(seed_interval([2.])["ci95_half_width"])
        interval = seed_interval([1., 2., 3.])
        self.assertAlmostEqual(interval["mean"], 2.)
        self.assertGreater(interval["ci95_half_width"], 2.)

    def test_signed_bias_and_nonfinite_rejection(self):
        self.assertEqual(error_metrics([1., 2.], [2., 3.])["bias"], -1.)
        with self.assertRaises(ValueError): error_metrics([np.nan], [0.])
        with self.assertRaises(ValueError): error_metrics([1., 2.], [1.])

    def test_job_budget_and_distinct_pressure_seed_identities(self):
        config = {"models": [{"label": "pretrained"}, {"label": "finetuned"}],
                  "pressures_bar": [.1, 1., 10.], "seeds": [17, 23, 31]}
        jobs = job_list(config)
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len({j["id"] for j in jobs}), 18)

    def test_input_validation_detects_changed_weights(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); asset = root / "model"
            asset.write_bytes(b"frozen weights")
            spec = {"path": "model", "sha256": sha256_file(asset)}
            config = {"schema_version": 1, "component": "CO2", "device": "cuda", "dtype": "float64",
                "models": [{**spec, "label": label, "profile": "mh", "head": head}
                    for label, head in (("pretrained", "r2scan+d4"), ("finetuned", "Default"))],
                "framework": spec, "adsorbate": spec, "references": [spec],
                "seeds": [17, 23, 31], "pressures_bar": [.1, 1., 10.],
                "gcmc": {"equilibration_steps": 10, "production_steps": 20, "checkpoint_interval": 5},
                "overlap_covalent_radius_scale": .8}
            self.assertEqual(len(validate(config, root)), 18)
            asset.write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "Changed asset"): validate(config, root)

    def test_result_reuse_requires_identity_and_result_checksum(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); result = root / "result.json"
            result.write_text('{"value": 1}')
            identity = {"config_sha256": "fixed", "runtime_versions": {"torch": "fixed"}}
            (root / "completed.json").write_text(json.dumps({"identity": identity, "result_path": "result.json", "result_sha256": sha256_file(result)}))
            self.assertEqual(completed_result(root, identity), {"value": 1})
            with self.assertRaises(ValueError): completed_result(root, {"config_sha256": "changed"})
            result.write_text('{"value": 2}')
            with self.assertRaisesRegex(ValueError, "checksum"): completed_result(root, identity)

    def test_shared_worker_selects_mh_instead_of_polar(self):
        from ase import Atoms
        host = Atoms("Zn", positions=[[0, 0, 0]], cell=[20,20,20], pbc=True)
        guest = Atoms("OCO", positions=[[0,0,-1.16],[0,0,0],[0,0,1.16]])
        config = {"framework": {"path": "host"}, "adsorbate": {"path": "gas"},
            "models": [{"label": "finetuned", "path": "model", "profile": "mh", "head": "Default"}],
            "device": "cpu", "dtype": "float64", "overlap_covalent_radius_scale": .8,
            "gcmc": {"equilibration_steps": 2, "production_steps": 4, "checkpoint_interval": 1},
            "component": "CO2", "material": "fixture", "temperature_K": 298.}
        result = {"pressure_points": [{"mean_uptake_molecules_per_unit_cell": 1., "uptake_std_molecules_per_unit_cell": .2, "transition_log_path": "unused"}]}
        with TemporaryDirectory() as tmp, patch("ase.io.read", side_effect=[host, guest]), \
             patch("modules.module_c_mlips.umof_profiles.calculator", return_value=SimpleNamespace(head="Default", r_max=6.)) as calc, \
             patch.object(adapter, "consistent_calculator", return_value=SimpleNamespace(gas_energy_eV=0.)), \
             patch.object(adapter, "engine_classes", return_value=(object, object)), \
             patch.object(adapter, "production_diagnostics", return_value={}), \
             patch("engines.mlip_mc_runner.run_mlip_mc_gcmc", return_value=result):
            adapter.worker(config, "finetuned", 17, "gcmc", 1., Path(tmp))
            self.assertEqual(calc.call_args.kwargs["model"], "mh")


if __name__ == "__main__":
    unittest.main()
