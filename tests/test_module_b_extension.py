"""Domain-extension regressions; no production simulations."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modules import module_b_campaign as al
from modules.module_b_extension import extension_design, prepare_extension, validate_bounds, verify_source, digest, refresh_evidence_checksums
from modules.module_b_profiles import point
from modules.module_b_active_learning import _save


def fixture():
    make = lambda p,y: point(p,298.15,{'CO2': y,'N2':1-y})
    initial = [make(p,y) for y in (.15,.3,.7) for p in (.1,1,5,10)]
    held = [make(.2+i*.01,.4) for i in range(16)]
    return dict(base_config={'conditions': {'pressures_bar':[.1,10.], 'temperature_K':298.15},
                             'output':{}}, candidates=initial+held, initial_design=initial,
                holdout=held, results=[dict(initial[0],precision_met=True,precision_retries=0)],
                holdout_results=[], options={}, history=[], imports=[], model_identity='physics',
                campaign_kind='production',material_key='mof5',pending_request={'not':'copied'})


class ExtensionTests(unittest.TestCase):
    def test_bounds(self):
        for a,b in [(0,20),(-1,20),(20,20),(21,20),(.1,float('nan')),(.1,float('inf'))]:
            with self.assertRaises(ValueError): validate_bounds(a,b)
        validate_bounds(.1,20)

    def test_design_and_next_request(self):
        s=fixture();before=deepcopy(s)
        candidates,initial,held=extension_design(s,.1,20)
        self.assertEqual((candidates,initial,held),extension_design(s,.1,20))
        self.assertEqual(s,before)
        self.assertEqual(len(held),20)
        self.assertEqual(initial[0],s['initial_design'][0])
        self.assertEqual(initial[1],point(20,298.15,{'CO2':.15,'N2':.85}))
        self.assertEqual(initial[2:],s['initial_design'][1:])
        self.assertEqual(len(candidates),len(s['candidates'])+4096+4+4)
        self.assertEqual({p['pressure_bar'] for p in held[16:]},{12.,18.})
        self.assertFalse({p['candidate_id'] for p in held} & {p['candidate_id'] for p in initial})
        s.update(candidates=candidates,initial_design=initial,holdout=held,pending_request=None)
        s['options']={'max_training_points':24,'max_precision_retries':1}
        self.assertEqual(al.recommend(s)['candidate'],initial[1])

    def test_invalid_source_design(self):
        s=fixture()
        with self.assertRaises(ValueError):extension_design(s,.2,20)
        with self.assertRaises(ValueError):extension_design(s,.1,5)
        s['holdout'].append(s['initial_design'][0])
        with self.assertRaisesRegex(ValueError,'overlap'):extension_design(s,.1,20)

    def test_separate_prepare_idempotency_and_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            origin=Path(d)/'origin';origin.mkdir();source=origin/'campaign.json'
            s=fixture();evidence=origin/'evidence.log';evidence.write_text('completed evidence')
            s['results'][0]['source_summary']=str(evidence)
            _save(source,s);before=source.read_bytes()
            root=Path(d)/'target'
            with patch('modules.module_b_extension.verify_source'), \
                 patch('modules.module_b_cluster.freeze_config',side_effect=lambda cfg,folder:cfg), \
                 patch.object(al,'identity',return_value='physics'), \
                 patch('pipeline.planning.build_run_plan',return_value={}), \
                 patch('pipeline.prepare.prepare_benchmark',return_value={}), \
                 patch('pipeline.materialization.materialize_benchmark',return_value={}) as materialize, \
                 patch('modules.module_b_cluster.archive_prepared_manifest'):
                p=prepare_extension(root,source,.1,20);state=json.loads(p.read_text())
                self.assertIsNone(state['pending_request'])
                self.assertEqual(state['results'][0]['precision_retries'],0)
                self.assertEqual(Path(state['results'][0]['source_summary']).read_text(),'completed evidence')
                self.assertEqual(state['options']['max_training_points'],24)
                self.assertFalse((p.parent/'report').exists())
                saved=p.read_bytes();prepare_extension(root,source,.1,20)
                self.assertEqual(p.read_bytes(),saved);self.assertEqual(materialize.call_count,1)
                with self.assertRaisesRegex(ValueError,'settings differ'):prepare_extension(root,source,.1,30)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(evidence.read_text(),'completed evidence')

    def test_reject_model_and_asset_changes(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);asset=folder/'asset';asset.write_text('original')
            s=fixture();_save(folder/'model_manifest.json',{'model_identity':'physics','assets':{str(asset):{'sha256':digest(asset)}}})
            with patch.object(al,'identity',return_value='different'):
                with self.assertRaisesRegex(ValueError,'identity mismatch'):verify_source(s,folder)
            asset.write_text('changed')
            with patch.object(al,'identity',return_value='physics'):
                with self.assertRaisesRegex(ValueError,'asset changed'):verify_source(s,folder)

    def test_relocated_json_input_checksums(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);asset=folder/'reference.json';asset.write_text('{"reference": true}\n')
            meta=folder/'prepare_summary.json'
            _save(meta,{'result':{'files':{'source_sha256':{str(asset):'old-checksum'}}}})
            before=asset.read_bytes()
            refresh_evidence_checksums(folder)
            self.assertEqual(json.loads(meta.read_text())['result']['files']['source_sha256'][str(asset)],digest(asset))
            self.assertEqual(asset.read_bytes(),before)

    def test_observation_evidence_validation(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);s=fixture();r=s['results'][0]
            log=folder/'seed.log';log.write_text('completed log')
            params={'temperature_K':298.15,'mixture':r['mixture'],'production_steps':50,'equilibration_steps':5,'run_steps':55}
            summary=folder/'isotherm_summary.json'
            summary_data={'status':'completed','results':[{'pressure_bar':r['pressure_bar'],'summary':{'components':{'CO2':{'mean_molecules':1.}}}}],
                          'replicate_results':[{'pressure_bar':r['pressure_bar'],'seed':1,'log_file':str(log)}]}
            _save(summary,summary_data)
            _save(folder/'prepare_summary.json',{'prepare_plan':{'parameters':params,'inputs':{}}})
            _save(folder/'model_manifest.json',{'model_identity':'physics','assets':{}})
            r.update(source_summary=str(summary),source_summary_sha256=digest(summary),seeds=[1],replicate_count=1,
                     components={'CO2':{'mean_molecules':1.}},raw_diagnostics={'logs':[{'log_file':str(log),'sha256':digest(log)}]})
            with patch.object(al,'identity',return_value='physics'), \
                 patch('pipeline.planning.build_run_plan',return_value={}), \
                 patch('pipeline.prepare.prepare_benchmark',return_value={'parameters':params,'inputs':{}}), \
                 patch('parsers.lammps_log_parser.get_log',return_value='unused'), \
                 patch('parsers.lammps_log_parser.parse_lammps_log',return_value={'rows':[{'Step':55}]}) as parse:
                verify_source(s,folder)
                parse.return_value={'rows':[{'Step':54}]}
                with self.assertRaisesRegex(ValueError,'Incomplete seed'):verify_source(s,folder)
                parse.return_value={'rows':[{'Step':55}]}
                r['components']['CO2']['mean_molecules']=2.
                with self.assertRaisesRegex(ValueError,'uptake differs'):verify_source(s,folder)
                r['components']['CO2']['mean_molecules']=1.
                log.write_text('tampered log')
                with self.assertRaisesRegex(ValueError,'checksum mismatch'):verify_source(s,folder)

    def test_region_metrics_report(self):
        from tests.test_module_b_general import GeneralCampaignTests
        s=GeneralCampaignTests().state()
        # Exercise actual report output, suppressing expensive plotting only.
        pts=s['candidates'][:3];s['results']=[dict(p,replicate_count=3,total_wall_time_seconds=1,precision_retries=0) for p in pts]
        s['holdout_results']=[];s['initial_design']=pts
        s['extension']={'previous_pressure_max':10,'previously_inspected_holdouts':[]}
        pred={'components':{},'kernels':{}}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'campaign.json';_save(p,s)
            def artifacts(state,pred,target):target.mkdir(parents=True,exist_ok=True);return {}
            with patch.object(al,'predict',return_value=pred),patch('modules.module_b_reporting.write_artifacts',side_effect=artifacts):
                r=al.report(p)
            self.assertEqual(r['metrics_by_region']['extended_domain']['status'],'missing_holdout_data')
            self.assertEqual(r['acquisition_progress']['initial_design_remaining'],0)


if __name__=='__main__':unittest.main()
