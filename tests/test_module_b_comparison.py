"""Random control, independent holdouts and no writes to a running AL campaign."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modules import module_b_campaign as al
from modules import module_b_comparison as compare
from modules.module_b_active_learning import _save
from modules.module_b_profiles import simplex_candidates


def state_fixture():
    points=simplex_candidates(['CO2','N2'],[.1,20],[298.15],32)
    with tempfile.TemporaryDirectory() as d,patch.object(al,'identity',return_value='physics'):
        state=al.initialize({'material':{'name':'MOF5'},'adsorbates':{'components':['CO2','N2'],'mixture':{'CO2':.15,'N2':.85}},
                             'conditions':{'temperature_K':298.15,'pressures_bar':[.1,20]}},Path(d)/'state.json',candidates=points,budget=24)
    state.update(initial_design=points[:3],holdout=points[-4:],campaign_kind='production',material_key='mof5')
    def row(p):
        return dict(p,precision_met=True,precision_retries=0,replicate_count=3,total_wall_time_seconds=30,
                    source_summary=p['candidate_id']+'.json',
                    components={c:{'mean_molecules':1.,'standard_error_molecules':.1} for c in ('CO2','N2')})
    state['results']=[row(p) for p in points[:3]]
    state['holdout_results']=[row(points[-1])]
    return state


def setup(root,state):
    source=root/'source'/'campaign.json';_save(source,state)
    random=deepcopy(state);random.update(comparison_role='random',shared_initial_ready=False)
    random['options'].update(acquisition_strategy='random',random_acquisition_seed=2026)
    worker=deepcopy(state);worker.update(comparison_role='holdouts',results=[])
    _save(compare.campaign(root/'comparison','random'),random)
    _save(compare.campaign(root/'comparison','holdouts'),worker)
    manifest=dict(source=str(source),protocol_sha256=compare.signature(compare.protocol(state)),
                  random=str(compare.campaign(root/'comparison','random')),holdouts=str(compare.campaign(root/'comparison','holdouts')),
                  criteria={})
    _save(root/'comparison'/'comparison.json',manifest)
    return source,root/'comparison'


class ComparisonTests(unittest.TestCase):
    def test_random_reproducible_and_independent_of_labels(self):
        state=state_fixture();state.update(comparison_role='random',shared_initial_ready=True)
        state['options'].update(acquisition_strategy='random',random_acquisition_seed=2026)
        before=deepcopy(state)
        with patch.object(al,'predict',side_effect=AssertionError('random must not fit a GP')):
            first=al.recommend(state)['candidate']
            state['holdout_results']=[{'poison':'unreadable'}]
            self.assertEqual(first,al.recommend(state)['candidate'])
        excluded={p['candidate_id'] for p in state['holdout']+state['results']}
        self.assertNotIn(first['candidate_id'],excluded)
        self.assertEqual(before['results'],state['results'])

    def test_random_cannot_duplicate_initial_states(self):
        state=state_fixture();state.update(comparison_role='random',shared_initial_ready=False)
        state['results']=state['results'][:1]
        state['options']['acquisition_strategy']='random'
        self.assertEqual(al.recommend(state)['status'],'waiting_for_shared_initial_design')

    def test_worker_role_guards(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'campaign.json';s=state_fixture();s['comparison_role']='holdouts';_save(p,s)
            with self.assertRaisesRegex(ValueError,'holdout execution'):al.propose(p)
            s['comparison_role']='random';_save(p,s)
            with self.assertRaisesRegex(ValueError,'separate shared holdout'):al.propose(p,holdout=True)

    def test_holdout_retry_uses_holdout_role_and_fourfold_length(self):
        with tempfile.TemporaryDirectory() as d:
            state=state_fixture();state['comparison_role']='holdouts'
            prototype=state['holdout_results'][0]
            state['holdout_results']=[dict(deepcopy(prototype),**p) for p in state['holdout']]
            state['holdout_results'][0]['precision_met']=False
            state['options']['max_precision_retries']=1
            p=Path(d)/'campaign.json';_save(p,state)
            with patch.object(al,'identity',return_value='physics'):
                decision=al.propose(p,holdout=True)
            self.assertEqual(decision['action'],'improve_holdout_precision')
            config=compare.read(decision['request']['config_path'])
            self.assertEqual(config['simulation']['cycles'],4*state['base_config']['simulation'].get('cycles',10000))
            self.assertEqual(len(compare.read(p)['results']),len(state['results']))

    def test_execute_accepts_holdout_precision_request(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=state_fixture();state['comparison_role']='holdouts'
            p=root/'campaign.json';cfg=root/'config.json';_save(p,state);_save(cfg,state['base_config'])
            request={'status':'awaiting_simulation','request':{'action':'improve_holdout_precision','config_path':str(cfg)}}
            with patch.object(al,'propose',return_value=request), \
                 patch('pipeline.planning.build_run_plan',return_value={}), \
                 patch('pipeline.prepare.prepare_benchmark',return_value={}), \
                 patch('pipeline.materialization.materialize_benchmark',return_value={}), \
                 patch('pipeline.runners.run_isotherm',return_value={'summary_file':'unused'}), \
                 patch.object(al,'ingest') as ingest:
                self.assertEqual(al.execute(p,1,holdout=True,jobs=3)['status'],'execution_budget_exhausted')
                ingest.assert_called_once()

    def test_holdout_failed_precision_is_explicit_after_retry(self):
        with tempfile.TemporaryDirectory() as d:
            state=state_fixture();state['comparison_role']='holdouts'
            prototype=state['holdout_results'][0]
            state['holdout_results']=[dict(deepcopy(prototype),**p) for p in state['holdout']]
            state['holdout_results'][0].update(precision_met=False,precision_retries=1)
            state['options']['max_precision_retries']=1
            p=Path(d)/'campaign.json';_save(p,state)
            with patch.object(al,'identity',return_value='physics'):
                self.assertEqual(al.propose(p,holdout=True)['status'],'holdout_precision_unmet')
            self.assertIsNone(compare.read(p)['pending_request'])

    def test_sync_while_source_locked_and_precision_gate(self):
        with tempfile.TemporaryDirectory() as d:
            state=state_fixture();source,root=setup(Path(d),state)
            random_path=compare.campaign(root,'random');random=compare.read(random_path)
            random['results']=random['results'][:1];_save(random_path,random)
            original=source.read_bytes()
            with compare.campaign_lock(source),patch.object(compare,'verify_source'):
                result=compare.synchronize_initial(root)
            self.assertEqual(result['status'],'ready_for_random_selection')
            self.assertEqual(len(compare.read(random_path)['results']),3)
            self.assertEqual(source.read_bytes(),original)
            state['results'][0]['precision_met']=False;_save(source,state)
            with patch.object(compare,'verify_source'):
                self.assertEqual(compare.synchronize_initial(root)['pending_initial_precision_retries'],1)
            self.assertFalse(compare.read(random_path)['shared_initial_ready'])

    def test_wait_does_not_execute(self):
        with tempfile.TemporaryDirectory() as d:
            state=state_fixture();state['results']=state['results'][:1];source,root=setup(Path(d),state)
            with patch.object(compare,'verify_source'),patch.object(al,'execute') as execute:
                result=compare.run(root,'random',2)
            self.assertEqual(result['status'],'waiting_for_shared_initial_design')
            execute.assert_not_called()

    def test_checkpoint_ignores_pending_requests(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=state_fixture();compare.capture(root,'al',state)
            state['history'].append({'request':{'candidate':{'candidate_id':'pending'}}})
            state['pending_request']={'not':'complete'}
            compare.capture(root,'al',state)
            self.assertEqual(len(list((root/'checkpoints/al').glob('*.json'))),1)

    def test_cost_includes_retries_without_double_counting(self):
        state=state_fixture();old=deepcopy(state['results'][0]);old['source_summary']='old.json'
        state['history']=[{'replaced_observation':old},{'replaced_observation':old}]
        self.assertEqual(compare.cost(state)['total_seed_wall_seconds'],120)
        self.assertEqual(compare.cost(state)['acquisition_seed_wall_seconds'],0)

    def test_protocol_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source,root=setup(Path(d),state_fixture());state=compare.read(source)
            state['options']['max_training_points']=25;_save(source,state)
            with self.assertRaisesRegex(ValueError,'protocol changed'):compare.checked_source(compare.read(root/'comparison.json'))

    def test_report_never_imports_holdouts_into_live_campaign(self):
        with tempfile.TemporaryDirectory() as d:
            state=state_fixture();source,root=setup(Path(d),state)
            original=source.read_bytes();random=compare.campaign(root,'random');random_before=random.read_bytes()
            prediction={'components':{c:[{'value':1.,'lower':.5,'upper':1.5} for p in state['candidates']] for c in state['components']}}
            with compare.campaign_lock(source),patch.object(compare,'verify_source'),patch.object(al,'predict',return_value=prediction),patch.object(compare,'plot'):
                result=compare.report(root)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(random.read_bytes(),random_before)
            self.assertEqual(result['completed_holdouts'],1)
            self.assertEqual(result['missing_holdouts'],3)
            self.assertEqual(result['current']['al']['regions']['overall']['metrics']['CO2']['rmse_molecules_per_cell'],0)
            self.assertEqual(result['matched_budgets'][0]['phase'],'shared_initial_design')

    def test_prepare_idempotent_and_pending_request_not_cloned(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);source=folder/'source'/'campaign.json';state=state_fixture()
            state['pending_request']={'candidate':state['candidates'][5],'action':'sample'};_save(source,state)
            original=source.read_bytes();root=folder/'comparison'
            with patch.object(compare,'verify_source'),patch.object(al,'identity',return_value='physics'), \
                 patch('modules.module_b_cluster.freeze_config',side_effect=lambda cfg,folder:cfg), \
                 patch('modules.module_b_cluster.archive_prepared_manifest'), \
                 patch('pipeline.planning.build_run_plan',return_value={}), \
                 patch('pipeline.prepare.prepare_benchmark',return_value={}), \
                 patch('pipeline.materialization.materialize_benchmark',return_value={}):
                compare.prepare(root,source);first=(root/'comparison.json').read_bytes()
                compare.prepare(root,source)
            self.assertEqual(first,(root/'comparison.json').read_bytes())
            self.assertEqual(source.read_bytes(),original)
            self.assertIsNone(compare.read(compare.campaign(root,'random'))['pending_request'])
            self.assertEqual(compare.read(compare.campaign(root,'holdouts'))['results'],[])


if __name__=='__main__':unittest.main()
