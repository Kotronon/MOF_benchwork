"""Independent MOF-5 random control and shared holdout worker.

Live campaigns are read through their atomic JSON snapshots and are never written
by this module. Only completed observations with verified evidence are reused.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
from pathlib import Path

from modules import module_b_campaign as al
from modules.module_b_active_learning import _save
from modules.module_b_extension import verify_source
from modules.module_b_runtime import campaign_lock

DEFAULT_SOURCE = Path('outputs/module_B_active_learning_mof5_20bar/production/mof5/campaign.json')
DEFAULT_ROOT = Path('outputs/module_B_mof5_al_comparison')


def read(path):
    return json.loads(Path(path).read_text())


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def protocol(state):
    return dict(model_identity=state['model_identity'], mode=state['mode'],
                candidates=sorted(state['candidates'], key=lambda p: p['candidate_id']),
                initial_design=state['initial_design'],
                holdout=sorted(state['holdout'], key=lambda p: p['candidate_id']),
                options={k: state['options'].get(k) for k in (
                    'max_training_points', 'max_precision_retries', 'absolute_loading_floor',
                    'relative_uncertainty_target', 'raw_log_uncertainty', 'exclude_holdout_from_stopping', 'random_seed')},
                temperature_K=state['base_config']['conditions']['temperature_K'],
                pressure_bounds=state['base_config']['conditions']['pressures_bar'],
                sampling={k: state['base_config']['simulation'].get(k) for k in (
                    'seeds', 'cycles', 'initialization_cycles', 'thermo_every_steps')})


def campaign(root, role):
    return Path(root)/role/'production'/'mof5'/'campaign.json'


def checked_source(manifest):
    state = read(manifest['source'])  # _save uses atomic replace; no live writer lock needed.
    if signature(protocol(state)) != manifest['protocol_sha256']:
        raise ValueError('Live source protocol changed; prepare a new comparison root.')
    return state


def completed_history(state, ids):
    return [deepcopy(event) for event in state['history']
            if 'replaced_observation' in event and event['replaced_observation']['candidate_id'] in ids]


def clone(state, target, role, random_seed):
    from modules.module_b_cluster import freeze_config, archive_prepared_manifest
    from pipeline.planning import build_run_plan
    from pipeline.prepare import prepare_benchmark
    from pipeline.materialization import materialize_benchmark

    with campaign_lock(target):
        if target.exists():
            existing = read(target)
            if existing.get('comparison_role') != role or signature(protocol(existing)) != signature(protocol(state)):
                raise ValueError(f'Existing {role} campaign has a different protocol.')
            if role == 'random' and existing['options'].get('random_acquisition_seed') != random_seed:
                raise ValueError('Existing random campaign uses a different seed.')
            return
        result = deepcopy(state)
        cfg = deepcopy(state['base_config'])
        cfg['output'].update(directory=str(target.parent/'simulation'), run_id='prepared_model', resume=True, overwrite=False)
        cfg = freeze_config(cfg, target.parent)
        if al.identity(cfg) != state['model_identity']:
            raise ValueError('Comparison physics differs from the source.')
        _save(target.parent/'config.json', cfg)
        prepared = materialize_benchmark(prepare_benchmark(build_run_plan(cfg)))
        archive_prepared_manifest(target.parent, prepared)
        result.update(base_config=cfg, pending_request=None, status='ready', comparison_role=role)
        result.pop('last_error', None)
        result.pop('last_error_logs', None)
        if role == 'random':
            result['options'].update(acquisition_strategy='random', random_acquisition_seed=random_seed)
            result['shared_initial_ready']=False
        else:
            result.update(results=[], history=[])
            # Make high-pressure validation available early, without changing membership.
            split = state.get('extension', {}).get('previous_pressure_max', 10.)
            result['holdout'] = sorted(state['holdout'], key=lambda p: (p['pressure_bar'] <= split, p['pressure_bar'], p['candidate_id']))
        _save(target.parent/'readiness.json', {'status':'prepared','physical_validation':'open', 'comparison_role':role})
        _save(target, result)


def prepare(root=DEFAULT_ROOT, source=DEFAULT_SOURCE, random_seed=2026):
    root, source = Path(root), Path(source)
    target = root/'comparison.json'
    settings = dict(source=str(source.resolve()), random_seed=random_seed)
    # The writer's output must never overlap any comparison worker or report directory.
    if root.resolve().is_relative_to(source.parent.resolve()) or source.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError('Comparison root and source directory must not overlap.')
    with campaign_lock(target):
        if target.exists():
            manifest=read(target)
            if manifest['settings'] != settings:
                raise ValueError('Comparison settings differ; choose a separate root.')
            checked_source(manifest)
            return manifest
        snapshot = read(source)
        verify_source(snapshot, source.parent)
        initial_ids = {p['candidate_id'] for p in snapshot['initial_design']}
        # Preserve the pre-extension shared starting point, even if AL is now running.
        origin = snapshot.get('extension', {}).get('source_archive')
        baseline_ids = {r['candidate_id'] for r in read(Path(origin)/'campaign.json')['results']} if origin else initial_ids
        available = {r['candidate_id'] for r in snapshot['results']}
        if not baseline_ids <= initial_ids or not baseline_ids <= available:
            raise ValueError('Common baseline must consist of available initial-design observations.')
        baseline = deepcopy(snapshot)
        baseline.update(results=[r for r in snapshot['results'] if r['candidate_id'] in baseline_ids],
                        history=completed_history(snapshot, baseline_ids), pending_request=None, status='ready')
        clone(baseline, campaign(root,'random'), 'random', random_seed)
        clone(baseline, campaign(root,'holdouts'), 'holdouts', random_seed)
        manifest = dict(schema_version=1, settings=settings, source=str(source),
                        protocol_sha256=signature(protocol(snapshot)), baseline_ids=sorted(baseline_ids),
                        baseline_path=str(root/'baseline.json'),
                        random=str(campaign(root,'random')), holdouts=str(campaign(root,'holdouts')),
                        checkpoints=[len(snapshot['initial_design']),16,20,24],
                        criteria={'primary':['CO2 RMSE','N2 RMSE','CO2/N2 selectivity MAE'],
                                  'relative_error_diagnostic_target':.2,
                                  'complete_holdouts_required':len(snapshot['holdout']),
                                  'precision_required_for_final_claim':True,
                                  'random_trajectories':1,
                                  'interpretation':'Single-seed pilot; no statistically established AL superiority. External physical validation remains open.'})
        _save(root/'baseline.json', baseline)
        _save(target, manifest)
        capture(root,'al',baseline)
        capture(root,'random',read(campaign(root,'random')))
        return manifest


def observation_key(row):
    return (row['candidate_id'], signature(row.get('raw_diagnostics', {}).get('logs', row.get('source_summary'))))


def synchronize_initial(root):
    """Import common initial results read-only; never simulate them twice."""
    root=Path(root);manifest=read(root/'comparison.json');snapshot=checked_source(manifest)
    target=campaign(root,'random');initial={p['candidate_id'] for p in snapshot['initial_design']}
    shared=[r for r in snapshot['results'] if r['candidate_id'] in initial]
    verification=deepcopy(snapshot);verification.update(results=shared,holdout_results=[])
    verify_source(verification, Path(manifest['source']).parent)
    with campaign_lock(target):
        state=read(target)
        by_id={r['candidate_id']:r for r in state['results']}
        for row in shared:
            previous=by_id.get(row['candidate_id'])
            if previous and observation_key(previous) != observation_key(row):
                state['history'].append({'replaced_observation':previous})
            by_id[row['candidate_id']]=deepcopy(row)
        state['results']=list(by_id.values())
        keys={observation_key(e['replaced_observation']) for e in state['history'] if 'replaced_observation' in e}
        for event in completed_history(snapshot,initial):
            if observation_key(event['replaced_observation']) not in keys:
                state['history'].append(event);keys.add(observation_key(event['replaced_observation']))
        missing=initial-{r['candidate_id'] for r in shared}
        retry=[r for r in shared if not r['precision_met'] and r.get('precision_retries',0)<snapshot['options']['max_precision_retries']]
        state['shared_initial_ready']=not missing and not retry
        _save(target,state)
        return dict(status='waiting_for_shared_initial_design' if missing or retry else 'ready_for_random_selection',
                    missing_initial_states=len(missing), pending_initial_precision_retries=len(retry))


def capture(root, strategy, state):
    """Immutable completed-data checkpoints; partial simulations are never evaluated."""
    data={k:deepcopy(state[k]) for k in ('results','model_identity')}
    data['history']=completed_history(state,{r['candidate_id'] for r in state['results']})
    key=signature(data)
    folder=Path(root)/'checkpoints'/strategy
    with campaign_lock(folder/'index.json'):
        target=folder/(key+'.json')
        if not target.exists():
            data.update(captured_at=datetime.now(timezone.utc).isoformat(), checkpoint_sha256=key)
            _save(target,data)


def run(root, role, max_requests, jobs=3):
    if max_requests < 1 or jobs not in (1,2,3):
        raise ValueError('Use a positive request budget and one to three seed jobs.')
    root=Path(root);manifest=read(root/'comparison.json');checked_source(manifest)
    target=campaign(root,role)
    # Hold the worker lock across synchronization and execution; two launchers cannot race.
    with campaign_lock(target):
        if role=='random':
            status=synchronize_initial(root)
            if status['status']!='ready_for_random_selection':
                return status
        for _ in range(max_requests):
            result=al.execute(target,1,holdout=role=='holdouts',jobs=jobs)
            if role=='random':capture(root,'random',read(target))
            if result['status']!='execution_budget_exhausted':return result
    return dict(status='execution_budget_exhausted', requests=max_requests)


def cost(state):
    initial={p['candidate_id'] for p in state['initial_design']}
    rows=state['results']+[e['replaced_observation'] for e in state['history'] if 'replaced_observation' in e]
    unique={observation_key(r):r for r in rows}
    total=sum(r.get('total_wall_time_seconds') or 0 for r in unique.values())
    common=sum(r.get('total_wall_time_seconds') or 0 for r in unique.values() if r['candidate_id'] in initial)
    return dict(total_seed_wall_seconds=total, common_initial_seed_wall_seconds=common,
                acquisition_seed_wall_seconds=total-common,
                note='Sum of seed runtimes including replaced observations; not elapsed campaign time or measured CPU time.')


def evaluate(state, labels):
    prediction=al.predict(state,state['candidates']) if len(state['results'])>=3 else None
    initial={p['candidate_id'] for p in state['initial_design']};seen={r['candidate_id'] for r in state['results']}
    split=state.get('extension',{}).get('previous_pressure_max',10.)
    regions={}
    for name,rows in (('overall',labels),('previous_domain',[r for r in labels if r['pressure_bar']<=split]),
                      ('extended_domain',[r for r in labels if r['pressure_bar']>split])):
        regions[name]=dict(holdout_points=len(rows),precision_unmet=sum(not r['precision_met'] for r in rows),
                           metrics=al.validation_metrics(state,prediction,rows),
                           status='available' if rows and prediction else 'missing_holdout_data' if not rows else 'insufficient_training_data')
    return dict(training_points=len(seen), initial_remaining=len(initial-seen),
                adaptive_points=len(seen-initial), training_precision_unmet=sum(not r['precision_met'] for r in state['results']),
                regions=regions, cost=cost(state))


def report(root=DEFAULT_ROOT):
    root=Path(root)
    with campaign_lock(root/'comparison.json'):
        manifest=read(root/'comparison.json');source=checked_source(manifest)
        random=read(manifest['random']);worker=read(manifest['holdouts'])
        for state in (random,worker):
            if signature(protocol(state))!=manifest['protocol_sha256']:
                raise ValueError('Comparison worker protocol mismatch.')
        held={p['candidate_id'] for p in source['holdout']}
        labels=worker['holdout_results']
        if len({r['candidate_id'] for r in labels})!=len(labels) or any(r['candidate_id'] not in held for r in labels):
            raise ValueError('Unexpected or duplicate holdout labels.')
        # Verify completed worker observations while the worker may continue another request.
        verify=deepcopy(worker);verify.update(results=[],holdout_results=labels)
        verify_source(verify,Path(manifest['holdouts']).parent)
        capture(root,'al',source);capture(root,'random',random)
        curve=[]
        for strategy,base in (('al',source),('random',random)):
            checkpoints=[read(p) for p in (root/'checkpoints'/strategy).glob('*.json')]
            for checkpoint in sorted(checkpoints,key=lambda x:x['captured_at']):
                snapshot=deepcopy(base);snapshot.update(results=checkpoint['results'],history=checkpoint['history'])
                curve.append(dict(strategy=strategy,captured_at=checkpoint['captured_at'],
                                  checkpoint_sha256=checkpoint['checkpoint_sha256'],**evaluate(snapshot,labels)))
        current={name:evaluate(state,labels) for name,state in (('al',source),('random',random))}
        missing=len(held)-len(labels)
        ready=not missing and all(r['precision_met'] for r in labels)
        matched=[]
        for count in sorted({r['training_points'] for r in curve}):
            rows={name:[r for r in curve if r['strategy']==name and r['training_points']==count] for name in ('al','random')}
            if all(rows.values()):
                pair={name:values[-1] for name,values in rows.items()}
                matched.append(dict(training_points=count,phase='adaptive_comparison' if all(r['adaptive_points']>0 and r['initial_remaining']==0 for r in pair.values()) else 'shared_initial_design',strategies=pair))
        result=dict(protocol_sha256=manifest['protocol_sha256'],test_set_sha256=signature(labels),
                    criteria=manifest['criteria'],validation_status='holdout_precision_met' if ready else 'incomplete_or_imprecise_holdouts',
                    completed_holdouts=len(labels),missing_holdouts=missing,
                    previously_inspected_holdouts=source.get('extension',{}).get('previously_inspected_holdouts',[]),
                    current=current,matched_budgets=matched,learning_curve=curve,external_validation='open')
        holdout_cost_state=deepcopy(worker);holdout_cost_state['results']=worker['holdout_results']
        result['shared_holdout_cost']=cost(holdout_cost_state)
        result['simulation_diagnostics']=diagnostics(source,worker)
        target=root/'report';_save(target/'comparison_report.json',result)
        diagnostic_rows=result['simulation_diagnostics']
        if diagnostic_rows:
            with (target/'simulation_diagnostics.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(diagnostic_rows[0]));writer.writeheader();writer.writerows(diagnostic_rows)

        with (target/'learning_curve.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['strategy','training_points','component','rmse_molecules_per_cell','mae_molecules_per_cell','total_seed_hours','acquisition_seed_hours','holdout_points','test_set_sha256'])
            for row in curve:
                for component,metrics in row['regions']['overall']['metrics'].items():
                    writer.writerow([row['strategy'],row['training_points'],component,metrics.get('rmse_molecules_per_cell'),metrics.get('mae_molecules_per_cell',metrics.get('mae')),row['cost']['total_seed_wall_seconds']/3600,row['cost']['acquisition_seed_wall_seconds']/3600,len(labels),result['test_set_sha256']])
        plot(curve,target)
        lines=['# MOF-5: AL versus random selection','',
               f"Shared holdouts: {len(labels)}/{len(held)}; status: {result['validation_status']}.",
               'The same completed labels evaluate every checkpoint. No labels are imported into live training campaigns.',
               'Random selection reuses the shared initial design. Strategy comparison begins only after it is complete.',
               'Primary metrics: component RMSE and selectivity MAE. The 20% relative-error target is diagnostic, not proof of scientific validity.',
               'Costs include precision retries and are sums of seed runtimes; elapsed time must be measured separately.',
               'One random trajectory is a pilot comparison, not statistical evidence of superiority.',
               'Previously inspected holdouts are not a fresh final test set. External physical validation remains open.','']
        for name,value in current.items():
            lines.append(f"{name}: {value['training_points']} training states; {value['initial_remaining']} initial states remaining; {value['adaptive_points']} adaptive/random-acquired states.")
        (target/'report.md').write_text('\n'.join(lines)+'\n')
        return result


def diagnostics(source, worker):
    rows=[]
    for role, records in (('training',source['results']),('holdout',worker['holdout_results'])):
        for record in records:
            for component, values in record['components'].items():
                rows.append(dict(role=role,candidate_id=record['candidate_id'],pressure_bar=record['pressure_bar'],
                                 component=component,gas_mole_fraction=record['mixture'][component],
                                 mean_molecules=values['mean_molecules'],
                                 relative_ci95_half_width=values.get('relative_ci95_half_width'),
                                 component_precision_met=values.get('precision_met'),
                                 point_precision_met=record['precision_met'],precision_retries=record.get('precision_retries',0),
                                 sum_seed_runtime_hours=(record.get('total_wall_time_seconds') or 0)/3600))
    return rows


def plot(curve,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for column,component in enumerate(('CO2','N2')):
        for strategy in ('al','random'):
            rows=[r for r in curve if r['strategy']==strategy and component in r['regions']['overall']['metrics']]
            for ax,axis in ((axes[0,column],'points'),(axes[1,column],'cost')):
                x=[r['training_points'] if axis=='points' else r['cost']['total_seed_wall_seconds']/3600 for r in rows]
                y=[r['regions']['overall']['metrics'][component]['rmse_molecules_per_cell'] for r in rows]
                ax.plot(x,y,'o-',label=strategy)
                ax.set(xlabel='Training states' if axis=='points' else 'Sum of seed runtimes / h',ylabel='Holdout RMSE / molecules per cell',title=component)
                if rows:ax.legend()
    fig.suptitle('MOF-5 pilot comparison: shared holdouts, common initial design')
    fig.savefig(target/'learning_curve.png',dpi=160);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','holdout','random','report','status'])
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--random-seed',type=int,default=2026)
    parser.add_argument('--max-requests',type=int)
    parser.add_argument('--jobs',type=int,choices=[1,2,3],default=3)
    args=parser.parse_args(argv)
    if args.root.is_absolute():parser.error('Use a repository-relative comparison root.')
    if args.command in ('holdout','random') and (args.max_requests is None or args.max_requests<1):
        parser.error('An explicit positive --max-requests is required.')
    if args.command=='prepare':
        from modules.module_b_cluster import check_dependencies
        check_dependencies();result=prepare(args.root,args.source,args.random_seed)
    elif args.command in ('holdout','random'):
        result=run(args.root,'holdouts' if args.command=='holdout' else 'random',args.max_requests,args.jobs)
        report(args.root)
    elif args.command=='report':
        full=report(args.root);result={k:full[k] for k in ('validation_status','completed_holdouts','missing_holdouts','current')}
    else:
        manifest=read(args.root/'comparison.json');source=checked_source(manifest)
        result={role:{'status':state['status'],'training_points':len(state['results']),'holdout_points':len(state['holdout_results']), 'pending_request':state.get('pending_request')} for role,state in (
            ('al',source),('random',read(manifest['random'])),('holdouts',read(manifest['holdouts'])))}
    print(json.dumps(result,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
