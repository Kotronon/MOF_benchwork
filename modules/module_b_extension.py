"""Prepare a separate pressure-domain extension from a verified MOF-5 campaign."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

from modules import module_b_campaign as al
from modules.module_b_active_learning import _save
from modules.module_b_profiles import point, simplex_candidates
from modules.module_b_runtime import campaign_lock, asset_fingerprints


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_bounds(lower, upper):
    if not all(math.isfinite(x) and x > 0 for x in (lower, upper)) or lower >= upper:
        raise ValueError('Pressure bounds must be finite, positive and strictly increasing.')


def extension_design(source, lower, upper):
    """Preserve locked states and add a reproducible upper-pressure domain."""
    validate_bounds(lower, upper)
    old_lower, old_upper = min(source['base_config']['conditions']['pressures_bar']), max(source['base_config']['conditions']['pressures_bar'])
    if lower != old_lower or upper <= old_upper:
        raise ValueError('Import requires the same lower bound and a larger upper bound.')
    t = source['base_config']['conditions']['temperature_K']
    make = lambda p, y: point(p, t, {'CO2': y, 'N2': 1-y})
    new = simplex_candidates(['CO2', 'N2'], [old_upper, upper], [t], count=4096, seed=2026)
    anchors = [make(upper, y) for y in (0., .15, .5, 1.)]
    # Fixed fractions reproduce 12 and 18 bar for the 10 -> 20 bar extension.
    holdout = deepcopy(source['holdout']) + [make(old_upper + f*(upper-old_upper), y)
                                            for f in (.2, .8) for y in (.15, .5)]
    initial = deepcopy(source['initial_design'])
    anchor = anchors[1]
    seen = {r['candidate_id'] for r in source['results']}
    insertion = next((i for i, p in enumerate(initial) if p['candidate_id'] not in seen), len(initial))
    initial.insert(insertion, anchor)
    candidates = list({p['candidate_id']: p for p in source['candidates'] + new + anchors + holdout}.values())
    held = {p['candidate_id'] for p in holdout}
    if held & {p['candidate_id'] for p in initial + source['results']}:
        raise ValueError('Training states overlap locked holdouts.')
    for p in candidates + initial + holdout:
        if point(p['pressure_bar'], p['temperature_K'], p['mixture']) != p or not lower <= p['pressure_bar'] <= upper:
            raise ValueError('Invalid candidate ID or pressure outside the requested domain.')
    return candidates, initial, holdout


def verify_source(source, folder):
    """Validate frozen physics, per-observation context and completed raw logs."""
    from pipeline.planning import build_run_plan
    from pipeline.prepare import prepare_benchmark
    from parsers.lammps_log_parser import get_log, parse_lammps_log

    if source.get('material_key') != 'mof5' or source.get('campaign_kind') != 'production':
        raise ValueError('Import requires a production MOF-5 campaign.')
    manifest = json.loads((folder/'model_manifest.json').read_text())
    if al.identity(source['base_config']) != source['model_identity'] or manifest['model_identity'] != source['model_identity']:
        raise ValueError('Source physical model identity mismatch.')
    for filename, entry in manifest['assets'].items():
        if not Path(filename).is_file() or digest(filename) != entry['sha256']:
            raise ValueError(f'Source model asset changed or missing: {filename}')
    expected = prepare_benchmark(build_run_plan(source['base_config']))
    ignored = {'pressures_bar', 'mixture', 'production_steps', 'equilibration_steps', 'run_steps',
               'save_dumps', 'save_restarts', 'dump_every_steps', 'restart_every_steps'}
    observations = source['results'] + source['holdout_results']
    ids = [r['candidate_id'] for r in observations]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate or overlapping training/holdout observations.')
    for r in observations:
        if point(r['pressure_bar'], r['temperature_K'], r['mixture'])['candidate_id'] != r['candidate_id']:
            raise ValueError('Observation candidate ID mismatch.')
        summary_path = Path(r['source_summary'])
        summary = json.loads(summary_path.read_text())
        if summary.get('status') != 'completed':
            raise ValueError(f'Incomplete simulation evidence: {summary_path}')
        if r.get('source_summary_sha256') and digest(summary_path) != r['source_summary_sha256']:
            raise ValueError(f'Simulation summary changed: {summary_path}')
        prep = json.loads((summary_path.parent/'prepare_summary.json').read_text())
        plan = prep['prepare_plan']
        params = plan['parameters']
        for key, value in expected['parameters'].items():
            if key not in ignored and params.get(key) != value:
                raise ValueError(f'Observation physical/sampling context differs: {key}')
        if params['temperature_K'] != r['temperature_K'] or params['mixture'] != r['mixture']:
            raise ValueError('Observation temperature/composition differs from simulation evidence.')
        if asset_fingerprints(plan['inputs']) != asset_fingerprints(expected['inputs']):
            raise ValueError('Observation input assets differ from frozen model.')
        for filename, checksum in prep.get('result', {}).get('files', {}).get('source_sha256', {}).items():
            if not Path(filename).is_file() or digest(filename) != checksum:
                raise ValueError(f'Simulation input checksum mismatch: {filename}')
        matching = [x for x in summary['results'] if math.isclose(x['pressure_bar'], r['pressure_bar'], rel_tol=1e-10)]
        if len(matching) != 1:
            raise ValueError('Observation pressure does not match simulation summary.')
        for component, values in r['components'].items():
            if not math.isclose(values['mean_molecules'], matching[0]['summary']['components'][component]['mean_molecules'], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError('Observation uptake differs from simulation summary.')
        factor = 4 ** r.get('precision_retries', 0)
        for key in ('production_steps', 'equilibration_steps'):
            if params[key] != expected['parameters'][key] * factor:
                raise ValueError(f'Observation sampling length differs: {key}')
        runs = [x for x in summary['replicate_results'] if math.isclose(x['pressure_bar'], r['pressure_bar'], rel_tol=1e-10)]
        if sorted(x['seed'] for x in runs) != sorted(r['seeds']) or len(runs) != r['replicate_count']:
            raise ValueError('Completed seed evidence does not match observation.')
        hashes = {x['log_file']: x['sha256'] for x in r.get('raw_diagnostics', {}).get('logs', [])}
        for run in runs:
            filename = run['log_file']
            if filename not in hashes or digest(filename) != hashes[filename]:
                raise ValueError(f'Raw log checksum mismatch or missing: {filename}')
            rows = parse_lammps_log(get_log(Path(filename)))['rows']
            if not any(x.get('Step') == params['run_steps'] for x in rows):
                raise ValueError(f'Incomplete seed log: {filename}')
    return manifest


def refresh_evidence_checksums(evidence):
    """Refresh relocated input hashes; byte-exact originals stay in provenance."""
    for filename in Path(evidence).rglob('*.json'):
        data = json.loads(filename.read_text())
        def refresh(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in ('source_sha256', 'rendered_sha256') and isinstance(child, dict):
                        for path in child:
                            if Path(path).is_file():
                                child[path] = digest(path)
                    else:
                        refresh(child)
            elif isinstance(value, list):
                for child in value:
                    refresh(child)
        original = deepcopy(data)
        refresh(data)
        if data != original:
            _save(filename, data)


def prepare_extension(root, source_path, lower, upper):
    from modules.module_b_cluster import freeze_config, archive_prepared_manifest
    from pipeline.planning import build_run_plan
    from pipeline.prepare import prepare_benchmark
    from pipeline.materialization import materialize_benchmark

    source_path = Path(source_path)
    folder = Path(root)/'production'/'mof5'
    target = folder/'campaign.json'
    if source_path.resolve() == target.resolve():
        raise ValueError('Source and destination must be separate campaigns.')
    validate_bounds(lower, upper)
    settings = {'source': str(source_path.resolve()), 'pressure_min': lower, 'pressure_max': upper}
    with campaign_lock(target):
        if target.exists():
            existing = json.loads(target.read_text())
            if existing.get('extension', {}).get('settings') != settings:
                raise ValueError('Existing campaign preparation settings differ; choose a separate root.')
            if al.identity(existing['base_config']) != existing['model_identity']:
                raise ValueError('Prepared destination physical model changed.')
            return target
        with campaign_lock(source_path):
            source = json.loads(source_path.read_text())
            verify_source(source, source_path.parent)
            candidates, initial, holdout = extension_design(source, lower, upper)
            archive = folder/'provenance'/'source'
            if folder.resolve().is_relative_to(source_path.parent.resolve()) or source_path.parent.resolve().is_relative_to(folder.resolve()):
                raise ValueError('Source and destination directories must not overlap.')
            # Copy evidence without expensive trajectories/restarts. Preserve source bytes and
            # checksums in this archive; mapped observation paths refer to these copies.
            shutil.copytree(source_path.parent, archive, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('dumps', 'restarts', '.writer.lock'))
            source_checksum = digest(source_path)
            evidence = folder/'evidence'
            shutil.copytree(archive, evidence, dirs_exist_ok=True, ignore=shutil.ignore_patterns('report'))
            prefix = str(source_path.parent)
            absolute_prefix = str(source_path.parent.resolve())
            def relocate(value):
                if isinstance(value, dict):
                    return {relocate(k): relocate(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [relocate(v) for v in value]
                if isinstance(value, str):
                    for old in (absolute_prefix, prefix):
                        if value == old or value.startswith(old+'/'):
                            return str(evidence) + value[len(old):]
                return value
            for filename in evidence.rglob('*'):
                if filename.is_file() and filename.suffix == '.json':
                    original = json.loads(filename.read_text())
                    relocated = relocate(original)
                    if relocated != original:
                        _save(filename, relocated)
                elif filename.is_file() and filename.suffix == '.in':
                    text = filename.read_text()
                    for old in (absolute_prefix, prefix):
                        text = text.replace(old+'/', str(evidence)+'/')
                    filename.write_text(text)
            refresh_evidence_checksums(evidence)
            def observations(rows):
                rows = relocate(rows)
                for row in rows:
                    if row.get('source_summary_sha256'):
                        row['original_source_summary_sha256'] = row['source_summary_sha256']
                        row['source_summary_sha256'] = digest(row['source_summary'])
                return rows
            cfg = deepcopy(source['base_config'])
            cfg['conditions']['pressures_bar'] = [lower, upper]
            cfg['output'].update(directory=str(folder/'simulation'), run_id='prepared_model', resume=True, overwrite=False)
            cfg = freeze_config(cfg, folder)
            if al.identity(cfg) != source['model_identity']:
                raise ValueError('Destination model differs from source.')
            _save(folder/'config.json', cfg)
            prepared = materialize_benchmark(prepare_benchmark(build_run_plan(cfg)))
            archive_prepared_manifest(folder, prepared)
            state = deepcopy(source)
            state.update(base_config=cfg, candidates=candidates, initial_design=initial, holdout=holdout,
                         results=observations(source['results']), holdout_results=observations(source['holdout_results']),
                         history=relocate(source['history']), pending_request=None, status='ready')
            for event in state['history']:
                if 'replaced_observation' in event:
                    event['replaced_observation'] = observations([event['replaced_observation']])[0]
            state.pop('last_error', None)
            state.pop('last_error_logs', None)
            state['options'].update(max_training_points=24, max_precision_retries=1)
            state['extension'] = {'settings': settings, 'source_sha256': source_checksum,
                                  'source_archive': str(archive), 'previous_pressure_max': max(source['base_config']['conditions']['pressures_bar']),
                                  'previously_inspected_holdouts': [r['candidate_id'] for r in source['holdout_results']],
                                  'evidence': str(evidence),
                                  'note': 'Original evidence is archived byte-for-byte; working evidence paths and summary checksums are relocated.'}
            _save(folder/'readiness.json', {'status': 'prepared', 'production_started': False,
                                           'physical_validation': 'open', 'campaign_kind': 'production'})
            _save(target, state)
    return target
