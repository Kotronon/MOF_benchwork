"""Opt-in safety and raw-log diagnostics for the three fixed AL campaigns."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import fcntl
import hashlib
import json
import math

_LOCKS = ContextVar('module_b_locks', default=frozenset())

@contextmanager
def campaign_lock(path):
    key = str(Path(path).resolve())
    if key in _LOCKS.get():
        yield
        return
    target = Path(path).parent / '.writer.lock'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Campaign already has an active writer: {target}') from exc
        token = _LOCKS.set(_LOCKS.get() | {key})
        try:
            yield
        finally:
            _LOCKS.reset(token)
            fcntl.flock(handle, fcntl.LOCK_UN)


def locked(function):
    @wraps(function)
    def wrapped(path, *args, **kwargs):
        with campaign_lock(path):
            return function(path, *args, **kwargs)
    return wrapped


def asset_fingerprints(value):
    if isinstance(value, dict):
        return {k: asset_fingerprints(v) for k, v in value.items() if k != 'reference_files'}
    if isinstance(value, list):
        return [asset_fingerprints(v) for v in value]
    if isinstance(value, str) and Path(value).is_file():
        return {'sha256': hashlib.sha256(Path(value).read_bytes()).hexdigest()}
    return value


def add_raw_uncertainty(record, summary_path):
    """Conservative seed + within-seed variance; not a calibrated equilibrium test.

    Read the actual (possibly combined restart) log recorded by the runner, not
    a glob of all log segments, which would double-count restarted seeds.
    """
    from analysis.module_b_convergence_report import (
        _batch_statistics, _combined_component_interval, _combined_fieller_interval,
    )
    from parsers.lammps_log_parser import get_log, parse_lammps_log
    source = json.loads(Path(summary_path).read_text())
    prep = json.loads((Path(summary_path).parent / 'prepare_summary.json').read_text())['prepare_plan']
    params = prep['parameters']
    start, end = params['equilibration_steps'], params['run_steps']
    replicates = []
    for run in source['replicate_results']:
        if not math.isclose(run['pressure_bar'], record['pressure_bar'], rel_tol=1e-10):
            continue
        log = Path(run['log_file'])
        rows = [r for r in parse_lammps_log(get_log(log))['rows']
                if start <= r.get('Step', -1) <= end and 'v_nco2' in r and 'v_nn2' in r]
        # Combined logs can contain a repeated boundary step after a restart.
        rows = list({r['Step']: r for r in rows}.values())
        rows.sort(key=lambda r: r['Step'])
        if not rows or rows[-1]['Step'] != end:
            raise ValueError(f'Incomplete component production log: {log}')
        replicates.append({'seed': run['seed'], 'rows': rows, 'log_file': str(log),
                           'sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
    if len(replicates) < 3 or len({r['seed'] for r in replicates}) != len(replicates):
        raise ValueError('Raw-log uncertainty requires three distinct completed seeds.')
    record['classic_seed_components'] = json.loads(json.dumps(record['components']))
    record['classic_seed_selectivity'] = json.loads(json.dumps(record['selectivity']))
    record['classic_seed_precision_met'] = record['precision_met']
    means, diagnostics = {}, {}
    component_ok = []
    for c in ('CO2', 'N2'):
        stats = [_batch_statistics([r[f'v_n{c.lower()}'] for r in rep['rows']]) for rep in replicates]
        means[c] = [s['mean'] for s in stats]
        interval = _combined_component_interval(means[c], stats)
        se = interval.get('standard_error_combined')
        if se is None or not math.isfinite(se):
            raise ValueError(f'Missing batch uncertainty for {c}; cannot train with zero noise.')
        relative = interval.get('relative_ci95_half_width')
        absent = record['mixture'][c] == 0
        # A constant nonzero-fugacity trace does not establish zero MC error.
        constant = all(s['standard_error_autocorrelation'] == 0 for s in stats)
        floor = 0.05 if constant and not absent else 0.0
        ok = absent or (not constant and relative is not None and relative <= .2)
        record['components'][c].update(interval, standard_error_molecules=max(se, floor),
                                      precision_met=ok)
        diagnostics[c] = {'seed_statistics': stats, 'constant_traces': constant,
                          'noise_floor_molecules': floor,
                          'identical_seed_means': len(set(means[c])) == 1}
        component_ok.append(ok)
    fractions = record['mixture']
    if all(fractions[c] > 0 for c in fractions):
        interval = _combined_fieller_interval(replicates, means, fractions['N2']/fractions['CO2'])
        relative = interval.get('relative_ci95_half_width')
        selectivity_ok = interval['ci95_status'] == 'bounded' and relative is not None and relative <= .2
        record['selectivity']['CO2/N2'].update(interval, precision_met=selectivity_ok)
    else:
        selectivity_ok = True  # No defined pair at a pure-gas boundary.
        record['selectivity']['CO2/N2'].update(value=None, ci95_status='undefined_gas_fraction', precision_met=False)
    record['precision_met'] = all(component_ok) and selectivity_ok
    record['raw_diagnostics'] = {
        'components': diagnostics,
        'logs': [{k:v for k,v in r.items() if k != 'rows'} for r in replicates],
        'interpretation': 'Conservative sum of between-seed and batch variances; may double-count sampling noise. Statistical precision is not proof of equilibrium.',
    }
    return record
