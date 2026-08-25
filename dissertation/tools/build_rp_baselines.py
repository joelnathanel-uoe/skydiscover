"""Build the RiskPricing reference cache for each type/split instance directory.

Phase 3 scores against RiskPricing (the Phase 2 p2D champion) rather than
against CS. The reason is that CS is a set of dispatching rules whose relative
quality degrades as due dates loosen -- Sobeyko and Monch's own Table 7 shows
delta over best-SDR roughly doubling from g=0.25 to g=0.75 for every method
they test. Scoring the three types against CS therefore inflates the loose
types for free, and a min over the three would pin to the moderate type
permanently. Measuring every type against the same fixed program removes that:
the seed scores exactly 0.0 everywhere by construction.

Writes .rp_baselines.pkl next to the existing .cs_baselines.pkl in each
directory, keyed by absolute instance path.
"""
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = '/root/skydiscover'
APPEND = os.path.join(ROOT, 'benchmarks', 'math', 'fjsp_twt_append')
INST = os.path.join(ROOT, 'benchmarks', 'math', 'fjsp_twt', 'instances')
sys.path.insert(0, APPEND)

from evaluator_generalise import (  # noqa: E402
    parse_generated, compute_twt, run_program_on_instance,
    validate_schedule, SUBPROCESS_FALLBACK_S, MAX_WORKERS,
)

REFERENCE = os.path.normpath(os.path.join(
    ROOT, 'dissertation', 'programs', '07_phase2_run3_riskpricing.py'))

DIRS = os.environ.get('RP_DIRS', '').split(',') if os.environ.get('RP_DIRS') else [
    'type_a_search', 'type_b_search', 'type_c_search',
    'type_a_extra', 'type_b_extra', 'type_c_extra']


def twt_for(path):
    inst = parse_generated(path)
    result = run_program_on_instance(REFERENCE, inst,
                                     timeout_seconds=SUBPROCESS_FALLBACK_S)
    schedule = result['schedule']
    valid, msg = validate_schedule(inst, schedule)
    if not valid:
        raise RuntimeError(f'{os.path.basename(path)}: {msg}')
    return path, compute_twt(inst, schedule)


if __name__ == '__main__':
    print(f'reference: {REFERENCE}', flush=True)
    for d in DIRS:
        dd = os.path.join(INST, d)
        paths = sorted(os.path.join(dd, f) for f in os.listdir(dd)
                       if f.endswith('.txt'))
        t0 = time.time()
        out = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for path, twt in ex.map(twt_for, paths):
                out[path] = twt
        vals = sorted(out.values())
        with open(os.path.join(dd, '.rp_baselines.pkl'), 'wb') as f:
            pickle.dump(out, f)
        print(f'{d:16s} n={len(out):3d}  min={vals[0]:12,.0f}  '
              f'median={vals[len(vals)//2]:12,.0f}  max={vals[-1]:12,.0f}  '
              f'({time.time()-t0:.0f}s)', flush=True)
