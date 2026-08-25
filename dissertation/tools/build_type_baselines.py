"""Build the CS baseline cache for each of the three type instance directories.

Type A is a KEY REWRITE, not a recomputation: type_a_search holds files
byte-identical to generated/000-049, so its baselines are copied from the
existing cache with the path keys rewritten. That keeps every number already
reported provably the same value rather than a recomputed one.

Type B and Type C are built from scratch with baselines.cs_solve, the same
best-of-78-classical-rules baseline used everywhere else.

Each directory gets its own .cs_baselines.pkl, keyed by absolute instance path.
"""
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = '/root/skydiscover'
PHASE1 = os.path.join(ROOT, 'benchmarks', 'math', 'fjsp_twt')
INST = os.path.join(PHASE1, 'instances')
sys.path.insert(0, PHASE1)
sys.path.insert(0, os.path.join(ROOT, 'benchmarks', 'math', 'fjsp_twt_append'))

from evaluator_penalize_failures import parse_generated, compute_twt  # noqa: E402


def cache_path(d):
    return os.path.join(INST, d, '.cs_baselines.pkl')


def files_in(d):
    p = os.path.join(INST, d)
    return sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith('.txt'))


def rewrite(dest_dir, src_cache, src_dir):
    """Copy baselines for dest_dir's files out of src_cache, rekeyed to dest_dir."""
    with open(src_cache, 'rb') as f:
        src = pickle.load(f)
    out = {}
    for path in files_in(dest_dir):
        old = os.path.join(INST, src_dir, os.path.basename(path))
        if old not in src:
            raise KeyError(f'{old} absent from {src_cache}')
        out[path] = src[old]
    with open(cache_path(dest_dir), 'wb') as f:
        pickle.dump(out, f)
    print(f'{dest_dir:16s} rewrote {len(out)} keys from {os.path.basename(src_cache)}',
          flush=True)


def _solve(path):
    from baselines import cs_solve
    inst = parse_generated(path)
    return path, compute_twt(inst, cs_solve(inst))


def build(d, workers):
    paths = files_in(d)
    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (path, twt) in enumerate(ex.map(_solve, paths), 1):
            out[path] = twt
            if i % 10 == 0:
                print(f'  {d}: {i}/{len(paths)}  {time.time()-t0:.0f}s', flush=True)
    with open(cache_path(d), 'wb') as f:
        pickle.dump(out, f)
    print(f'{d:16s} built {len(out)} baselines in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 14

    rewrite('type_a_search', os.path.join(PHASE1, '.cs_baselines.pkl'), 'generated')

    for d in ('type_b_search', 'type_c_search'):
        build(d, workers)
