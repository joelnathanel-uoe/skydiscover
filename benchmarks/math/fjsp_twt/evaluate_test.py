"""
Evaluate a program on the held-out test instances.

Usage:
    python evaluate_test.py <program_path> [--verbose]

Reports median, mean, per-instance scores vs CS baseline, and win/loss counts.
Run at the END of an experiment — test instances are never seen during evolution.
"""
import sys, os, argparse, statistics, importlib.util, copy, time

BENCH    = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BENCH, 'instances', 'test')
sys.path.insert(0, BENCH)

from evaluator import (
    parse_generated, compute_twt, validate_schedule,
    _get_cs_baselines, HARNESS_PATH, run_program_on_instance,
    SUBPROCESS_FALLBACK_S,
)

def load_test_instances():
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")
    return sorted(
        os.path.join(TEST_DIR, f)
        for f in os.listdir(TEST_DIR)
        if f.endswith('.txt')
    )

def compute_cs_baselines_test(paths):
    """Compute CS baselines for test instances (cached separately from train)."""
    import pickle
    from baselines import cs_solve

    cache_path = os.path.join(TEST_DIR, '.cs_baselines_test.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if all(p in cached for p in paths):
            return cached

    print("Computing CS baselines for test instances...", flush=True)
    baselines = {}
    for i, path in enumerate(paths):
        inst = parse_generated(path)
        sched = cs_solve(inst)
        baselines[path] = compute_twt(inst, sched)
        print(f"  {i+1}/{len(paths)}: {os.path.basename(path)} CS={baselines[path]:.0f}", flush=True)

    with open(cache_path, 'wb') as f:
        pickle.dump(baselines, f)
    return baselines

def main():
    parser = argparse.ArgumentParser(description='Evaluate program on test instances')
    parser.add_argument('program', help='Path to best_program.py')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.program):
        print(f"Program not found: {args.program}"); sys.exit(1)

    paths = load_test_instances()
    print(f"Test instances: {len(paths)}")
    print(f"Program: {args.program}")
    print()

    cs_baselines = compute_cs_baselines_test(paths)

    scores = []
    beats = ties = loses = errors = 0

    if args.verbose:
        print(f"{'Instance':<22} {'CS TWT':>10} {'Our TWT':>10} {'Score':>8}  Result")
        print("-" * 65)

    t0 = time.perf_counter()
    for path in paths:
        name   = os.path.basename(path)
        cs_twt = cs_baselines.get(path)
        if cs_twt is None:
            print(f"{name:<22}  SKIP (no CS baseline)")
            continue
        try:
            result = run_program_on_instance(args.program, parse_generated(path),
                                             timeout_seconds=SUBPROCESS_FALLBACK_S)
            inst   = parse_generated(path)
            twt    = compute_twt(inst, result['schedule'])
            score  = (cs_twt - twt) / cs_twt if cs_twt else 0.0
            scores.append(score)
            if score > 1e-9:   beats += 1; tag = "BEAT"
            elif score < -1e-9: loses += 1; tag = "lose"
            else:               ties  += 1; tag = "tie"
            if args.verbose:
                print(f"{name:<22} {cs_twt:>10.0f} {twt:>10.0f} {score:>8.4f}  {tag}")
        except Exception as e:
            errors += 1
            print(f"{name:<22}  ERROR: {e}")

    elapsed = time.perf_counter() - t0

    print()
    print(f"{'='*45}")
    print(f"TEST SET RESULTS ({len(paths)} instances, {elapsed:.1f}s)")
    print(f"{'='*45}")
    if scores:
        print(f"Median score : {statistics.median(scores):+.4f}")
        print(f"Mean score   : {sum(scores)/len(scores):+.4f}")
        scores_sorted = sorted(scores)
        print(f"Q10          : {scores_sorted[int(len(scores)*0.10)]:+.4f}")
        print(f"Q25          : {scores_sorted[int(len(scores)*0.25)]:+.4f}")
        print(f"Q75          : {scores_sorted[int(len(scores)*0.75)]:+.4f}")
        print(f"Beats CS     : {beats}/{len(scores)}")
        print(f"Ties CS      : {ties}/{len(scores)}")
        print(f"Loses to CS  : {loses}/{len(scores)}")
    if errors:
        print(f"Errors       : {errors}")

if __name__ == '__main__':
    main()
