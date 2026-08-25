"""
Evaluate an appending program on the held-out test instances.

Usage:
    python evaluate_test.py <program_path> [--verbose] [--workers N]

Reports median, mean, per-instance scores vs CS baseline, and win/loss counts.
Run at the END of an experiment -- test instances are never seen during evolution.

Test instances and their CS baselines are SHARED with the insertion benchmark (same absolute
paths, same .cs_baselines_test.pkl), so scores are directly comparable and the
held-out set stays held out. Evaluates instances concurrently: safe because the
per-call program budget is charged in CPU time (ITIMER_VIRTUAL) inside each
instance's own subprocess, not wall clock.
"""
import sys, os, argparse, statistics, time, pickle
from concurrent.futures import ThreadPoolExecutor

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)

from evaluator_penalize_failures import (
    parse_generated, compute_twt, PHASE1_DIR, SUBPROCESS_FALLBACK_S,
    run_program_on_instance,
)

TEST_DIR = os.path.join(PHASE1_DIR, 'instances', 'test')
TEST_CACHE_PATH = os.path.join(TEST_DIR, '.cs_baselines_test.pkl')


def load_test_instances():
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")
    return sorted(
        os.path.join(TEST_DIR, f)
        for f in os.listdir(TEST_DIR)
        if f.endswith('.txt')
    )


def load_cs_baselines_test():
    """Reuse Phase 1's cache directly -- CS baselines are a property of the
    instance and the classical portfolio, not of the harness, so nothing needs
    recomputing. Fails loudly if Phase 1 hasn't built it yet."""
    if not os.path.exists(TEST_CACHE_PATH):
        raise FileNotFoundError(
            f"No cached CS baselines at {TEST_CACHE_PATH}. "
            f"Run Phase 1's evaluate_test.py once first to build it."
        )
    with open(TEST_CACHE_PATH, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a Phase 2 program on test instances')
    parser.add_argument('program', help='Path to best_program.py')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--workers', '-w', type=int, default=6,
                        help='Concurrent instance evaluations (default 6)')
    args = parser.parse_args()

    if not os.path.exists(args.program):
        print(f"Program not found: {args.program}"); sys.exit(1)

    program_path = os.path.abspath(args.program)
    paths = load_test_instances()
    print(f"Test instances: {len(paths)}")
    print(f"Program: {program_path}")
    print(f"Workers: {args.workers}")
    print()

    cs_baselines = load_cs_baselines_test()

    def eval_one(path):
        name = os.path.basename(path)
        cs_twt = cs_baselines.get(path)
        if cs_twt is None:
            return name, None, None, None, "SKIP (no CS baseline)"
        try:
            inst = parse_generated(path)
            result = run_program_on_instance(program_path, inst,
                                             timeout_seconds=SUBPROCESS_FALLBACK_S)
            twt = compute_twt(inst, result['schedule'])
            score = (cs_twt - twt) / cs_twt if cs_twt else 0.0
            return name, score, cs_twt, twt, None
        except Exception as e:
            return name, None, None, None, str(e)

    scores = []
    beats = ties = loses = errors = 0

    if args.verbose:
        print(f"{'Instance':<22} {'CS TWT':>10} {'Our TWT':>10} {'Score':>8}  Result", flush=True)
        print("-" * 65, flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for name, score, cs_twt, twt, err in ex.map(eval_one, paths):
            if err is not None:
                if err.startswith("SKIP"):
                    print(f"{name:<22}  {err}", flush=True)
                else:
                    errors += 1
                    print(f"{name:<22}  ERROR: {err}", flush=True)
                continue
            scores.append(score)
            if score > 1e-9:   beats += 1; tag = "BEAT"
            elif score < -1e-9: loses += 1; tag = "lose"
            else:               ties  += 1; tag = "tie"
            if args.verbose:
                print(f"{name:<22} {cs_twt:>10.0f} {twt:>10.0f} {score:>8.4f}  {tag}", flush=True)

    elapsed = time.perf_counter() - t0

    print()
    print(f"{'='*45}")
    print(f"HELD-OUT TEST SET RESULTS ({len(paths)} instances, {elapsed:.1f}s)")
    print(f"{'='*45}")
    if scores:
        scores_sorted = sorted(scores)
        print(f"Median score : {statistics.median(scores):+.4f}")
        print(f"Mean score   : {sum(scores)/len(scores):+.4f}")
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
