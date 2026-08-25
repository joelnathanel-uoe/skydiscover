"""
Evaluate an appending program on the academic instance sets
(Set_1..Set_4, .fjs format, Sobeyko & Moench 2016 style).

Usage:
    python evaluate_academic.py <program_path> [--sets Set_1,Set_2,...] [--verbose] [--workers N]

Mirrors the insertion benchmark's evaluate_academic.py (same CS-baseline scoring, same per-program
results cache so an interrupted run resumes instead of restarting), but drives
the append-only harness (choose_next) and reuses the insertion benchmark's academic CS-baseline
cache directly, since baselines depend only on the instance and the classical
portfolio, not on which harness scored the program.

Evaluates instances concurrently: safe because the per-call program budget is
charged in CPU time (ITIMER_VIRTUAL) inside each instance's own subprocess,
not wall clock -- same reasoning as the main evaluator's FJSP_EVAL_WORKERS.
"""
import sys, os, argparse, statistics, time, pickle, json, hashlib
from concurrent.futures import ThreadPoolExecutor

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)

from evaluator_penalize_failures import (
    parse_instance, compute_twt, PHASE1_DIR, SUBPROCESS_FALLBACK_S,
    run_program_on_instance,
)

ALL_SETS = ["Set_1", "Set_2", "Set_3", "Set_4"]
ACADEMIC_CACHE_PATH = os.path.join(PHASE1_DIR, "instances", ".cs_baselines_academic.pkl")

# This benchmark's own results cache directory, kept separate from the insertion one so the
# two phases' per-program caches (keyed by program path hash) never collide.
RESULTS_DIR = os.path.join(BENCH, "instances")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_set_instances(set_names):
    paths = []
    for set_name in set_names:
        folder = os.path.join(PHASE1_DIR, "instances", set_name)
        if not os.path.isdir(folder):
            print(f"Warning: {folder} not found, skipping")
            continue
        for fname in sorted(f for f in os.listdir(folder) if f.endswith(".fjs")):
            paths.append((set_name, os.path.join(folder, fname)))
    return paths


def load_cs_baselines_academic():
    if not os.path.exists(ACADEMIC_CACHE_PATH):
        raise FileNotFoundError(
            f"No cached CS baselines at {ACADEMIC_CACHE_PATH}. "
            f"Run the insertion benchmark's evaluate_academic.py once first to build it."
        )
    with open(ACADEMIC_CACHE_PATH, "rb") as f:
        return pickle.load(f)


def _results_cache_path(program_path):
    program_hash = hashlib.sha256(os.path.abspath(program_path).encode()).hexdigest()[:12]
    return os.path.join(RESULTS_DIR, f".academic_results_p2_{program_hash}.json")


def _load_results_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


_save_lock = None  # set in main() to a threading.Lock


def _save_result(cache_path, cache, instance_path, record):
    cache[instance_path] = record
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, cache_path)


def main():
    import threading
    global _save_lock
    _save_lock = threading.Lock()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="Path to best_program.py")
    parser.add_argument("--sets", default=",".join(ALL_SETS), help="Comma-separated set names")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--workers", "-w", type=int, default=6,
                        help="Concurrent instance evaluations (default 6)")
    args = parser.parse_args()

    if not os.path.exists(args.program):
        print(f"Program not found: {args.program}")
        sys.exit(1)

    program_path = os.path.abspath(args.program)
    set_names = args.sets.split(",")
    entries = load_set_instances(set_names)
    print(f"Instances: {len(entries)} across {set_names}", flush=True)
    print(f"Program: {program_path}", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    results_cache_path = _results_cache_path(program_path)
    results_cache = _load_results_cache(results_cache_path)
    n_cached = sum(1 for _, p in entries if p in results_cache)
    if n_cached:
        print(f"Resuming: {n_cached}/{len(entries)} instances already evaluated "
              f"(cached at {results_cache_path})", flush=True)
    print(flush=True)

    cs_baselines = load_cs_baselines_academic()

    per_set = {s: [] for s in set_names}
    beats = ties = loses = errors = 0

    if args.verbose:
        print(f"{'Set':<8} {'Instance':<32} {'CS TWT':>10} {'Our TWT':>10} {'Score':>8}  Result", flush=True)
        print("-" * 80, flush=True)

    def eval_one(entry):
        set_name, path = entry
        name = os.path.basename(path)
        cs_twt = cs_baselines.get(path)
        if not cs_twt:
            return set_name, name, path, None, None, None, "SKIP (no/zero CS baseline)", False

        cached = results_cache.get(path)
        if cached is not None:
            if "error" in cached:
                return set_name, name, path, None, None, None, cached["error"], True
            return set_name, name, path, cached["score"], cs_twt, cached["twt"], None, True

        try:
            inst = parse_instance(path)
            result = run_program_on_instance(program_path, inst, timeout_seconds=SUBPROCESS_FALLBACK_S)
            twt = compute_twt(inst, result["schedule"])
            score = (cs_twt - twt) / cs_twt
            with _save_lock:
                _save_result(results_cache_path, results_cache, path, {"score": score, "twt": twt})
            return set_name, name, path, score, cs_twt, twt, None, False
        except Exception as e:
            with _save_lock:
                _save_result(results_cache_path, results_cache, path, {"error": str(e)})
            return set_name, name, path, None, None, None, str(e), False

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for set_name, name, path, score, cs_twt, twt, err, was_cached in ex.map(eval_one, entries):
            if err is not None:
                if err.startswith("SKIP"):
                    if args.verbose:
                        print(f"{set_name:<8} {name:<32}  {err}", flush=True)
                else:
                    errors += 1
                    if args.verbose:
                        tag = "(cached)" if was_cached else ""
                        print(f"{set_name:<8} {name:<32}  ERROR {tag}: {err}", flush=True)
                continue
            per_set[set_name].append(score)
            if score > 1e-9: beats += 1; tag = "BEAT"
            elif score < -1e-9: loses += 1; tag = "lose"
            else: ties += 1; tag = "tie"
            if args.verbose:
                cachetag = " (cached)" if was_cached else ""
                print(f"{set_name:<8} {name:<32} {cs_twt:>10.1f} {twt:>10.1f} {score:>8.4f}  {tag}{cachetag}", flush=True)

    elapsed = time.perf_counter() - t0
    all_scores = [s for scores in per_set.values() for s in scores]

    print(flush=True)
    print("=" * 60, flush=True)
    print(f"ACADEMIC INSTANCE SET RESULTS ({len(entries)} instances, {elapsed:.1f}s this run)", flush=True)
    print("=" * 60, flush=True)
    for set_name in set_names:
        scores = per_set[set_name]
        if scores:
            print(f"{set_name}: n={len(scores):<4} median={statistics.median(scores):+.4f}  "
                  f"mean={sum(scores)/len(scores):+.4f}  "
                  f"beats={sum(1 for s in scores if s > 1e-9)}/{len(scores)}", flush=True)
        else:
            print(f"{set_name}: no scored instances", flush=True)
    print("-" * 60, flush=True)
    if all_scores:
        print(f"OVERALL: n={len(all_scores)}  median={statistics.median(all_scores):+.4f}  "
              f"mean={sum(all_scores)/len(all_scores):+.4f}  "
              f"beats CS={beats}/{len(all_scores)}  ties={ties}  loses={loses}", flush=True)
    if errors:
        print(f"Errors: {errors}", flush=True)


if __name__ == "__main__":
    main()
