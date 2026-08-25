"""
Evaluate a program on the academic instance sets (Set_1..Set_4, .fjs format,
Sobeyko & Mönch 2016 style: n_jobs/n_machines header, then per-job operations
followed by inline release_time/due_date/weight).

Usage:
    python evaluate_academic.py <program_path> [--sets Set_1,Set_2,...] [--verbose]

Mirrors evaluate_test.py (same CS-baseline scoring), but parses instances with
evaluator.parse_instance (the academic instances' inline format) instead of parse_generated
(the generated-instance DUE_DATES:/WEIGHTS: footer format), and reads from
instances/Set_1..Set_4 instead of instances/test.
"""
import sys, os, argparse, statistics, time, pickle, json, hashlib

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)

from evaluator import (
    parse_instance, compute_twt, validate_schedule,
    run_program_on_instance, SUBPROCESS_FALLBACK_S,
)

ALL_SETS = ["Set_1", "Set_2", "Set_3", "Set_4"]


def load_set_instances(set_names):
    paths = []
    for set_name in set_names:
        folder = os.path.join(BENCH, "instances", set_name)
        if not os.path.isdir(folder):
            print(f"Warning: {folder} not found, skipping")
            continue
        for fname in sorted(f for f in os.listdir(folder) if f.endswith(".fjs")):
            paths.append((set_name, os.path.join(folder, fname)))
    return paths


def compute_cs_baselines_academic(entries):
    """CS baselines for the academic instances, cached per-set (separate from generated/test caches)."""
    from baselines import cs_solve

    cache_path = os.path.join(BENCH, "instances", ".cs_baselines_academic.pkl")
    cached = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)

    baselines = {}
    to_compute = [(s, p) for s, p in entries if p not in cached]
    if to_compute:
        print(f"Computing CS baselines for {len(to_compute)} instances (not cached)...", flush=True)
        for i, (set_name, path) in enumerate(to_compute):
            inst = parse_instance(path)
            sched = cs_solve(inst)
            cached[path] = compute_twt(inst, sched)
            print(f"  {i+1}/{len(to_compute)}: {set_name}/{os.path.basename(path)} CS={cached[path]:.1f}", flush=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cached, f)

    for set_name, path in entries:
        baselines[path] = cached[path]
    return baselines


def _results_cache_path(program_path):
    """Per-program incremental results cache, so an interrupted run (killed,
    timed out, crashed) never loses completed instance evaluations -- a
    rerun of the same program resumes instead of restarting from scratch."""
    program_hash = hashlib.sha256(os.path.abspath(program_path).encode()).hexdigest()[:12]
    return os.path.join(BENCH, "instances", f".academic_results_{program_hash}.json")


def _load_results_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_result(cache_path, cache, instance_path, record):
    cache[instance_path] = record
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, cache_path)  # atomic -- never leaves a half-written cache file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="Path to best_program.py")
    parser.add_argument("--sets", default=",".join(ALL_SETS), help="Comma-separated set names")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.program):
        print(f"Program not found: {args.program}")
        sys.exit(1)

    set_names = args.sets.split(",")
    entries = load_set_instances(set_names)
    print(f"Instances: {len(entries)} across {set_names}", flush=True)
    print(f"Program: {args.program}", flush=True)

    results_cache_path = _results_cache_path(args.program)
    results_cache = _load_results_cache(results_cache_path)
    n_cached = sum(1 for _, p in entries if p in results_cache)
    if n_cached:
        print(f"Resuming: {n_cached}/{len(entries)} instances already evaluated (cached at {results_cache_path})", flush=True)
    print(flush=True)

    cs_baselines = compute_cs_baselines_academic(entries)

    per_set = {s: [] for s in set_names}
    beats = ties = loses = errors = 0

    if args.verbose:
        print(f"{'Set':<8} {'Instance':<32} {'CS TWT':>10} {'Our TWT':>10} {'Score':>8}  Result", flush=True)
        print("-" * 80, flush=True)

    t0 = time.perf_counter()
    for set_name, path in entries:
        name = os.path.basename(path)
        cs_twt = cs_baselines.get(path)
        if not cs_twt:
            print(f"{set_name:<8} {name:<32}  SKIP (no/zero CS baseline)", flush=True)
            continue

        cached = results_cache.get(path)
        if cached is not None:
            if "error" in cached:
                errors += 1
                if args.verbose:
                    print(f"{set_name:<8} {name:<32}  ERROR (cached): {cached['error']}", flush=True)
                continue
            score = cached["score"]
            per_set[set_name].append(score)
            if score > 1e-9: beats += 1; tag = "BEAT"
            elif score < -1e-9: loses += 1; tag = "lose"
            else: ties += 1; tag = "tie"
            if args.verbose:
                print(f"{set_name:<8} {name:<32} {cs_twt:>10.1f} {cached['twt']:>10.1f} {score:>8.4f}  {tag} (cached)", flush=True)
            continue

        try:
            inst = parse_instance(path)
            result = run_program_on_instance(args.program, inst, timeout_seconds=SUBPROCESS_FALLBACK_S)
            twt = compute_twt(inst, result["schedule"])
            score = (cs_twt - twt) / cs_twt
            per_set[set_name].append(score)
            _save_result(results_cache_path, results_cache, path, {"score": score, "twt": twt})
            if score > 1e-9:
                beats += 1; tag = "BEAT"
            elif score < -1e-9:
                loses += 1; tag = "lose"
            else:
                ties += 1; tag = "tie"
            if args.verbose:
                print(f"{set_name:<8} {name:<32} {cs_twt:>10.1f} {twt:>10.1f} {score:>8.4f}  {tag}", flush=True)
        except Exception as e:
            errors += 1
            _save_result(results_cache_path, results_cache, path, {"error": str(e)})
            print(f"{set_name:<8} {name:<32}  ERROR: {e}", flush=True)

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
