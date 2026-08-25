#!/usr/bin/env python3
"""Compare evolved programs against the published state-of-the-art bounds.

Joins three sources per instance:
  * UB   -- VNSG upper bound from 'Best bounds.xlsx' (the published SOTA table)
  * CS   -- our dispatching-rule portfolio reference
  * prog -- total weighted tardiness achieved by each evolved program,
            read from the per-program cache written by evaluate_academic.py

Usage:
    python dissertation/tools/compare_to_academic_bounds.py PROGRAM.py [PROGRAM2.py ...]
                                            [--labels A,B] [--csv OUT.csv]
"""
import argparse
import hashlib
import json
import os
import statistics
import sys

BENCH = "/root/skydiscover/benchmarks/math/fjsp_twt_insertion"
BENCH_P2 = "/root/skydiscover/benchmarks/math/fjsp_twt_append"
DISS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def cache_for(program_path):
    """Phase 1's evaluate_academic.py caches at BENCH/instances/.academic_results_<h>.json;
    Phase 2's at BENCH_P2/instances/.academic_results_p2_<h>.json. Check both."""
    h = hashlib.sha256(os.path.abspath(program_path).encode()).hexdigest()[:12]
    candidates = [
        os.path.join(BENCH, "instances", f".academic_results_{h}.json"),
        os.path.join(BENCH_P2, "instances", f".academic_results_p2_{h}.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f), p
    return None, candidates[-1]


def load_reference():
    """UB and CS per instance, from the per-instance bounds CSV (it carries both)."""
    ref = {}
    path = os.path.join(DISS, "academic_bounds_per_instance.csv")
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
        icol, ucol = header.index("Instance"), header.index("UB")
        ccol = header.index("CS") if "CS" in header else None
        lcol = header.index("LLM") if "LLM" in header else None
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= ucol:
                continue
            key = parts[icol]
            try:
                ref[key] = {
                    "ub": float(parts[ucol]),
                    "cs": float(parts[ccol]) if ccol is not None and parts[ccol] else None,
                    "llm": float(parts[lcol]) if lcol is not None and parts[lcol] else None,
                }
            except ValueError:
                continue
    return ref


def norm(instance_path):
    """Cache keys are absolute paths; the reference uses 'Set_1/WT1.fjs'."""
    parts = instance_path.replace("\\", "/").split("/")
    return "/".join(parts[-2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("programs", nargs="+")
    ap.add_argument("--labels", default="")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else \
        [f"P{i+1}" for i in range(len(args.programs))]

    ref = load_reference()
    print(f"reference rows: {len(ref)}")

    results = {}
    for label, prog in zip(labels, args.programs):
        cache, path = cache_for(prog)
        if cache is None:
            print(f"  {label}: NO CACHE yet at {path} (evaluate_academic.py not finished)")
            continue
        results[label] = {norm(k): v for k, v in cache.items()}
        print(f"  {label}: {len(results[label])} instances cached")

    if not results:
        sys.exit("nothing to compare yet")

    rows = []
    for key, r in sorted(ref.items()):
        row = {"instance": key, "ub": r["ub"], "cs": r["cs"], "llm_gpt55": r["llm"]}
        for label, res in results.items():
            rec = res.get(key)
            row[label] = rec.get("twt") if rec else None
        rows.append(row)

    def gap(twt, ub):
        if twt is None or ub is None:
            return None
        if ub == 0:
            return None
        return (twt - ub) / ub

    print()
    hdr = f"{'set':8} {'n':>4} " + " ".join(f"{c:>16}" for c in
          ["median gap UB"] + [f"{l} med gap UB" for l in results])
    print(hdr); print("-" * len(hdr))
    for setname in ["Set_1", "Set_2", "Set_3", "Set_4"]:
        sub = [r for r in rows if r["instance"].startswith(setname)]
        if not sub:
            continue
        cs_gaps = [g for g in (gap(r["cs"], r["ub"]) for r in sub) if g is not None]
        line = f"{setname:8} {len(sub):>4} {statistics.median(cs_gaps) if cs_gaps else float('nan'):>16.3f}"
        for label in results:
            gs = [g for g in (gap(r[label], r["ub"]) for r in sub) if g is not None]
            line += f" {statistics.median(gs) if gs else float('nan'):>16.3f}"
        print(line)

    print("\n(gap = (TWT - UB) / UB; 0 means matching the VNSG bound, lower is better)")

    beats = {l: sum(1 for r in rows if r[l] is not None and r["ub"] is not None
                    and r[l] <= r["ub"]) for l in results}
    for l, n in beats.items():
        print(f"  {l}: matches or beats the VNSG bound on {n} of {len(rows)} instances")

    if args.csv:
        cols = ["instance", "ub", "cs", "llm_gpt55"] + list(results)
        with open(args.csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
