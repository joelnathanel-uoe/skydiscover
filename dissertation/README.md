# Dissertation material

Everything this study added to SkyDiscover beyond the two benchmarks: the
programs the write-up names, the tools that produced its tables, and the bounds
data those tables are measured against.

The benchmarks themselves stay where SkyDiscover looks for them:

- [`benchmarks/math/fjsp_twt_append`](../benchmarks/math/fjsp_twt_append) — the
  appending harness, its evaluators and its run configs. Every run reported in
  the write-up was made here.
- [`benchmarks/math/fjsp_twt_insertion`](../benchmarks/math/fjsp_twt_insertion) —
  the earlier insertion harness, and the home of the instances and cached
  baselines both benchmarks share.

## Layout

```
programs/     the eleven programs the write-up names, with their scores
tools/        the tools the study was run and analysed with
tools/data/   the bounds the results are measured against
```

## The tools

| | |
|---|---|
| `compare_dashboard/` | builds a multi-run comparison dashboard from finished runs |
| `jss-instance-gen/` | the Rust generator that produced every generated instance |
| `build_type_baselines.py` | builds the combined-scheduler baseline cache for an instance directory |
| `build_rp_baselines.py` | builds the reference cache used when scoring against a program rather than the baseline |
| `extract_run.py` | writes a run's iteration log out as text or HTML |
| `extract_champion.py` | pulls the best program out of a finished run |
| `bundle_monitor_snapshot.py` | freezes a live monitor view as a standalone HTML file |
| `compare_to_academic_bounds.py` | scores programs against the published best-known values |

## The bounds data

`tools/data/best_bounds.csv` holds the best-known total weighted tardiness for
each of the 209 academic instances, from the VNSG results of Sobeyko and Mönch
(2016). `tools/data/academic_bounds_per_instance.csv` carries the same bound
alongside the baseline and each reported program, per instance, and is what
`compare_to_academic_bounds.py` reads.

## Reproducing a result

The tools that read finished runs — the dashboard, the extraction scripts —
need the run directories under `outputs/adaevolve/`, which are not in this
repository. The tools that score programs need only what is here:

```bash
# a program against the academic sets
uv run python benchmarks/math/fjsp_twt_append/evaluate_academic.py \
    dissertation/programs/07_phase2_run3_riskpricing.py --verbose

# and against the published best-known values
uv run python dissertation/tools/compare_to_academic_bounds.py \
    dissertation/programs/07_phase2_run3_riskpricing.py
```

The second reads the per-program cache the first writes, so run them in that
order. Caches are keyed by the program's absolute path, so a program scored
under one path is not found again under another.
