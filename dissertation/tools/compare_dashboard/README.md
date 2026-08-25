# compare_dashboard

Builds one HTML page that compares several finished AdaEvolve runs side by side.
A run's own monitor shows one run as it goes; this shows a batch of them after
the fact, which is what a comparison between search settings needs.

## Run

```bash
uv run python dissertation/tools/compare_dashboard/build_comparison.py \
    outputs/adaevolve/<run_a> outputs/adaevolve/<run_b> --output comparison.html
```

`--list` prints the run directories it can see, newest first. `--experiment
NAME` builds a named comparison instead, taking its run list and its written
brief from `experiments.json`; `--list-experiments` prints those names. With
neither, it falls back to a fixed batch of run directories that must already
exist under `outputs/adaevolve/`.

Run directories are not in this repository. The page is built from a run's log,
its JSONL record and its checkpoint directory.

## What the page holds

Seven tabs: **Overview** (settings and headline scores per run), **Scores**
(the trajectory of each run, and of each island within it), **Programs** (every
archived program, with its code and its lineage), **Islands** (membership and
migration), **Paradigm** (when the paradigm mechanism fired and what it did),
**Diversity** (distance between programs, within an island and between
islands), and **Errors** (what failed, and how often).

Programs can be gathered by island, by score, by a substring of their code, or
by the migration that carried them, and exported together.

## Caching

Per-run bundles are cached under `run_summaries/.cache/`, keyed by a
fingerprint of the run's log, its JSONL file and its checkpoint listing, so an
unchanged run is not parsed twice. `--no-cache` forces a rebuild.

## experiments.json

Titles, questions, design notes and caveats only — prose. Every number in the
brief is computed from the run bundles when the page is built, so nothing in
the file can go stale against the runs it describes.
