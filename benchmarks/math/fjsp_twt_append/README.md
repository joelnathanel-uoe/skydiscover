# FJSP-TWT — appending construction heuristics

A SkyDiscover discovery benchmark: the flexible job shop scheduling problem with
total weighted tardiness, searched over **appending** construction heuristics. A
program builds a schedule one dispatch at a time, appending the operation it
chooses to the end of a machine's sequence, started as early as it can be.

`../fjsp_twt_insertion/` is the sibling benchmark, where a program may place an
operation anywhere in a machine's sequence. Appending is the restricted case:
machine sequences cannot overlap and precedence cannot be violated, so every
schedule is feasible by construction — no cycle checks, no rollback, and no
class of infeasible-move failures.

## What a program implements

The harness generates the legal candidates and applies the one returned. The
program decides which candidates compete and which of them wins, in a single
function:

```python
def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    # Restriction: which candidates compete
    # Selection:   which of them wins
    return one_candidate
```

The split into a restriction and a selection is asked for in the prompt rather
than enforced by the interface. Every config carries its prompt inline, so the
design space a run was given can be read off the config alone.

## Scoring

Each instance carries a baseline: the lowest total weighted tardiness reached by
the combined scheduler of Sobeyko and Mönch (2016) — the classical dispatching
rules, with the apparent tardiness cost rule at seventy values of its parameter.
A program's score is the median proportional reduction on that baseline over the
evaluation set. A program that returns an illegal move, raises, or exceeds the
per-call time budget scores −100 for the whole set, so one that broke is never
mistaken for one that merely scheduled badly.

## Layout

```
harness.py                       the appending harness
seeds/                           seed programs a run starts from
configs/                         run configs; each carries its prompt inline
evaluator_penalize_failures.py   scoring on generated instances
evaluator_generalise.py          scoring over three instance types at once
evaluator_generalise_extra.py    the same, on instances generated after the runs
evaluate_academic.py             scoring on the academic sets
evaluate_test.py                 scoring on the held-out generated instances
prompts/                         unused; the configs carry the prompt inline
```

Instances and their cached baselines live in `../fjsp_twt_insertion/instances/`
and are shared rather than copied, so scores from the two benchmarks are
comparable. The generator that produced them is in
`dissertation/tools/jss-instance-gen/`.

## Run

```bash
uv run skydiscover-run \
    benchmarks/math/fjsp_twt_append/seeds/initial_program_append.py \
    benchmarks/math/fjsp_twt_append/evaluator_penalize_failures.py \
    --config benchmarks/math/fjsp_twt_append/configs/config_p2_run1.yaml \
    --search adaevolve --model <model> --iterations 200
```

`FJSP_EVAL_WORKERS=N` sets how many instances are scored in parallel. Left
unset, the evaluators use `min(8, cpus - 2)`. The per-call budget is charged in
CPU time rather than wall clock, so raising it does not penalise a program for
contention.
