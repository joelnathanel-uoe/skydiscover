# FJSP-TWT — inserting construction heuristics

A SkyDiscover discovery benchmark for the flexible job shop scheduling problem
with total weighted tardiness, in which a program may place an operation
anywhere in a machine's sequence rather than only at the end.

The program implements one function, returning the operation, the machine it
goes on, and where in that machine's sequence it is placed:

```python
def schedule_next(unscheduled, machine_sequences, start_times, end_times,
                  job_completion, instance):
    return op, machine, position
```

Because a placement can be made anywhere, the harness maintains a DAG of the
committed schedule so it can reject an insertion that would create a cycle,
which brings a class of failure the appending benchmark does not have:
infeasible moves, deadlock, and the cost of the checks themselves.

`../fjsp_twt_append/` is the restricted sibling, where a program appends to the
end of a machine's sequence and every schedule is feasible by construction.

## Instances

This directory holds the instances and cached baselines **both** benchmarks
use:

```
instances/generated/                     the generated instances runs are scored on
instances/test/                          held-out generated instances
instances/type_{a,b,c}_search/           50 each, three instance types
instances/type_{a,b,c}_heldout/          50 each, unseen during search
instances/type_{a,b,c}_extra/            200 each, generated after the runs
instances/Set_1 .. Set_4/                the academic sets of Sobeyko and Mönch (2016)
.cs_baselines.pkl                        cached combined-scheduler baselines
```

The generator that produced the generated sets is in
[`dissertation/tools/jss-instance-gen`](../../../dissertation/tools/jss-instance-gen).

## Run

```bash
uv run skydiscover-run \
    benchmarks/math/fjsp_twt_insertion/initial_program.py \
    benchmarks/math/fjsp_twt_insertion/evaluator_penalize_failures.py \
    --config benchmarks/math/fjsp_twt_insertion/config_p1_A_1island.yaml \
    --search adaevolve --model <model> --iterations 200
```

`FJSP_EVAL_WORKERS=N` sets how many instances are scored in parallel; unset,
the evaluators use `min(8, cpus - 2)`.
