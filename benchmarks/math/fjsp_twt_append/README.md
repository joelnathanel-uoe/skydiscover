# FJSP-TWT — appending construction heuristics (Phase 2)

Discovery benchmark for the second experimental stage: evolving **appending**
construction heuristics for the flexible job shop with total weighted tardiness.

Phase 1 lives in `../fjsp_twt/` and is deliberately left untouched, so its runs
stay reproducible. This directory is a sibling, not a replacement.

## Scope: appending only

Karim scoped Stage 2 to appending schemes, excluding the more general inserting
scheme (too complex for the timeline). The Phase 1 trajectory data independently
supports that scoping — in run `fjsp_twt_0714_1406`:

- every program scoring >= 0.35 used strict non-delay dispatch (earliest
  achievable start as primary key, i.e. effectively appending);
- every attempt at delaying/shifting insertion failed (infeasible moves,
  deadlock, or O(ops^3) timeouts);
- even benign non-delaying gap-backfill scored bit-identically to its parent.

So restricting to appending removes an entire failure class and loses nothing
that ever helped. The design space comes from a supervisor-authored guidelines
document (kept locally, not tracked in this repo), which frames an appending
heuristic as two choices: a **restriction rule** (who competes) and a
**selection function** (who wins). Both are reproduced in the system prompt in
`configs/config_p2_A_1island.yaml`, so the design space the search was given is
recoverable from the config alone.

## Decisions, settled 2026-07-27

**Definition of appending.** Strict append to the end of the machine's sequence,
started as early as possible — the definition in §1 of the guidelines. No
non-delaying gap insertion. Enforced structurally: the program never sees a
position argument and cannot express any other placement.

**Harness interface.** The harness owns the two steps the guidelines fix
(candidate generation, update) and the program implements the two that are free
(restriction, selection) inside a **single** function:

    def choose_next(candidates, machine_sequences, machine_free, job_ready,
                    job_next_op, instance):
        # --- Restriction: who competes ---
        # --- Selection: who wins ---
        return one_candidate

The restriction/selection split is requested **in the prompt**, not enforced by
the interface. This is the cheaper first attempt: if the search produces
programs where the two stages are fused or incoherent, the fallback is to split
`choose_next` into separate `restrict` and `select` functions, which needs no
harness change — the candidate set and state are already exactly what those two
functions would receive.

**Simplicity.** The guidelines judge designs partly on simplicity (few
parameters, interpretable ones). This is stated in the prompt and each program
is asked to self-report its restriction rule, selection function and parameters
in a docstring. It is deliberately **not** scored: every automatic proxy (code
length, constant count) is crude, and adding one to the fitness signal would
make Phase 1 and Phase 2 scores non-comparable.

## What the appending harness removes

Phase 1's harness maintained a DAG so it could validate insertions: cycle
checks, topological sort, rollback on infeasible moves. None of that survives
here. With append-only placement at the earliest start, machine sequences cannot
overlap and precedence cannot be violated, so **every schedule is feasible by
construction** and there is no infeasible-move failure class at all.

Confirmed on the seed program: `-9.241673534595261` (q10 `-11.378846252758477`,
fraction_positive `0.0`), bit-identical to the Phase 1 seed, which is
behaviourally the same heuristic. Evaluation takes 2.5s against Phase 1's 11.4s.

## Layout

    harness.py                        appending harness
    evaluator_penalize_failures.py    scoring; -100 for any failing instance
    configs/config_p2_A_1island.yaml  run config (prompt is inline here)
    seeds/initial_program_append.py   seed program
    prompts/                          unused — configs carry the prompt inline

Instances and CS baselines are shared with Phase 1 (`../fjsp_twt/instances/`,
`../fjsp_twt/.cs_baselines.pkl`) rather than copied: 100 generated train
instances, 50 held-out generated test instances, and Karim's four real
benchmark sets. Reusing the cached baselines makes Phase 1 and Phase 2 scores
comparable by construction, and preserves the held-out status of the test set.

## Run

    cd /root/skydiscover
    FJSP_EVAL_WORKERS=14 uv run skydiscover-run \
        benchmarks/math/fjsp_twt_append/seeds/initial_program_append.py \
        benchmarks/math/fjsp_twt_append/evaluator_penalize_failures.py \
        --config benchmarks/math/fjsp_twt_append/configs/config_p2_A_1island.yaml \
        --search adaevolve --model gpt-5.5 --iterations 200

Monitor on port 8790. Search settings copy Phase 1's `config_p1_A_1island.yaml`
exactly, so the phases differ only in design space and guidance.

## Not built yet

- **Held-out test evaluation.** Phase 1's `evaluate_test.py` has no Phase 2
  counterpart; needed after a run to report the generalization gap.
- **A multi-island / multi-objective config** mirroring Phase 1's
  `config_p1_B_2island_multiobj.yaml`, if the A/B contrast is wanted here too.
- **Per-run config provenance** — configs are still edited in place and never
  copied into the run directory, so a past run's exact config is unrecoverable.
  Worth fixing before these runs, not after.
