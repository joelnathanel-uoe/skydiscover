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
that ever helped. The design space is described in
`Dissertation Project/construction_heuristic_design_guidelines.md`, which frames
an appending heuristic as two choices: a **restriction rule** (who competes) and
a **selection function** (who wins).

## Layout

    configs/    run configs (one file per experiment)
    prompts/    system-message text
    seeds/      initial programs

Instances are shared with Phase 1 (`../fjsp_twt/instances/`) rather than copied:
100 generated train instances, 50 held-out generated test instances, and Karim's
four real benchmark sets. Reusing them keeps Phase 1 and Phase 2 scores directly
comparable and preserves the held-out status of the test set.

## Open decisions

1. **Harness interface.** How literally the harness should encode the
   restriction/selection split — from "Phase 1's `schedule_next` minus the
   `position` argument" through to the harness generating the candidate set and
   evolving `restrict` and `select` as separate functions.
2. **Definition of appending.** Strict append to the end of a machine's
   sequence, or also non-delaying gap insertion (fill an idle gap only when it
   delays nothing). To confirm with Karim. Whichever is chosen should be
   enforced by the harness structurally, not by prompt guidance.

Nothing here is wired up yet — the harness, evaluator, seed, prompt, and config
all follow once decision 1 is settled.
