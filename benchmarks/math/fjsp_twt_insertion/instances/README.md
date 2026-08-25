# Instances

Both benchmarks read from this directory. The generated sets come from
[`dissertation/tools/jss-instance-gen`](../../../../dissertation/tools/jss-instance-gen),
which is deterministic: the same seed and the same parameters reproduce an
instance byte for byte. Every directory carries a `manifest.csv` giving each
instance's file, scenario, size and seed. File names restart at
`instance_000.txt` in each directory, so the seed in the manifest, not the file
name, is what identifies an instance.

## How the generated instances were made

Every generated instance is the generator's `fjsp_medium` scenario: 50 jobs, 30
machines, 30 operation types, flexibility 3.0, flow structure 0.4, product mix
ratio 1.0, re-entrance level 0.2. Three parameters vary between the types, set
through the generator's environment variables:

| Type | `JSS_G` | `JSS_SPEED_MIN` | `JSS_SPEED_MAX` | |
|---|---|---|---|---|
| A | 0.5 | 0.8 | 1.2 | moderate due dates |
| B | 0.75 | 0.8 | 1.2 | loose due dates |
| C | 0.75 | 0.2 | 1.8 | loose due dates, wide spread of machine speeds |

`JSS_G` is the due-date tightness $g$; the two speed factors bound the
multiplier applied to an operation type's base processing time on each eligible
machine, so a narrow band makes the machines near-identical and a wide one
makes them unrelated.

To regenerate a set, build the generator and run it with those variables set
and the seed range below.

## What is here

| Directory | n | seeds | |
|---|---|---|---|
| `generated/` | 100 | 0–99 | Type A; the set programs are scored on during a run |
| `test/` | 50 | 100–149 | Type A, held out from that scoring |
| `type_a_search/` | 50 | 0–49 | the first 50 of `generated/`, byte for byte |
| `type_b_search/` | 50 | 150–199 | |
| `type_c_search/` | 50 | 250–299 | |
| `type_a_extra/` | 200 | 1000–1199 | generated after the runs finished |
| `type_b_extra/` | 200 | 2000–2199 | |
| `type_c_extra/` | 200 | 3000–3199 | |

The `search` sets are what a three-type run is scored on; `extra` are unseen
instances of the same three types, generated afterwards to test whether a
program's margin holds outside the set it was searched against.

## The academic sets

`Set_1/` to `Set_4/` are not generated. They are the 209 instances of Sobeyko
and Mönch (2016), in their own format, read by `evaluate_academic.py` in either
benchmark. `FJSPDD.txt` is a single instance in the same format, kept for
parsing checks.

## Baselines

`.cs_baselines.pkl` caches the combined-scheduler baseline for the generated
instances, so the seventy-eight classical schedules behind each one are
computed once rather than on every evaluation.
