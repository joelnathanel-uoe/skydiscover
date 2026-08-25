# Programs referenced in the dissertation

Every program named in the write-up, copied here unmodified so that each one
cited in the text has a stable location. Each file is the `choose_next`
function the harness calls, between its `EVOLVE-BLOCK-START` and
`EVOLVE-BLOCK-END` markers, exactly as the search produced it.

The harness, evaluator and instance generator these were produced and scored
with are in [`benchmarks/math/fjsp_twt_append`](../benchmarks/math/fjsp_twt_append).

| File | Referred to as | Origin |
|---|---|---|
| `00_seed.py` | the seed program | `seeds/initial_program_append.py` |
| `01_phase1_run1_champion.py` | Phase 1, Run 1 program (the champion) | `p1mech01_reference` |
| `02_phase1_run2.py` | Phase 1, Run 2 program | `p1mech02_nomig` |
| `03_phase1_run3.py` | Phase 1, Run 3 program | `p1mech03_isolated` |
| `04_phase1_run4.py` | Phase 1, Run 4 program | `p1mech04_homogeneous_r2` |
| `05_phase2_run1.py` | Phase 2, Run 1 program | `p2E_champ_23fa39b9` |
| `06_phase2_run2.py` | Phase 2, Run 2 program | `p2C_champ_cb4db0df` |
| `07_phase2_run3_riskpricing.py` | Phase 2, Run 3 program (RiskPricing), and the Phase 3 seed | `p2D_champ_2438e6be` |
| `08_phase3_run1.py` | Phase 3, Run 1 program | `p3_riskpricing_champ_c1b5ffd7` |
| `09_phase3_run2.py` | Phase 3, Run 2 program | `p3_random_champ_04466b6b` |
| `10_phase3_specialist.py` | the Type C specialist | `p3_loosevaried_champ_e2a65a78` |

## Scores

`train` and `held-out` are the median improvement on the Combined Scheduler
baseline over 50 generated instances each; `academic` is the median over the
209 instances of Sobeyko and Mönch (2016). Phase 3 programs are additionally
scored against `07_phase2_run3_riskpricing.py` on 150 held-out instances, 50 of each
generated type, which is the objective their search used.

| File | Train | Held-out | Academic |
|---|---|---|---|
| `00_seed.py` | −9.2417 | — | — |
| `01_phase1_run1_champion.py` | 0.4462 | +0.4179 | +0.2223 |
| `02_phase1_run2.py` | 0.4424 | +0.4111 | +0.2359 |
| `03_phase1_run3.py` | 0.4236 | +0.3754 | +0.1989 |
| `04_phase1_run4.py` | 0.4193 | +0.3656 | +0.2233 |
| `05_phase2_run1.py` | 0.4666 | +0.4293 | +0.2422 |
| `06_phase2_run2.py` | 0.4340 | +0.4018 | +0.1935 |
| `07_phase2_run3_riskpricing.py` | 0.4306 | +0.4040 | +0.2236 |
| `08_phase3_run1.py` | +0.0181 | — | +0.2327 |
| `09_phase3_run2.py` | +0.0228 | +0.0221 | +0.2494 |
| `10_phase3_specialist.py` | +0.2581 | +0.2066 | +0.2448 |

Phase 3 `train` figures are against `07_phase2_run3_riskpricing.py` rather than against
the baseline, and the specialist's is on Type C alone, which is the only type
it was scored on. `08_phase3_run1.py` exceeds the harness per-call time
budget on one held-out instance, so it has no held-out figure.

## Running one

```python
import sys, importlib

sys.path.insert(0, "benchmarks/math/fjsp_twt_append")
sys.path.insert(0, "dissertation/programs")

from harness import run_harness
import evaluator_penalize_failures as ev

instance = ev.parse_generated(
    "benchmarks/math/fjsp_twt_insertion/instances/type_a_search/instance_000.txt")
program = importlib.import_module("10_phase3_specialist")

result = run_harness(instance, program.choose_next)
print(result["twt"])          # total weighted tardiness of the schedule built
```

`parse_generated` reads the generated instances; the academic sets are in a
different format and are read by `ev.parse_instance` instead.
