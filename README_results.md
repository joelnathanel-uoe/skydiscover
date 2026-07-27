# FJSP-TWT Run Results
**Joel Nathanael — MSc Dissertation, University of Edinburgh**
*LLM-Driven Algorithm Discovery for Construction Heuristics (FJSSP-TWT)*

---

## Run Configurations

| Run | Model | Iters | Paradigm Breakthrough | Seed Program | Resumed From | Best Score |
|---|---|---|---|---|---|---|
| `fjsp_twt_0615_1548` | gpt-5.5 | 200 | On | initial_program_sequential.py | — | **0.6791** |
| `fjsp_twt_0614_0036` | gpt-4o-mini | 700 | On | initial_program_sequential.py | fjsp_twt_0613_2027 @ iter 200 | 0.5829 |
| `fjsp_twt_0614_0031` | gpt-4o-mini | 500 | On | initial_program_sequential.py | — | 0.2022 |
| `fjsp_twt_0613_2027` | gpt-4o-mini | 200 | On | initial_program_sequential.py | — | 0.5800 |
| `fjsp_twt_0613_2028` | gpt-4o-mini | 200 | Off | initial_program_sequential.py | — | 0.2271 |
| `fjsp_twt_0613_2011` | gpt-4o-mini | 200 | On | initial_program_sequential.py | — | 0.2271 |
| `fjsp_twt_0613_2012` | gpt-4o-mini | 200 | Off | initial_program_sequential.py | — | 0.2265 |

**Shared settings across all runs:** 3 islands · decay=0.9 · UCB selection · adaptive search · dynamic islands · no multiobjective

Score = `combined_score` (higher is better, range 0–1).

---

## Viewing the Results

### Option 1 — HTML summaries (no install needed)

Open any file from `run_summaries/` directly in your browser. Each file contains the full iteration log for that run — metrics, code, and LLM interactions.

### Option 2 — Live dashboard (interactive)

#### Requirements
- Python 3.10+
- SkyDiscover installed from the repo:
  ```bash
  git clone https://github.com/joelnathanel-uoe/skydiscover.git
  pip install ./skydiscover
  ```

#### Launch

```bash
python -m skydiscover.extras.monitor.viewer outputs/adaevolve/fjsp_twt_0615_1548
```

Replace the run name with any from the table above. Then open:
```
http://localhost:8765/
```
Press `Ctrl+C` to stop.

### Best Discovered Heuristic

The best program from each run is at:
```
outputs/adaevolve/<run_name>/best/best_program.py
```
