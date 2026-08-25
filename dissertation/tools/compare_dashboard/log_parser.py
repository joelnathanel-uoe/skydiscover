"""
Regex-based parsing of AdaEvolve's plain-text run logs (logs/adaevolve_*.log).

These logs are the only place some data survives (e.g. the outcome of every
evaluated candidate, including ones the archive rejects outright before a
checkpoint is ever written), and the only place the *resolved* run config is
recorded (scattered across the first ~20 lines, not a single JSON dump).

All parsing here is line-oriented and single-pass: the log is strictly
chronological, so an "iteration N:" line establishes the current iteration
for any subsequent event lines (paradigm generator/tracker messages) until
the next "Iteration N:" line updates it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Config header (first ~20 INFO lines, format confirmed identical across runs)
# ---------------------------------------------------------------------------

DB_INIT_RE = re.compile(
    r"AdaEvolveDatabase initialized: num_islands=(?P<num_islands>\d+), "
    r"decay=(?P<decay>[\d.]+), "
    r"intensity=\[(?P<imin>[\d.]+), (?P<imax>[\d.]+)\], "
    r"migration=(?P<migration>True|False)(?: \(interval=(?P<interval>\d+)\))?, "
    r"unified_archive=(?P<unified>True|False), "
    r"adaptive_search=(?P<adaptive>True|False), "
    r"ucb_selection=(?P<ucb>True|False), "
    r"dynamic_islands=(?P<dyn>True|False), "
    r"paradigm_breakthrough=(?P<paradigm>True|False), "
    r"multiobjective=(?P<mo>True|False)"
)
LLM_RE = re.compile(r"OpenAI LLM: (?P<model>\S+)")
RUNNER_RE = re.compile(r"Runner ready: search=(?P<search>\S+), program=(?P<program>\S+)")
EVALUATOR_RE = re.compile(r"Initialized evaluator with (?P<path>\S+)")
CONTROLLER_RE = re.compile(r"DiscoveryController initialized: num_context_programs=(?P<n>\d+)")
GUIDE_MODELS_RE = re.compile(r"Paradigm LLM: using guide_models \[(?P<models>[^\]]+)\]")

_BOOL = {"True": True, "False": False}


def parse_config_header(log_text: str) -> Dict[str, Any]:
    """Extract the resolved run config from the log's startup lines.

    Only scans the first 60 lines -- the config header always appears there
    and this keeps parsing cheap even on multi-MB logs.
    """
    head = "\n".join(log_text.splitlines()[:60])
    config: Dict[str, Any] = {}

    m = DB_INIT_RE.search(head)
    if m:
        config.update(
            num_islands=int(m["num_islands"]),
            decay=float(m["decay"]),
            intensity_min=float(m["imin"]),
            intensity_max=float(m["imax"]),
            use_migration=_BOOL[m["migration"]],
            migration_interval=int(m["interval"]) if m["interval"] else None,
            use_unified_archive=_BOOL[m["unified"]],
            use_adaptive_search=_BOOL[m["adaptive"]],
            use_ucb_selection=_BOOL[m["ucb"]],
            use_dynamic_islands=_BOOL[m["dyn"]],
            use_paradigm_breakthrough=_BOOL[m["paradigm"]],
            multiobjective=_BOOL[m["mo"]],
        )

    m = RUNNER_RE.search(head)
    if m:
        config["search_type"] = m["search"]
        config["initial_program_path"] = m["program"]

    m = LLM_RE.search(head)
    if m:
        config["llm_model"] = m["model"]

    m = EVALUATOR_RE.search(head)
    if m:
        config["evaluator_path"] = m["path"]

    m = CONTROLLER_RE.search(head)
    if m:
        config["num_context_programs"] = int(m["n"])

    m = GUIDE_MODELS_RE.search(head)
    if m:
        config["paradigm_guide_models"] = [s.strip() for s in m["models"].split(",")]

    # Parameters confirmed to never appear in the log or JSONL for any run --
    # always surfaced explicitly rather than silently omitted.
    config["not_recoverable"] = [
        "fitness_weight",
        "novelty_weight",
        "pareto_weight",
        "elite_ratio",
        "llm_temperature",
        "evaluator_timeout",
    ]
    return config


# ---------------------------------------------------------------------------
# Per-evaluation lines -- exists for every candidate, even ones the archive
# rejects outright (i.e. finer-grained than the JSONL's iteration_result,
# which only records the one child program actually produced this iteration)
# ---------------------------------------------------------------------------

EVAL_RE = re.compile(
    r"Evaluated program (?P<pid>[0-9a-f-]{36}) in (?P<t>[\d.]+)s: (?P<rest>.*)$"
)
SCORE_RE = re.compile(r"combined_score=(?P<score>-?[\d.]+)")
ERROR_RE = re.compile(r"error=(?P<file>[^:]+): (?P<message>.*)$")


def _bucket_error(message: Optional[str]) -> str:
    if message is None:
        return "success"
    low = message.lower()
    if "exceeded per-call limit" in low or "timeout" in low:
        return "timeout"
    return "code_error"


def parse_evaluations(log_text: str) -> List[Dict[str, Any]]:
    """One record per 'Evaluated program ...' log line, in file order."""
    events = []
    for line in log_text.splitlines():
        m = EVAL_RE.search(line)
        if not m:
            continue
        rest = m["rest"]
        score_m = SCORE_RE.search(rest)
        error_m = ERROR_RE.search(rest)
        message = error_m["message"] if error_m else None
        events.append(
            {
                "program_id": m["pid"],
                "eval_time": float(m["t"]),
                "score": float(score_m["score"]) if score_m else None,
                "error_file": error_m["file"] if error_m else None,
                "error_message": message,
                "bucket": _bucket_error(message),
            }
        )
    return events


# ---------------------------------------------------------------------------
# Iteration-completion lines, migration events, "new best" markers
# ---------------------------------------------------------------------------

ITERATION_RE = re.compile(
    r"Iteration (?P<iter>\d+): Program (?P<child>\w+) \(parent: (?P<parent>\w+)\) "
    r"completed in (?P<total>[\d.]+)s \(llm: (?P<llm>[\d.]+)s, eval: (?P<eval>[\d.]+)s\)"
)
NEW_BEST_RE = re.compile(r"New best solution found at iteration (?P<iter>\d+)")
MIGRATION_RE = re.compile(r"Migration completed at iteration (?P<iter>\d+)")
CHECKPOINT_RE = re.compile(r"Checkpoint interval reached at iteration (?P<iter>\d+)")


def parse_iteration_lines(log_text: str) -> List[Dict[str, Any]]:
    records = []
    for line in log_text.splitlines():
        m = ITERATION_RE.search(line)
        if m:
            records.append(
                {
                    "iteration": int(m["iter"]),
                    "child_short_id": m["child"],
                    "parent_short_id": m["parent"],
                    "iteration_time_seconds": float(m["total"]),
                    "llm_time_seconds": float(m["llm"]),
                    "eval_time_seconds": float(m["eval"]),
                }
            )
    return records


def parse_migration_events(log_text: str) -> List[int]:
    return [int(m["iter"]) for m in MIGRATION_RE.finditer(log_text)]


def parse_new_best_iterations(log_text: str) -> List[int]:
    return [int(m["iter"]) for m in NEW_BEST_RE.finditer(log_text)]


# ---------------------------------------------------------------------------
# Paradigm-breakthrough events (stagnation trigger, generated ideas, which
# paradigm is active, success/failure outcome) -- correlated to the nearest
# preceding "Iteration N:" line since the log is strictly chronological.
# ---------------------------------------------------------------------------

STAGNATION_RE = re.compile(r"Global paradigm stagnation detected")
GENERATED_RE = re.compile(r"Generated (?P<n>\d+) paradigms:")
PARADIGM_LINE_RE = re.compile(r"^\s*\[(?P<idx>\d+)\] (?P<idea>.+?) \(approach: (?P<approach>.+)\)$")
USING_RE = re.compile(
    r"Using paradigm (?P<idx>\d+)/(?P<total>\d+) \((?P<use>\d+)/(?P<max_use>\d+)\): (?P<idea>.+)$"
)
OUTCOME_RE = re.compile(r"^\s*\[(?P<outcome>SUCCESS|FAILED)\] (?P<idea>.+?) \(uses: (?P<uses>\d+)\)$")


def parse_paradigm_events(log_text: str) -> List[Dict[str, Any]]:
    """Extract paradigm-breakthrough events with best-effort iteration correlation."""
    events: List[Dict[str, Any]] = []
    current_iteration = 0

    for line in log_text.splitlines():
        m = ITERATION_RE.search(line)
        if m:
            current_iteration = int(m["iter"])
            continue

        if STAGNATION_RE.search(line):
            events.append({"type": "stagnation_detected", "iteration": current_iteration})
            continue

        m = GENERATED_RE.search(line)
        if m:
            events.append(
                {"type": "paradigms_generated", "iteration": current_iteration, "count": int(m["n"])}
            )
            continue

        m = PARADIGM_LINE_RE.match(line.split(" - INFO - ", 1)[-1] if " - INFO - " in line else line)
        if m:
            events.append(
                {
                    "type": "paradigm_proposed",
                    "iteration": current_iteration,
                    "index": int(m["idx"]),
                    "idea": m["idea"],
                    "approach": m["approach"],
                }
            )
            continue

        m = USING_RE.search(line)
        if m:
            events.append(
                {
                    "type": "paradigm_start",
                    "iteration": current_iteration,
                    "index": int(m["idx"]),
                    "total": int(m["total"]),
                    "use": int(m["use"]),
                    "max_use": int(m["max_use"]),
                    "idea": m["idea"],
                }
            )
            continue

        m = OUTCOME_RE.match(line.split(" - INFO - ", 1)[-1] if " - INFO - " in line else line)
        if m:
            events.append(
                {
                    "type": "paradigm_outcome",
                    "iteration": current_iteration,
                    "outcome": m["outcome"],
                    "idea": m["idea"],
                    "uses": int(m["uses"]),
                }
            )
            continue

    return events


def segment_dirs(run_dir: Path) -> List[Path]:
    """A resumed run's earlier trajectory segments are parked under _parts/NN/
    (see the _parts/README the queue writes), each holding its own logs/ and
    iteration-stats JSONL; the live segment stays in the run dir itself and is
    always last. Reading only the run dir would show a 193-iteration run as its
    final 17 iterations, so every log/JSONL consumer walks the segments in
    order instead."""
    parts_root = run_dir / "_parts"
    parked = []
    if parts_root.is_dir():
        parked = sorted(
            (p for p in parts_root.iterdir() if p.is_dir()),
            key=lambda p: (len(p.name), p.name),
        )
    return parked + [run_dir]


ITERATION_LINE_RE = re.compile(r"^.*? - Iteration (\d+):", re.MULTILINE)


def _first_iteration(text: str) -> Optional[int]:
    m = ITERATION_LINE_RE.search(text)
    return int(m.group(1)) if m else None


def _drop_discarded_tail(text: str, next_first: Optional[int]) -> str:
    """A resume restarts from the last *checkpoint*, not the iteration the run
    actually reached, so a parked segment's final iterations are redone in the
    next segment (e.g. parked reaches 177, live restarts at 171). Those redone
    iterations are discarded work and must not be counted twice, so a segment
    is cut where the next one picks up."""
    if next_first is None:
        return text
    for m in ITERATION_LINE_RE.finditer(text):
        if int(m.group(1)) >= next_first:
            return text[: m.start()]
    return text


def load_log_text(run_dir: Path) -> str:
    texts = []
    for seg in segment_dirs(run_dir):
        logs = sorted((seg / "logs").glob("adaevolve_*.log"))
        texts.append(logs[-1].read_text(encoding="utf-8", errors="replace") if logs else "")
    if not any(texts):
        raise FileNotFoundError(f"No adaevolve_*.log found under {run_dir / 'logs'}")
    firsts = [_first_iteration(t) for t in texts]
    kept = [
        _drop_discarded_tail(t, next((f for f in firsts[i + 1 :] if f is not None), None))
        for i, t in enumerate(texts)
    ]
    return "\n".join(t for t in kept if t)
