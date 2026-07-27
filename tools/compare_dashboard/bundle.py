"""
Per-run and cross-run JSON bundle assembly for the comparison dashboard.

Combines three data sources per run:
  - checkpoints/checkpoint_N/{programs/*.json, adaevolve_metadata.json}
  - adaevolve_iteration_stats_*.jsonl (one line per iteration)
  - logs/adaevolve_*.log (plain text -- config header + per-evaluation lines
    + migration/paradigm events; see log_parser.py)

Checkpoint availability is irregular per run (confirmed empirically -- gaps
at different iterations for different runs), so every checkpoint-driven
computation enumerates whatever checkpoint_N dirs actually exist for a given
run rather than assuming a fixed cadence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from skydiscover.search.base_database import Program

from . import log_parser
from .diversity import compute_run_diversity

SCHEMA_VERSION = 4


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def find_checkpoints(run_dir: Path) -> List[int]:
    ckpts_dir = run_dir / "checkpoints"
    if not ckpts_dir.is_dir():
        return []
    out = []
    for p in ckpts_dir.glob("checkpoint_*"):
        try:
            out.append(int(p.name.split("_")[-1]))
        except ValueError:
            continue
    return sorted(out)


def load_checkpoint_islands(run_dir: Path, iteration: int) -> List[List[str]]:
    meta_path = run_dir / "checkpoints" / f"checkpoint_{iteration}" / "adaevolve_metadata.json"
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text())
    return meta.get("islands", [])


def load_all_programs(run_dir: Path) -> Dict[str, dict]:
    """Union of every program dict seen across every checkpoint, deduped by id.

    Programs evicted from the archive before ever reaching a checkpoint are
    NOT recoverable here -- their bare outcome only survives in the log's
    per-evaluation lines (see log_parser.parse_evaluations).
    """
    programs: Dict[str, dict] = {}
    for ckpt in find_checkpoints(run_dir):
        progs_dir = run_dir / "checkpoints" / f"checkpoint_{ckpt}" / "programs"
        if not progs_dir.is_dir():
            continue
        for pf in progs_dir.glob("*.json"):
            try:
                data = json.loads(pf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            pid = data.get("id")
            if pid and pid not in programs:
                programs[pid] = data
    return programs


def build_island_history(islands_by_checkpoint: Dict[int, List[List[str]]]) -> Dict[str, List[List[int]]]:
    """Invert checkpoint->islands->[program_ids] into program_id->[(checkpoint, island_idx), ...].

    A program can appear under more than one island across checkpoints (a
    migration clone gets its own id, but the *original* program can itself be
    read into a later checkpoint's snapshot if it's still resident -- in
    practice each id tends to stay in one island, but this makes no such
    assumption). History is ordered by checkpoint ascending.
    """
    history: Dict[str, List[List[int]]] = {}
    for ckpt in sorted(islands_by_checkpoint.keys()):
        for island_idx, pids in enumerate(islands_by_checkpoint[ckpt]):
            for pid in pids:
                history.setdefault(pid, []).append([ckpt, island_idx])
    return history


def find_migration_clones(
    programs: Dict[str, dict],
    islands_by_checkpoint: Dict[int, List[List[str]]],
    migration_iterations: List[int],
) -> List[Dict[str, Any]]:
    """Programs created by migration carry metadata.migrated_from/to and a
    parent_id pointing at the pre-migration original (see database.py's
    _migrate_archives: migration clones into a NEW id, it never moves one).

    IMPORTANT: the clone's own `iteration_found` is NOT when the migration
    happened -- database.py's _migrate_archives constructs the migrant with
    `iteration_found=program.iteration_found`, copying the *original*
    program's creation iteration verbatim, and the `add()` call that follows
    passes no `iteration=` kwarg, so it's never overwritten. Confirmed on a
    real clone: `iteration_found=13` while this run's only migration events
    are at iterations 30/60/90 -- 13 can't be a real migration point, it's
    just when the pre-migration original was first created, possibly many
    iterations earlier. The true migration iteration is derivable instead:
    check which checkpoints the clone id is present/absent in, and match
    against the run's known migration_iterations (there's no per-program
    DEBUG-level migration log to read this from directly, per
    project-data-availability-audit).

    Checkpoint-vs-migration ordering (confirmed from controller.py): within
    one iteration, `checkpoint_callback` fires in `_run_iteration` BEFORE
    `end_iteration()` (which runs migration) is called in the `finally`
    block. So a checkpoint taken AT iteration N does NOT yet include a
    migration that happens at that same iteration N -- the clone first
    appears in the *next* checkpoint after N. That means a migration event
    exactly at a clone's `last_absent` checkpoint is a valid candidate, not
    just events strictly after it (hence `>=`, not `>`, below).

    Checkpoint cadence is irregular (confirmed: gaps appear when a stretch of
    iterations all fail, e.g. an OpenAI quota outage -- checkpoints seemingly
    only get taken after productive iterations). When more than one migration
    event falls inside a clone's (last_absent, first_present] window, this is
    a genuine ambiguity -- checkpoint presence alone cannot tell which one
    produced the clone. Reported honestly as `migration_iteration_candidates`
    (plural) rather than silently guessing one.
    """
    checkpoints_sorted = sorted(islands_by_checkpoint.keys())
    presence: Dict[str, List[int]] = {}
    for ckpt in checkpoints_sorted:
        for island in islands_by_checkpoint[ckpt]:
            for pid in island:
                presence.setdefault(pid, []).append(ckpt)

    clones = []
    for pid, data in programs.items():
        meta = data.get("metadata") or {}
        if "migrated_from" in meta and "migrated_to" in meta:
            metrics = data.get("metrics") or {}
            ckpts_present = sorted(presence.get(pid, []))
            first_present = ckpts_present[0] if ckpts_present else None
            last_absent = max(
                (c for c in checkpoints_sorted if first_present is not None and c < first_present),
                default=0,
            )
            # Migration events at-or-after the last checkpoint where the clone
            # was absent (checkpoint-before-migration ordering, see above),
            # and at-or-before the first checkpoint where it appears.
            candidates = sorted(
                m for m in migration_iterations
                if m >= last_absent and (first_present is None or m <= first_present)
            )
            if len(candidates) == 1:
                inferred_iter, approx = candidates[0], False
            elif candidates:
                # Genuinely ambiguous -- report the earliest as a best guess
                # but flag it and keep the full candidate list.
                inferred_iter, approx = candidates[0], True
            else:
                # No candidate in-window at all (e.g. checkpoint gap swallowed
                # every migration event) -- fall back to nearest before.
                before = [m for m in migration_iterations if first_present is not None and m <= first_present]
                inferred_iter, approx = (max(before), True) if before else (None, True)
                candidates = [inferred_iter] if inferred_iter is not None else []

            clones.append(
                {
                    "clone_id": pid,
                    "original_id": data.get("parent_id"),
                    "from_island": meta["migrated_from"],
                    "to_island": meta["migrated_to"],
                    "migration_iteration": inferred_iter,
                    "migration_iteration_approx": approx,
                    "migration_iteration_candidates": candidates,
                    "original_iteration_found": data.get("iteration_found"),
                    "score": metrics.get("combined_score"),
                }
            )
    return clones


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


def load_iterations_jsonl(run_dir: Path) -> List[dict]:
    jsonl_files = list(run_dir.glob("adaevolve_iteration_stats_*.jsonl"))
    if not jsonl_files:
        return []
    records = []
    with open(jsonl_files[0]) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _bucket_iteration_error(error: Optional[str]) -> str:
    if error is None:
        return "success"
    low = error.lower()
    if "llm error" in low:
        return "llm_error"
    if "exceeded per-call limit" in low or "timeout" in low:
        return "eval_timeout"
    return "eval_other"


def flatten_iterations(raw_lines: List[dict]) -> List[Dict[str, Any]]:
    flattened = []
    for line in raw_lines:
        g = line.get("global", {})
        ir = line.get("iteration_result", {})
        child = ir.get("child_program") or {}
        error = ir.get("error")

        flattened.append(
            {
                "iteration": line.get("iteration"),
                "global_best_score": g.get("global_best_score"),
                "current_island_idx": g.get("current_island_idx"),
                "islands": [
                    {
                        "island_idx": isl.get("island_idx"),
                        "population_size": isl.get("population_size"),
                        "search_intensity": isl.get("search_intensity"),
                        "productivity": isl.get("productivity"),
                        "config_name": isl.get("config_name"),
                        "best_score": isl.get("best_score"),
                        "archive_stats": isl.get("archive_stats"),
                    }
                    for isl in line.get("islands", [])
                ],
                "paradigm": {
                    "is_stagnating": line.get("paradigm", {}).get("is_stagnating"),
                    "improvement_rate": line.get("paradigm", {}).get("improvement_rate"),
                    "has_active_paradigm": line.get("paradigm", {}).get("has_active_paradigm"),
                },
                "dynamic_islands": {
                    "would_spawn": line.get("dynamic_islands", {}).get("would_spawn"),
                    "current_num_islands": line.get("dynamic_islands", {}).get("current_num_islands"),
                },
                "iteration_result": {
                    "success": ir.get("success"),
                    "error": error,
                    "bucket": _bucket_iteration_error(error),
                    "iteration_time_seconds": ir.get("iteration_time_seconds"),
                    "llm_generation_time_seconds": ir.get("llm_generation_time_seconds"),
                    "eval_time_seconds": ir.get("eval_time_seconds"),
                    "child_program_id": child.get("id"),
                    "child_score": (child.get("metrics") or {}).get("combined_score"),
                    "child_generation": child.get("generation"),
                    "parent_id": child.get("parent_id"),
                },
            }
        )
    return flattened


def last_productive_iteration(flat_iterations: List[Dict[str, Any]]) -> int:
    """Last iteration where the run actually produced a scored child program.

    Used to cap chart x-axes for runs that died mid-search (quota exhaustion),
    rather than letting a flat/absent tail read as "converged".
    """
    productive = [it["iteration"] for it in flat_iterations if it["iteration_result"]["success"]]
    return max(productive) if productive else 0


# ---------------------------------------------------------------------------
# Per-run bundle assembly
# ---------------------------------------------------------------------------


def build_run_bundle(run_dir: Path, include_full_code: bool = False) -> Dict[str, Any]:
    run_name = run_dir.name
    log_text = log_parser.load_log_text(run_dir)
    config = log_parser.parse_config_header(log_text)

    raw_iterations = load_iterations_jsonl(run_dir)
    flat_iterations = flatten_iterations(raw_iterations)
    total_iterations = len(flat_iterations)
    last_productive = last_productive_iteration(flat_iterations)

    # Cross-check num_islands/migration agreement between log header and JSONL
    # config block -- cheap sanity check, catches log/JSONL truncation issues.
    conflicts = {}
    if raw_iterations:
        jsonl_cfg = raw_iterations[0].get("config", {})
        jsonl_global = raw_iterations[0].get("global", {})
        if jsonl_global.get("num_islands") is not None and config.get("num_islands") is not None:
            if jsonl_global["num_islands"] != config["num_islands"]:
                conflicts["num_islands"] = [config["num_islands"], jsonl_global["num_islands"]]
        config["population_size"] = (
            raw_iterations[0].get("islands", [{}])[0].get("archive_stats", {}).get("max_size")
        )
        config["migration_count"] = jsonl_cfg.get("migration_count")
        config["local_context_program_ratio"] = jsonl_cfg.get("local_context_program_ratio")
    config["_conflicts"] = conflicts

    checkpoints = find_checkpoints(run_dir)
    raw_programs = load_all_programs(run_dir)
    islands_by_checkpoint = {ckpt: load_checkpoint_islands(run_dir, ckpt) for ckpt in checkpoints}
    island_history = build_island_history(islands_by_checkpoint)
    migration_iterations = log_parser.parse_migration_events(log_text)
    migration_clones = find_migration_clones(raw_programs, islands_by_checkpoint, migration_iterations)
    new_best_iterations = log_parser.parse_new_best_iterations(log_text)
    paradigm_events = log_parser.parse_paradigm_events(log_text)
    evaluation_events = log_parser.parse_evaluations(log_text)

    # Diversity needs Program objects (not persisted to the JSON bundle --
    # recomputed here, results are just the aggregate distance series).
    program_objects = {pid: Program.from_dict(data) for pid, data in raw_programs.items()}
    diversity = compute_run_diversity(islands_by_checkpoint, program_objects)

    # Lightweight program records for the bundle -- code_preview only by
    # default (see project_html_slowness.md: avoid repeating the past
    # 4.7MB-per-file slow-load problem by not inlining every program's
    # full source unless explicitly requested).
    programs_out = {}
    for pid, data in raw_programs.items():
        solution = data.get("solution") or ""
        history = island_history.get(pid, [])
        islands_seen = sorted({isl for _, isl in history})
        rec = {
            "generation": data.get("generation", 0),
            "iteration_found": data.get("iteration_found", 0),
            "parent_id": data.get("parent_id"),
            "other_context_ids": data.get("other_context_ids") or [],
            "metrics": data.get("metrics") or {},
            "metadata": data.get("metadata") or {},
            "code_length": len(solution),
            "code_preview": solution[:500],
            "island_history": history,
            "islands_seen": islands_seen,
            "last_island": history[-1][1] if history else None,
        }
        if include_full_code:
            rec["solution"] = solution
        programs_out[pid] = rec

    iteration_bucket_counts: Dict[str, int] = {}
    for it in flat_iterations:
        b = it["iteration_result"]["bucket"]
        iteration_bucket_counts[b] = iteration_bucket_counts.get(b, 0) + 1

    eval_bucket_counts: Dict[str, int] = {}
    for ev in evaluation_events:
        eval_bucket_counts[ev["bucket"]] = eval_bucket_counts.get(ev["bucket"], 0) + 1

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config": config,
        "total_iterations": total_iterations,
        "last_productive_iteration": last_productive,
        "checkpoints_present": checkpoints,
        "iterations": flat_iterations,
        "programs": programs_out,
        "islands_by_checkpoint": {str(k): v for k, v in islands_by_checkpoint.items()},
        "migration_events": {
            "iterations": migration_iterations,
            "clones": migration_clones,
        },
        "new_best_iterations": new_best_iterations,
        "diversity": diversity,
        "errors": {
            "iteration_level": iteration_bucket_counts,
            "evaluation_level": eval_bucket_counts,
            "evaluation_events": evaluation_events,
        },
        "paradigm_events": paradigm_events,
    }


def build_comparison_bundle(run_dirs: List[Path], include_full_code: bool = False) -> Dict[str, Any]:
    from datetime import datetime, timezone

    runs = {}
    for run_dir in run_dirs:
        runs[run_dir.name] = build_run_bundle(run_dir, include_full_code=include_full_code)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
    }
