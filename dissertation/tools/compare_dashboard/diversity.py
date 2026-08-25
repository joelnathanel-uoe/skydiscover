"""
Within-island and between-island code diversity, computed post-hoc from
checkpoint program code.

Reuses skydiscover's own CodeDiversity (the exact class the framework uses
internally for novelty-driven sampling) rather than reinventing a distance
metric -- see skydiscover/search/adaevolve/archive/diversity.py. The
framework computes novelty/elite scores in-memory during a run but never
persists them, so this recomputes distance() from saved code, adding
memoization since the same program participates in many pairwise
comparisons (within its island, and against every other island).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

from skydiscover.search.adaevolve.archive.diversity import CodeDiversity
from skydiscover.search.base_database import Program


class CachedCodeDiversity(CodeDiversity):
    """CodeDiversity with per-program-id memoization of tokens/features.

    The base class re-tokenizes both programs on every distance() call;
    here each program's tokens/features are computed once and reused across
    every pair it appears in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tok_cache: Dict[str, set] = {}
        self._feat_cache: Dict[str, set] = {}

    def distance_by_id(self, pid_a: str, prog_a: Program, pid_b: str, prog_b: Program) -> float:
        if prog_a.solution == prog_b.solution:
            return 0.0

        t1 = self._tok_cache.setdefault(pid_a, self._tokenize(prog_a.solution))
        t2 = self._tok_cache.setdefault(pid_b, self._tokenize(prog_b.solution))
        f1 = self._feat_cache.setdefault(pid_a, self._extract_features(prog_a.solution))
        f2 = self._feat_cache.setdefault(pid_b, self._extract_features(prog_b.solution))

        token_dist = self._jaccard_distance(t1, t2)
        struct_dist = self._jaccard_distance(f1, f2)
        max_len = max(len(prog_a.solution), len(prog_b.solution), 1)
        len_dist = abs(len(prog_a.solution) - len(prog_b.solution)) / max_len

        return (
            token_dist * self.token_weight
            + struct_dist * self.structure_weight
            + len_dist * self.length_weight
        )


def mean_pairwise_distance(
    diversity: CachedCodeDiversity,
    ids_a: List[str],
    programs: Dict[str, Program],
    ids_b: List[str] = None,
) -> float:
    """Mean pairwise distance within one group (ids_b=None) or between two groups."""
    if ids_b is None:
        pairs: List[Tuple[str, str]] = list(combinations(ids_a, 2))
    else:
        pairs = [(a, b) for a in ids_a for b in ids_b]

    if not pairs:
        return 0.0

    total = 0.0
    n = 0
    for pid_a, pid_b in pairs:
        prog_a = programs.get(pid_a)
        prog_b = programs.get(pid_b)
        if prog_a is None or prog_b is None:
            continue
        total += diversity.distance_by_id(pid_a, prog_a, pid_b, prog_b)
        n += 1

    return total / n if n else 0.0


def compute_run_diversity(
    islands_by_checkpoint: Dict[int, List[List[str]]],
    programs: Dict[str, Program],
) -> Dict[str, object]:
    """
    Args:
        islands_by_checkpoint: {checkpoint_iteration: [[program_ids island 0], [island 1], ...]}
        programs: {program_id: Program}, union across all checkpoints of this run

    Returns:
        {"checkpoints": [...], "within_island": {"0": [...], ...},
         "between_island": {"0-1": [...], ...}, "population_mean": [...]}
    """
    diversity = CachedCodeDiversity()
    checkpoints = sorted(islands_by_checkpoint.keys())

    num_islands = max((len(v) for v in islands_by_checkpoint.values()), default=0)
    within: Dict[str, List[float]] = {str(i): [] for i in range(num_islands)}
    between: Dict[str, List[float]] = {}
    pop_mean: List[float] = []

    island_pairs = list(combinations(range(num_islands), 2))
    for i, j in island_pairs:
        between[f"{i}-{j}"] = []

    for ckpt in checkpoints:
        island_lists = islands_by_checkpoint[ckpt]

        for i in range(num_islands):
            ids = island_lists[i] if i < len(island_lists) else []
            within[str(i)].append(mean_pairwise_distance(diversity, ids, programs))

        for i, j in island_pairs:
            ids_i = island_lists[i] if i < len(island_lists) else []
            ids_j = island_lists[j] if j < len(island_lists) else []
            between[f"{i}-{j}"].append(mean_pairwise_distance(diversity, ids_i, programs, ids_j))

        all_ids = [pid for island in island_lists for pid in island]
        pop_mean.append(mean_pairwise_distance(diversity, all_ids, programs))

    return {
        "checkpoints": checkpoints,
        "within_island": within,
        "between_island": between,
        "population_mean": pop_mean,
    }
