"""
Baseline dispatching rules for FJSP-TWT from Sobeyko & Mönch (2016).

Rules implemented (Section 4.1):
    1.  FIFO           — smallest ready time r_v
    2.  WFIFO          — smallest r_v / w_i
    3.  ODD            — smallest op due date d_v = d_i - FF * Σ_{θ_v} p_u
    4.  WODD           — smallest d_v / w_i
    5.  MOD            — smallest max(d_v, t + p_v)
    6.  WMOD           — smallest (1/w_i) * max(d_v - t, p_v)
    7.  SPT            — largest 1/p_v
    8.  WSPT           — largest w_i/p_v
    9.  ATC            — largest (w_i/p_v)*exp(-(d_v + r_i - t)^+ / (κ*p̄))
    CS  Combined Scheduler — best TWT across all SDRs + ATC(κ=0.1..7.0)

All rules use EFT machine selection (smallest completion time).
θ_v = remaining operations of job i AFTER the current one.
p_u in Σ_{θ_v} = min processing time across eligible machines (lower bound).
p_v = processing time on EFT-selected machine for priority calculation.
FF  = instance-specific flow factor.
t   = EFT start time for the candidate operation (decision time proxy).
"""

import math
import os
import sys

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _suffix_min_p(jobs):
    """For each job, suffix sum of min processing times over remaining ops.

    suffix[j][k] = sum of min_p for ops k, k+1, ..., n_ops-1
    suffix[j][op_idx + 1] = remaining work after op op_idx
    """
    result = []
    for job in jobs:
        ops = job['operations']
        n = len(ops)
        suf = [0] * (n + 1)
        for k in range(n - 1, -1, -1):
            suf[k] = suf[k + 1] + min(p for _, p in ops[k])
        result.append(suf)
    return result


def _compute_flow_factor(instance, schedule):
    """FF = mean over jobs of (C_j - r_j) / sum(min_p for all ops of j).

    Computed from a given schedule (typically FIFO).
    """
    jobs = instance['jobs']
    job_completion = {}
    for entry in schedule:
        j = entry['job']
        if j not in job_completion or entry['end'] > job_completion[j]:
            job_completion[j] = entry['end']

    ratios = []
    for j, job in enumerate(jobs):
        c_j = job_completion.get(j, job['release_time'])
        r_j = job['release_time']
        total_p = sum(min(p for _, p in op) for op in job['operations'])
        if total_p > 0:
            ratios.append((c_j - r_j) / total_p)

    return sum(ratios) / len(ratios) if ratios else 1.0


def _list_schedule(instance, priority_fn, maximize=False, suf=None, p_bar_fn=None):
    """Generic list scheduler with EFT machine selection.

    Args:
        priority_fn(j, op_idx, r_j, start, p_v, suf, p_bar) -> float
            r_j   = job ready time at this scheduling step
            start = EFT start time for this candidate (used as t)
            p_v   = processing time on EFT machine
            p_bar = average remaining proc time of all current candidates
        maximize: True for rules where larger priority = better (SPT, WSPT, ATC)
        suf:      precomputed suffix min_p table (reused across calls)
    """
    jobs = instance['jobs']
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    machine_available = [0] * (n_machines + 1)
    job_op_idx = [0] * n_jobs
    job_ready = [jobs[j]['release_time'] for j in range(n_jobs)]
    schedule = []
    total_ops = sum(len(jobs[j]['operations']) for j in range(n_jobs))

    if suf is None:
        suf = _suffix_min_p(jobs)

    while len(schedule) < total_ops:
        # Build candidate list with EFT machine for each ready operation
        raw = []
        for j in range(n_jobs):
            op_idx = job_op_idx[j]
            if op_idx >= len(jobs[j]['operations']):
                continue
            op = jobs[j]['operations'][op_idx]
            best_end, best_start, best_m, best_p = float('inf'), None, None, None
            for m, p in op:
                start = max(machine_available[m], job_ready[j])
                end = start + p
                if end < best_end:
                    best_end, best_start, best_m, best_p = end, start, m, p
            if best_m is not None:
                raw.append((j, op_idx, job_ready[j], best_start, best_end, best_m, best_p))

        if not raw:
            break

        # p̄ = average remaining processing time of all candidates (incl. current op)
        p_bar = sum(suf[j][op_idx] for j, op_idx, *_ in raw) / len(raw)

        candidates = []
        for j, op_idx, r_j, start, end, m, p_v in raw:
            pri = priority_fn(j, op_idx, r_j, start, p_v, suf, p_bar)
            candidates.append((pri, j, op_idx, m, start, end))

        if maximize:
            _, j, op_idx, m, start, end = max(candidates, key=lambda x: x[0])
        else:
            _, j, op_idx, m, start, end = min(candidates, key=lambda x: x[0])

        schedule.append({'job': j, 'op': op_idx, 'machine': m, 'start': start, 'end': end})
        machine_available[m] = end
        job_ready[j] = end
        job_op_idx[j] += 1

    return schedule


# ---------------------------------------------------------------------------
# Flow factor helpers
# ---------------------------------------------------------------------------

def _ff_from_fifo(instance):
    sched = fifo_solve(instance)
    return _compute_flow_factor(instance, sched)


def _ff_from_best_sdr(instance):
    """FF for ATC: computed from best schedule among the 8 SDRs."""
    suf = _suffix_min_p(instance['jobs'])
    ff_fifo = _ff_from_fifo(instance)
    best_twt = float('inf')
    best_sched = None
    for sched in _run_sdrs(instance, suf, ff_fifo):
        twt = _compute_twt(instance, sched)
        if twt < best_twt:
            best_twt = twt
            best_sched = sched
    return _compute_flow_factor(instance, best_sched) if best_sched else ff_fifo


# ---------------------------------------------------------------------------
# TWT helper
# ---------------------------------------------------------------------------

def _compute_twt(instance, schedule):
    jobs = instance['jobs']
    job_completion = {}
    for entry in schedule:
        j = entry['job']
        if j not in job_completion or entry['end'] > job_completion[j]:
            job_completion[j] = entry['end']
    twt = 0.0
    for j, job in enumerate(jobs):
        c = job_completion.get(j, float('inf'))
        twt += job['weight'] * max(0, c - job['due_date'])
    return twt


# ---------------------------------------------------------------------------
# Rule 1: FIFO — smallest r_v
# ---------------------------------------------------------------------------

def fifo_solve(instance, suf=None):
    def priority(j, op_idx, r_j, start, p_v, suf, p_bar):
        return r_j
    return _list_schedule(instance, priority, maximize=False, suf=suf)


# ---------------------------------------------------------------------------
# Rule 2: WFIFO — smallest r_v / w_i
# ---------------------------------------------------------------------------

def wfifo_solve(instance, suf=None):
    jobs = instance['jobs']
    def priority(j, op_idx, r_j, start, p_v, suf, p_bar):
        return r_j / jobs[j]['weight']
    return _list_schedule(instance, priority, maximize=False, suf=suf)


# ---------------------------------------------------------------------------
# Rule 3: ODD — smallest d_v = d_i - FF * Σ_{θ_v} p_u
# ---------------------------------------------------------------------------

def odd_solve(instance, ff=None, suf=None):
    if ff is None:
        ff = _ff_from_fifo(instance)
    jobs = instance['jobs']
    def priority(j, op_idx, r_j, start, p_v, suf, p_bar):
        remaining_after = suf[j][op_idx + 1]
        return jobs[j]['due_date'] - ff * remaining_after
    return _list_schedule(instance, priority, maximize=False, suf=suf)


# ---------------------------------------------------------------------------
# Rule 4: WODD — smallest d_v / w_i
# ---------------------------------------------------------------------------

def wodd_solve(instance, ff=None, suf=None):
    if ff is None:
        ff = _ff_from_fifo(instance)
    jobs = instance['jobs']
    def priority(j, op_idx, r_j, start, p_v, suf, p_bar):
        remaining_after = suf[j][op_idx + 1]
        d_v = jobs[j]['due_date'] - ff * remaining_after
        return d_v / jobs[j]['weight']
    return _list_schedule(instance, priority, maximize=False, suf=suf)


# ---------------------------------------------------------------------------
# Rule 5: MOD — smallest max(d_v, t + p_v)
# ---------------------------------------------------------------------------

def mod_solve(instance, ff=None, suf=None):
    if ff is None:
        ff = _ff_from_fifo(instance)
    jobs = instance['jobs']
    def priority(j, op_idx, r_j, start, p_v, suf, p_bar):
        remaining_after = suf[j][op_idx + 1]
        d_v = jobs[j]['due_date'] - ff * remaining_after
        return max(d_v, start + p_v)   # t = start
    return _list_schedule(instance, priority, maximize=False, suf=suf)


# ---------------------------------------------------------------------------
# Rule 6: WMOD — smallest (1/w_i) * max(d_v - t, p_v)
# ---------------------------------------------------------------------------

def wmod_solve(instance, ff=None, suf=None):
    if ff is None:
        ff = _ff_from_fifo(instance)
    jobs = instance['jobs']
    def priority(j, op_idx, r_j, start, p_v, suf, p_bar):
        remaining_after = suf[j][op_idx + 1]
        d_v = jobs[j]['due_date'] - ff * remaining_after
        return (1.0 / jobs[j]['weight']) * max(d_v - start, p_v)  # t = start
    return _list_schedule(instance, priority, maximize=False, suf=suf)


# ---------------------------------------------------------------------------
# Rule 7: SPT — largest 1/p_v
# ---------------------------------------------------------------------------

def spt_solve(instance, suf=None):
    def priority(j, op_idx, r_j, start, p_v, suf, p_bar):
        return 1.0 / p_v if p_v > 0 else float('inf')
    return _list_schedule(instance, priority, maximize=True, suf=suf)


# ---------------------------------------------------------------------------
# Rule 8: WSPT — largest w_i/p_v
# ---------------------------------------------------------------------------

def wspt_solve(instance, suf=None):
    jobs = instance['jobs']
    def priority(j, op_idx, r_j, start, p_v, suf, p_bar):
        return jobs[j]['weight'] / p_v if p_v > 0 else float('inf')
    return _list_schedule(instance, priority, maximize=True, suf=suf)


# ---------------------------------------------------------------------------
# Rule 9: ATC — largest (w_i/p_v)*exp(-(d_v + r_i - t)^+ / (κ*p̄))
# FF computed from best SDR schedule.
# ---------------------------------------------------------------------------

def atc_solve(instance, kappa, ff=None, suf=None):
    if ff is None:
        ff = _ff_from_best_sdr(instance)
    jobs = instance['jobs']
    def priority(j, op_idx, r_j, start, p_v, suf, p_bar):
        remaining_after = suf[j][op_idx + 1]
        d_v = jobs[j]['due_date'] - ff * remaining_after
        # t = start (EFT start time = decision time proxy)
        slack = d_v + (r_j - start)
        denom = kappa * p_bar if p_bar > 0 else 1e-9
        return (jobs[j]['weight'] / p_v) * math.exp(-max(slack, 0.0) / denom)
    return _list_schedule(instance, priority, maximize=True, suf=suf)


# ---------------------------------------------------------------------------
# SDR runner (rules 1–8, used internally by CS and ATC FF computation)
# ---------------------------------------------------------------------------

def _run_sdrs(instance, suf, ff):
    """Yield schedules for all 8 simple dispatching rules."""
    yield fifo_solve(instance, suf=suf)
    yield wfifo_solve(instance, suf=suf)
    yield odd_solve(instance, ff=ff, suf=suf)
    yield wodd_solve(instance, ff=ff, suf=suf)
    yield mod_solve(instance, ff=ff, suf=suf)
    yield wmod_solve(instance, ff=ff, suf=suf)
    yield spt_solve(instance, suf=suf)
    yield wspt_solve(instance, suf=suf)


# ---------------------------------------------------------------------------
# Combined Scheduler (CS)
# CS = best TWT among: 8 SDRs + ATC(κ=0.1, 0.2, ..., 7.0)
# FF for ATC = from best SDR schedule (per paper Section 5.2)
# ---------------------------------------------------------------------------

def cs_solve(instance):
    jobs = instance['jobs']
    suf = _suffix_min_p(jobs)

    # FF for ODD/MOD/WMOD/WODD = from FIFO schedule
    ff_fifo = _ff_from_fifo(instance)

    best_twt = float('inf')
    best_sched = None

    # 8 SDRs
    for sched in _run_sdrs(instance, suf, ff_fifo):
        twt = _compute_twt(instance, sched)
        if twt < best_twt:
            best_twt = twt
            best_sched = sched

    # FF for ATC = from best SDR schedule
    ff_atc = _compute_flow_factor(instance, best_sched)

    # ATC: κ = 0.1k for k = 1..70
    for k in range(1, 71):
        kappa = 0.1 * k
        sched = atc_solve(instance, kappa=kappa, ff=ff_atc, suf=suf)
        twt = _compute_twt(instance, sched)
        if twt < best_twt:
            best_twt = twt
            best_sched = sched

    return best_sched


# ---------------------------------------------------------------------------
# Default solve = CS (used when this file is an initial_program seed)
# ---------------------------------------------------------------------------

def solve(instance):
    return cs_solve(instance)


# ---------------------------------------------------------------------------
# Standalone runner — prints TWT for every rule on all eval instances
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(__file__))
    from evaluator import parse_instance, EVAL_INSTANCES

    RULES = [
        ('FIFO',  lambda inst: fifo_solve(inst)),
        ('WFIFO', lambda inst: wfifo_solve(inst)),
        ('ODD',   lambda inst: odd_solve(inst)),
        ('WODD',  lambda inst: wodd_solve(inst)),
        ('MOD',   lambda inst: mod_solve(inst)),
        ('WMOD',  lambda inst: wmod_solve(inst)),
        ('SPT',   lambda inst: spt_solve(inst)),
        ('WSPT',  lambda inst: wspt_solve(inst)),
        ('ATC',   lambda inst: atc_solve(inst, kappa=2.0)),
        ('CS',    lambda inst: cs_solve(inst)),
    ]

    results = {name: [] for name, _ in RULES}

    for path in EVAL_INSTANCES:
        if not os.path.exists(path):
            continue
        inst = parse_instance(path)
        for name, fn in RULES:
            sched = fn(inst)
            twt = _compute_twt(inst, sched)
            results[name].append(twt)

    import statistics
    print(f"\n{'Rule':<8}  {'Median TWT':>12}  {'Mean TWT':>12}  {'n':>4}")
    print('-' * 46)
    for name, twts in results.items():
        if twts:
            print(f"{name:<8}  {statistics.median(twts):>12.1f}  {sum(twts)/len(twts):>12.1f}  {len(twts):>4}")
