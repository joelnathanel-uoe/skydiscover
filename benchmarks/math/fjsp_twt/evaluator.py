"""
Evaluator for FJSP-TWT (Flexible Job Shop Scheduling Problem with Total Weighted Tardiness).

Scoring: combined_score = mean over instances of modd_twt / (modd_twt + heuristic_twt)
  - 1.0 = zero tardiness (perfect)
  - 0.5 = matches MODD baseline
  - 0.0 = heuristic TWT >> baseline

Baseline: MODD (Modified Operation Due Date) — assigns each operation a deadline by
propagating the job due date backwards: op_due = job_due - sum(min_p of all later ops).
Dispatches the operation with the earliest op_due. EFT machine selection.
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback


INSTANCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instances')

def _load_all_instances():
    tight, moderate, loose, unlabelled = [], [], [], []
    for set_name in ['Set_1', 'Set_2', 'Set_3', 'Set_4']:
        folder = os.path.join(INSTANCES_DIR, set_name)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.fjs'):
                continue
            path = os.path.join(folder, fname)
            if 'Tight' in fname:
                tight.append(path)
            elif 'Moderate' in fname or 'Mod' in fname:
                moderate.append(path)
            elif 'Loose' in fname:
                loose.append(path)
            else:
                unlabelled.append(path)
    return tight, moderate, loose, unlabelled

TIGHT_INSTANCES, MODERATE_INSTANCES, LOOSE_INSTANCES, UNLABELLED_INSTANCES = _load_all_instances()
EVAL_INSTANCES = TIGHT_INSTANCES + MODERATE_INSTANCES + LOOSE_INSTANCES + UNLABELLED_INSTANCES

# Per-instance timeout for the generated program (seconds).
INSTANCE_TIMEOUT_S = 1

# Pre-computed WSPT baselines — computed once at import time so we don't
# re-run WSPT on every program evaluation.
def _precompute_wspt_baselines():
    baselines = {}
    for path in EVAL_INSTANCES:
        if os.path.exists(path):
            inst = parse_instance(path)
            sched = wspt_solve(inst)
            baselines[path] = compute_twt(inst, sched)
    return baselines


# ---------------------------------------------------------------------------
# Instance parsing
# ---------------------------------------------------------------------------

def parse_instance(filepath):
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    first = list(map(int, lines[0].split()))
    n_jobs, n_machines = first[0], first[1]

    jobs = []
    for i in range(1, n_jobs + 1):
        tokens = list(map(int, lines[i].split()))
        pos = 0
        n_ops = tokens[pos]; pos += 1

        operations = []
        for _ in range(n_ops):
            n_eligible = tokens[pos]; pos += 1
            eligible = []
            for _ in range(n_eligible):
                machine_id = tokens[pos]; pos += 1
                proc_time = tokens[pos]; pos += 1
                eligible.append([machine_id, proc_time])
            operations.append(eligible)

        release_time = tokens[pos]; pos += 1
        due_date = tokens[pos]; pos += 1
        weight = float(tokens[pos]); pos += 1

        jobs.append({
            'id': i - 1,
            'release_time': release_time,
            'due_date': due_date,
            'weight': weight,
            'operations': operations,
        })

    return {'n_jobs': n_jobs, 'n_machines': n_machines, 'jobs': jobs}


# ---------------------------------------------------------------------------
# Schedule utilities
# ---------------------------------------------------------------------------

def compute_twt(instance, schedule):
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


def validate_schedule(instance, schedule):
    jobs = instance['jobs']
    n_machines = instance['n_machines']

    lookup = {}
    for entry in schedule:
        key = (entry['job'], entry['op'])
        if key in lookup:
            return False, f"Duplicate operation {key}"
        lookup[key] = entry

    for j, job in enumerate(jobs):
        for o in range(len(job['operations'])):
            if (j, o) not in lookup:
                return False, f"Operation ({j},{o}) not scheduled"

    for j, job in enumerate(jobs):
        if (j, 0) in lookup:
            if lookup[(j, 0)]['start'] < job['release_time']:
                return False, f"Job {j} violates release time"
        for o in range(1, len(job['operations'])):
            if lookup[(j, o)]['start'] < lookup[(j, o - 1)]['end']:
                return False, f"Precedence violation job {j} op {o}"
        for o, op_machines in enumerate(job['operations']):
            entry = lookup[(j, o)]
            eligible = {m: t for m, t in op_machines}
            if entry['machine'] not in eligible:
                return False, f"Machine {entry['machine']} not eligible for job {j} op {o}"
            if entry['end'] - entry['start'] != eligible[entry['machine']]:
                return False, f"Wrong processing time job {j} op {o}"

    machine_intervals = {}
    for entry in schedule:
        m = entry['machine']
        machine_intervals.setdefault(m, []).append((entry['start'], entry['end']))
    for m, intervals in machine_intervals.items():
        intervals.sort()
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i - 1][1]:
                return False, f"Machine {m} conflict"

    return True, "OK"


# ---------------------------------------------------------------------------
# WSPT baseline (runs inline for normalization)
# ---------------------------------------------------------------------------

def wspt_solve(instance):
    jobs = instance['jobs']
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']

    machine_available = [0] * (n_machines + 1)
    job_op_idx = [0] * n_jobs
    job_ready = [jobs[j]['release_time'] for j in range(n_jobs)]
    schedule = []
    total_ops = sum(len(jobs[j]['operations']) for j in range(n_jobs))

    while len(schedule) < total_ops:
        candidates = []
        for j in range(n_jobs):
            op_idx = job_op_idx[j]
            if op_idx >= len(jobs[j]['operations']):
                continue
            job = jobs[j]
            best_end, best_start, best_machine, best_proc = float('inf'), None, None, None
            for machine_id, proc_time in job['operations'][op_idx]:
                start = max(machine_available[machine_id], job_ready[j])
                end = start + proc_time
                if end < best_end:
                    best_end, best_start, best_machine, best_proc = end, start, machine_id, proc_time
            if best_machine is None:
                continue
            candidates.append({
                'job': j, 'op': op_idx, 'machine': best_machine,
                'start': best_start, 'end': best_end,
                'priority': job['weight'] / best_proc,
            })
        if not candidates:
            break
        best = max(candidates, key=lambda x: x['priority'])
        schedule.append({'job': best['job'], 'op': best['op'],
                         'machine': best['machine'], 'start': best['start'], 'end': best['end']})
        machine_available[best['machine']] = best['end']
        job_ready[best['job']] = best['end']
        job_op_idx[best['job']] += 1

    return schedule


def modd_solve(instance):
    """MODD — Modified Operation Due Date.
    Each operation gets a deadline: op_due = job_due - sum(min_p of all later ops).
    Dispatches the operation with the earliest op_due. EFT machine selection.
    """
    jobs = instance['jobs']
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    machine_available = [0] * (n_machines + 1)
    job_op_idx = [0] * n_jobs
    job_ready = [jobs[j]['release_time'] for j in range(n_jobs)]
    schedule = []
    total_ops = sum(len(jobs[j]['operations']) for j in range(n_jobs))

    while len(schedule) < total_ops:
        candidates = []
        for j in range(n_jobs):
            op_idx = job_op_idx[j]
            if op_idx >= len(jobs[j]['operations']):
                continue
            job = jobs[j]
            op = job['operations'][op_idx]

            best_end, best_start, best_machine = float('inf'), None, None
            for machine_id, proc_time in op:
                start = max(machine_available[machine_id], job_ready[j])
                end = start + proc_time
                if end < best_end:
                    best_end, best_start, best_machine = end, start, machine_id

            future_ops = job['operations'][op_idx + 1:]
            p_remain_after = sum(min(pt for _, pt in fop) for fop in future_ops)
            op_due = job['due_date'] - p_remain_after

            candidates.append({
                'job': j, 'op': op_idx,
                'machine': best_machine,
                'start': best_start, 'end': best_end,
                'priority': -op_due,
            })

        if not candidates:
            break
        best = max(candidates, key=lambda x: x['priority'])
        schedule.append({'job': best['job'], 'op': best['op'],
                         'machine': best['machine'], 'start': best['start'], 'end': best['end']})
        machine_available[best['machine']] = best['end']
        job_ready[best['job']] = best['end']
        job_op_idx[best['job']] += 1

    return schedule


def _precompute_modd_baselines():
    baselines = {}
    for path in EVAL_INSTANCES:
        if os.path.exists(path):
            inst = parse_instance(path)
            sched = modd_solve(inst)
            baselines[path] = compute_twt(inst, sched)
    return baselines


MODD_BASELINES = _precompute_modd_baselines()


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_program_on_instance(program_path, instance, timeout_seconds=60):
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        json.dump(instance, f)
        instance_path = f.name

    results_path = instance_path + '.results'

    script = f"""
import sys, os, json, pickle, traceback, importlib.util

try:
    with open('{instance_path}') as f:
        instance = json.load(f)

    spec = importlib.util.spec_from_file_location("program", '{program_path}')
    program = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname('{program_path}'))
    spec.loader.exec_module(program)

    schedule = program.solve(instance)
    with open('{results_path}', 'wb') as f:
        pickle.dump({{'schedule': schedule}}, f)

except Exception as e:
    traceback.print_exc()
    with open('{results_path}', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
"""

    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
        f.write(script)
        script_path = f.name

    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}", flush=True)
            if process.returncode != 0:
                raise RuntimeError(f"Process exited with code {process.returncode}")
            if not os.path.exists(results_path):
                raise RuntimeError("Results file not found")
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
            if 'error' in results:
                raise RuntimeError(f"Program error: {results['error']}")
            return results['schedule']
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError(f"Timeout after {timeout_seconds}s")
    finally:
        for path in [script_path, instance_path, results_path]:
            try:
                os.unlink(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def _score_instance(program_path, instance_path):
    """Run heuristic on one instance, return (score, error_msg)."""
    try:
        instance = parse_instance(instance_path)
        modd_twt = MODD_BASELINES[instance_path]
        schedule = run_program_on_instance(program_path, instance, timeout_seconds=INSTANCE_TIMEOUT_S)
        valid, msg = validate_schedule(instance, schedule)
        if not valid:
            return 0.0, f"{os.path.basename(instance_path)}: {msg}"
        twt = compute_twt(instance, schedule)
        if modd_twt == 0 and twt == 0:
            return 1.0, None
        elif modd_twt == 0:
            return 0.0, None
        return modd_twt / (modd_twt + twt), None
    except Exception as e:
        return 0.0, f"{os.path.basename(instance_path)}: {e}"


def evaluate(program_path):
    start_time = time.time()

    # Copy to a stable path we own — the framework may delete the original
    # temp file while our subprocesses are still running (race condition).
    import shutil, tempfile as _tf
    with _tf.NamedTemporaryFile(suffix='.py', delete=False) as _f:
        stable_path = _f.name
    shutil.copy2(program_path, stable_path)
    program_path = stable_path

    def _cleanup():
        try:
            os.unlink(stable_path)
        except OSError:
            pass

    def _run_group(instance_paths):
        """Score all instances in a group. Returns (scores, error_msg) where
        error_msg is set on the first failure and evaluation stops immediately."""
        scores = []
        for path in instance_paths:
            if os.path.exists(path):
                score, err = _score_instance(program_path, path)
                if err:
                    return None, err
                scores.append(score)
        return scores, None

    FAIL = {
        'combined_score': 0.0,
    }

    tight_scores, err = _run_group(TIGHT_INSTANCES)
    if err:
        _cleanup()
        return {**FAIL, 'eval_time': time.time() - start_time, 'errors': err}

    moderate_scores, err = _run_group(MODERATE_INSTANCES)
    if err:
        _cleanup()
        return {**FAIL, 'eval_time': time.time() - start_time, 'errors': err}

    loose_scores, err = _run_group(LOOSE_INSTANCES)
    if err:
        _cleanup()
        return {**FAIL, 'eval_time': time.time() - start_time, 'errors': err}

    unlabelled_scores, err = _run_group(UNLABELLED_INSTANCES)
    if err:
        _cleanup()
        return {**FAIL, 'eval_time': time.time() - start_time, 'errors': err}

    _cleanup()
    eval_time = time.time() - start_time

    all_scores = tight_scores + moderate_scores + loose_scores + unlabelled_scores
    if not all_scores:
        return {'combined_score': 0.0, 'validity': 0.0, 'eval_time': eval_time,
                'error': 'No instances evaluated'}

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        'combined_score': _mean(all_scores),
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else __file__.replace('evaluator.py', 'initial_program.py')
    result = evaluate(path)
    for k, v in result.items():
        print(f"  {k}: {v}")
