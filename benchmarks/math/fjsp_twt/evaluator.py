"""
Evaluator for FJSP-TWT (Flexible Job Shop Scheduling Problem with Total Weighted Tardiness).

Programs implement:
    schedule_next(unscheduled, machine_sequences, start_times, end_times,
                  job_completion, instance) -> (op, machine, position)

The harness calls this repeatedly to build a complete schedule one placement at a time.

Scoring: median over instances of (cs_twt - heuristic_twt) / cs_twt
  - 0.0  = ties CS baseline
  - > 0  = beats CS (better)
  - < 0  = worse than CS

Baseline: Combined Scheduler (CS) from Sobeyko & Mönch (2016) — runs 8 SDRs +
ATC(κ=0.1..7.0) and returns the schedule with the lowest TWT.
"""

import json
import os
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
import traceback


INSTANCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instances')
GENERATED_DIR = os.path.join(INSTANCES_DIR, 'generated')
HARNESS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness.py')

# Use the first N_EVAL generated instances for evaluation.
# Instances are sorted by filename so the set is deterministic.
N_EVAL = 100

def _load_generated_instances():
    if not os.path.isdir(GENERATED_DIR):
        return []
    files = sorted(f for f in os.listdir(GENERATED_DIR) if f.endswith('.txt'))
    return [os.path.join(GENERATED_DIR, f) for f in files[:N_EVAL]]

EVAL_INSTANCES = _load_generated_instances()

# Per-instance cumulative schedule_next time budget (seconds).
INSTANCE_TIMEOUT_S    = 5
# Per-call timeout = PER_CALL_K_S * n_feasible_pairs + PER_CALL_BASE_S
PER_CALL_K_S          = 0.0001   # seconds per feasible (ready_op, machine) pair
PER_CALL_BASE_S       = 0.010    # minimum per-call budget (10ms)
# Subprocess wall-clock fallback — hung-process guard only.
SUBPROCESS_FALLBACK_S = 60


# ---------------------------------------------------------------------------
# Instance parsing
# ---------------------------------------------------------------------------

def parse_instance(filepath):
    """Parse standard FJSP instance with inline release/due/weight per job."""
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


def parse_generated(filepath):
    """Parse generated FJSP instances (DUE_DATES:/WEIGHTS: footer format)."""
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    n_jobs, n_machines = map(int, lines[0].split())
    due_dates = list(map(int, lines[-2].replace('DUE_DATES:', '').split()))
    weights   = list(map(int, lines[-1].replace('WEIGHTS:', '').split()))

    jobs = []
    for i in range(1, n_jobs + 1):
        tokens = list(map(int, lines[i].split()))
        pos = 0
        n_ops = tokens[pos]; pos += 1
        operations = []
        for _ in range(n_ops):
            n_elig = tokens[pos]; pos += 1
            elig = []
            for _ in range(n_elig):
                m = tokens[pos]; pos += 1
                p = tokens[pos]; pos += 1
                elig.append([m, p])
            operations.append(elig)
        jobs.append({
            'id': i - 1,
            'release_time': 0,
            'due_date': due_dates[i - 1],
            'weight': float(weights[i - 1]),
            'operations': operations,
        })

    return {'n_jobs': n_jobs, 'n_machines': n_machines, 'jobs': jobs}


# ---------------------------------------------------------------------------
# Schedule utilities (kept for compatibility)
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
# CS baseline precomputation (lazy + disk-cached)
# ---------------------------------------------------------------------------

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cs_baselines.pkl')
_CS_BASELINES = None


def _build_cs_baselines():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from baselines import cs_solve
    baselines = {}
    for path in EVAL_INSTANCES:
        if not os.path.exists(path):
            continue
        try:
            inst = parse_generated(path)
            sched = cs_solve(inst)
            baselines[path] = compute_twt(inst, sched)
        except Exception:
            pass
    return baselines


def _get_cs_baselines():
    global _CS_BASELINES
    if _CS_BASELINES is not None:
        return _CS_BASELINES
    # Try loading from disk cache
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, 'rb') as f:
                _CS_BASELINES = pickle.load(f)
            return _CS_BASELINES
        except Exception:
            pass
    # Compute and save
    print("Computing CS baselines (first run only)...", flush=True)
    _CS_BASELINES = _build_cs_baselines()
    try:
        with open(_CACHE_PATH, 'wb') as f:
            pickle.dump(_CS_BASELINES, f)
    except Exception:
        pass
    return _CS_BASELINES


# ---------------------------------------------------------------------------
# Subprocess runner — calls run_harness(instance, program.schedule_next)
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

    import time as _time, signal as _signal

    _K      = {PER_CALL_K_S}
    _BASE   = {PER_CALL_BASE_S}
    _BUDGET = {INSTANCE_TIMEOUT_S}

    _t_import = _time.perf_counter()

    # Load harness
    harness_spec = importlib.util.spec_from_file_location("harness", '{HARNESS_PATH}')
    harness_mod = importlib.util.module_from_spec(harness_spec)
    harness_spec.loader.exec_module(harness_mod)

    # Load program
    spec = importlib.util.spec_from_file_location("program", '{program_path}')
    program = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname('{program_path}'))
    spec.loader.exec_module(program)

    def _count_feasible(unscheduled, instance):
        jobs = instance['jobs']
        return sum(
            len(jobs[j]['operations'][o])
            for j, o in unscheduled
            if o == 0 or (j, o - 1) not in unscheduled
        )

    _elapsed    = [0.0]
    _call_count = [0]
    _max_call   = [0.0]
    _call_times = []
    _fn         = program.schedule_next

    def _alarm(sig, frame):
        raise TimeoutError(f"schedule_next exceeded per-call limit")

    _signal.signal(_signal.SIGALRM, _alarm)

    def _timed(**kwargs):
        n_feas = _count_feasible(kwargs['unscheduled'], kwargs['instance'])
        limit  = _K * n_feas + _BASE
        _signal.setitimer(_signal.ITIMER_REAL, limit)
        _t0 = _time.perf_counter()
        try:
            r = _fn(**kwargs)
        finally:
            _signal.setitimer(_signal.ITIMER_REAL, 0)
        dt = _time.perf_counter() - _t0
        _elapsed[0]    += dt
        _call_count[0] += 1
        _max_call[0]    = max(_max_call[0], dt)
        _call_times.append(dt)
        if _elapsed[0] > _BUDGET:
            raise TimeoutError(f"cumulative schedule_next exceeded {{_BUDGET}}s")
        return r

    def _checked(**kwargs):
        if _fn.__dict__:
            raise ValueError(f"schedule_next has private attributes before call: {{list(_fn.__dict__.keys())}}")
        kwargs['instance'] = dict(kwargs['instance'])
        r = _timed(**kwargs)
        if _fn.__dict__:
            raise ValueError(f"schedule_next set private attributes: {{list(_fn.__dict__.keys())}}")
        return r

    _t_harness_start = _time.perf_counter()
    result = harness_mod.run_harness(instance, _checked)
    _t_harness_end   = _time.perf_counter()

    timing = {{
        'import_s':   _t_harness_start - _t_import,
        'harness_s':  _t_harness_end - _t_harness_start,
        'call_s':     _elapsed[0],
        'n_calls':    _call_count[0],
        'max_call_s': _max_call[0],
        'call_times': _call_times,
    }}

    with open('{results_path}', 'wb') as f:
        pickle.dump({{'result': result, 'timing': timing}}, f)

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
                data = pickle.load(f)
            if 'error' in data:
                raise RuntimeError(f"Program error: {data['error']}")
            result = data['result']
            result['timing'] = data.get('timing', {})
            return result
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
        inst = parse_generated(instance_path)
        cs_twt = _get_cs_baselines().get(instance_path)
        if cs_twt is None:
            return 0.0, f"{os.path.basename(instance_path)}: no CS baseline"

        result = run_program_on_instance(program_path, inst, timeout_seconds=SUBPROCESS_FALLBACK_S)
        schedule = result['schedule']
        valid, msg = validate_schedule(inst, schedule)
        if not valid:
            return -100.0, f"{os.path.basename(instance_path)}: {msg}"

        twt = compute_twt(inst, schedule)
        if cs_twt == 0 and twt == 0:
            return 1.0, None
        if cs_twt == 0:
            return 0.0, None
        return (cs_twt - twt) / cs_twt, None
    except Exception as e:
        return -100.0, f"{os.path.basename(instance_path)}: {e}"


def evaluate(program_path):
    start_time = time.time()

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

    scores = []
    for path in EVAL_INSTANCES:
        if not os.path.exists(path):
            continue
        score, err = _score_instance(program_path, path)
        if err:
            _cleanup()
            return {'validity': -1,
                    'eval_time': time.time() - start_time,
                    'error': err}
        scores.append(score)

    _cleanup()

    if not scores:
        return {'validity': -1,
                'eval_time': time.time() - start_time,
                'error': 'No instances evaluated'}

    return {'combined_score': statistics.median(scores)}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else __file__.replace('evaluator.py', 'initial_program_harness.py')
    result = evaluate(path)
    for k, v in result.items():
        print(f"  {k}: {v}")
