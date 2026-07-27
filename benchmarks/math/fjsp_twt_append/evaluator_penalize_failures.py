"""
Evaluator for FJSP-TWT under the appending scheme (Phase 2).

Programs implement:
    choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance) -> one of the candidates

The harness (`harness.py` in this directory) generates the candidate set each
step, appends the returned candidate to its machine, and updates the state.

This is the Phase 1 `evaluator_penalize_failures.py` with four changes:
  1. it drives the appending harness in this directory;
  2. it calls `choose_next` rather than `schedule_next`;
  3. the per-call budget counts the candidates the harness actually built,
     which is the same quantity Phase 1 counted as "feasible (ready op,
     machine) pairs" - so the budget per step is identical;
  4. instances and CS baselines are read from the Phase 1 benchmark rather
     than copied, so scores from the two phases are directly comparable and
     the held-out test set stays held out.

Everything else - scoring, the -100 whole-program penalty for any failing
instance, concurrent evaluation, CPU-time budgeting, the metric set - is
unchanged from Phase 1.

Scoring: median over instances of (cs_twt - heuristic_twt) / cs_twt
  - 0.0  = ties CS baseline
  - > 0  = beats CS (better)
  - < 0  = worse than CS
"""

import json
import os
import pickle
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor


HERE = os.path.dirname(os.path.abspath(__file__))

# Instances and CS baselines are SHARED with Phase 1, not copied. normpath so
# the resolved paths are byte-identical to the ones Phase 1 used as keys in the
# baseline cache below.
PHASE1_DIR    = os.path.normpath(os.path.join(HERE, '..', 'fjsp_twt'))
INSTANCES_DIR = os.path.join(PHASE1_DIR, 'instances')
GENERATED_DIR = os.path.join(INSTANCES_DIR, 'generated')

HARNESS_PATH  = os.path.join(HERE, 'harness.py')

# Use the first N_EVAL generated instances for evaluation.
# Instances are sorted by filename so the set is deterministic.
N_EVAL = 50

def _load_generated_instances():
    if not os.path.isdir(GENERATED_DIR):
        return []
    files = sorted(f for f in os.listdir(GENERATED_DIR) if f.endswith('.txt'))
    return [os.path.join(GENERATED_DIR, f) for f in files[:N_EVAL]]

EVAL_INSTANCES = _load_generated_instances()

# Per-call timeout = PER_CALL_K_S * n_candidates + PER_CALL_BASE_S.
# Same constants as Phase 1: there n_candidates was counted separately as the
# number of feasible (ready op, machine) pairs; here the harness has already
# built exactly that list.
PER_CALL_K_S          = 0.0001   # seconds per candidate
PER_CALL_BASE_S       = 0.010    # minimum per-call budget (10ms)
# Subprocess wall-clock fallback — hung-process guard only.
SUBPROCESS_FALLBACK_S = 60

# Fixed penalty applied to the whole program if any single instance fails
# (timeout, invalid schedule, or error).
FAILED_PROGRAM_SCORE = -100.0

# Instances are independent, so they are evaluated concurrently. Each one is
# already its own subprocess, so threads only wait on those - the GIL is not a
# constraint. Safe only because the per-call timeout is CPU-time based
# (ITIMER_VIRTUAL): under a wall-clock timer, concurrency would spuriously kill
# slow-scheduled programs.
def _default_workers():
    try:
        return max(1, min(8, (os.cpu_count() or 2) - 2))
    except Exception:
        return 4

MAX_WORKERS = int(os.environ.get('FJSP_EVAL_WORKERS', '0')) or _default_workers()

# Print failing subprocesses' full tracebacks. Off by default - see the call site.
VERBOSE_STDERR = os.environ.get('FJSP_EVAL_VERBOSE', '') not in ('', '0')


def _failed_objectives():
    """Worst possible value for every objective this evaluator reports.

    fraction_positive is bounded below by 0.0, so that is its floor; the other
    two use the standard failure penalty.
    """
    return {
        'combined_score':    FAILED_PROGRAM_SCORE,
        'q10_score':         FAILED_PROGRAM_SCORE,
        'fraction_positive': 0.0,
    }


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
    """Full independent feasibility check.

    The appending harness cannot produce an infeasible schedule, so this should
    never fail - it is kept as an independent check on the harness itself
    rather than on the program, and is identical to the Phase 1 version so a
    schedule from either phase is judged by the same code.
    """
    jobs = instance['jobs']

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
# CS baseline — SHARED with Phase 1, read from its cache
# ---------------------------------------------------------------------------
# Pointing at Phase 1's cache rather than building a second one guarantees both
# phases are scored against bit-identical baselines. The cache is keyed by
# absolute instance path, and EVAL_INSTANCES above resolves to the same paths.

_CACHE_PATH = os.path.join(PHASE1_DIR, '.cs_baselines.pkl')
_CS_BASELINES = None


def _build_cs_baselines():
    sys.path.insert(0, PHASE1_DIR)          # baselines.py lives in Phase 1
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
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, 'rb') as f:
                _CS_BASELINES = pickle.load(f)
            return _CS_BASELINES
        except Exception:
            pass
    print("Computing CS baselines (first run only)...", flush=True)
    _CS_BASELINES = _build_cs_baselines()
    try:
        with open(_CACHE_PATH, 'wb') as f:
            pickle.dump(_CS_BASELINES, f)
    except Exception:
        pass
    return _CS_BASELINES


# ---------------------------------------------------------------------------
# Subprocess runner — calls run_harness(instance, program.choose_next)
# ---------------------------------------------------------------------------

def run_program_on_instance(program_path, instance, timeout_seconds=60):
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        json.dump(instance, f)
        instance_path = f.name

    results_path = instance_path + '.results'

    script = f"""
import os as _os
# Pin BLAS/OpenMP to a single thread. This must happen before numpy is
# imported. Two reasons: (1) the per-call budget is charged in CPU time summed
# over all threads of the process, so a multi-threaded numpy call would burn
# the budget N times faster and be killed spuriously; (2) it stops N concurrent
# instances each spawning their own BLAS pool and oversubscribing the machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_var] = "1"

import sys, os, json, pickle, traceback, importlib.util
import numpy  # pre-import so a cold `import numpy` inside choose_next
              # (running under the per-call timer) doesn't blow the budget

try:
    with open('{instance_path}') as f:
        instance = json.load(f)

    import time as _time, signal as _signal

    _K      = {PER_CALL_K_S}
    _BASE   = {PER_CALL_BASE_S}

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

    _call_count = [0]
    _max_call   = [0.0]
    _call_times = []
    _fn         = program.choose_next

    def _alarm(sig, frame):
        raise TimeoutError(f"choose_next exceeded per-call limit")

    # ITIMER_VIRTUAL charges CPU time actually burned by this process rather
    # than wall clock, so a program is never killed merely for being
    # descheduled while sibling instances evaluate concurrently. It delivers
    # SIGVTALRM, not SIGALRM.
    _signal.signal(_signal.SIGVTALRM, _alarm)

    def _timed(**kwargs):
        # The harness has already built the candidate list, so the budget is
        # read straight off it rather than recounted.
        limit = _K * len(kwargs['candidates']) + _BASE
        _signal.setitimer(_signal.ITIMER_VIRTUAL, limit)
        _t0 = _time.process_time()
        try:
            r = _fn(**kwargs)
        finally:
            _signal.setitimer(_signal.ITIMER_VIRTUAL, 0)
        dt = _time.process_time() - _t0
        _call_count[0] += 1
        _max_call[0]    = max(_max_call[0], dt)
        _call_times.append(dt)
        return r

    def _checked(**kwargs):
        if _fn.__dict__:
            raise ValueError(f"choose_next has private attributes before call: {{list(_fn.__dict__.keys())}}")
        kwargs['instance'] = dict(kwargs['instance'])
        r = _timed(**kwargs)
        if _fn.__dict__:
            raise ValueError(f"choose_next set private attributes: {{list(_fn.__dict__.keys())}}")
        return r

    _t_harness_start = _time.perf_counter()
    result = harness_mod.run_harness(instance, _checked)
    _t_harness_end   = _time.perf_counter()

    timing = {{
        'import_s':   _t_harness_start - _t_import,
        'harness_s':  _t_harness_end - _t_harness_start,
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
            if stderr and VERBOSE_STDERR:
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

    paths = [p for p in EVAL_INSTANCES if os.path.exists(p)]

    # Prime the CS baseline cache before any worker starts, so threads don't
    # race to load the same pickle.
    _get_cs_baselines()

    abort = threading.Event()
    first_error = [None]
    scores = []
    scores_lock = threading.Lock()

    def _work(path):
        if abort.is_set():
            return
        score, err = _score_instance(program_path, path)
        if err:
            abort.set()
            with scores_lock:
                if first_error[0] is None:
                    first_error[0] = err
            return
        with scores_lock:
            scores.append(score)

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    try:
        list(executor.map(_work, paths))
    finally:
        executor.shutdown(wait=True)

    if first_error[0] is not None:
        _cleanup()
        # Penalize the whole program instead of dropping it, so later
        # iterations can see that this approach failed and why. Every
        # objective must get an explicitly terrible value: UnifiedArchive
        # reads objectives as metrics.get(key, 0.0), so an omitted objective
        # becomes 0.0 and would beat a legitimate program scoring below zero.
        return {**_failed_objectives(),
                'eval_time': time.time() - start_time,
                'error': first_error[0]}

    _cleanup()

    if not scores:
        return {**_failed_objectives(),
                'eval_time': time.time() - start_time,
                'error': 'No instances evaluated'}

    scores.sort()
    n = len(scores)
    return {
        'combined_score':    statistics.median(scores),          # fitness
        'q10_score':         scores[max(0, int(0.10 * n))],       # worst-decile robustness
        'fraction_positive': sum(1 for s in scores if s > 0) / n, # how often it beats CS
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, 'seeds', 'initial_program_append.py')
    result = evaluate(path)
    for k, v in result.items():
        print(f"  {k}: {v}")
