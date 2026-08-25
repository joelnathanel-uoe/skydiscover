"""
Insertion/DAG harness for FJSP-TWT.

The LLM-written program implements:

    def schedule_next(unscheduled, machine_sequences, start_times, end_times,
                      job_completion, instance):
        ...
        return op, machine, position

Parameters:
    unscheduled       set of (job_id, op_idx) tuples not yet placed
    machine_sequences dict of machine_id -> [(job_id, op_idx), ...] (placed only)
    start_times       dict of (job_id, op_idx) -> int
                        exact for placed ops; lower bound for unplaced
    end_times         dict of (job_id, op_idx) -> int
                        exact for placed ops; lower bound for unplaced
    job_completion    dict of job_id -> int or None (None = last op not placed)
    instance          the original FJSP instance dict
    op                (job_id, op_idx) to schedule next
    machine           machine_id to assign
    position          0-based index in machine_sequences[machine] to insert before
                        position == len(seq) means append at end

Harness guarantees:
    - Validates machine eligibility, position range, and DAG acyclicity.
    - On invalid moves: raises InvalidMoveError (program is rejected).
    - All times in start_times / end_times are non-negative integers.

Fast path:
    Appends (position == len(seq)) use O(1) incremental start-time update.
    Insertions use a full O(V+E) topological sort.
    Dispatching rules (always append) therefore run in O(n) total time.
"""

from collections import deque


# ---------------------------------------------------------------------------
# Lazy views passed to schedule_next each step (O(1) to construct)
# ---------------------------------------------------------------------------

class _TimeView:
    """Read-only dict-like view over an integer array keyed by (job, op) tuples."""
    __slots__ = ('_arr', '_op_id')
    def __init__(self, arr, op_id): self._arr = arr; self._op_id = op_id
    def __getitem__(self, op): return self._arr[self._op_id[op]]
    def get(self, op, default=None):
        i = self._op_id.get(op)
        return default if i is None else self._arr[i]
    def __contains__(self, op): return op in self._op_id
    def __iter__(self): return iter(self._op_id)
    def keys(self): return self._op_id.keys()
    def items(self):
        arr = self._arr
        return ((op, arr[i]) for op, i in self._op_id.items())


class _SeqList:
    """Read-only list-like view over an integer id sequence."""
    __slots__ = ('_seq', '_all_ops')
    def __init__(self, seq, all_ops): self._seq = seq; self._all_ops = all_ops
    def __getitem__(self, idx): return self._all_ops[self._seq[idx]]
    def __len__(self): return len(self._seq)
    def __bool__(self): return bool(self._seq)
    def __iter__(self):
        all_ops = self._all_ops
        for i in self._seq: yield all_ops[i]


class _MSeqView:
    """Read-only dict-like view over machine sequences."""
    __slots__ = ('_mseq', '_all_ops')
    def __init__(self, mseq, all_ops): self._mseq = mseq; self._all_ops = all_ops
    def __getitem__(self, m): return _SeqList(self._mseq[m], self._all_ops)
    def get(self, m, default=None):
        if not (0 <= m < len(self._mseq)): return default
        return _SeqList(self._mseq[m], self._all_ops)
    def __contains__(self, m): return 0 <= m < len(self._mseq) and bool(self._mseq[m])
    def __iter__(self): return (m for m, s in enumerate(self._mseq) if s)
    def keys(self): return [m for m, s in enumerate(self._mseq) if s]
    def items(self):
        all_ops = self._all_ops
        return ((m, _SeqList(s, all_ops)) for m, s in enumerate(self._mseq) if s)


# ---------------------------------------------------------------------------
# Harness state
# ---------------------------------------------------------------------------

class _H:
    """
    Incremental FJSP scheduling state.
    Operations are stored as integer ids for fast array access.
    """

    __slots__ = (
        'instance', 'jobs', 'n_jobs', 'n_machines',
        'all_ops', 'n', 'op_id',
        'pred_count', 'succ', 'pred_lists',
        'mseq', 'assigned', 'proc', 'min_proc',
        'release', 'st', 'et',
    )

    def __init__(self, instance):
        jobs = instance['jobs']
        n_jobs = instance['n_jobs']
        n_machines = instance['n_machines']

        self.instance = instance
        self.jobs = jobs
        self.n_jobs = n_jobs
        self.n_machines = n_machines

        # Enumerate ops in (job, op_idx) order = topological order for job edges
        ops = [(j, o) for j in range(n_jobs) for o in range(len(jobs[j]['operations']))]
        self.all_ops = ops
        self.n = len(ops)
        self.op_id = {op: i for i, op in enumerate(ops)}
        n = self.n

        # Adjacency: initialised from job-precedence edges
        self.pred_count = [0] * n
        self.succ = [[] for _ in range(n)]
        self.pred_lists = [[] for _ in range(n)]
        for j, job in enumerate(jobs):
            for o in range(len(job['operations']) - 1):
                a = self.op_id[(j, o)]
                b = self.op_id[(j, o + 1)]
                self.pred_count[b] += 1
                self.succ[a].append(b)
                self.pred_lists[b].append(a)

        self.mseq = [[] for _ in range(n_machines + 1)]
        self.assigned = [-1] * n
        self.proc = [0] * n
        self.min_proc = [
            min(p for _, p in jobs[j]['operations'][o])
            for j, o in ops
        ]
        self.release = [0] * n
        for j, job in enumerate(jobs):
            self.release[self.op_id[(j, 0)]] = job['release_time']

        # Start/end times: exact if placed, lower bound if unplaced
        self.st = [0] * n
        self.et = list(self.min_proc)
        # Propagate lower bounds through job chains (ops enumerated in topo order)
        for i in range(n):
            r = self.release[i]
            pf = max((self.et[p] for p in self.pred_lists[i]), default=r)
            s = pf if pf > r else r
            self.st[i] = s
            self.et[i] = s + self.min_proc[i]

    # ------------------------------------------------------------------
    # Edge maintenance (kept in sync for both fast-path and topo-sort path)
    # ------------------------------------------------------------------

    def _add_edge(self, a, b):
        self.pred_count[b] += 1
        self.succ[a].append(b)
        self.pred_lists[b].append(a)

    def _rem_edge(self, a, b):
        self.pred_count[b] -= 1
        sl = self.succ[a]
        for k in range(len(sl) - 1, -1, -1):
            if sl[k] == b:
                del sl[k]; break
        pl = self.pred_lists[b]
        for k in range(len(pl) - 1, -1, -1):
            if pl[k] == a:
                del pl[k]; break

    def _proc_of(self, op_id, machine):
        j, o = self.all_ops[op_id]
        for m, p in self.jobs[j]['operations'][o]:
            if m == machine:
                return p
        raise ValueError(f"machine {machine} not eligible")

    # ------------------------------------------------------------------
    # Full topological sort (used for insertions only)
    # ------------------------------------------------------------------

    def _topo_and_update(self):
        """Kahn's with local in_degree copy. Writes st/et only on success."""
        n = self.n
        in_deg = list(self.pred_count)
        succ = self.succ
        q = deque(i for i in range(n) if in_deg[i] == 0)
        order = []
        while q:
            i = q.popleft()
            order.append(i)
            for j in succ[i]:
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    q.append(j)
        if len(order) != n:
            return False

        et = self.et
        st = self.st
        rel = self.release
        assigned = self.assigned
        proc = self.proc
        min_proc = self.min_proc
        pred_lists = self.pred_lists

        for i in order:
            r = rel[i]
            pf = r
            for p in pred_lists[i]:
                v = et[p]
                if v > pf:
                    pf = v
            s = pf if pf > r else r
            st[i] = s
            et[i] = s + (proc[i] if assigned[i] >= 0 else min_proc[i])
        return True

    # ------------------------------------------------------------------
    # FAST PATH: append to end of machine sequence (dispatching rules)
    # ------------------------------------------------------------------

    def _append_commit(self, op_id, machine):
        """O(1) commit for appending at end. Always valid for dispatching rules."""
        j, o = self.all_ops[op_id]
        job = self.jobs[j]
        seq = self.mseq[machine]

        p = self._proc_of(op_id, machine)
        self.proc[op_id] = p
        self.assigned[op_id] = machine

        # Machine ordering edge: prev -> op_id
        if seq:
            self._add_edge(seq[-1], op_id)
        seq.append(op_id)

        # Start time: max(machine availability, job readiness, release)
        m_avail = self.et[seq[-2]] if len(seq) > 1 else 0
        if o > 0:
            job_ready = self.et[self.op_id[(j, o - 1)]]
        else:
            job_ready = self.release[op_id]
        r = self.release[op_id]
        s = m_avail
        if job_ready > s: s = job_ready
        if r > s: s = r
        self.st[op_id] = s
        self.et[op_id] = s + p

        # Propagate lower-bound update through unplaced job successors
        cur_et = s + p
        n_ops = len(job['operations'])
        for o2 in range(o + 1, n_ops):
            s_id = self.op_id[(j, o2)]
            if self.assigned[s_id] >= 0:
                break           # placed op — topo sort handles it; stop here
            r2 = self.release[s_id]
            new_st = cur_et if cur_et > r2 else r2
            new_et = new_st + self.min_proc[s_id]
            if new_et <= self.et[s_id]:
                break           # lower bound not tightened; successors unaffected
            self.st[s_id] = new_st
            self.et[s_id] = new_et
            cur_et = new_et

    def _append_cycle_free(self, op_id, machine):
        """Check if appending op_id to machine is cycle-free (O(n_ops_in_job))."""
        j, o = self.all_ops[op_id]
        job = self.jobs[j]
        seq = self.mseq[machine]
        # Cycle only if a job successor of op_id is already assigned to the same machine
        # AND appears before the end of the sequence (would force: op_id -> succ -> ... -> op_id)
        for o2 in range(o + 1, len(job['operations'])):
            if self.assigned[self.op_id[(j, o2)]] == machine:
                return False
        return True

    # ------------------------------------------------------------------
    # SLOW PATH: insertion at arbitrary position
    # ------------------------------------------------------------------

    def _insert_commit(self, op_id, machine, position):
        """Full cycle check + commit for insertion. Returns True if valid."""
        seq = self.mseq[machine]
        prev_id = seq[position - 1] if position > 0 else -1
        next_id = seq[position] if position < len(seq) else -1

        if prev_id >= 0 and next_id >= 0:
            self._rem_edge(prev_id, next_id)
        if prev_id >= 0:
            self._add_edge(prev_id, op_id)
        if next_id >= 0:
            self._add_edge(op_id, next_id)

        p = self._proc_of(op_id, machine)
        self.proc[op_id] = p
        self.assigned[op_id] = machine
        seq.insert(position, op_id)

        if self._topo_and_update():
            return True

        # Rollback
        seq.pop(position)
        self.proc[op_id] = 0
        self.assigned[op_id] = -1
        if next_id >= 0:
            self._rem_edge(op_id, next_id)
        if prev_id >= 0:
            self._rem_edge(prev_id, op_id)
        if prev_id >= 0 and next_id >= 0:
            self._add_edge(prev_id, next_id)
        return False

    # ------------------------------------------------------------------
    # Public placement
    # ------------------------------------------------------------------

    def try_commit(self, op, machine, position):
        op_id = self.op_id[op]
        seq = self.mseq[machine]
        if position == len(seq):
            if not self._append_cycle_free(op_id, machine):
                return False
            self._append_commit(op_id, machine)
            return True
        else:
            return self._insert_commit(op_id, machine, position)

    # ------------------------------------------------------------------
    # Views for the LLM — lazy wrappers, O(1) to construct
    # ------------------------------------------------------------------

    def start_times(self):
        return _TimeView(self.st, self.op_id)

    def end_times(self):
        return _TimeView(self.et, self.op_id)

    def machine_sequences(self):
        return _MSeqView(self.mseq, self.all_ops)

    def job_completion(self):
        et = self.et
        result = {}
        for j, job in enumerate(self.jobs):
            last_id = self.op_id[(j, len(job['operations']) - 1)]
            result[j] = et[last_id] if self.assigned[last_id] >= 0 else None
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class InvalidMoveError(Exception):
    pass


def _eligible_machines(instance, op):
    j, o = op
    return [m for m, _ in instance['jobs'][j]['operations'][o]]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_harness(instance, schedule_next_fn):
    """
    Run the insertion/DAG harness.

    Returns dict with keys: schedule, twt.
        schedule  list of {'job','op','machine','start','end'}
        twt       total weighted tardiness (float)

    Raises InvalidMoveError if schedule_next_fn returns an invalid move.
    """
    h = _H(instance)
    jobs = instance['jobs']
    all_ops = h.all_ops
    unscheduled = set(all_ops)

    while unscheduled:
        all_ops = h.all_ops
        op_id   = h.op_id
        machine_sequences = {m: [all_ops[i] for i in seq]
                             for m, seq in enumerate(h.mseq) if seq}
        start_times       = {op: h.st[i] for op, i in op_id.items()}
        end_times         = {op: h.et[i] for op, i in op_id.items()}
        job_completion    = {j: (h.et[op_id[(j, len(job['operations']) - 1)]]
                                 if h.assigned[op_id[(j, len(job['operations']) - 1)]] >= 0
                                 else None)
                             for j, job in enumerate(jobs)}

        result = schedule_next_fn(
            unscheduled=frozenset(unscheduled),
            machine_sequences=machine_sequences,
            start_times=start_times,
            end_times=end_times,
            job_completion=job_completion,
            instance=instance,
        )

        if not (isinstance(result, tuple) and len(result) == 3):
            raise InvalidMoveError(f"schedule_next must return a 3-tuple, got {type(result)}")
        op, machine, position = result

        if not (isinstance(op, tuple) and len(op) == 2
                and isinstance(op[0], int) and isinstance(op[1], int)):
            raise InvalidMoveError(f"op must be (int, int), got {type(op)}: {op}")
        if not isinstance(machine, int):
            raise InvalidMoveError(f"machine must be int, got {type(machine)}: {machine}")
        if not isinstance(position, int) or position < 0:
            raise InvalidMoveError(f"position must be non-negative int, got {position}")
        if op not in unscheduled:
            raise InvalidMoveError(f"op {op} is not in unscheduled")
        if machine not in _eligible_machines(instance, op):
            raise InvalidMoveError(f"machine {machine} not eligible for op {op}")
        if not (position <= len(h.mseq[machine])):
            raise InvalidMoveError(f"invalid position {position} for machine {machine}")
        if not h.try_commit(op, machine, position):
            raise InvalidMoveError(f"move {op} -> machine {machine} pos {position} creates a cycle")

        unscheduled.remove(op)

    schedule = []
    for i, (j, o) in enumerate(all_ops):
        m = h.assigned[i]
        s = h.st[i]
        p = h.proc[i]
        schedule.append({'job': j, 'op': o, 'machine': m, 'start': s, 'end': s + p})

    twt = 0.0
    for j, job in enumerate(jobs):
        last_id = h.op_id[(j, len(job['operations']) - 1)]
        c = h.et[last_id]
        twt += job['weight'] * max(0, c - job['due_date'])

    return {'schedule': schedule, 'twt': twt}
