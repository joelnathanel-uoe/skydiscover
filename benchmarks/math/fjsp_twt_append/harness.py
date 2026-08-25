"""
Appending harness for FJSP-TWT (Phase 2).

The appending scheme places every chosen operation at the *end* of its machine's
current sequence, started as early as possible. Compared with the Phase 1
insertion harness (`../fjsp_twt_insertion/harness.py`) this removes the DAG entirely:
there is no position argument, no cycle check and no topological sort, because
no placement can ever conflict with or delay an operation already committed.
Every schedule this harness produces is feasible by construction.

The harness owns the two steps the design guidelines fix — candidate generation
and update — and hands the program the candidate set:

    def choose_next(candidates, machine_sequences, machine_free, job_ready,
                    job_next_op, instance):
        ...
        return one_candidate

Arguments
    candidates        list[Candidate], one per (available operation, eligible
                        machine) pair, in deterministic order. An operation is
                        available if it is the next unscheduled operation of its
                        job. Never empty while work remains.
    machine_sequences dict m -> [(op, start, end), ...] already committed on m,
                        in processing order
    machine_free      dict m -> time m becomes free (0 if never used)
    job_ready         dict j -> time j becomes ready, i.e. the end of its last
                        committed operation, or its release time if none
    job_next_op       dict j -> index of j's next unscheduled operation,
                        or None if j is fully scheduled
    instance          the original FJSP instance dict

A Candidate is a namedtuple (op, machine, start, end, proc) where
    op       (job_id, op_idx)
    machine  machine id
    start    max(job_ready[job_id], machine_free[machine])
    end      start + proc
    proc     processing time of op on machine

The return value must be one of the candidates offered in that call. Any tuple
equal to one of them is accepted, so a program may rebuild a candidate rather
than return the object it was given; a fabricated candidate that matches none of
them is rejected.
"""

from collections import namedtuple


Candidate = namedtuple('Candidate', 'op machine start end proc')


class InvalidMoveError(Exception):
    pass


def run_harness(instance, choose_next_fn):
    """
    Run the appending harness.

    Returns dict with keys: schedule, twt.
        schedule  list of {'job','op','machine','start','end'}
        twt       total weighted tardiness (float)

    Raises InvalidMoveError if choose_next_fn returns something that is not one
    of the candidates it was offered.
    """
    jobs = instance['jobs']
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']

    # Per-job state: next unscheduled operation, and when the job becomes ready.
    job_next = [0] * n_jobs
    job_ready = [job['release_time'] for job in jobs]
    n_ops = [len(job['operations']) for job in jobs]
    remaining = sum(n_ops)

    # Per-machine state: committed sequence and free time. Machine ids in the
    # instances are 0-based but the eligible lists are trusted rather than
    # validated here, so size defensively.
    mseq = [[] for _ in range(n_machines + 1)]
    mfree = [0] * (n_machines + 1)

    schedule = []

    while remaining:
        # --- Candidate generation (fixed by the scheme) --------------------
        candidates = []
        for j in range(n_jobs):
            o = job_next[j]
            if o >= n_ops[j]:
                continue
            ready = job_ready[j]
            for m, p in jobs[j]['operations'][o]:
                s = ready if ready > mfree[m] else mfree[m]
                candidates.append(Candidate((j, o), m, s, s + p, p))

        if not candidates:
            # Unreachable: every job with work left contributes its next
            # operation, and every operation has at least one eligible machine.
            raise InvalidMoveError("no candidates while operations remain")

        views = {
            'candidates': candidates,
            'machine_sequences': {m: list(seq) for m, seq in enumerate(mseq) if seq},
            'machine_free': {m: mfree[m] for m in range(n_machines)},
            'job_ready': {j: job_ready[j] for j in range(n_jobs)},
            'job_next_op': {j: (job_next[j] if job_next[j] < n_ops[j] else None)
                            for j in range(n_jobs)},
            'instance': instance,
        }

        chosen = choose_next_fn(**views)

        # --- Validation ----------------------------------------------------
        # Equality rather than identity, so a program may rebuild the tuple it
        # wants to return. A candidate that matches none of the offered ones
        # cannot be honoured: its start time would be wrong.
        if not (isinstance(chosen, tuple) and len(chosen) == 5):
            raise InvalidMoveError(
                f"choose_next must return one of the candidates "
                f"(a 5-tuple), got {type(chosen).__name__}: {chosen!r}")
        if chosen not in candidates:
            raise InvalidMoveError(
                f"returned candidate {tuple(chosen)!r} was not among the "
                f"{len(candidates)} candidates offered this step")

        # --- Update (fixed by the scheme) ----------------------------------
        (j, o), m, s, e, p = chosen
        mseq[m].append(((j, o), s, e))
        mfree[m] = e
        job_ready[j] = e
        job_next[j] = o + 1
        remaining -= 1
        schedule.append({'job': j, 'op': o, 'machine': m, 'start': s, 'end': e})

    twt = 0.0
    for j, job in enumerate(jobs):
        twt += job['weight'] * max(0, job_ready[j] - job['due_date'])

    return {'schedule': schedule, 'twt': twt}
