# EVOLVE-BLOCK-START

def schedule_next(unscheduled, machine_sequences, start_times, end_times,
                  job_completion, instance):
    """
    Sequential seed: pick the ready operation with the smallest (job, op) index,
    assign to the machine with the earliest finish time, append at end.

    'Ready' means op 0 of any job, or any op whose job-predecessor is already placed.
    """
    jobs = instance['jobs']

    # Find the ready op with the smallest (j, o) index
    best_op = None
    for (j, o) in sorted(unscheduled):
        if o == 0 or (j, o - 1) not in unscheduled:
            best_op = (j, o)
            break

    # Fallback: if nothing is ready (shouldn't happen in normal scheduling),
    # just take the first unscheduled op
    if best_op is None:
        best_op = min(unscheduled)

    j, o = best_op
    job = jobs[j]

    # EFT machine selection
    best_m, best_eft = None, float('inf')
    for m, p in job['operations'][o]:
        seq = machine_sequences.get(m, [])
        m_avail = end_times[seq[-1]] if seq else 0
        job_ready = end_times[(j, o - 1)] if o > 0 else job['release_time']
        eft = max(m_avail, job_ready) + p
        if eft < best_eft:
            best_eft = eft
            best_m = m

    position = len(machine_sequences.get(best_m, []))
    return best_op, best_m, position

# EVOLVE-BLOCK-END

