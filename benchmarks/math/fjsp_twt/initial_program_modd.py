# EVOLVE-BLOCK-START

def solve(instance):
    jobs = instance['jobs']
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    machine_available = [0] * (n_machines + 1)
    job_op_idx = [0] * n_jobs
    job_ready = [jobs[j]['release_time'] for j in range(n_jobs)]
    schedule = []
    total_ops = sum(len(jobs[j]['operations']) for j in range(n_jobs))

    # Precompute op_due[j][k] = job_due - sum(min_p of ops after k) — static, computed once.
    op_due = []
    for j in range(n_jobs):
        job = jobs[j]
        ops = job['operations']
        n_ops = len(ops)
        suffix_min_p = [0] * (n_ops + 1)
        for k in range(n_ops - 1, -1, -1):
            suffix_min_p[k] = suffix_min_p[k + 1] + min(p for _, p in ops[k])
        op_due.append([job['due_date'] - (suffix_min_p[k + 1]) for k in range(n_ops)])

    while len(schedule) < total_ops:
        candidates = []
        for j in range(n_jobs):
            op_idx = job_op_idx[j]
            if op_idx >= len(jobs[j]['operations']):
                continue
            op = jobs[j]['operations'][op_idx]
            best_end, best_start, best_m = float('inf'), None, None
            for m, p in op:
                start = max(machine_available[m], job_ready[j])
                end = start + p
                if end < best_end:
                    best_end, best_start, best_m = end, start, m
            candidates.append((op_due[j][op_idx], j, op_idx, best_m, best_start, best_end))
        if not candidates:
            break
        _, j, op_idx, m, start, end = min(candidates, key=lambda x: x[0])
        schedule.append({'job': j, 'op': op_idx, 'machine': m, 'start': start, 'end': end})
        machine_available[m] = end
        job_ready[j] = end
        job_op_idx[j] += 1
    return schedule

# EVOLVE-BLOCK-END


def get_schedule(instance):
    return solve(instance)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from evaluator import evaluate

    result = evaluate(__file__)
    for k, v in result.items():
        print(f"  {k}: {v}")
