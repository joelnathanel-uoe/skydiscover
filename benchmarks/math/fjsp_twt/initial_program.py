# WSPT-like seed (weight / min_proc_time — no due date awareness):
# def solve(instance):
#     jobs = instance['jobs']
#     n_jobs = instance['n_jobs']
#     n_machines = instance['n_machines']
#     machine_available = [0] * (n_machines + 1)
#     job_op_idx = [0] * n_jobs
#     job_ready = [jobs[j]['release_time'] for j in range(n_jobs)]
#     schedule = []
#     total_ops = sum(len(j['operations']) for j in jobs)
#     while len(schedule) < total_ops:
#         candidates = []
#         for j_idx in range(n_jobs):
#             op_idx = job_op_idx[j_idx]
#             if op_idx >= len(jobs[j_idx]['operations']):
#                 continue
#             job = jobs[j_idx]
#             op = job['operations'][op_idx]
#             best_end, best_start, best_m = float('inf'), None, None
#             for m, p in op:
#                 start = max(machine_available[m], job_ready[j_idx])
#                 end = start + p
#                 if end < best_end:
#                     best_end, best_start, best_m = end, start, m
#             min_p = min(p for _, p in op)
#             pri = job['weight'] / min_p
#             candidates.append((pri, j_idx, op_idx, best_m, best_start, best_end))
#         if not candidates:
#             break
#         _, j_idx, op_idx, m, start, end = max(candidates, key=lambda x: x[0])
#         schedule.append({'job': j_idx, 'op': op_idx, 'machine': m, 'start': start, 'end': end})
#         machine_available[m] = end
#         job_ready[j_idx] = end
#         job_op_idx[j_idx] += 1
#     return schedule

# EVOLVE-BLOCK-START

def solve(instance):
    jobs = instance['jobs']
    n_machines = instance['n_machines']
    machine_available = [0] * (n_machines + 1)
    schedule = []
    for j, job in enumerate(jobs):
        job_ready = job['release_time']
        for op_idx, op in enumerate(job['operations']):
            best_end, best_start, best_machine = float('inf'), None, None
            for machine_id, proc_time in op:
                start = max(machine_available[machine_id], job_ready)
                end = start + proc_time
                if end < best_end:
                    best_end, best_start, best_machine = end, start, machine_id
            schedule.append({'job': j, 'op': op_idx, 'machine': best_machine,
                             'start': best_start, 'end': best_end})
            machine_available[best_machine] = best_end
            job_ready = best_end
    return schedule

# EVOLVE-BLOCK-END


def get_schedule(instance):
    return solve(instance)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from evaluator import parse_instance, compute_twt, evaluate

    result = evaluate(__file__)
    for k, v in result.items():
        print(f"  {k}: {v}")
