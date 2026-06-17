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
    from evaluator import evaluate

    result = evaluate(__file__)
    for k, v in result.items():
        print(f"  {k}: {v}")
