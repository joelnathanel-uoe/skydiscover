# EVOLVE-BLOCK-START
import random

_rng = random.Random(0)

def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    """
    Picks uniformly at random among the eligible candidates.
    """
    return _rng.choice(candidates)

# EVOLVE-BLOCK-END
