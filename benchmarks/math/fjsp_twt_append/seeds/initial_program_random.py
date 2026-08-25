# EVOLVE-BLOCK-START
import random


def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    """
    Random seed, stated as a restriction rule and a selection function.

    RESTRICTION: none. Every candidate competes.
    SELECTION:   uniformly at random among the competing candidates. The draw
                 is seeded from the current schedule state, so the same state
                 always yields the same choice and nothing is carried between
                 calls.
    PARAMETERS:  none.
    """

    # --- Restriction: who competes ---------------------------------------
    competing = candidates

    # --- Selection: who wins ---------------------------------------------
    # Seeded per call from the state this call was given. A module-level
    # generator would advance between calls, which is state carried across
    # calls and is not permitted.
    seed = (len(competing) * 1000003
            + sum(machine_free.values())
            + sum(job_ready.values())) & 0xFFFFFFFF
    return random.Random(seed).choice(competing)

# EVOLVE-BLOCK-END
