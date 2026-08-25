# EVOLVE-BLOCK-START

def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    """
    Sequential seed, stated as a restriction rule and a selection function.

    RESTRICTION: none. Every candidate competes.
    SELECTION:   the operation with the smallest (job, op) index; among that
                 operation's eligible machines, the one giving the earliest
                 completion. Ties broken by machine order in the instance.
    PARAMETERS:  none.
    """

    # --- Restriction: who competes ---------------------------------------
    competing = candidates

    # --- Selection: who wins ---------------------------------------------
    return min(competing, key=lambda c: (c.op, c.end))

# EVOLVE-BLOCK-END
