# EVOLVE-BLOCK-START

import time
from math import exp, log

import numpy as np

try:                                   # used when present, never depended on
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except Exception:
    _scipy_lsa = None


def _jv_assign(C):
    """Jonker-Volgenant shortest-augmenting-path assignment (numpy only) on a
    FINITE rectangular cost matrix; returns (rows, cols), a minimum-cost
    matching of min(n, m) rows to distinct columns.  Pure fallback for the
    environments where scipy is absent, so the coordination step below never
    depends on a package outside the stated set."""
    C = np.asarray(C, dtype=float)
    n, m = C.shape
    if n == 0 or m == 0:
        return [], []
    transposed = False
    if n > m:                              # the algorithm needs n <= m
        C = np.ascontiguousarray(C.T)
        n, m = m, n
        transposed = True
    INF = np.inf
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)    # p[j] = row (1-based) on column j
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        ok = False
        for _ in range(m + 2):             # safety bound, never binding
            used[j0] = True
            i0 = int(p[j0])
            cur = C[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            imp = free & (cur < minv[1:])
            if imp.any():
                mv = minv[1:]
                mv[imp] = cur[imp]
                wv = way[1:]
                wv[imp] = j0
            cand = np.where(free, minv[1:], INF)
            k = int(np.argmin(cand))
            delta = float(cand[k])
            if not np.isfinite(delta):
                delta = 0.0
            uidx = np.nonzero(used)[0]
            u[p[uidx]] += delta
            v[uidx] -= delta
            nfree = ~used
            minv[nfree] -= delta
            j0 = k + 1
            if p[j0] == 0:
                ok = True
                break
        if not ok:                         # leave this row unmatched
            continue
        while j0:                          # augment along the path
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
    rows = []
    cols = []
    for j in range(1, m + 1):
        if p[j] > 0:
            rows.append(int(p[j]) - 1)
            cols.append(j - 1)
    if transposed:
        rows, cols = cols, rows
    return rows, cols


def _assign(C):
    """Minimum-cost assignment on a finite rectangular matrix: scipy's
    linear_sum_assignment when available, the numpy Jonker-Volgenant solver
    above otherwise.  Both return the same optimal matching."""
    if _scipy_lsa is not None:
        r, c = _scipy_lsa(C)
        return list(r), list(c)
    return _jv_assign(C)


def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    """
    Appending dispatcher for FJSP-TWT.  A REGIME-INTERPOLATED SMITH core --
    weighted shortest REMAINING work for a job already committed to be late
    (whose tardiness then grows one-for-one with its completion, so the
    objective on the doomed set is weighted completion time and the greedy is
    job-level Smith, not per-operation WSPT), weighted minimum slack for a
    job with time to spare, with the job's end-of-route slack SHARED OUT over
    the operations that still have to consume it -- is
    corrected by three machine terms: capacity stranded, the operation's
    dependence on the machine it asks for, and the urgent queue that machine
    already carries.  Job time is counted by ONE-STEP LOOK-AHEAD (what an
    operation buys its job is when its SUCCESSOR can finish, not when it
    finishes itself) and contention is measured in URGENT, IMMINENT work two
    waves deep, so a machine whose claimant is merely one step away is not
    mistaken for a machine nobody wants -- the mistake that makes an
    appending rule strand capacity.

    RESTRICTION: two stages.  (i) with c* the smallest earliest completion of
                 the step, only candidates that can START strictly before c*
                 compete -- the Giffler-Thompson active conflict set relaxed
                 to keep every machine alternative, hence a superset of the
                 non-delay set.  (ii) HUNGARIAN COORDINATION OF THE READY
                 WAVE, the only mechanism in the design that reasons about
                 candidates JOINTLY.  A per-candidate index cannot resolve a
                 COLLISION: when two urgent operations both want the same
                 fast machine it hands the machine to one of them and says
                 nothing about where the loser should go, so the loser is
                 routed one step later under a state nothing anticipated.  So
                 before any preference is expressed the whole wave is routed
                 at once by one optimal assignment.  Rows are the available
                 operations, columns the machines appearing in the candidate
                 set, and a real (operation, machine) pair costs
                     urg_j*(g_jm - g*_j) + ubar*(s(o,m) - max(free_m, s*)),
                 the urgency-weighted job time the operation forfeits by
                 taking m instead of its best machine plus the capacity m
                 would strand, priced at the wave's mean urgency ubar -- the
                 same two time terms the index below trusts, so the matching
                 never fights the rest of the rule; an INELIGIBLE pair costs a
                 large FINITE penalty (10x the largest real entry, never an
                 infinity, which the solver would reject).  The minimum-cost
                 matching assigns DISTINCT machines to the operations, i.e. a
                 coordinated virtual dispatch of the whole wave minimising its
                 total urgency-weighted delay and idleness: the contested
                 machine goes to the claimant who suffers most and every
                 loser is EXPLICITLY rerouted to its best remaining
                 alternative.  It is then applied as a pure ROUTING
                 restriction -- a matched operation may compete only on the
                 machine it was given -- while an operation the matching
                 could not place (more operations than machines, or placed on
                 a penalty entry) keeps ALL of its machines.  Hence the
                 SEQUENCING decision, which operation goes first, is left
                 entirely to the index, the competing set can never collapse
                 to one arbitrary candidate, and if the intersection with the
                 active set were empty the unfiltered active set is restored.
                 Ties in the matrix are broken by a deterministic 1e-9 ripple
                 in (operation, machine) order, and the whole mechanism is
                 skipped when the matrix exceeds MATCH_CAP entries.
    SELECTION:   maximise the log index
                   log(w_j) - log(p + lam_j*min(RW_j, C_RW*pbar))
                   - log(1 + slack_j/(p + RW_j))
                   - waste_m/pbar + pi_j*adv_jm/pbar - cong_jm
                 whose first two terms are the REGIME-INTERPOLATED SMITH
                 core: the denominator moves continuously from the current
                 operation's processing time p (a comfortable job, lam ~ 0:
                 per-operation WSPT, today's rule exactly) to p + RW_j, the
                 job's whole remaining route (a job with no slack left,
                 lam ~ 1: weighted shortest REMAINING work, the correct
                 greedy for weighted completion once tardiness is certain),
                 the interpolation weight lam_j = exp(-slack_j/(K*(pbar +
                 RW_j))) in (0,1] being the tardiness likelihood the rule
                 already forms when it builds urg.  So a long doomed job
                 stops outbidding short high-weight jobs for the bottleneck
                 merely because its next operation happens to be quick.  RW_j
                 is MACHINE INDEPENDENT, so this term decides only WHICH
                 OPERATION wins, never which machine it is routed to, and is
                 damped at C_RW mean processing times so a very long tail
                 cannot send a job to the back of every queue in one shot;
                 for a last operation RW_j = 0 and the index is unchanged.
                 Here
                 g_jm, the job progress bought, is the earliest COMPLETION
                   the job's NEXT operation could reach given this candidate,
                   min over its eligible machines m' of
                   max(e(o,m), free(m')) + p(o+1,m'); g = e for a last
                   operation, where the completion IS the objective, so a
                   head start no successor machine can absorb is worth zero
                   and the contended slot is left to a job that can use it;
                 slack_j = max(0, d_j - g*_j - R_j) is machine independent
                   (g*_j the best g over this operation's machines, R_j the
                   FORECAST work plus forecast queueing left after operation
                   o+1, whose own wait g already prices), so a slow machine
                   can never inflate its own job's urgency;
                 R_j and RW_j are read off a WATER-FILLING FORECAST rather
                   than from a mean over eligible machines plus a flat queue
                   constant: once per step the operations that REMAIN are
                   swept in order of earliest possible availability and each
                   is greedily given the eligible machine that would finish
                   it soonest, which yields both an assigned processing time
                   for every future operation and a congestion clock T_m per
                   machine; a future operation is then charged its assigned
                   time plus q*phi_m, with phi_m = 2E_m/(E_m + Ebar) in
                   [0,2) and E_m = max(0, T_m - s*) the machine's forecast
                   busy horizon.  Work headed for the bottleneck is charged
                   several times the mean wait, work headed for a machine
                   nothing else needs is charged nothing, and on a balanced
                   shop every phi_m = 1 and the estimate collapses exactly
                   to the flat-q one;
                 RW_j is that forecast work plus queueing left after THIS
                   operation, so slack/(p + RW) is the share of the slack
                   this operation alone may spend -- the classical
                   proportional operation due date, which keeps a job with a
                   long route still ahead of it urgent per operation, and
                   which reduces to the plain w/(p + slack) rule for a last
                   operation and to WSPT for a job already late;
                 waste_m = (s - max(free_m, s*))*rho_m is recoverable
                   capacity given away, priced by rho_m = 2D_m/(D_m + Dbar)
                   in [0,2): free on a machine no urgent job can reach soon
                   -- the only deliberate idleness an appending scheme can
                   express -- and double on the bottleneck;
                 adv_jm = (best g on any OTHER machine) - g_jm is the Vogel
                   regret of being denied this machine, capped at pbar (and
                   equal to pbar when the operation has no alternative at
                   all), charged at the job's marginal tardiness rate
                   pi_j = urg_j/urg_max in (0,1], so a contended machine goes
                   to the claimant who suffers most while a slack job gives
                   it up cheaply and is routed by the load terms;
                 cong_jm = log((D_m + Dbar)/(D*_o + Dbar)) >= 0 is the urgent
                   queue of the machine asked for relative to the quietest
                   machine THIS operation could use, hence exactly zero for
                   machine-locked work: a pure routing term weighed against
                   adv, i.e. against the job time moving would forfeit;
                 plan_jm = GAMMA*(rank_jo - rank_min)/len(plan) is the
                   SHIFTING-BOTTLENECK PLAN-CONSISTENCY term, and it is
                   strictly zero on every machine except the bottleneck b.
                   The water-filling sweep routes future operations by
                   earliest completion, which is a LOAD rule and says nothing
                   about the ORDER in which the busiest machine should take
                   them -- exactly the decision that fixes the objective on a
                   bottleneck-dominated shop.  So the operations the sweep
                   assigned to b (at most PLAN_CAP of them, in forecast
                   release order) are RE-SEQUENCED by the classical
                   single-machine ATC list rule with release dates: a clock t
                   walks the queue, and among the operations forecast
                   available by t (or the earliest-released one when none is)
                   the maximiser of (w_j/p_b)*exp(-max(0, d_op - t - p_b)
                   /(K*pbar)) is appended and t advanced.  The operation due
                   date d_op = d_j - (forecast work + forecast queue of the
                   job's operations AFTER this one) is a proper backward pass
                   off the water-filling tables, not a proportional split.
                   The resulting rank is then charged, on b only, relative to
                   the smallest rank among the candidates b can actually take
                   now: the bottleneck goes to the job the single-machine
                   plan says is next, and a job that jumps its planned queue
                   position pays in proportion to how far it jumps.  The
                   whole mechanism is GATED OFF -- plan_jm identically zero,
                   the rule bit-for-bit as before -- unless the bottleneck's
                   forecast horizon exceeds the mean horizon by more than
                   GATE, i.e. unless the shop really has a bottleneck;
=======
                 D_m is the work the operations available now and their
                   successors expect to send to m (each spreading its time
                   over its eligible machines), every claim discounted by its
                   job's urgency urg_j = w_j*exp(-(slack_j - slack_min)
                   /(K*(pbar + RW_j))) -- slack measured against the RISK
                   still ahead of the job, so route length, not the clock
                   alone, sets the horizon -- and by its imminence
                   exp(-(s - free_m)/(K*pbar)), a claim that cannot reach the
                   machine until long after it falls idle not being
                   contention now.  Each D_m is finally scaled by its own
                   forecast factor phi_m, so a front queue on a machine the
                   whole residue must still pass through counts double,
                   while one on a machine nothing else needs counts for
                   nothing -- the deep horizon correcting the two-wave view.
                 Candidates within BAND of the best index count as equally
                 urgent, and that near-tie -- the one place where a myopic
                 surrogate is admittedly uninformative -- is settled by a
                 PILOT ROLLOUT rather than by a proxy: at most B band members
                 (today's proxy winner first, then the strongest indices) are
                 each provisionally appended, the shop is then simulated
                 forward H steps by a stripped-down version of this same rule
                 (active restriction, the same regime-interpolated core, and
                 the same stranded-capacity discount exp(-waste/pbar), so the
                 pilot simulates the policy it advises rather than a policy
                 that never leaves a machine idle; slack read off the
                 water-filling forecast tables), and the residue is
                 closed with the terminal bound C_j = max(ready_j, MEAN free
                 time of the machines able to take operation o) + forecast
                 remaining work -- the mean rather than the earliest, because
                 after the horizon no job can count on the least loaded of
                 its machines being held for it, and because the mean is the
                 only form in which the bound registers the LOAD A PILOT
                 LEAVES BEHIND, which is the whole consequence the rollout is
                 run to measure.  The band member with the smallest rolled-out
                 sum w_j*max(0, C_j - d_j) wins, EXACT TIES -- above all the
                 flat zero of a horizon in which nothing is yet predicted
                 late, where a tardiness surrogate carries no information at
                 all -- being broken by the smaller rolled-out WEIGHTED
                 COMPLETION sum w_j*C_j, the classical surrogate that
                 discriminates before any job is tardy and agrees with the
                 objective where it matters; further ties, aborted pilots and
                 an exhausted time budget all fall back to the old proxy key
                 (earliest g, then earliest completion, then shortest
                 processing time, then lowest (job, op), then lowest
                 machine), so the base policy is never abandoned.
    PARAMETERS:  K = 2.0, the look-ahead width in mean processing times, used
                 by the tardiness likelihood lam and by the urgency and
                 imminence discounts of the queue estimate; C_RW = 3.0, the
                 damping cap on the remaining-work denominator, in mean
                 processing times -- the length of tail beyond which a doomed
                 job is treated as simply long; BAND = 0.9,
                 the relative index tolerance defining indifference, which
                 fixes the PROXY (the fallback winner); PILOT_BAND = 0.7, the
                 wider tolerance -- an index gap of 30% is under half a mean
                 processing time of waste or regret, i.e. inside the
                 surrogate's own resolution -- deciding who is SIMULATED, so
                 the rollout also fires on steps where the narrow band holds
                 a single member; pbar,
                 the step's mean candidate processing time -- measured, not
                 tuned -- which also caps the dependence bonus and the
                 measured per-operation queueing delay q (itself the mean
                 wait min_m s(o,m) - ready_j of the available operations,
                 hence zero in an idle shop, and reused by the pilot's
                 terminal bound: without it every job comes back on time and
                 the rolled-out TWT is flat, hence uninformative), q now
                 being SPREAD over the machines by the water-filling factors
                 phi_m rather than charged flat; SWEEP_CAP = 300, the number
                 of future operations one call may forecast, which fixes the
                 sweep depth (the whole residue when it is small, otherwise
                 the next SWEEP_CAP/active-jobs operations of every job,
                 never fewer than two, the tail falling back to the old mean
                 work plus flat q); and,
                 for the pilot, B = 3 members simulated (2 on very wide
                 steps) and WORK_CAP = 800, the
                 number of candidate evaluations one pilot may spend, which
                 fixes the horizon H = WORK_CAP/(candidates this step), never
                 below 3 steps, so the cost of a step is bounded
                 independently of shop size; and FULL_CAP = 2400, the
                 EXACT-ENDGAME work cap: once the residue is small enough
                 that two whole rollouts fit inside it (2*cost <= FULL_CAP,
                 the cost of a full rollout being estimated as
                 rem_ops*(sum_j cnt_j*r_j/r_max + n_jobs/3) -- a job stops
                 generating candidates the moment it is FINISHED, so a job
                 with r_j operations left is active for only about r_j/r_max
                 of the rollout, and the simulator's own job scan costs about
                 a third of a candidate evaluation per job per step; the
                 estimate coincides with the old rem_ops*candidates when
                 every job has the same route left and is smaller otherwise,
                 so the exact rollout switches on earlier and covers more of
                 the endgame at an unchanged work cap) the pilot is run to the
                 LAST operation, H = rem_ops, so every job finishes inside
                 the simulation, the terminal bound (and with it the q
                 correction) is never reached, and the number returned is the
                 TRUE total weighted tardiness of a schedule this policy
                 could build -- an exact ordering of the band for the policy
                 class instead of a surrogate of a surrogate, which is
                 precisely what decides the one or two due-date crossings
                 that fix the objective on a lightly loaded instance.  In
                 that regime B rises to FULL_CAP/cost (at most
                 4) and the simulated band widens from PILOT_BAND to
                 EXACT_BAND = 0.40 -- a factor 2.5 of index, about one unit
                 of log priority, i.e. one mean processing time of stranded
                 capacity or one full displacement of the bottleneck plan,
                 which is inside the resolution of a myopic surrogate but not
                 of a rollout that returns the TRUE objective, so the wider
                 band mostly buys arbitration on the endgame steps where the
                 narrow band held a single member and no look-ahead was done
                 at all --
                 the signal now being trustworthy enough to overrule a larger
                 index gap; a shared evaluation counter (bail out past
                 FULL_CAP + two rollouts, the slack absorbing a cost estimate
                 that came out low) and a clock check every eighth
                 simulated step keep the cost of the step bounded whatever
                 the candidate count does as jobs advance, and every bail-out
                 returns the proxy, so the base policy is never abandoned.
                 On every earlier step -- wherever the residue is too big --
                 the truncated pilot and its terminal bound are used exactly
                 as before.  The
                 wall-clock guard (abort once 40% of the per-call CPU budget
                 0.0001*n_cand + 0.010 s is spent) is a safety net only: the
                 horizon, not the timer, decides how much look-ahead is done,
                 so the rule stays deterministic.  For the bottleneck plan:
                 GATE = 1.2, the ratio of the busiest machine's forecast
                 horizon to the mean horizon below which the shop counts as
                 balanced and the plan term is switched off entirely;
                 PLAN_CAP = 40, the number of forecast queue entries
                 re-sequenced, which caps the ATC list schedule at 40^2
                 comparisons per call; and GAMMA = 1.0, one unit of log index
                 -- a factor e of priority -- per FULL displacement of the
                 planned bottleneck queue, so half a queue length of jumping
                 costs half a unit.  The plan reuses pbar, K and the forecast
                 suffix tables already built, adds no state between calls,
                 and is deterministic: the queue is sorted by (forecast
                 release, job, operation) and every ATC tie is broken the
                 same way, so the rank vector is stable from step to step.
    """
    t0 = time.process_time()
    K = 2.0
    C_RW = 3.0
    BAND = 0.9
    PILOT_BAND = 0.7
    jobs = instance['jobs']

    # --- Restriction: who competes ---------------------------------------
    cstar = min(c.end for c in candidates)
    competing = [c for c in candidates if c.start < cstar]
    if not competing:                      # degenerate (zero-length ops)
        competing = list(candidates)

    # --- Selection: who wins ---------------------------------------------
    # Time scale: mean processing time over ALL candidates of the step, a
    # steadier yardstick than the mean over the restricted set alone.
    pbar = sum(c.proc for c in candidates) / len(candidates)
    if pbar <= 0.0:
        pbar = 1.0

    # One-step look-ahead: the JOB PROGRESS a candidate really buys.  Not its
    # own completion e, but g = min over the machines m' eligible for
    # operation o+1 of [max(e, free(m')) + p(o+1,m')] -- the earliest the
    # job's next operation could actually be FINISHED.  Taking the minimum
    # over the successor's COMPLETIONS rather than over its starts prices the
    # availability and the speed of the successor's machines together: the
    # machine that is free soonest is credited only if it is also quick
    # enough to turn that head start into an earlier finish.  For a job's
    # last operation g = e, the completion being the objective itself.
    # Finishing early on a contended machine is worth nothing when every
    # successor machine is busy past e, and g says so, so the fast slot is
    # left to a claimant that can use it.  Exact with respect to what is
    # already committed, and optimistic about what is not.
    gval = {}
    for c in candidates:
        j, o = c.op
        ops = jobs[j]['operations']
        if o + 1 < len(ops):
            g = None
            for m2, p2 in ops[o + 1]:
                f = machine_free.get(m2, 0)
                t = (c.end if c.end > f else f) + p2
                if g is None or t < g:
                    g = t
        else:
            g = c.end
        gval[(c.op, c.machine)] = g

    # Per available operation: the best progress (with the machine that
    # attains it, ties going to the earlier completion), the best progress
    # reachable on any OTHER machine, and the best start.  The first is the
    # job-level reference for urgency (a slow machine cannot buy priority by
    # making its own job later); the first two together measure how much the
    # operation DEPENDS on a given machine; the last exposes the wait it is
    # suffering right now.
    bestg = {}
    bestkey = {}
    bestmach = {}
    beststart = {}
    bestend = {}
    for c in candidates:
        g = gval[(c.op, c.machine)]
        k = (g, c.end)
        b = bestkey.get(c.op)
        if b is None or k < b:
            bestkey[c.op] = k
            bestg[c.op] = g
            bestmach[c.op] = c.machine
        b = beststart.get(c.op)
        if b is None or c.start < b:
            beststart[c.op] = c.start
        # Earliest this operation could possibly finish: the arrival time of
        # the claim its SUCCESSOR will make on the machines it can use.
        b = bestend.get(c.op)
        if b is None or c.end < b:
            bestend[c.op] = c.end
    altg = {}                              # best progress elsewhere
    for c in candidates:
        if c.machine == bestmach[c.op]:
            continue
        g = gval[(c.op, c.machine)]
        b = altg.get(c.op)
        if b is None or g < b:
            altg[c.op] = g

    # Measured queueing delay: how long, on average, an operation that is
    # ready now must still wait before any machine can take it.  This is the
    # shop's own answer to "how long will one more operation queue?" -- zero
    # in an idle shop, growing with congestion, and capped at one mean
    # processing time so the look-ahead can never much more than double a
    # job's remaining work.
    q = 0.0
    for op, s0 in beststart.items():
        wait = s0 - job_ready[op[0]]
        if wait > 0.0:
            q += wait
    q /= len(beststart)
    if q > pbar:
        q = pbar

    # Reference against which wasted capacity is measured: the earliest
    # moment machine m could plausibly be at work, i.e. the later of its own
    # free time and the shop's earliest start this step.  Anything later
    # strands recoverable capacity, whether or not a rival for that machine
    # happens to be visible right now -- and postponing such a commitment
    # costs nothing, since a candidate no rival can displace comes back next
    # step with exactly the same start time.
    sstar = min(c.start for c in competing)
    mstart = {}
    for c in competing:
        if c.machine not in mstart:
            f = machine_free.get(c.machine, 0)
            mstart[c.machine] = f if f > sstar else sstar

    # Per available operation, computed once: the work its job must still
    # survive after this operation (expected, not optimistic -- the mean over
    # the eligible machines of each operation from o+2 on -- plus the queue
    # each of them must survive; operation o+1 is excluded from R ENTIRELY,
    # work and wait alike, because g now measures its actual earliest
    # completion), the resulting machine-independent slack, the total work
    # RW still ahead of the job AFTER this operation (o+1 now INCLUDED: this
    # is the yardstick the end-of-route slack is shared out against, not a
    # term of the slack itself), and the urgency of the job.
    # Suffix table, built once per call in O(total operations): remexp[j][o]
    # is the expected work of operations o..end of job j (the mean processing
    # time over the eligible machines of each).  It replaces the repeated
    # inner summations this rule used to do per available operation, and it
    # is the same table the pilot rollout below reads, so the look-ahead
    # costs no extra bookkeeping.
    # Only the SUFFIX from each job's current operation is ever read -- by
    # the water-filling sweep, by the forecast tables and by the pilot, all
    # of which touch operations at or beyond the current one.  Everything
    # before that, and every operation of a finished job, is dead work
    # repeated at every one of the thousands of calls; skipping it makes the
    # table cheaper as the schedule fills and hands the saved time straight
    # to the rollout, whose horizon is what the per-call budget binds.  The
    # machine scan is likewise confined to the remaining operations --
    # exactly the machines the pilot can touch, which is all the sizing of
    # its flat machine array needs.
    n_jobs = instance['n_jobs']
    n_machines_i = instance['n_machines']
    nops = [len(jobs[j]['operations']) for j in range(n_jobs)]
    remexp = []
    maxm = 0
    for j in range(n_jobs):
        opsj = jobs[j]['operations']
        suf = [0.0] * (nops[j] + 1)
        o0 = job_next_op[j]
        if o0 is None:                     # fully scheduled: never read
            remexp.append(suf)
            continue
        acc = 0.0
        for k in range(nops[j] - 1, o0 - 1, -1):
            alts = opsj[k]
            t = 0.0
            for m2, p in alts:
                t += p
                if m2 > maxm:
                    maxm = m2
            acc += t / len(alts)
            suf[k] = acc
        remexp.append(suf)

    # --- Water-filling congestion clocks ---------------------------------
    # One deterministic greedy list schedule of the work that REMAINS.  The
    # remaining operations are swept in order of earliest possible
    # availability (job ready time plus the expected work of the operations
    # of the same job that precede them, ties by job then operation index),
    # and each is given the eligible machine that would COMPLETE it soonest,
    # max(job avail, load[m]) + p.  Two things fall out of that single pass:
    #   * an assigned processing time for every future operation -- a
    #     routing forecast, strictly better than the mean over eligible
    #     machines, which assumes the job is routed at random;
    #   * a congestion clock T_m = load[m] per machine, i.e. how far into
    #     the future that machine is committed once the residue is spread
    #     over the shop.
    # The clocks are used ONLY relatively (as phi_m below), never as an
    # absolute completion prediction: greedy list scheduling is optimistic,
    # but its IMBALANCE across machines is exactly the information the flat
    # queue constant q was missing.
    nm = maxm + 1
    if n_machines_i > nm:
        nm = n_machines_i
    msize = nm + 1

    jn0 = []
    rem_ops = 0
    active = 0
    for j in range(n_jobs):
        o = job_next_op[j]
        if o is None:
            o = nops[j]
        jn0.append(o)
        rem_ops += nops[j] - o
        if o < nops[j]:
            active += 1

    # Sweep depth: the whole residue when it is small, otherwise the next
    # few operations of every job, so one call never forecasts much more
    # than SWEEP_CAP operations and the per-step cost stays bounded by the
    # cost of the suffix table that is built anyway.
    SWEEP_CAP = 300
    if active == 0:
        depth = 1
    elif rem_ops <= SWEEP_CAP:
        depth = rem_ops + 1
    else:
        depth = SWEEP_CAP // active
        if depth < 2:
            depth = 2

    sweep = []
    for j in range(n_jobs):
        o0 = jn0[j]
        last = nops[j]
        if last > o0 + depth:
            last = o0 + depth
        acc = 0.0
        rj = job_ready[j]
        rex = remexp[j]
        for o in range(o0, last):
            sweep.append((rj + acc, j, o))
            acc += rex[o] - rex[o + 1]
    sweep.sort()                            # fully deterministic order

    load = [0.0] * msize
    for m in range(nm):
        f = machine_free.get(m, 0)
        load[m] = f if f > sstar else sstar
    avail = [job_ready[j] for j in range(n_jobs)]
    asgp = {}                               # (j,o) -> forecast proc time
    asgm = {}                               # (j,o) -> forecast machine
    asgr = {}                               # (j,o) -> forecast RELEASE, i.e.
    #                                         when the job is forecast to be
    #                                         available for that operation.
    #                                         Free here, and it is precisely
    #                                         the release date the
    #                                         single-machine plan below needs.
    for _, j, o in sweep:
        r = avail[j]
        bv = None
        bm = -1
        bp = 0.0
        for m, p in jobs[j]['operations'][o]:
            L = load[m]
            s = r if r > L else L
            v = s + p
            if bv is None or v < bv:
                bv = v
                bm = m
                bp = p
        load[bm] = bv
        avail[j] = bv
        asgp[(j, o)] = bp
        asgm[(j, o)] = bm
        asgr[(j, o)] = r

    # Forecast busy horizon of each machine, and the congestion factor it
    # implies.  phi_m = 2E_m/(E_m + Ebar) is 1 at average horizon (so the
    # measured mean wait q keeps its calibration), 0 on a machine the
    # residue never needs, and tends to 2 on the bottleneck.  Same shape as
    # the rho used to price stranded capacity, so the two agree.
    Ebar = 0.0
    for m in range(nm):
        e = load[m] - sstar
        if e > 0.0:
            Ebar += e
    Ebar /= nm if nm else 1
    phi = [1.0] * msize
    if Ebar > 0.0:
        for m in range(nm):
            e = load[m] - sstar
            if e < 0.0:
                e = 0.0
            phi[m] = 2.0 * e / (e + Ebar)

    # Per-job suffix tables of FORECAST work and FORECAST queueing: each
    # remaining operation contributes the processing time the water-filling
    # assigned it, plus q*phi of the machine it was assigned to.  Operations
    # beyond the sweep depth fall back to the old mean work and flat q, so
    # the estimate degrades gracefully on very large instances.
    asgw = []
    qsuf = []
    for j in range(n_jobs):
        nj = nops[j]
        aw = [0.0] * (nj + 1)
        qs = [0.0] * (nj + 1)
        rex = remexp[j]
        for o in range(nj - 1, jn0[j] - 1, -1):
            pa = asgp.get((j, o))
            if pa is None:
                pa = rex[o] - rex[o + 1]
                qc = q
            else:
                qc = q * phi[asgm[(j, o)]]
            aw[o] = aw[o + 1] + pa
            qs[o] = qs[o + 1] + qc
        asgw.append(aw)
        qsuf.append(qs)

    # Flat per-(job, operation) table of the forecast work AND queueing that
    # FOLLOWS an operation, rwtab[j][o] = asgw[j][o+1] + qsuf[j][o+1].  The
    # pilot reads exactly this quantity once per candidate per simulated
    # step -- the hottest loop in the call -- so it is formed here once
    # rather than rebuilt from two suffix tables inside every rollout.  Only
    # the remaining operations are filled: nothing else is ever read.
    rwtab = []
    for j in range(n_jobs):
        aw = asgw[j]
        qs = qsuf[j]
        nj = nops[j]
        rwt = [0.0] * (nj + 1)
        for o in range(jn0[j], nj):
            rwt[o] = aw[o + 1] + qs[o + 1]
        rwtab.append(rwt)

    # --- Shifting-bottleneck plan: ATC re-sequencing of the busiest machine
    # The water-filling sweep decided WHICH machine each future operation
    # goes to, by earliest completion.  That is a load rule; it says nothing
    # about the ORDER in which the busiest machine should take the work it
    # was given, and on a bottleneck-dominated instance that order is very
    # nearly the whole objective.  So we take the forecast queue of the
    # bottleneck and re-sequence it with the classical single-machine ATC
    # list rule -- release dates from the sweep, operation due dates from a
    # backward pass over the forecast suffix tables -- and keep the resulting
    # RANK as a plan for the selection stage to stay consistent with.
    #
    # Hard gate: unless the busiest machine's forecast horizon exceeds the
    # mean horizon by more than GATE, the shop is balanced, there is no
    # bottleneck to decompose, and the whole mechanism is skipped so the rule
    # is bit-for-bit what it was.
    GATE = 1.2
    PLAN_CAP = 40
    GAMMA = 1.0
    planrank = {}                           # (j,o) -> position in the plan
    plan_b = -1
    plan_len = 0
    if Ebar > 0.0 and asgm:
        Eb = -1.0
        bmach = -1
        for m in range(nm):
            e = load[m] - sstar
            if e > Eb:
                Eb = e
                bmach = m
        if bmach >= 0 and Eb > GATE * Ebar:
            queue = []
            for key, m in asgm.items():
                if m != bmach:
                    continue
                j, o = key
                # Backward-derived operation due date: the job's own due date
                # less everything the forecast says still has to happen after
                # this operation (work AND queueing).  A proper backward pass,
                # not a proportional split of the end-of-route slack.
                dop = (jobs[j]['due_date']
                       - asgw[j][o + 1] - qsuf[j][o + 1])
                queue.append((asgr[key], j, o, asgp[key], dop))
            if len(queue) > 1:
                queue.sort()                # (release, job, op): deterministic
                if len(queue) > PLAN_CAP:
                    del queue[PLAN_CAP:]
                nq = len(queue)
                used = [False] * nq
                t = machine_free.get(bmach, 0)
                if t < sstar:
                    t = sstar
                hor = K * pbar
                for pos in range(nq):
                    pick = -1
                    bk = None
                    for i in range(nq):
                        if used[i]:
                            continue
                        rel, j, o, p, dop = queue[i]
                        if rel > t:
                            continue
                        pp = p if p > 0.0 else 1e-9
                        w = jobs[j]['weight']
                        if w <= 0.0:
                            w = 1e-9
                        sl = dop - t - p
                        if sl < 0.0:
                            sl = 0.0
                        k = (w / pp) * exp(-sl / hor)
                        key2 = (-k, j, o)   # ties by (job, op): stable plan
                        if bk is None or key2 < bk:
                            bk = key2
                            pick = i
                    if pick < 0:            # nothing released: jump the clock
                        for i in range(nq):
                            if used[i]:
                                continue
                            rel, j, o, p, dop = queue[i]
                            key2 = (rel, j, o)
                            if bk is None or key2 < bk:
                                bk = key2
                                pick = i
                        t = queue[pick][0]
                    used[pick] = True
                    rel, j, o, p, dop = queue[pick]
                    if t < rel:
                        t = rel
                    t += p
                    planrank[(j, o)] = pos
                plan_b = bmach
                plan_len = nq

    slackval = {}
    rwork = {}
    for op in bestg:
        j, o = op
        nj = nops[j]
        # Forecast work plus forecast queueing of operations o+2..end
        # (o+1 is inside g, whose own wait is already exact).
        R = (asgw[j][o + 2] + qsuf[j][o + 2]) if o + 2 <= nj else 0.0
        s = jobs[j]['due_date'] - bestg[op] - R
        slackval[op] = s if s > 0.0 else 0.0
        # Work plus queueing left after THIS operation.  Zero for a last
        # operation, where the slack is the job's own and nothing downstream
        # can claim a share of it.
        if o + 1 < nj:
            rwork[op] = asgw[j][o + 1] + qsuf[j][o + 1]
        else:
            rwork[op] = 0.0

    # Urgency: the rate at which delaying this job would cost the objective,
    # i.e. roughly the chance it ends up tardy times its weight.  The slack
    # is measured against the RISK STILL AHEAD OF THE JOB, K*(pbar + RW),
    # rather than against a flat two mean processing times: a job with a long
    # route left is exposed to that much more queueing, so the same slack
    # protects it far less, while a job one operation from the end is nearly
    # certain of its own completion.  This is the classical slack-per-
    # remaining-work correction, and it introduces no new constant.
    # Exponents are shifted by the least slack of the step, so the multiplier
    # can never underflow to a flat zero -- only RATIOS of urgencies are used
    # below, so the shift is harmless.
    smin = min(slackval.values())
    urg = {}
    lamval = {}
    umax = 0.0
    for op in slackval:
        w = jobs[op[0]]['weight']
        if w <= 0.0:
            w = 1e-9
        horizon = K * (pbar + rwork[op])
        # Tardiness LIKELIHOOD of the job, in (0,1]: 1 when its slack has run
        # out, decaying to 0 as the slack grows past the risk still ahead of
        # it.  Built from the MACHINE-INDEPENDENT slack, so a slow machine
        # can never inflate its own job's priority.  It is the same quantity
        # the urgency below uses (urg = w*lam up to a common shift), and it
        # is what interpolates the priority core between per-operation WSPT
        # and weighted shortest REMAINING work.
        lamval[op] = exp(-slackval[op] / horizon)
        # Shifted by the least slack of the step so the multiplier cannot
        # underflow to a flat zero; only ratios of urgencies are used.
        u = w * exp(-(slackval[op] - smin) / horizon)
        urg[op] = u
        if u > umax:
            umax = u
    if umax <= 0.0:
        umax = 1.0

    # --- Restriction, second stage: HUNGARIAN COORDINATION OF THE WAVE ---
    # Every score below is formed one candidate at a time, so nothing in the
    # rule can resolve a COLLISION: if two urgent operations both want the
    # same fast machine, the index gives it to one of them and is silent
    # about where the loser should go -- the loser is routed one step later,
    # under a state the rule never anticipated.  Here the whole ready wave is
    # routed JOINTLY, once, by an optimal assignment: operations as rows,
    # the machines of the candidate set as columns, and the cost of a real
    # pair the urgency-weighted job time it forfeits by taking that machine
    # instead of its best one, plus the capacity that machine would strand
    # (priced at the wave's mean urgency, so both terms are urgency x time).
    # Ineligible pairs take a large FINITE penalty.  The matching is used as
    # a ROUTING restriction only: a matched operation competes solely on the
    # machine it was given, an unmatched one keeps all its machines, and WHO
    # GOES FIRST remains the index's decision.
    MATCH_CAP = 4000 if _scipy_lsa is not None else 900
    if len(competing) > 1 and len(bestg) > 1:
        ops_list = []
        opidx = {}
        for c in candidates:
            if c.op not in opidx:
                opidx[c.op] = len(ops_list)
                ops_list.append(c.op)
        machs = sorted(set(c.machine for c in candidates))
        nr = len(ops_list)
        nc = len(machs)
        if nc > 1 and nr * nc <= MATCH_CAP:
            midx = {}
            for k_, m_ in enumerate(machs):
                midx[m_] = k_
            ubar = sum(urg.values()) / len(urg)
            ent = []
            hi = 0.0
            for c in candidates:
                f = machine_free.get(c.machine, 0)
                base = f if f > sstar else sstar
                idle = c.start - base
                if idle < 0.0:
                    idle = 0.0
                # Urgency-weighted job time forfeited + capacity stranded.
                val = (urg[c.op] * (gval[(c.op, c.machine)] - bestg[c.op])
                       + ubar * idle)
                if val > hi:
                    hi = val
                ent.append((opidx[c.op], midx[c.machine], val))
            BIG = 10.0 * hi + 1.0          # finite: never an infinity
            eps = 1e-9 * (hi + 1.0)        # deterministic tie ripple
            Cm = np.full((nr, nc), BIG)
            for i_, k_, val in ent:
                Cm[i_, k_] = val + eps * (i_ * nc + k_)
            try:
                rows_, cols_ = _assign(Cm)
            except Exception:              # never let coordination fail hard
                rows_, cols_ = (), ()
            mach_of = {}
            for i_, k_ in zip(rows_, cols_):
                if Cm[i_, k_] < BIG:       # a real pair, not a penalty cell
                    mach_of[ops_list[i_]] = machs[k_]
            if mach_of:
                keep = [c for c in competing
                        if c.op not in mach_of
                        or c.machine == mach_of[c.op]]
                if keep:                   # never empty the competing set
                    competing = keep

    # Front queue per machine, measured in URGENT work: the work the
    # operations available RIGHT NOW expect to send to each machine, every
    # operation spreading its own processing time evenly over the machines
    # that could take it, and every claim discounted (by a factor at most
    # one) by how urgent its job is next to the most urgent job of the step.
    # A machine only slack jobs want therefore looks empty, one an urgent job
    # is counting on looks full.  This is the only load-balancing information
    # in the rule, it costs one pass over the candidates, and it is used only
    # RELATIVE to the alternatives of the same operation -- so an operation
    # with no lighter alternative (a machine-locked one above all) is never
    # penalised for the queue it cannot avoid.
    nalt = {}
    for c in candidates:
        nalt[c.op] = nalt.get(c.op, 0) + 1
    demand = {}
    for c in candidates:
        f = machine_free.get(c.machine, 0)
        if f < sstar:
            f = sstar
        # Imminence discount.  Every unfinished job is a candidate at every
        # step, so without this a job whose predecessor runs until far in the
        # future would inflate the queue at a machine it cannot reach for a
        # long time.  A claim counts in full while the machine is idle
        # waiting for it and fades over the same look-ahead width K*pbar the
        # urgencies use.
        imm = exp(-(c.start - f) / (K * pbar)) if c.start > f else 1.0
        demand[c.machine] = (demand.get(c.machine, 0.0)
                             + c.proc * urg[c.op] * imm
                             / (umax * nalt[c.op]))

    # Second wave: the operations that will FOLLOW the available ones.  The
    # successor's alternatives were already read to build g, so seeing one
    # step further costs nothing, and it repairs the one blind spot of a
    # front-only queue: a machine whose claimant is merely one step away
    # otherwise looks EMPTY, which both makes stranding it look free (rho ~ 0)
    # and invites flexible work onto exactly the machine that is about to be
    # needed.  A successor cannot arrive before its predecessor's earliest
    # completion, so the same imminence discount prices it -- a claim two
    # steps out fades of its own accord, and no new constant is introduced.
    for op, u in urg.items():
        j, o = op
        opsj = jobs[j]['operations']
        if o + 1 >= len(opsj):
            continue
        alts = opsj[o + 1]
        e0 = bestend[op]
        share = u / (umax * len(alts))
        for m2, p2 in alts:
            f2 = machine_free.get(m2, 0)
            if f2 < sstar:
                f2 = sstar
            imm2 = exp(-(e0 - f2) / (K * pbar)) if e0 > f2 else 1.0
            demand[m2] = demand.get(m2, 0.0) + p2 * share * imm2

    # Deep-horizon correction: the front queue of a machine, seen through
    # its water-filled congestion clock.  Two waves of claims say who wants
    # the machine NOW; phi says how long the machine stays committed once
    # ALL the remaining work is spread over the shop.  On a balanced shop
    # every phi_m = 1 and this line is the identity.
    for m in demand:
        if m < msize:
            demand[m] *= phi[m]
    dbar = sum(demand.values()) / len(demand)
    if dbar <= 0.0:
        dbar = 1.0
    dbest = {}                             # lightest queue this op can reach
    for c in candidates:
        d = demand[c.machine]
        b = dbest.get(c.op)
        if b is None or d < b:
            dbest[c.op] = d

    # Reference position of the plan: the smallest planned rank among the
    # candidates the bottleneck could actually take at THIS step.  The
    # penalty is measured against it, so the candidate the single-machine
    # plan says is next pays nothing and the term never becomes a blanket
    # discount on the bottleneck.  If no competing candidate on b appears in
    # the plan the term is switched off for this step.
    plan_min = 0
    if plan_b >= 0:
        found = False
        for c in competing:
            if c.machine == plan_b:
                r_ = planrank.get(c.op)
                if r_ is not None and (not found or r_ < plan_min):
                    plan_min = r_
                    found = True
        if not found:
            plan_b = -1

    scored = []
    best_log = None
    for c in competing:
        j, o = c.op
        job = jobs[j]

        p = c.proc if c.proc > 0 else 1e-9
        w = job['weight'] if job['weight'] > 0.0 else 1e-9
        gb = bestg[c.op]
        g = gval[(c.op, c.machine)]
        # Machine independent, and already floored at zero for a job that is
        # committed to be tardy: computed once, above, with the urgencies.
        slack = slackval[c.op]
        # Machine capacity given away, PRICED by how contended the machine
        # is: rho = 2*D_m/(D_m + Dbar) equals 1 at average urgent demand (so
        # the idle rate stays calibrated), falls to 0 on a machine no urgent
        # job can reach soon -- exactly when holding it for a claimant still
        # on its way costs nothing, the one deliberate idleness an appending
        # scheme can express -- and rises towards 2 on the bottleneck, where
        # a stranded slot is paid for by every job queueing behind it.
        dm = demand[c.machine]
        rho = 2.0 * dm / (dm + dbar)
        waste = (c.start - mstart[c.machine]) * rho

        # Machine dependence (Vogel-style regret): what this operation gains
        # from THIS machine over its best alternative.  On a machine that is
        # not the best one it is exactly minus the job time given away; on
        # the best machine it is the gap to the runner-up, capped at one mean
        # processing time.  That job time is then charged at the job's
        # MARGINAL TARDINESS RATE pi = urg/umax in (0,1] -- the rate at which
        # a delay of this job actually costs the objective -- so an urgent
        # job outbids a slack rival for the machine it depends on, while a
        # slack job gives its preferred machine up cheaply and is routed by
        # the congestion and idleness terms.  An operation with no
        # alternative machine at all keeps the full, undiscounted claim: for
        # every candidate of an inflexible shop that is the same constant,
        # so such instances are ranked exactly as before.
        pi = urg[c.op] / umax
        if c.machine == bestmach[c.op]:
            a = altg.get(c.op)
            if a is None:
                adv = pbar                    # no substitute machine at all
            else:
                adv = a - g
                if adv > pbar:
                    adv = pbar
                adv *= pi
        else:
            adv = (gb - g) * pi               # = -(g - g*_j)*pi <= 0

        # Congestion of the requested machine, measured against the quietest
        # machine this same operation could use: zero when there is nothing
        # quieter, growing like the log of the queue ratio otherwise.  It is
        # weighed directly against adv, i.e. against the time the operation
        # would forfeit by moving, so a flexible operation steps aside for a
        # contended machine only when stepping aside is cheap.
        cong = log((demand[c.machine] + dbar) / (dbest[c.op] + dbar))

        # Weighted modified OPERATION due date, in logs.  The slack belongs
        # to the whole remaining route, not to this operation alone, so only
        # its proportional share slack*p/(p + RW) is charged here -- the
        # classical operation-due-date allocation.  The index is then
        # (w/p)/(1 + slack/(p + RW)): exactly WSPT for a job already
        # committed to be late (the correct greedy for weighted completion),
        # weighted minimum slack for a job with time to spare, and the plain
        # w/(p + slack) rule for a last operation, where RW = 0.  Because a
        # job with a long route left keeps only a small share of its slack
        # per operation, it stops looking comfortable while there is still
        # time to save it.  Unlike an exponential the index saturates rather
        # than decaying without bound, so when every competing job is
        # comfortably early the machine terms below still decide the
        # routing, the regime in which routing errors compound.
        # Regime-interpolated Smith denominator.  For a comfortable job
        # (lam ~ 0) this is p and the index is the old per-operation WSPT;
        # for a job already committed to be tardy (lam ~ 1) it is the job's
        # whole remaining route p + RW, i.e. weighted shortest REMAINING
        # work -- the correct greedy once tardiness grows one-for-one with
        # completion.  RW is machine independent, so this changes which
        # OPERATION wins, never how it is routed; it is damped at C_RW mean
        # processing times so a very long tail cannot bury a job in one
        # shot; and for a last operation RW = 0 leaves the index untouched.
        rw = rwork[c.op]
        rwc = rw if rw < C_RW * pbar else C_RW * pbar
        # Plan consistency, on the bottleneck machine ONLY: how far this
        # candidate would jump ahead of the single-machine ATC plan for that
        # machine, as a fraction of the planned queue, at GAMMA units of log
        # index per full displacement.  Exactly zero on every other machine
        # and whenever no bottleneck was identified, so the rule degrades to
        # its previous behaviour on balanced shops.
        plan_pen = 0.0
        if c.machine == plan_b:
            r_ = planrank.get(c.op)
            if r_ is not None:
                plan_pen = GAMMA * (r_ - plan_min) / plan_len
        logpri = (log(w) - log(p + lamval[c.op] * rwc)
                  - log(1.0 + slack / (p + rw))
                  - (waste - adv) / pbar - cong - plan_pen)
        scored.append((logpri, g, c))
        if best_log is None or logpri > best_log:
            best_log = logpri

    # --- Indifference band, settled by a PILOT ROLLOUT --------------------
    # The index declares these candidates equivalent; rather than guessing
    # between them with a proxy, we actually simulate the shop forward and
    # keep the one whose estimated total weighted tardiness is lowest.  The
    # base policy is never abandoned: the pilot only REORDERS candidates the
    # index already judged near-equal, and every failure path returns the
    # old proxy winner.
    # Two tolerances, one purpose.  BAND (10%) still defines what the index
    # calls a tie and therefore fixes the PROXY -- the candidate returned
    # whenever the pilot cannot or does not overrule it.  PILOT_BAND (30%)
    # defines who is SIMULATED: an index gap of that size is less than half a
    # mean processing time of stranded capacity or forfeited regret, well
    # inside the resolution of a myopic surrogate, so those candidates are
    # worth an actual look-ahead even though the narrow band would have
    # declared one of them the outright winner.
    # --- Endgame detection: can a pilot afford to run to COMPLETION? -----
    # A pilot costs about one candidate evaluation per remaining operation
    # per step, i.e. rem_ops*len(candidates) for a rollout that reaches the
    # LAST operation (rem_ops was already counted for the water-filling
    # sweep).  When two such rollouts fit inside FULL_CAP the pilot needs no
    # terminal bound at all: every job is finished in the simulation, so the
    # number it returns is the TRUE weighted tardiness of a schedule this
    # policy could actually build, and the ordering it induces is exact for
    # the policy class rather than a proxy of a proxy.  Because the signal is
    # then trustworthy, the band that is SIMULATED is widened a little in
    # that regime (0.60 instead of 0.70); on every earlier step nothing
    # changes.
    FULL_CAP = 2400
    # Cost of a rollout that reaches the LAST operation.  rem_ops*candidates
    # assumes every job keeps generating candidates until the very last step;
    # in fact a job stops the moment it is FINISHED, so a job with r_j
    # operations left contributes its cnt_j candidates for only about
    # r_j/r_max of the rollout.  Adding the simulator's own job scan
    # (n_jobs per step, about a third the cost of a candidate evaluation)
    # completes the model.  The estimate coincides with the old one when
    # every job has the same amount of route left and is smaller otherwise,
    # so the exact rollout -- the only regime in which the pilot returns the
    # TRUE objective rather than a surrogate -- switches on earlier and
    # covers more of the endgame at the same work cap.
    cnt = {}
    for c in candidates:
        jc = c.op[0]
        cnt[jc] = cnt.get(jc, 0) + 1
    rmax = 1
    for j in range(n_jobs):
        r = nops[j] - jn0[j]
        if r > rmax:
            rmax = r
    est = 0.0
    for jc, n_ in cnt.items():
        est += n_ * (nops[jc] - jn0[jc])
    per_pilot = int(rem_ops * (est / rmax + n_jobs / 3.0)) + 1
    exact = 2 * per_pilot <= FULL_CAP
    EXACT_BAND = 0.40
    threshold = best_log + log(BAND)
    pthreshold = best_log + log(EXACT_BAND if exact else PILOT_BAND)
    band = []
    wide = []
    for logpri, g, c in scored:
        if logpri < pthreshold:
            continue
        item = (logpri, (g, c.end, c.proc, c.op, c.machine), c)
        wide.append(item)
        if logpri >= threshold:
            band.append(item)

    band.sort(key=lambda t: t[1])          # old proxy order
    proxy = band[0][2]
    if len(wide) == 1:                     # nothing to arbitrate: free
        return proxy

    # Pilot set: the proxy winner FIRST (so an aborted pilot phase can only
    # reproduce today's decision), then the strongest indices, at most B.
    if exact:
        B = FULL_CAP // per_pilot          # as many as the work cap buys
        if B > 4:
            B = 4
    else:
        B = 3 if len(candidates) <= 80 else 2
    pilots = [band[0]]
    taken = {(proxy.op, proxy.machine)}
    for t in sorted(wide, key=lambda t: (-t[0], t[1])):
        if len(pilots) >= B:
            break
        k = (t[2].op, t[2].machine)
        if k not in taken:
            taken.add(k)
            pilots.append(t)
    if len(pilots) == 1:
        return proxy

    # CPU guard: a pure safety net, never the thing that decides how much
    # look-ahead is done (the horizon H does that, deterministically).
    clock = time.process_time
    tend = t0 + (0.70 if exact else 0.40) * (0.0001 * len(candidates) + 0.010)
    if clock() > tend:
        return proxy

    # Flat state, copied per pilot: no dicts and no numpy inside the loop.
    # msize, jn0 and rem_ops were built for the water-filling forecast
    # above and are reused verbatim, so the pilot costs no extra setup.
    mf0 = [0.0] * msize
    for m, v in machine_free.items():
        mf0[m] = v
    jr0 = [job_ready[j] for j in range(n_jobs)]
    due = [jobs[j]['due_date'] for j in range(n_jobs)]
    wgt = [jobs[j]['weight'] for j in range(n_jobs)]
    allops = [jobs[j]['operations'] for j in range(n_jobs)]

    # Horizon from a work cap, not from the clock: one pilot may spend at
    # most WORK_CAP candidate evaluations, so a wide step buys a short
    # look-ahead and a narrow one a long look-ahead, at constant cost.
    WORK_CAP = 800
    if exact:
        # Rollout to COMPLETION: no terminal bound, no q correction, the
        # exact objective.  spent[] is a shared counter over all pilots of
        # this step, so the cost of the endgame is bounded whatever the
        # candidate count does as jobs advance.
        H = rem_ops
        # Two rollouts of slack above the work cap: a cost estimate that came
        # out low then costs at most one wasted pilot instead of aborting the
        # arbitration altogether.  The clock guard still bounds the step.
        evcap = FULL_CAP + 2 * per_pilot
    else:
        H = WORK_CAP // len(candidates)
        if H < 3:                          # below three steps a pilot is
            H = 3                          # little more than its own bound
        if H > rem_ops:
            H = rem_ops
        evcap = 1 << 30                    # truncated path: unchanged
    spent = [0]
    INF = float('inf')
    crw = C_RW * pbar                      # damped remaining-work cap

    def pilot(c0):
        """Append c0, run the stripped-down base policy H steps, then close
        the residue with the terminal bound; returns the pair (estimated
        TWT, estimated weighted completion sum) -- the second entry being the
        finer surrogate that decides when the first is tied, and in
        particular when it is flat at zero -- or None if time ran out."""
        jn = list(jn0)
        jr = list(jr0)
        mf = list(mf0)
        j0, o0 = c0.op
        mf[c0.machine] = c0.end
        jr[j0] = c0.end
        jn[j0] = o0 + 1
        nstep = 0
        for _ in range(H):
            # Clock guard every eighth step only: on a rollout that reaches
            # the last operation the check itself would otherwise cost more
            # than the simulation it protects.
            nstep += 1
            if not (nstep & 7) and clock() > tend:
                return None
            cmin = INF
            s_min = INF
            cl = []
            ap = cl.append
            for j in range(n_jobs):
                o = jn[j]
                if o >= nops[j]:
                    continue
                r = jr[j]
                for m, p in allops[j][o]:
                    f = mf[m]
                    s = r if r > f else f
                    e = s + p
                    if e < cmin:
                        cmin = e
                    if s < s_min:
                        s_min = s
                    ap((j, o, m, p, s, e))
            if not cl:
                break
            spent[0] += len(cl)
            if spent[0] > evcap:           # hard work bound for the step
                return None
            # Active restriction, then the SAME regime-interpolated core the
            # base rule uses, w(p+RW)/((p + lam*min(RW,C_RW*pbar))(p+RW+sl)):
            # weighted shortest remaining work for a job already late,
            # weighted slack-per-remaining-work otherwise.  The pilot is a
            # simulation of the base policy, so it must simulate the same
            # regime, or it predicts the wrong future exactly on the
            # overloaded instances where the interpolation matters.
            bkey = None
            bt = None
            for t in cl:
                j, o, m, p, s, e = t
                if s >= cmin:
                    continue
                rw = rwtab[j][o]
                sl = due[j] - e - rw
                if sl < 0.0:
                    sl = 0.0
                pp = p if p > 0.0 else 1e-9
                rwc = rw if rw < crw else crw
                lam = exp(-sl / (K * (pbar + rw)))
                pr = (wgt[j] * (pp + rw)
                      / ((pp + lam * rwc) * (pp + rw + sl)))
                # The SAME stranded-capacity discount the base rule applies:
                # committing machine m to a start later than the earliest
                # moment anyone could have occupied it (its own free time, or
                # the shop's earliest start, whichever is later) loses that
                # capacity for good, and postponing the commitment would have
                # cost nothing.  Without it the pilot simulates a policy that
                # never leaves a machine idle and so predicts a future the
                # real rule will not build -- precisely on the congested
                # steps where the arbitration is worth doing.
                f = mf[m]
                base = f if f > s_min else s_min
                if s > base:
                    pr *= exp((base - s) / pbar)
                key = (-pr, e, j, o, m)
                if bkey is None or key < bkey:
                    bkey = key
                    bt = t
            if bt is None:
                break
            j, o, m, p, s, e = bt
            mf[m] = e
            jr[j] = e
            jn[j] = o + 1
        # Terminal bound: a job cannot finish before its own ready time, nor
        # before the earliest machine that could take its next operation,
        # plus the expected work it still has to get through -- and plus the
        # queue it still has to survive, one measured delay q per operation
        # that must WAIT for a machine after the first (whose wait the
        # machine-free term above already prices).  Without that congestion
        # term the bound is purely optimistic, nearly every job comes back on
        # time, and the rolled-out TWT is flat zero for every pilot: the term
        # is what makes the surrogate discriminate BEFORE jobs are certainly
        # late, which is the whole point of running the pilot.
        twt = 0.0
        wcs = 0.0
        for j in range(n_jobs):
            o = jn[j]
            if o >= nops[j]:
                cj = jr[j]
            else:
                # MEAN free time of the machines that could take the job's
                # next operation, not the earliest.  The earliest is a lower
                # bound, and in a flexible shop it is nearly always the
                # current time, so the residue becomes insensitive to which
                # machines the pilot has just loaded -- exactly the
                # difference between the pilots that is being measured.  The
                # mean says: a job cannot count on the least loaded of its
                # machines still being free by the time it gets there.
                r = jr[j]
                alts = allops[j][o]
                tot = 0.0
                for m, p in alts:
                    tot += mf[m]
                mm = tot / len(alts)
                if mm > r:
                    r = mm
                # Forecast work of the whole tail, plus the forecast queue
                # of every operation after the first (whose wait the
                # machine-free term above already prices).
                cj = r + asgw[j][o] + qsuf[j][o + 1]
            wcs += wgt[j] * cj
            late = cj - due[j]
            if late > 0.0:
                twt += wgt[j] * late
        return twt, wcs

    best_c = None
    best_pair = None
    for rank, item in enumerate(pilots):
        if clock() > tend:
            break
        c = item[2]
        v = pilot(c)
        if v is None:                      # aborted: keep what we have
            break
        # Lexicographic.  Rolled-out weighted tardiness first; then -- when
        # the horizon predicts the SAME tardiness for both, which for a
        # comfortably loaded horizon means zero for both and leaves the
        # objective completely uninformative -- the rolled-out WEIGHTED
        # COMPLETION sum, which still separates the futures and is the
        # standard surrogate that becomes the objective itself once jobs
        # start to be late.  Rank 0 is the proxy and the rest are in
        # decreasing index order, so a tie on BOTH keys is still resolved in
        # favour of the base policy: the pilot can only overrule it by
        # predicting a strictly better future.
        pair = (v[0], v[1], rank)
        if best_pair is None or pair < best_pair:
            best_pair = pair
            best_c = c

    return best_c if best_c is not None else proxy

# EVOLVE-BLOCK-END
