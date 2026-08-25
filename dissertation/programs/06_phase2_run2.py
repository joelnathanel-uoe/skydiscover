# EVOLVE-BLOCK-START
import math
import time
from bisect import bisect_left

import numpy as np

try:                                    # optional: fast C implementation
    from scipy.optimize import linear_sum_assignment as _LSA
except Exception:                       # numpy-only environments
    _LSA = None


def _fld(obj, name):
    """Read a field from either a dict-like or an attribute-like record."""
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _hungarian(cost, n, m):
    """Jonker-Volgenant / e-maxx assignment for n <= m, all costs finite.

    Returns a list `col_of_row` of length n (every row is matched).
    """
    INF = float('inf')
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            row = cost[i0 - 1]
            ui = u[i0]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - ui - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    col_of_row = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            col_of_row[p[j] - 1] = j - 1
    return col_of_row


def _solve_lap(cost, nr, nc):
    """Minimum-cost matching of a rectangular finite matrix.

    Returns a list of (row, col) matched pairs.  Uses scipy when available,
    otherwise the pure-python Hungarian above (matrices here are tiny).
    """
    if _LSA is not None:
        ri, ci = _LSA(np.asarray(cost, dtype=float))
        return list(zip(ri.tolist(), ci.tolist()))
    if nr <= nc:
        ans = _hungarian(cost, nr, nc)
        return [(i, ans[i]) for i in range(nr) if ans[i] >= 0]
    trans = [[cost[i][j] for i in range(nr)] for j in range(nc)]
    ans = _hungarian(trans, nc, nr)
    return [(ans[j], j) for j in range(nc) if ans[j] >= 0]


def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    """
    Softplus-tardiness appender with a depth-synchronised, clock-bounded
    rollout that becomes an exact evaluation whenever it reaches the end.

    RESTRICTION: (a) BOUNDED IDLENESS -- with c* the minimum earliest
                 completion, only candidates that can START STRICTLY BEFORE c*
                 compete (machine-flexible Giffler-Thompson conflict set; the
                 full set is the fallback).  (b) ROUTING COORDINATION -- one
                 min-cost assignment of the surviving operations (rows) to the
                 machines they may use (columns), cell cost
                 (w_j*sigma_j + UFLOOR) * ((e(o,m) - s*) - T) with T = worst
                 resource offered + one mean operation and SENT = +1 on cells
                 that are not offered; only the matched, genuinely offered
                 pairs compete, so two urgent jobs cannot both claim the same
                 fast machine.  The matching is rebuilt from scratch each call:
                 a rolling plan, not a commitment.
    SELECTION:   maximise the index  w_j*sigma_j / cost(o,m) -- marginal
                 decrease of the smooth surrogate
                 sum_j w_j*tau*log(1+exp((Chat_j - d_j)/tau)) per unit of shop
                 resource consumed -- where
                   sigma_j = 1/(1+exp(-(Chat_j - d_j)/tau)) is job j's forecast
                     probability of ending up tardy (WSPT when all jobs are
                     doomed, earliest completion when all are safe);
                   Chat_j  = max(chain bound, machine-load bound), the chain
                     bound being j's best reachable completion now plus its
                     fastest remaining work, the load bound the worst
                     remaining operation's mean availability horizon
                     A_m = free_m + HALF * expected remaining load of m, load
                     being routed as Pr(m) ~ 1/p(o,m);
                   cost    = (e - s*) plus the displacement imposed on rivals
                     queued on the same machine, each discounted by its own
                     urgency and its number of eligible machines, capped at
                     (e - s*) so contention at most doubles the cost.
                 When the leader does not beat the runner-up by DOM, the TOPK
                 best plus the most urgent outsider are rolled out IN LOCK-STEP
                 -- one ATC-per-shop-time dispatch per round for every state,
                 under the same bounded-idleness rule, urgency read off each
                 job's own optimistic completion -- so all states are always
                 compared at EQUAL DEPTH.  Rounds stop only at the clock or
                 when every state holds a COMPLETE schedule; states are then
                 ranked by EXACT weighted tardiness if complete (soft surrogate
                 as tie-break) and by the soft surrogate otherwise, ties by
                 index rank, so the rule degrades gracefully to the index.
                 A TRUNCATED state is priced with ITS OWN congestion, not the
                 root's: every state carries the remaining expected machine
                 load, decremented as it dispatches, and an unfinished job is
                 forecast at max(chain bound, root load bound, mean
                 availability mf[m] + HALF*rl[m] of the machines its next
                 operation may use, plus its fastest remaining work) -- so a
                 state that piled work onto a busy machine is charged for it.
                 Rollout urgency changes for exactly ONE job per dispatch and
                 is cached per state, which buys depth at no accuracy cost.
                 BREADTH EXPANSION: a completed rollout is priced exactly, so
                 depth-synchronisation is automatic and any further candidate
                 rolled out TO COMPLETION is directly comparable; whenever the
                 lock-step phase finished early, the rest of the window is
                 spent widening the shortlist -- one more competing candidate
                 at a time, admitted only while the clock still holds the
                 measured cost of one full rollout -- instead of deepening
                 what is already exact.  Near the end of the construction, and
                 on small instances, the look-ahead thereby becomes an
                 exhaustive one-step-plus-policy evaluation of every competitor.
                 Index ties: earliest completion, shortest processing time,
                 then (job, op, machine).
    PARAMETERS:  K = 2.0, softness floor in mean competing processing times:
                   tau = max(K*p_bar, mean positive slack), measured per call;
                 HALF = 0.5, share of a machine's remaining load an arriving
                   operation queues behind; CLIP = 40, exponent guard;
                 SENT = 1.0 / UFLOOR = 1% of mean urgency, the assignment's
                   not-offered sentinel and urgency floor;
                 TOPK = 3 lock-step candidates plus one diversity seat, after
                   which breadth expansion is limited only by the clock;
                 DOM = 2.5, index margin above which the leader is committed
                   without a rollout; NO depth ceiling and NO breadth ceiling
                   -- both phases share the single budget 40% of
                   0.0001*|candidates| + 0.010 s per call.
    """
    t_start = time.process_time()
    K = 2.0
    HALF = 0.5
    CLIP = 40.0
    TOPK = 3
    DOM = 2.5

    # --- Restriction: who competes ---------------------------------------
    # Bounded idleness: no candidate may keep the shop waiting beyond the
    # instant c* at which some operation could already have been completed.
    c_star = min(c.end for c in candidates)
    competing = [c for c in candidates if c.start < c_star]
    if not competing:
        competing = list(candidates)

    # --- Selection: who wins ---------------------------------------------
    jobs = _fld(instance, 'jobs')
    n_jobs = _fld(instance, 'n_jobs')
    s_star = min(c.start for c in competing)
    p_bar = sum(c.proc for c in competing) / float(len(competing))
    if p_bar <= 0.0:
        p_bar = 1.0

    # (i-a) Expected remaining load of every machine and its availability
    # horizon: one pass over all unscheduled operations, never per candidate.
    load = {}
    share_of = {}          # (j, k) -> expected load one dispatch of (j,k) frees
    for j in range(n_jobs):
        o = job_next_op.get(j)
        if o is None:
            continue
        ops = _fld(jobs[j], 'operations')
        for k in range(o, len(ops)):
            alt = ops[k]
            # Expected time this operation adds to each eligible machine under
            # inverse-processing-time routing, Pr(m) ~ 1/p(o,m): the
            # contribution p(o,m)*Pr(m) = 1/S is the same for every eligible
            # machine, so a slow alternative no longer inflates that machine's
            # phantom load as if it were chosen as often as the fast one.
            s_inv = 0.0
            for mid, pt in alt:
                s_inv += (1.0 / pt) if pt > 0.0 else 1e9
            share = 1.0 / s_inv if s_inv > 0.0 else 0.0
            share_of[(j, k)] = share
            for mid, pt in alt:
                load[mid] = load.get(mid, 0.0) + share
    horizon = {}
    for m, t in machine_free.items():
        horizon[m] = t + HALF * load.get(m, 0.0)
    for m, l in load.items():
        if m not in horizon:
            horizon[m] = HALF * l

    # Earliest completion each available job can reach on ANY machine offered
    # (the forecast must not depend on the machine we happen to be trying).
    best_end = {}
    op_of = {}
    for c in candidates:
        j, o = c.op
        op_of[j] = o
        cur = best_end.get(j)
        if cur is None or c.end < cur:
            best_end[j] = c.end

    # (i-b) Congestion-aware completion forecast: job critical path versus
    # machine-load horizon, whichever is later.  The two suffix arrays built
    # here are RETAINED: they let the rollout below price any state a job may
    # reach in O(1), so look-ahead costs no extra structural work.
    chat = {}
    base = {}          # j -> index of j's current next operation
    tailsum = {}       # j -> ts[i] = fastest work left from operation base+i on
    lbs = {}           # j -> lb[i] = machine-load bound from operation base+i on
    for j, be in best_end.items():
        ops = _fld(jobs[j], 'operations')
        o = op_of[j]
        n = len(ops) - o
        fast = []
        mean_h = []
        for k in range(o, len(ops)):
            alt = ops[k]
            fmin = None
            hsum = 0.0
            for mid, pt in alt:
                if fmin is None or pt < fmin:
                    fmin = pt
                hsum += horizon.get(mid, 0.0)
            fast.append(float(fmin))
            mean_h.append(hsum / len(alt))
        ts = [0.0] * (n + 1)                # suffix sums of fastest work
        for i in range(n - 1, -1, -1):
            ts[i] = ts[i + 1] + fast[i]
        lb = [-1e30] * (n + 1)              # suffix max of the load bound
        for i in range(n - 1, -1, -1):
            v = mean_h[i] + ts[i + 1]
            lb[i] = v if v > lb[i + 1] else lb[i + 1]
        est = be + ts[1]                    # chain bound off the live start
        if lb[0] > est:
            est = lb[0]
        chat[j] = est
        base[j] = o
        tailsum[j] = ts
        lbs[j] = lb

    # (ii) Softness width, self-normalising: never sharper than K mean
    # operations, flattened to the average remaining slack while jobs are safe.
    slack = {}
    pos_sum = 0.0
    pos_cnt = 0
    for j, ch in chat.items():
        sl = _fld(jobs[j], 'due_date') - ch
        slack[j] = sl
        if sl > 0.0:
            pos_sum += sl
            pos_cnt += 1
    tau = K * p_bar
    if pos_cnt:
        mean_pos = pos_sum / pos_cnt
        if mean_pos > tau:
            tau = mean_pos
    if tau <= 0.0:
        tau = 1.0

    # Marginal tardiness rate of every available job: the softplus derivative.
    urg = {}
    for j, sl in slack.items():
        x = -sl / tau                       # = (Chat_j - d_j)/tau
        if x > CLIP:
            x = CLIP
        elif x < -CLIP:
            x = -CLIP
        urg[j] = _fld(jobs[j], 'weight') / (1.0 + math.exp(-x))
    u_bar = sum(urg.values()) / len(urg)
    if u_bar <= 0.0:
        u_bar = 1e-9

    # --- Restriction, stage (b): global routing coordination -------------
    # Solve "which available operation should go on which machine next" as a
    # single assignment problem instead of choosing the operation first and
    # the machine second.  Cheap cell cost: shop time consumed per unit of
    # marginal tardiness rate, so the urgent jobs claim the machines that
    # finish them earliest and the rest are priced on their alternatives.
    row_of_op = {}
    ops_list = []
    col_of_m = {}
    mach_list = []
    pair = {}
    for c in competing:
        if c.op not in row_of_op:
            row_of_op[c.op] = len(ops_list)
            ops_list.append(c.op)
        if c.machine not in col_of_m:
            col_of_m[c.machine] = len(mach_list)
            mach_list.append(c.machine)
        k = (c.op, c.machine)
        prev = pair.get(k)
        if prev is None or c.end < prev.end:
            pair[k] = c
    nr = len(ops_list)
    nc = len(mach_list)
    # Skip the LAP when it cannot coordinate anything (one operation or one
    # machine left, the common case near the end of the schedule).
    if nr > 1 and nc > 1 and nr * nc <= 4000:
        # Resource each offered pair consumes, and the reference "worst option"
        # against which its saving is measured: one mean operation beyond the
        # worst offer, so EVERY offered cell is strictly profitable and every
        # row wants to be matched (with more operations than machines, the
        # rectangular LAP then keeps the urgent-and-fast rows, not the cheap
        # ones a plain weighted cost would keep).
        res_of = {}
        t_ref = 0.0
        for k, c in pair.items():
            r = c.end - s_star                  # idleness + processing
            if r < 0.0:
                r = 0.0
            res_of[k] = r
            if r > t_ref:
                t_ref = r
        t_ref += p_bar
        u_floor = 0.01 * u_bar                  # LAP stays meaningful while no
                                                # job is yet forecast tardy
        cost_m = [[1.0] * nc for _ in range(nr)]        # 1.0 = not offered
        for (op, mid), r in res_of.items():
            # MINUS the urgency-weighted shop time this pair saves.  Summing a
            # weighted completion time is what makes the assignment hand the
            # machine that finishes an urgent job earliest to THAT job;
            # dividing by urgency instead (a ratio inside a sum) would make the
            # LAP optimise the routing of the least important jobs first and
            # delete the urgent job's best pair from the competing set.
            cost_m[row_of_op[op]][col_of_m[mid]] = \
                (urg[op[0]] + u_floor) * (r - t_ref)
        try:
            matching = _solve_lap(cost_m, nr, nc)
        except Exception:
            matching = []
        matched = []
        for r, cc in matching:
            if 0 <= r < nr and 0 <= cc < nc:
                cand = pair.get((ops_list[r], mach_list[cc]))
                if cand is not None:            # discard big-M (non-offered) cells
                    matched.append(cand)
        if matched:                             # otherwise keep the full set
            competing = matched

    # (iii) Displacement bookkeeping: per machine, the urgency-weighted,
    # flexibility-discounted queue, sorted by earliest start with prefix sums
    # so each candidate's total imposed delay costs one binary search.
    n_elig = {}
    per_m = {}
    for c in candidates:
        key = c.op
        ne = n_elig.get(key)
        if ne is None:
            j, o = key
            ne = float(len(_fld(jobs[j], 'operations')[o]))
            n_elig[key] = ne
        per_m.setdefault(c.machine, []).append(
            (c.start, urg[c.op[0]] / (u_bar * ne)))
    stats = {}
    for m, lst in per_m.items():
        lst.sort()
        starts = []
        cw = [0.0]
        cws = [0.0]
        for st, w in lst:
            starts.append(st)
            cw.append(cw[-1] + w)
            cws.append(cws[-1] + w * st)
        stats[m] = (starts, cw, cws)

    # Score: marginal weighted tardiness saved per unit of resource consumed.
    scored = []
    for c in competing:
        j, o = c.op
        starts, cw, cws = stats[c.machine]
        i = bisect_left(starts, c.end)      # rivals that would be pushed back
        disp = c.end * cw[i] - cws[i]
        disp -= (urg[j] / (u_bar * n_elig[c.op])) * (c.end - c.start)
        own = c.end - s_star                # resource the candidate itself takes
        if disp < 0.0:
            disp = 0.0
        elif disp > own:
            disp = own                      # contention never more than doubles
        cost = own + disp
        if cost < 1e-9:
            cost = 1e-9
        idx_val = urg[j] / cost
        scored.append(((-idx_val, c.end, c.proc, j, o, c.machine), idx_val, c))
    scored.sort(key=lambda t: t[0])
    best = scored[0][2]                     # index winner = guaranteed fallback

    # --- Look-ahead: depth-synchronised, clock-bounded rollout -------------
    # Only invoked when the index leader is contested.  The shortlisted states
    # advance in lock-step, one dispatch per round, so a clock stop truncates
    # every state at the same depth and the comparison stays fair; any failure
    # falls back to the index winner.
    try:
        if len(scored) >= 2 and scored[0][1] < DOM * scored[1][1]:
            deadline = t_start + 0.40 * (0.0001 * len(candidates) + 0.010)
            if deadline - time.process_time() > 0.0015:
                # Lock-step seats, held as indices into `scored` so that the
                # breadth-expansion phase below can tell which competitors have
                # already been evaluated and can break ties by index rank.
                sel = list(range(min(TOPK, len(scored))))
                if len(scored) > TOPK:
                    # Diversity seat: the index divides by cost, so a job that
                    # is very urgent but can only be run slowly right now is
                    # starved by it and never reaches the shortlist.  Give the
                    # most urgent outsider a state of its own and let the
                    # horizon surrogate, not the index, arbitrate.
                    sel.append(min(
                        range(TOPK, len(scored)),
                        key=lambda i: (-urg[scored[i][2].op[0]],
                                       scored[i][2].end, scored[i][2].op,
                                       scored[i][2].machine)))
                # Flat per-job tables: allocation-free rollout inner loop.
                jobs_ops = []
                nops = []
                due = []
                wgt = []
                for jj in range(n_jobs):
                    jb = jobs[jj]
                    opl = _fld(jb, 'operations')
                    jobs_ops.append(opl)
                    nops.append(len(opl))
                    due.append(float(_fld(jb, 'due_date')))
                    wgt.append(float(_fld(jb, 'weight')))
                # Machine clock template covering every machine still needed
                # (machine_free plus every machine eligible for remaining work).
                mf0 = dict(machine_free)
                for m in horizon:
                    if m not in mf0:
                        mf0[m] = 0.0

                def _advance(jr, mf, jn, active, rl, uc):
                    """One ATC-per-shop-time dispatch, two allocation-free passes.

                    Pass 1 finds c* (min earliest completion) and s* (min
                    earliest start); pass 2 scores only the pairs that respect
                    bounded idleness.  Urgency is machine-INDEPENDENT -- read
                    off the job's own optimistic completion -- so it changes for
                    exactly ONE job per dispatch and is CACHED in `uc`, which is
                    what buys the extra rollout depth.  The state's remaining
                    expected machine load `rl` is decremented by the dispatched
                    operation's share, so the pricing of a truncated state sees
                    that state's own congestion.
                    """
                    cs = 1e30
                    sm = 1e30
                    for aj in active:
                        rj = jr[aj]
                        for mid, pt in jobs_ops[aj][jn[aj]]:
                            fm = mf[mid]
                            st = rj if rj > fm else fm
                            if st < sm:
                                sm = st
                            en = st + pt
                            if en < cs:
                                cs = en
                    if cs > 1e29:
                        return False
                    bk = 1e30
                    ben = 0.0
                    bj = -1
                    bo = 0
                    bm = None
                    for aj in active:
                        ao = jn[aj]
                        rj = jr[aj]
                        uj = uc.get(aj)
                        if uj is None:              # only ever one job per round
                            sl = due[aj] - (rj + tailsum[aj][ao - base[aj]])
                            uj = wgt[aj] * (math.exp(-sl / tau)
                                            if sl > 0.0 else 1.0)
                            if uj < 1e-12:
                                uj = 1e-12
                            uc[aj] = uj
                        for mid, pt in jobs_ops[aj][ao]:
                            fm = mf[mid]
                            st = rj if rj > fm else fm
                            if st >= cs:            # bounded idleness, as above
                                continue
                            en = st + pt
                            k2 = (en - sm + 1e-9) / uj      # shop time per urgency
                            if k2 < bk or (k2 == bk and
                                           (en < ben or (en == ben and aj < bj))):
                                bk = k2
                                ben = en
                                bj = aj
                                bo = ao
                                bm = mid
                    if bj < 0:
                        return False
                    jr[bj] = ben
                    mf[bm] = ben
                    uc.pop(bj, None)                # this job's urgency moved
                    sh = share_of.get((bj, bo))
                    if sh:                          # work removed from the pool
                        for mid, pt in jobs_ops[bj][bo]:
                            rl[mid] = rl.get(mid, 0.0) - sh
                    if bo + 1 < nops[bj]:
                        jn[bj] = bo + 1
                    else:
                        jn[bj] = None
                        active.remove(bj)
                    return True

                def _price(jr, mf, jn, rl):
                    """(exact weighted tardiness, soft surrogate) of a state.

                    On a COMPLETE state the first entry is the true objective;
                    while operations remain, an unfinished job is forecast at
                    max(chain bound, root load bound, THIS state's congestion
                    bound) -- the mean availability mf[m] + HALF*rl[m] of the
                    machines its next operation may use, plus its fastest
                    remaining work -- and the soft entry discriminates.
                    """
                    hard = 0.0
                    soft = 0.0
                    for aj in range(n_jobs):
                        ao = jn.get(aj)
                        if ao is None:
                            ch = jr.get(aj, 0.0)
                        else:
                            ix = ao - base[aj]
                            tl = tailsum[aj][ix]
                            ch = jr[aj] + tl
                            lv = lbs[aj][ix]
                            if lv > ch:
                                ch = lv
                            alt = jobs_ops[aj][ao]
                            hs = 0.0
                            for mid, pt in alt:
                                r = rl.get(mid, 0.0)
                                hs += mf.get(mid, 0.0) + (HALF * r
                                                          if r > 0.0 else 0.0)
                            lv = hs / len(alt) + tl
                            if lv > ch:             # state's own congestion
                                ch = lv
                        dv = ch - due[aj]
                        if dv > 0.0:
                            hard += wgt[aj] * dv
                        x = dv / tau
                        if x > CLIP:                     # linear regime
                            soft += wgt[aj] * dv
                        elif x > -CLIP:                  # softplus regime
                            soft += wgt[aj] * tau * math.log1p(math.exp(x))
                    return hard, soft

                def _seed(c0):
                    """Private state obtained by committing candidate c0.

                    Carries its own remaining-load pool `rl` and urgency cache.
                    """
                    jr = dict(job_ready)
                    mf = dict(mf0)
                    jn = dict(job_next_op)
                    rl = dict(load)
                    jj, oo = c0.op
                    jr[jj] = c0.end
                    mf[c0.machine] = c0.end
                    sh = share_of.get((jj, oo))
                    if sh:
                        for mid, pt in jobs_ops[jj][oo]:
                            rl[mid] = rl.get(mid, 0.0) - sh
                    jn[jj] = oo + 1 if oo + 1 < nops[jj] else None
                    active = [aj for aj in range(n_jobs)
                              if jn.get(aj) is not None]
                    return jr, mf, jn, active, rl, {}

                # --- Phase 1: depth, in lock-step ------------------------
                # No depth ceiling: rounds end at the clock or when every state
                # has run out of operations -- in which case all states hold a
                # COMPLETE schedule and the pricing below is exact.
                t_roll = time.process_time()
                states = [_seed(scored[i][2]) for i in sel]
                while time.process_time() < deadline:
                    alive = False
                    for jr, mf, jn, active, rl, uc in states:
                        if active and _advance(jr, mf, jn, active, rl, uc):
                            alive = True
                    if not alive:
                        break                   # truncate ALL states together

                all_done = True
                for st in states:
                    if st[3]:
                        all_done = False
                        break
                outs = []
                for rank in range(len(states)):
                    jr, mf, jn, active, rl, uc = states[rank]
                    h, s = _price(jr, mf, jn, rl)
                    # complete: exact tardiness first, surrogate as tie-break;
                    # truncated: surrogate first.  Index rank breaks all ties.
                    outs.append((h, s, sel[rank]) if all_done
                                else (s, h, sel[rank]))

                # --- Phase 2: breadth, once depth is exhausted ------------
                # Every phase-1 state ran to completion, so each is priced by
                # its TRUE weighted tardiness and depth-synchronisation is
                # automatic: a further competitor rolled out to completion is
                # directly comparable.  Spend what is left of the SAME window
                # widening the shortlist rather than deepening what is already
                # exact.  One full rollout has just been timed, so a new seat
                # is opened only while the clock still holds that cost; the
                # in-loop check every 16 rounds is a safety net, and a
                # competitor that has to be truncated is simply dropped (never
                # compared at unequal depth) and expansion stops.
                if all_done and len(scored) > len(sel):
                    cost1 = (time.process_time() - t_roll) / len(states)
                    seen = set(sel)
                    for i in range(len(scored)):
                        if i in seen:
                            continue
                        if time.process_time() + cost1 > deadline:
                            break
                        jr, mf, jn, active, rl, uc = _seed(scored[i][2])
                        rounds = 0
                        while active:
                            if not _advance(jr, mf, jn, active, rl, uc):
                                break
                            rounds += 1
                            if not (rounds & 15) and \
                                    time.process_time() > deadline:
                                break
                        if active:              # truncated: not comparable
                            break
                        h, s = _price(jr, mf, jn, rl)
                        outs.append((h, s, i))

                outs.sort()
                best = scored[outs[0][2]][2]
    except Exception:
        pass                                    # any failure -> index winner

    return best

# EVOLVE-BLOCK-END
