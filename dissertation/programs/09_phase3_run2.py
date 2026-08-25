# EVOLVE-BLOCK-START
import math
import bisect


def _fld(obj, name):
    """Field access that works for dicts and for objects with attributes."""
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _ptime(alt):
    """Processing time of one (machine, time) alternative."""
    if isinstance(alt, dict):
        for k in ('proc_time', 'processing_time', 'time', 'p'):
            if k in alt:
                return alt[k]
    return alt[1]


def _mid(alt):
    """Machine id of one (machine, time) alternative."""
    if isinstance(alt, dict):
        for k in ('machine_id', 'machine', 'm', 'id'):
            if k in alt:
                return alt[k]
        return None
    return alt[0]


def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    """
    RESTRICTION: two filters, applied in this order.
      (R1) VALUE-WEIGHTED CONGESTION-PRICED ROUTING -- every available
           operation is reduced to the single machine minimising
               (w_j / wbar) * u_j * (end - now)  +  (D_m / Dbar) * cap_m,
           the total cost of that routing to the shop.  Both terms are a
           time multiplied by a dimensionless RELATIVE price: the delay it
           buys the job, valued at that job's tardiness risk u_j times its
           weight relative to the shop mean wbar, plus the capacity it
           consumes there, cap_m = proc + b*idle (the duration plus the
           AVOIDABLE idle it strands -- the gap between its start and the
           earliest instant at which some available operation could have put
           that machine to work, charged at the rate b at which destroyed
           capacity turns into delay), valued at the relative shadow price
           of that machine's capacity (D_m = machine m's committed backlog
           at the decision epoch plus the VALUE-WEIGHTED near-horizon demand
           headed for it: each job's forecast share is weighted by its own
           relative weighted risk vhat_j = (w_j/wbar)*u_j normalised to mean
           one over the available jobs, so a machine is scarce when VALUABLE
           work needs it, not merely when much work does; Dbar is the mean
           over machines).  A machine the urgent jobs need therefore prices
           itself out of everyone else's routing while staying open to those
           urgent jobs, whose pull outbids the price.
           u_j = 1/(1 + slack_j/F_j) is the same
           probability-like risk the selection uses, computed BEFORE routing
           from the operation's fastest alternative so it does not depend on
           the choice being made.  A heavy job about to be late buys speed:
           it takes the machine that finishes it earliest.  A job with a
           comfortable buffer (u small) buys efficiency: it takes the machine
           that destroys the least capacity and simply waits if that machine
           is busy.  One cost covers both regimes, and a bottleneck prices
           itself out of work no urgent job needs it for.
      (R2) ACTIVE FRONTIER -- of the routed pairs keep those starting
           strictly before c*, the earliest completion the routed set can
           deliver, so no machine is booked past the shop's next completion
           epoch; a routed pair that starts later simply waits for a later
           step, where it is routed afresh.  The c*-attainer starts before
           its own end, so the survivors are never empty.
    SELECTION:   maximise, IN LOG SPACE,
        ln(w_j) + ln(pref_m) - ln(cap) - THETA*P_j*ln(R_j/Rbar) - u_j
      = ln( w_j * P(tardy) / [ RELATIVE capacity committed * (relative
      remaining work)^(THETA*P_j) ] ), where
        cap    = proc + b*idle + block, the machine time this dispatch
                 costs the shop:
                 (i) its duration;
                 (ii) the AVOIDABLE idle it strands, i.e. the gap between
                 its start and the earliest instant some available operation
                 could have put that machine to work -- idle that NO
                 available operation could have filled is not this
                 candidate's doing and is not charged to it -- turned into
                 delay at the shop's own capacity stretch b, so idling is
                 nearly free in a path-bound shop and nearly prohibited in a
                 capacity-bound one, with no tuning constant;
                 (iii) block, the BLOCKING EXTERNALITY: appending here
                 pushes every other available operation i eligible on this
                 machine back by max(0, end - s_i(m)), and one unit of that
                 push is priced at rel_i = vhat_i / k_i, job i's weighted
                 tardiness risk divided by the number of machines it could
                 use instead -- its reliance on this one.  The total,
                 sum_i rel_i*max(0, end - s_i(m)), is divided by the shop's
                 mean value rate to convert it back into machine time, so it
                 is commensurable with the two terms above and needs no
                 constant.  This is the one irreversible act of an appending
                 harness -- a slot once taken is taken -- and it makes the
                 index prefer short operations exactly where the queue is
                 long and valuable, and defer the bottleneck's decision
                 until the contender set for it is richest;
        pref_m = mean duration of the near-horizon work eligible on m, i.e.
                 that machine's own time scale, so an operation is expensive
                 only if it is long FOR THIS MACHINE -- what keeps a slow
                 machine usable when speeds vary widely across eligible
                 machines, and a constant (hence inert) when they do not;
        tail_j = sum over the job's remaining operations of that operation's
                 FASTEST alternative: the optimistic path still open to it,
                 which is the right question to ask of a slack ("can this
                 job still be saved?");
        qhat_j = mean over those operations of the SMALLEST committed backlog
                 among the machines the operation is eligible on -- the wait
                 the job faces if it always takes its least busy eligible
                 machine, the queueing counterpart of the fastest-alternative
                 tail: a job whose whole route is congested is projected
                 later than one whose route is free;
        b      = max(1, (remaining fastest work / n_machines) / longest
                 remaining path), the stretch capacity imposes on a path when
                 capacity, not precedence, is what binds (b = 1 in a
                 path-bound shop, and b -> 1 again as the shop drains);
        flow_j = max(tail_j + n_rem_j*qhat_j, b*tail_j), the remaining path
                 inflated by the MORE PESSIMISTIC of the two congestion
                 signals the shop currently shows;
        slack_j = max(0, d_j - (end + flow_j)) and F_j = flow_j + proc, the
                 job's own projected remaining flow, floored at one typical
                 operation pbar since no projection is sharper than that;
        u_j    = min(slack_j / F_j, SAFE) and P_j = exp(-u_j), the chance the
                 job ends up tardy.  SATURATED, because past SAFE units of
                 its own flow a job is out of danger and further buffer says
                 nothing useful -- those jobs must be separated by capacity,
                 not by comfort.  Kept in logs so that a large slack keeps
                 ranking jobs instead of underflowing them into a tie;
        R_j    = proc + tail_j, the work still standing between the job and
                 its completion, Rbar its mean over the available jobs.
                 Among jobs of comparable urgency the nearer one is served
                 first: finishing it stops its tardiness accruing for good,
                 while each rival loses only one operation -- the correct
                 marginal rule precisely where the due-date term goes inert
                 because everybody is already late.  GATED by P_j so it
                 decides only among jobs in danger and never delays a long
                 job that is still comfortably on time.
      Where due dates bite the index is weighted shortest RELATIVE capacity
      graded by remaining work; where they are loose it is a weighted
      critical ratio protecting the tardy tail.
      Ties: earliest completion, then shortest duration, then lowest
      (job, operation, machine) index.
    PARAMETERS:  HORIZON = 3 operations -- how far ahead the demand forecast
      and the machine-speed sample reach.  THETA = 1/2 -- a square-root,
      i.e. deliberately mild, preference for jobs nearer completion.
      SAFE = 4 projected-flow units -- the buffer past which a job is out of
      danger.  Everything else (now, m_first, qhat_j, b, vhat_j, rel_i,
      D_m, Dbar, pref_m, pbar, wbar, vbar, Rbar) is read off the current
      state; the blocking charge introduces no constant of its own.
    """
    jobs = _fld(instance, 'jobs')
    now = min(c.start for c in candidates)          # decision epoch

    # earliest instant at which each machine could actually be put to work by
    # SOME available operation: idle before that instant is unavoidable at
    # this step and must not be charged to any candidate.
    m_first = {}
    for c in candidates:
        s0 = m_first.get(c.machine)
        if s0 is None or c.start < s0:
            m_first[c.machine] = c.start

    # ---- shop state at the epoch ----------------------------------------
    # Near-horizon sample: every unfinished job will send its next HORIZON
    # operations to the machines soon.  Each of them loads every eligible
    # machine with the fair share 1/sum_m(1/p_m) (an operation with one fast
    # and several slow machines loads mainly the fast one), and the same
    # durations give each machine its own time scale pref_m.  The shares are
    # kept PER JOB, because the price of a machine is set later by the VALUE
    # of the work headed for it, which needs each job's risk.
    HORIZON = 3          # operations ahead the demand forecast reaches
    THETA = 0.5          # square-root discount for the job's remaining work
    SAFE = 4.0           # buffer, in projected-flow units, past which a job
                         # is out of danger and extra buffer means nothing
    shares = {}
    psum = {}
    pcnt = {}
    gsum = 0.0
    gcnt = 0
    for j, k in job_next_op.items():
        if k is None:
            continue
        ops = _fld(jobs[j], 'operations')
        last = len(ops)
        if last > k + HORIZON:
            last = k + HORIZON
        lst = []
        for i in range(k, last):
            alts = ops[i]
            inv = 0.0
            for a in alts:
                p = _ptime(a)
                m = _mid(a)
                psum[m] = psum.get(m, 0.0) + p
                pcnt[m] = pcnt.get(m, 0) + 1
                gsum += p
                gcnt += 1
                if p > 0.0:
                    inv += 1.0 / p
            if inv <= 0.0:
                continue
            share = 1.0 / inv
            for a in alts:
                lst.append((_mid(a), share))
        shares[j] = lst

    pbar = (gsum / gcnt) if gcnt > 0 else 1.0       # shop-wide time scale
    if pbar <= 0.0:
        pbar = 1.0
    pref = {}
    for m, s in psum.items():
        n = pcnt.get(m, 0)
        pref[m] = (s / n) if (n > 0 and s > 0.0) else pbar

    machs = set(machine_free)
    machs.update(psum)
    nm = len(machs)
    if nm <= 0:
        nm = 1
    backlog = {}
    for m in machs:
        bk = machine_free.get(m, 0.0) - now
        if bk < 0.0:
            bk = 0.0
        backlog[m] = bk                             # committed queue on m

    # ---- per available operation: remaining path, and pre-routing risk ---
    by_op = {}
    for c in candidates:
        by_op.setdefault(c.op, []).append(c)

    tails = {}
    nrems = {}
    qhats = {}
    pmins = {}
    starts = {}
    rem_work = 0.0                                  # fastest work still to do
    longest = 0.0                                   # longest remaining path
    for op, grp in by_op.items():
        j, k = op
        ops = _fld(jobs[j], 'operations')
        t = 0.0
        n = 0
        qs = 0.0
        for i in range(k + 1, len(ops)):
            mn = None
            qm = None
            for a in ops[i]:
                p = _ptime(a)
                if mn is None or p < mn:
                    mn = p                          # fastest alternative
                q = backlog.get(_mid(a), 0.0)
                if qm is None or q < qm:
                    qm = q                          # least busy eligible mach.
            if mn is not None:
                t += mn
            if qm is not None:
                qs += qm
            n += 1
        # routing-independent projection: fastest alternative, earliest start
        pm = None
        s0 = None
        for c in grp:
            if pm is None or c.proc < pm:
                pm = c.proc
            if s0 is None or c.start < s0:
                s0 = c.start
        tails[op] = t
        nrems[op] = n
        qhats[op] = (qs / n) if n > 0 else 0.0      # route-specific wait
        pmins[op] = pm
        starts[op] = s0
        # every unfinished job contributes exactly one available operation,
        # so these two sums see the whole of the remaining shop
        rem_work += pm + t
        head = s0 - now
        if head < 0.0:
            head = 0.0
        path = head + pm + t
        if path > longest:
            longest = path

    # b: the stretch capacity imposes on every remaining path when capacity,
    # not precedence, is the binding constraint (b = 1 when the shop is
    # path-bound, and b -> 1 again as the shop drains).
    b = 1.0
    if longest > 0.0:
        b = (rem_work / float(nm)) / longest
        if b < 1.0:
            b = 1.0


    # projected remaining flow and tardiness risk, per available operation
    flows = {}
    urg = {}
    wsum = 0.0
    for op in by_op:
        j = op[0]
        t = tails[op]
        f = t + nrems[op] * qhats[op]               # route-queue signal
        f2 = b * t                                  # capacity-stretch signal
        if f2 > f:
            f = f2                                  # the more pessimistic one
        flows[op] = f
        F = f + pmins[op]
        if F < pbar:                                # projection floor
            F = pbar
        sl = _fld(jobs[j], 'due_date') - (starts[op] + pmins[op] + f)
        if sl < 0.0:
            sl = 0.0
        urg[op] = 1.0 / (1.0 + sl / F)              # tardiness risk in (0,1]
        w = _fld(jobs[j], 'weight')
        if w > 0.0:
            wsum += w
    wbar = wsum / float(len(by_op))                 # shop-mean weight
    if wbar <= 0.0:
        wbar = 1.0
    # mean remaining work per available job -- the reference the remaining-
    # work discount is measured against (every unfinished job contributes
    # exactly one available operation, so this sees the whole remaining shop)
    rbar = rem_work / float(len(by_op))
    if rbar <= 0.0:
        rbar = 1.0

    # ---- value-weighted shadow price of every machine --------------------
    # A machine is scarce not because much work is headed for it but because
    # VALUABLE work is.  Each job's near-horizon forecast share is weighted
    # by vhat_j, its weighted tardiness risk relative to the shop mean, so
    # the machines the urgent jobs need price themselves out of everybody
    # else's routing (a comfortable job has a small pull and yields at once)
    # while staying open to the urgent jobs, whose pull outbids the price.
    # Normalising vhat to mean one keeps D_m in time units, so the committed
    # backlog and the forecast remain commensurable.
    vraw = {}
    vsum = 0.0
    for op in by_op:
        j = op[0]
        w = _fld(jobs[j], 'weight')
        if w <= 0.0:
            w = 1e-12
        v = (w / wbar) * urg[op]        # relative value of one unit of delay
        vraw[j] = v
        vsum += v
    vbar = vsum / float(len(by_op))
    if vbar <= 0.0:
        vbar = 1.0
    demand = {}
    for j, lst in shares.items():
        vh = vraw.get(j, vbar) / vbar               # mean one over the shop
        for m, share in lst:
            demand[m] = demand.get(m, 0.0) + vh * share
    load = {}
    dsum = 0.0
    for m in machs:
        d = backlog[m] + demand.get(m, 0.0)
        load[m] = d
        dsum += d
    dbar = dsum / float(nm)                         # mean projected value

    # ---- blocking externality: what a dispatch costs the machine's queue --
    # Whatever is appended to machine m first pushes every OTHER available
    # operation eligible on m back behind it -- the one irreversible act of
    # an appending harness.  A unit of that push is priced at
    # rel_i = vhat_i / k_i: job i's weighted tardiness risk divided by the
    # number of machines it could use instead, i.e. its reliance on this one
    # (a highly flexible rival is barely hurt, a captive rival fully).  A
    # cumulative profile per machine lets the total push
    # sum_i rel_i * max(0, e - s_i) be read off in O(log k) per candidate,
    # so the step stays O(C log C).
    kelig = {}
    for c in candidates:
        kelig[c.op] = kelig.get(c.op, 0) + 1
    relv = {}
    riv = {}
    for c in candidates:
        r = vraw[c.op[0]] / kelig[c.op]
        relv[c.op] = r
        riv.setdefault(c.machine, []).append((c.start, r))
    prof = {}
    for m, lst in riv.items():
        lst.sort()
        ss = []
        pa = [0.0]
        pb = [0.0]
        acc = 0.0
        accs = 0.0
        for s, r in lst:
            ss.append(s)
            acc += r
            accs += r * s
            pa.append(acc)
            pb.append(accs)
        prof[m] = (ss, pa, pb)

    # --- Restriction: who competes ---------------------------------------
    # (R1) urgency-discounted congestion-priced routing: one machine per
    #      available operation, the one minimising the delay it buys the job
    #      (priced at that job's risk) plus the machine time it burns (priced
    #      at that machine's relative shadow price).  Urgent jobs buy speed,
    #      comfortable jobs buy efficiency and are content to wait.
    routed = []
    for op, grp in by_op.items():
        pull = vraw[op[0]]              # relative value of one unit of delay
        bc = None
        bk = None
        for c in grp:
            price = (load.get(c.machine, 0.0) / dbar) if dbar > 0.0 else 0.0
            base = m_first.get(c.machine, c.start)  # first usable instant
            idl = c.start - base                    # AVOIDABLE idle only
            if idl < 0.0:
                idl = 0.0
            capm = c.proc + b * idl     # capacity this routing consumes
            key = (pull * (c.end - now) + price * capm,
                   c.end, c.proc, c.machine)
            if bk is None or key < bk:
                bk = key
                bc = c
        routed.append(bc)

    # (R2) active frontier: never book a machine past the earliest completion
    #      the routed set can deliver; a later-starting routed pair waits.
    cstar = min(c.end for c in routed)
    competing = [c for c in routed if c.start < cstar]
    if not competing:                               # zero-length ops only
        competing = routed

    # --- Selection: who wins ---------------------------------------------
    best = None
    best_key = None
    for c in competing:
        j, k = c.op
        job = jobs[j]

        # projected completion: the remaining path inflated by the more
        # pessimistic of the two congestion signals the shop shows
        flow = flows[c.op]
        slack = _fld(job, 'due_date') - (c.end + flow)
        if slack < 0.0:
            slack = 0.0
        F = flow + c.proc                           # projected remaining flow
        if F < pbar:                                # projection floor
            F = pbar

        # capacity this dispatch commits: duration + the AVOIDABLE idle it
        # strands (idle no available operation could have filled is not this
        # candidate's doing), measured on the machine's own time scale so
        # that machines of different speeds are comparable
        base = m_first.get(c.machine, c.start)      # first usable instant
        idle = c.start - base
        if idle < 0.0:
            idle = 0.0
        # blocking externality: the value-weighted delay this dispatch
        # imposes on the OTHER available operations queued for this machine,
        # converted back into machine time at the shop's mean value rate so
        # that it is commensurable with the duration and the stranded idle.
        ss, pa, pb = prof[c.machine]
        i = bisect.bisect_right(ss, c.end)
        push = pa[i] * c.end - pb[i] - relv[c.op] * (c.end - c.start)
        if push < 0.0:
            push = 0.0
        cap = c.proc + b * idle + push / vbar
        if cap <= 0.0:
            cap = 1e-9
        scale = pref.get(c.machine, pbar)
        if scale <= 0.0:
            scale = pbar

        w = _fld(job, 'weight')
        if w <= 0.0:
            w = 1e-12

        # urgency SATURATED: past SAFE units of its own projected flow a job
        # is out of danger, and how much further buffer it holds says nothing
        # useful -- those jobs must be separated by capacity, not by comfort.
        u = slack / F
        if u > SAFE:
            u = SAFE
        risk = math.exp(-u)                         # P(tardy) in [e^-SAFE, 1]

        # work still standing between this job and its completion
        rem = c.proc + tails[c.op]
        if rem <= 0.0:
            rem = 1e-9

        # log( w * P(tardy) / [relative capacity * (relative remaining
        # work)^(THETA*risk)] ).  The last factor is the weighted-shortest-
        # REMAINING-work half of the rule -- the correct marginal rule once a
        # job is projected tardy, since finishing it stops its tardiness
        # accruing for good -- GATED by the risk so that it decides only
        # among jobs already in danger and never delays a long job that is
        # still comfortably on time.
        score = (math.log(w) + math.log(scale) - math.log(cap)
                 - THETA * risk * math.log(rem / rbar) - u)

        key = (-score, c.end, c.proc, j, k, c.machine)
        if best_key is None or key < best_key:
            best_key = key
            best = c

    return best

# EVOLVE-BLOCK-END
