# EVOLVE-BLOCK-START
import math

# Size cap (in machine-alternative entries) above which the per-job route
# lookahead is skipped; keeps the per-call cost proportional to the work the
# heuristic must do anyway.
_LOOKAHEAD_CAP = 20000


def _fld(obj, name):
    """Read a field from either a dict-like or an attribute-style record."""
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _log_sigmoid(x):
    """log(1/(1+e^-x)) : log-probability of being late, computed stably."""
    if x > 30.0:
        return 0.0
    if x < -30.0:
        return x
    return -math.log1p(math.exp(-x))


def _lrisk(x, xmax, slope):
    """log-risk of lateness at x standard deviations, FLATTENED beyond xmax
    sigma of earliness to a residual slope of `slope` per sigma instead of
    falling at full slope.  Due dates being loose, nearly every job lives in
    that region for nearly the whole construction, so this is what fixes how
    much due dates say at all: safe jobs stay ordered by their margin, ten
    times less sharply than where the curve is bent, so slack informs the
    index without ever swamping the weights."""
    if x < -xmax:
        return _log_sigmoid(-xmax) + slope * (x + xmax)
    return _log_sigmoid(x)


def _route_end(t, ops_j, k0, j, shares, avail, now, half):
    """Forecast completion of job j when its operation k0-1 ends at time t:
    its remaining operations are routed greedily, each taking the machine that
    completes it soonest given that machine's projected queue, and a job never
    waits behind its own projected share of a queue."""
    for k in range(k0, len(ops_j)):
        alts = ops_j[k]
        if not alts:
            continue
        sh = shares.get((j, k), 0.0)
        best = None
        for a in alts:
            av = avail.get(a[0], now) - half * sh
            st = t if t > av else av
            e2 = st + a[1]
            if best is None or e2 < best:
                best = e2
        if best is not None:
            t = best
    return t


def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    """
    Tardiness-risk pricing: every candidate is priced as the weighted
    tardiness RATE it relieves per unit of machine time it burns, urgency
    being the probability the job ends late rather than a hard slack.

    RESTRICTION: three nested stages, each waived if it would empty the set.
                 (a) CROSS-MACHINE ACTIVE FILTER - with c* the smallest
                 earliest-completion among all candidates, only candidates
                 able to START STRICTLY BEFORE c* compete (Giffler-Thompson
                 applied on every machine at once, so every routing option
                 survives the filter and the schedule built is active).
                 (b) ROUTING ECONOMY AT A WORK-FOR-TIME EXCHANGE RATE -
                 durations vary widely across machines, so the currency of
                 congestion is the WORK injected into the shop: a candidate
                 spending more than ALPHA times the least work its own
                 operation could be done with is dropped, its fast machine
                 being busy the operation simply waits a step, which loose due
                 dates can afford.  Two exemptions readmit it, both demanding
                 that it really completes SOONER than the best economical
                 option offered for the same operation: either the exchange is
                 at least AT PAR - the time it saves is no less than the extra
                 work it spends, so the shop as a whole loses no capacity while
                 one job gains - or the job is already FORECAST LATE and pays
                 w_j per unit of further delay right now.  Nothing that merely
                 burns capacity is ever admitted.
                 (c) BOUNDED DELIBERATE IDLENESS - a machine may not be held
                 open more than IDLE times the duration of the operation it
                 waits for, counted from the earliest instant an ECONOMICAL
                 claimant could have taken it.  Such a decision is not lost but
                 DEFERRED: the machine stays free and the same candidate is
                 offered again next step, when more of the schedule is known,
                 so the rule caps the damage the active filter's permissiveness
                 can do at no cost in reachability.
    SELECTION:   maximise
                   log(rho_j) - log(p_eff) - (end - e_best)/tau
                   - THETA*log(price(m)/price(cheapest m for the operation))
                   + PSI*(1 - exp(-(alt_end - end)/tau))
                 with tau = K*p_bar, p_bar the mean duration of an operation
                 still to be scheduled.  rho_j = w_j*sigmoid((Chat_j-d_j)/
                 sigma_j) is job j's marginal weighted-tardiness rate - its
                 weight times the probability it ends late - with Chat_j
                 obtained by WALKING THE JOB'S REMAINING ROUTE THROUGH THE
                 PROJECTED MACHINE QUEUES: every unscheduled operation of the
                 shop spreads its work over its eligible machines IN
                 PROPORTION TO THE SPEED each offers it (routing probability
                 proportional to 1/p_a, so every eligible machine expects the
                 same amount 1/sum(1/p_a) and a machine merely SLOW for an
                 operation is never charged with work it will never see),
                 giving each machine a residual workload q_m, and a newcomer
                 waits for the COMMITTED backlog IN FULL - that work is certain
                 to run - plus HALF*q_m, MINUS ITS OWN SHARE OF THAT QUEUE (a
                 job never waits behind itself); the job's next operations are
                 then routed greedily through those projected queues, so a job
                 whose route crosses the bottleneck is seen as at risk while
                 one with a clear road ahead is not.
                 sigma_j = K*sqrt(p_bar*(p_bar + W_j)) is the
                 width of the risk curve: the forecast error grows like the
                 SQUARE ROOT OF THE WORK W_j still to do behind this
                 operation, a random walk in work rather than in operation
                 count (it coincides with K*p_bar*sqrt(1+r_j) when the job's
                 operations are of typical length).  Beyond XMAX standard
                 deviations of earliness the risk curve is FLATTENED to a
                 residual slope SLOPE per sigma rather than allowed to fall at
                 full slope, so the comfortably safe jobs - nearly all of them,
                 nearly all the time, due dates being loose - stay ordered by
                 their margin as a WEAK slack rule underneath a strong WSPT
                 rule instead of drowning the weights; it saturates at w_j once
                 a job is certainly late, where the index is WSPT among the
                 tardy.  The curve is RE-READ AT THE JOB COMPLETION EACH
                 CANDIDATE IMPLIES - the job's remaining route re-walked
                 through the projected queues from that candidate's own
                 completion - so a routing loss a downstream queue would have
                 absorbed anyway is charged nothing, while one that really
                 pushes the job out is charged in full: dear for a job at risk,
                 nearly free for one forecast comfortably early.  p_eff = p +
                 max(0, start - earliest instant any available operation could
                 occupy that machine) is exactly the machine time burned:
                 unavoidable idle is free, self-inflicted idle is paid for.
                 (end - e_best) is the completion thrown away relative to the
                 best machine offered for the same operation, charged at
                 rho_j*delay and normalised by rho_j*tau, hence urgency-free.
                 price(m) is the geometric mean of D_m - the summed rho of the
                 operations contending for m, each discounted by tau/(tau+lag)
                 so only imminent rivals count and shared over its own
                 eligible machines so a rival that can go elsewhere presses
                 only its fair share - and p_bar + q_m, how long m stays busy;
                 it is compared against the cheapest machine still eligible for
                 the operation, so a machine is penalised only when an
                 alternative exists.  The last term is the SCARCITY BONUS:
                 alt_end is the best completion this operation could still
                 reach WITHOUT machine m (infinite when m is its only machine,
                 zero bonus when an equally good machine remains), so an
                 operation with nowhere else to go outbids an equally urgent
                 rival that can be served elsewhere - least flexible first.
                 Ties: indices within BAND of the best (the index being
                 logarithmic, a fixed ratio) are equivalent and settled by
                 earliest completion, then earliest start, then smallest job,
                 operation, machine index.
    PARAMETERS:  ALPHA = 2.0 - the most work an operation may spend relative
                 to the least work it could be done with; IDLE = 1.0 - a
                 machine may be left deliberately idle for at most this many
                 durations of the operation it waits for; HALF = 0.5 - an
                 operation joining a machine waits for all the work already
                 committed there and for half the work not yet scheduled;
                 K = 2.0 - urgency width, in mean operation durations per
                 sqrt(unit of remaining work); XMAX = 3.0 - how many standard
                 deviations early a job must be forecast before it counts as
                 simply safe; SLOPE = 0.1 - residual log-risk slope per sigma
                 kept beyond that point, a tenth of the slope where the curve
                 is bent; THETA = 0.5 - log-priority
                 price of one e-fold of externality; PSI = 0.5 - log-priority
                 worth of being irreplaceable on a machine; BAND = 0.05 -
                 relative index tolerance within which two candidates are
                 indistinguishable.
    """
    ALPHA = 2.0
    IDLE = 1.0
    HALF = 0.5
    K = 2.0
    XMAX = 3.0
    SLOPE = 0.1
    THETA = 0.5
    PSI = 0.5
    BAND = 0.05

    # --- Shop measurement (read by BOTH stages below) --------------------
    jobs = _fld(instance, 'jobs')
    now = min(c.start for c in candidates)

    # Best completion attainable now for each available operation, the least
    # work it could be done with, and the two cheapest completions of every
    # operation - the fallback left to it if it is denied a given machine.
    e_best = {}
    p_min = {}
    b1 = {}
    b2 = {}
    for c in candidates:
        e = e_best.get(c.op)
        if e is None or c.end < e:
            e_best[c.op] = c.end
        v = p_min.get(c.op)
        if v is None or c.proc < v:
            p_min[c.op] = c.proc
        p1 = b1.get(c.op)
        if p1 is None or c.end < p1[0]:
            b2[c.op] = p1
            b1[c.op] = (c.end, c.machine)
        else:
            p2 = b2.get(c.op)
            if p2 is None or c.end < p2[0]:
                b2[c.op] = (c.end, c.machine)

    # Earliest instant each machine could be put to work by an ECONOMICAL
    # claimant - an operation for which this machine is within ALPHA of its
    # fastest.  A machine offered only work it is far too slow for is not
    # idling while it waits: charging that wait to whoever finally takes it
    # would penalise exactly the right routing decision, both in p_eff and in
    # the deliberate-idleness cap.  Machines with no economical claimant fall
    # back to the earliest start offered to them at all.
    first_use = {}
    first_any = {}
    for c in candidates:
        f = first_any.get(c.machine)
        if f is None or c.start < f:
            first_any[c.machine] = c.start
        if c.proc <= ALPHA * p_min[c.op]:
            f = first_use.get(c.machine)
            if f is None or c.start < f:
                first_use[c.machine] = c.start
    for m, t in first_any.items():
        if m not in first_use:
            first_use[m] = t

    # One pass over EVERY unscheduled operation of the shop: best-case work
    # still to do behind each operation of a job (suffix), the mean duration
    # of an operation still to be scheduled (the time scale), and the
    # residual workload each machine must expect.  Routing follows SPEED: an
    # operation goes to machine a with probability proportional to 1/p_a, so
    # every eligible machine expects the same amount 1/sum(1/p_a) of it, and a
    # machine merely slow for an operation is no longer charged for work it
    # will never see - decisive here, durations varying widely across the
    # machines eligible for the same operation.
    suffix = {}
    shares = {}
    load = {}
    tot_p = 0.0
    cnt = 0
    alt_cnt = 0
    for j, job in enumerate(jobs):
        nxt = job_next_op.get(j)
        if nxt is None:
            continue
        ops_j = _fld(job, 'operations')
        L = len(ops_j)
        if nxt >= L:
            continue
        suf = [0.0] * (L + 1)
        for k in range(L - 1, nxt - 1, -1):
            alts = ops_j[k]
            if not alts:
                suf[k] = suf[k + 1]
                continue
            alt_cnt += len(alts)
            mn = alts[0][1]
            inv = 0.0
            for a in alts:
                p = a[1]
                if p < mn:
                    mn = p
                inv += (1.0 / p) if p > 0.0 else 1e9
            sh = (1.0 / inv) if inv > 0.0 else 0.0
            for a in alts:
                load[a[0]] = load.get(a[0], 0.0) + sh
            shares[(j, k)] = sh
            suf[k] = suf[k + 1] + mn        # luckiest routing: a bound
            tot_p += mn
            cnt += 1
        suffix[j] = suf
    p_bar = (tot_p / cnt) if cnt else 1.0
    if p_bar <= 0.0:
        p_bar = 1.0

    # When a newly arriving operation can hope to take up each machine: the
    # COMMITTED backlog waited for IN FULL - that work is certain to run -
    # plus half of the residual workload still headed there, which is only
    # expected.  (Halving the committed backlog too, as a plain queue estimate
    # does, makes a heavily booked machine look free far too soon.)
    avail = {}
    for m in load:
        mf = machine_free.get(m, 0.0)
        avail[m] = (mf if mf > now else now) + HALF * load[m]
    for c in candidates:
        if c.machine not in avail:
            mf = machine_free.get(c.machine, 0.0)
            avail[c.machine] = mf if mf > now else now
    tau = K * p_bar

    # Marginal weighted-tardiness rate of each available operation's job,
    # the risk being read off a forecast completion obtained by routing the
    # job's remaining operations greedily through the projected queues.
    look = alt_cnt <= _LOOKAHEAD_CAP
    fc = {}
    rho = {}
    for op, eb in e_best.items():
        j, o = op
        job = jobs[j]
        w = float(_fld(job, 'weight'))
        if w <= 0.0:
            w = 1e-9
        d = float(_fld(job, 'due_date'))
        ops_j = _fld(job, 'operations')
        suf = suffix.get(j)
        rem = suf[o + 1] if suf is not None else 0.0   # work still behind o
        if look:
            # (a job's own projected share is part of load[m], so the queue it
            # faces never falls below the machine's real free time)
            t = _route_end(eb, ops_j, o + 1, j, shares, avail, now, HALF)
        else:                       # oversized instance: work-limited bound
            t = eb + rem
        # forecast error: a random walk over the work still to be done
        sigma = K * math.sqrt(p_bar * (p_bar + rem))
        if sigma <= 0.0:
            sigma = 1.0
        # the whole risk curve is kept, so it can be re-read later at the job
        # completion each candidate implies
        lw = math.log(w)
        fc[op] = (lw, t, d, sigma)
        rho[op] = max(math.exp(lw + _lrisk((t - d) / sigma, XMAX, SLOPE)),
                      1e-6)

    # --- Restriction: who competes ---------------------------------------
    # (a) Cross-machine active filter: nobody may leave a machine idle past
    #     the instant at which some available operation could already be
    #     finished, so the schedule built is active.
    c_star = min(c.end for c in candidates)
    competing = [c for c in candidates if c.start < c_star]
    if not competing:                       # only if some p(o,m) == 0
        competing = list(candidates)
    # (b) Routing economy at a work-for-time exchange rate.  An operation may
    #     not spend more than ALPHA times the least work it could be done
    #     with - its fast machine being busy, it simply waits a step, which
    #     loose due dates can afford.  An extravagant option is readmitted
    #     only if it truly finishes SOONER than the best economical option
    #     for the same operation AND either the exchange is at least at par
    #     (time saved >= extra work spent, so the shop loses no capacity on
    #     balance) or the job is already forecast late and pays w_j per unit
    #     of further delay.  Nothing that merely burns capacity gets in.
    lean_best = {}          # op -> (end, proc) of its best economical option
    for c in candidates:
        if c.proc <= ALPHA * p_min[c.op]:
            v = lean_best.get(c.op)
            if v is None or c.end < v[0]:
                lean_best[c.op] = (c.end, c.proc)
    lean = []
    for c in competing:
        if c.proc <= ALPHA * p_min[c.op]:
            lean.append(c)
            continue
        lb = lean_best.get(c.op)
        if lb is None:
            lean.append(c)                  # no economical option exists
            continue
        gain = lb[0] - c.end                # time saved over the lean option
        if gain <= 0.0:
            continue                        # more work and no sooner: never
        if gain >= c.proc - lb[1]:
            lean.append(c)                  # the exchange is at least at par
            continue
        if fc[c.op][1] > fc[c.op][2]:       # forecast late, and truly faster
            lean.append(c)
    if lean:
        competing = lean
    # (c) Bounded deliberate idleness: a machine may not be held open longer
    #     than the operation it waits for, counted from the earliest instant
    #     an economical claimant could have taken it.  Nothing is lost by
    #     this, only deferred - the machine stays free and the candidate is
    #     offered again next step, decided then with more information.
    prompt = [c for c in competing
              if c.start - first_use[c.machine] <= IDLE * c.proc]
    if prompt:
        competing = prompt

    # --- Selection: who wins ---------------------------------------------
    # Externality of each candidate: the urgency pressing on its machine -
    # each rival discounted by how soon it can really turn up and shared over
    # its eligible machines in proportion to the speed each offers it, the
    # same routing law the queue forecast uses, so a rival that can be served
    # much faster elsewhere barely presses here - times the machine time this
    # candidate actually consumes; and the cheapest option of each operation.
    invs = {}
    for c in candidates:
        invs[c.op] = invs.get(c.op, 0.0) + ((1.0 / c.proc) if c.proc > 0.0
                                            else 1e9)
    dem = {}
    for c in candidates:
        lag = c.start - now
        disc = tau / (tau + lag) if lag > 0.0 else 1.0
        ip = (1.0 / c.proc) if c.proc > 0.0 else 1e9
        dem[c.machine] = (dem.get(c.machine, 0.0)
                          + rho[c.op] * disc * ip / invs[c.op])
    price = {}
    pmin = {}
    for c in candidates:
        occ = c.end - first_use[c.machine]   # machine time really consumed
        if occ <= 0.0:
            occ = 1e-9
        pr = dem[c.machine] * occ
        price[(c.op, c.machine)] = pr
        cur = pmin.get(c.op)
        if cur is None or pr < cur:
            pmin[c.op] = pr

    scored = []
    walked = {}
    best_s = None
    for c in competing:
        waste = c.start - first_use[c.machine]
        if waste < 0.0:
            waste = 0.0
        p_eff = c.proc + waste
        if p_eff <= 0.0:
            p_eff = 1e-9
        # externality relative to the cheapest machine open to this operation
        cont = math.log(price[(c.op, c.machine)] / pmin[c.op])
        # routing loss: completion thrown away versus the best machine
        # offered for this same operation
        delay = c.end - e_best[c.op]
        # scarcity bonus: what this operation could still reach WITHOUT m
        bb = b1[c.op]
        if bb[1] != c.machine:
            alt = bb[0]                     # its best machine is still free
        else:
            sec = b2.get(c.op)
            alt = sec[0] if sec is not None else None
        if alt is None:                     # m is its only machine
            excl = 1.0
        else:
            g = alt - c.end
            excl = (1.0 - math.exp(-g / tau)) if g > 0.0 else 0.0
        # tardiness risk READ AT THE JOB COMPLETION THIS CANDIDATE IMPLIES:
        # the job's remaining route is re-walked through the projected queues
        # from this candidate's own completion, so a routing loss a downstream
        # queue would have absorbed anyway costs nothing, while one that
        # really pushes the job out is charged in full - dear for a job at
        # risk, nearly free for one forecast comfortably early
        lw, t_hat, dd, sig = fc[c.op]
        if delay <= 0.0:                    # its best machine: already walked
            ch = t_hat
        elif look:
            key = (c.op, c.end)
            ch = walked.get(key)
            if ch is None:
                jj = c.op[0]
                ch = _route_end(c.end, _fld(jobs[jj], 'operations'),
                                c.op[1] + 1, jj, shares, avail, now, HALF)
                walked[key] = ch
        else:                               # oversized: full propagation
            ch = t_hat + delay
        lr = lw + _lrisk((ch - dd) / sig, XMAX, SLOPE)
        s = (lr - math.log(p_eff)
             - delay / tau
             - THETA * cont
             + PSI * excl)
        scored.append((s, c))
        if best_s is None or s > best_s:
            best_s = s

    # Indices within BAND are indistinguishable: finish sooner.
    band = [c for (s, c) in scored if s >= best_s - BAND]
    return min(band, key=lambda c: (c.end, c.start, c.op[0], c.op[1],
                                    c.machine))

# EVOLVE-BLOCK-END
