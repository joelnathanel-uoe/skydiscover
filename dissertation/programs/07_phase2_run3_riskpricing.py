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


def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    """
    Tardiness-risk pricing: every candidate is priced as the weighted
    tardiness RATE it relieves per unit of machine time it burns, urgency
    being the probability the job ends late rather than a hard slack.

    RESTRICTION: cross-machine active filter - with c* the smallest
                 earliest-completion among all candidates, only candidates
                 able to START STRICTLY BEFORE c* compete (Giffler-Thompson
                 applied on every machine at once, so all routing options
                 survive the filter and the schedule built is active).
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
                 shop spreads its duration evenly over its eligible machines,
                 giving each machine a residual workload q_m on top of its
                 committed backlog, and the job's next operations are routed
                 greedily, each arriving at machine m no earlier than
                 now + HALF*q_m MINUS ITS OWN SHARE OF THAT QUEUE (a job never
                 waits behind itself), so a job whose route crosses the
                 bottleneck is seen as at risk while one with a clear road
                 ahead is not.  sigma_j = K*sqrt(p_bar*(p_bar + W_j)) is the
                 width of the risk curve: the forecast error grows like the
                 SQUARE ROOT OF THE WORK W_j still to do behind this
                 operation, a random walk in work rather than in operation
                 count (it coincides with K*p_bar*sqrt(1+r_j) when the job's
                 operations are of typical length).  In the far tail
                 log(rho_j) degrades to log(w_j) + slack/sigma_j, a linear
                 slack rule that still discriminates while every job is on
                 time, and it saturates at w_j once a job is certainly late,
                 where the index becomes WSPT among the tardy.  p_eff = p +
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
    PARAMETERS:  HALF = 0.5 - an operation joining a queue expects to wait
                 half of it; K = 2.0 - urgency width, in mean operation
                 durations per sqrt(operation of remaining work); THETA = 0.5
                 - price of machine contention, split evenly between rival
                 urgency and projected busy time; PSI = 0.5 - log-priority
                 worth of being irreplaceable on a machine; BAND = 0.05 -
                 relative index tolerance within which two candidates are
                 indistinguishable.
    """
    HALF = 0.5
    K = 2.0
    THETA = 0.5
    PSI = 0.5
    BAND = 0.05

    # --- Restriction: who competes ---------------------------------------
    # Nobody may leave a machine idle past the instant at which some
    # available operation could already be finished.
    c_star = min(c.end for c in candidates)
    competing = [c for c in candidates if c.start < c_star]
    if not competing:                       # only if some p(o,m) == 0
        competing = list(candidates)

    # --- Selection: who wins ---------------------------------------------
    jobs = _fld(instance, 'jobs')

    # Best completion attainable now for each available operation, and the
    # earliest instant each machine could be put to work by anybody.
    e_best = {}
    first_use = {}
    b1 = {}
    b2 = {}
    for c in candidates:
        e = e_best.get(c.op)
        if e is None or c.end < e:
            e_best[c.op] = c.end
        f = first_use.get(c.machine)
        if f is None or c.start < f:
            first_use[c.machine] = c.start
        # two cheapest completions of the operation: the fallback left to it
        # if it is denied a given machine
        p1 = b1.get(c.op)
        if p1 is None or c.end < p1[0]:
            b2[c.op] = p1
            b1[c.op] = (c.end, c.machine)
        else:
            p2 = b2.get(c.op)
            if p2 is None or c.end < p2[0]:
                b2[c.op] = (c.end, c.machine)

    # One pass over EVERY unscheduled operation of the shop: best-case work
    # still to do behind each operation of a job (suffix), the mean duration
    # of an operation still to be scheduled (the time scale), and the
    # residual workload each machine must expect - every operation spreads
    # its duration evenly over the machines eligible for it.
    now = min(c.start for c in candidates)
    suffix = {}
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
            na = len(alts)
            alt_cnt += na
            mn = alts[0][1]
            for a in alts:
                p = a[1]
                if p < mn:
                    mn = p
                load[a[0]] = load.get(a[0], 0.0) + p / na
            suf[k] = suf[k + 1] + mn
            tot_p += mn
            cnt += 1
        suffix[j] = suf
    p_bar = (tot_p / cnt) if cnt else 1.0
    if p_bar <= 0.0:
        p_bar = 1.0

    # Queue a newly arriving operation faces on each machine: committed
    # backlog still to run plus the residual workload headed for it.
    q = {}
    for m in load:
        b = machine_free.get(m, 0.0) - now
        q[m] = (b if b > 0.0 else 0.0) + load[m]
    for c in candidates:
        if c.machine not in q:
            b = machine_free.get(c.machine, 0.0) - now
            q[c.machine] = b if b > 0.0 else 0.0
    tau = K * p_bar

    # Marginal weighted-tardiness rate of each available operation's job,
    # the risk being read off a forecast completion obtained by routing the
    # job's remaining operations greedily through the projected queues.
    look = alt_cnt <= _LOOKAHEAD_CAP
    lrho = {}
    rho = {}
    for op, eb in e_best.items():
        j, o = op
        job = jobs[j]
        w = float(_fld(job, 'weight'))
        if w <= 0.0:
            w = 1e-9
        d = float(_fld(job, 'due_date'))
        ops_j = _fld(job, 'operations')
        L = len(ops_j)
        suf = suffix.get(j)
        rem = suf[o + 1] if suf is not None else 0.0   # work still behind o
        if look:
            t = eb
            for k in range(o + 1, L):
                alts = ops_j[k]
                na = len(alts)
                best = None
                for a in alts:
                    # a job does not queue behind its own projected share
                    qm = q.get(a[0], 0.0) - a[1] / na
                    st = t
                    if qm > 0.0:
                        av = now + HALF * qm
                        if av > st:
                            st = av
                    e2 = st + a[1]
                    if best is None or e2 < best:
                        best = e2
                if best is not None:
                    t = best
        else:                       # oversized instance: work-limited bound
            t = eb + rem
        # forecast error: a random walk over the work still to be done
        sigma = K * math.sqrt(p_bar * (p_bar + rem))
        if sigma <= 0.0:
            sigma = 1.0
        lr = math.log(w) + _log_sigmoid((t - d) / sigma)
        lrho[op] = lr
        rho[op] = max(math.exp(lr), 1e-6)

    # Price of each machine: the urgency pressing on it - each rival
    # discounted by how soon it can really turn up and shared over the
    # machines it could equally well use - times how long the machine stays
    # busy; and the cheapest machine still eligible for each operation.
    nalt = {}
    for c in candidates:
        nalt[c.op] = nalt.get(c.op, 0) + 1
    dem = {}
    for c in candidates:
        lag = c.start - now
        disc = tau / (tau + lag) if lag > 0.0 else 1.0
        dem[c.machine] = (dem.get(c.machine, 0.0)
                          + rho[c.op] * disc / nalt[c.op])
    price = {}
    pmin = {}
    for c in candidates:
        pr = dem[c.machine] * (p_bar + q.get(c.machine, 0.0))
        price[(c.op, c.machine)] = pr
        cur = pmin.get(c.op)
        if cur is None or pr < cur:
            pmin[c.op] = pr

    scored = []
    best_s = None
    for c in competing:
        waste = c.start - first_use[c.machine]
        if waste < 0.0:
            waste = 0.0
        p_eff = c.proc + waste
        if p_eff <= 0.0:
            p_eff = 1e-9
        # geometric mean of the two contention factors (hence the 0.5)
        cont = 0.5 * math.log(price[(c.op, c.machine)] / pmin[c.op])
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
        s = (lrho[c.op] - math.log(p_eff)
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
