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

    RESTRICTION: cross-machine active filter PLUS PROFITABLE WAITS - with c*
                 the smallest earliest-completion among all candidates, a
                 candidate competes if it can START STRICTLY BEFORE c*
                 (Giffler-Thompson applied on every machine at once, so every
                 routing option survives the filter); in addition a candidate
                 invisible to that test competes when the completion it BUYS,
                 (best end reachable without waiting) - end, exceeds the
                 avoidable machine idle it COSTS, start - (earliest instant
                 anybody could use m).  One currency, machine time, the same
                 the selection prices in, and no parameter of its own; the
                 clause is vacuous when a duration does not depend on the
                 machine (a later start then always means a later finish), so
                 it widens the reachable schedule class exactly where machine
                 speeds differ.
    SELECTION:   maximise
                   log(rho_j) - log(p_eff + delay + harm/rho_j)
                   - THETA*log(price(o,m)/price(o, cheapest m))
                   + PSI*(1 - exp(-(alt_end - end)/tau))
                 with tau = K*p_bar, p_bar the mean duration of an operation
                 still to be scheduled.  rho_j = w_j*sigmoid((Chat_j-d_j)/
                 sigma_j) is job j's marginal weighted-tardiness rate - its
                 weight times the probability it ends late - with Chat_j
                 obtained by WALKING THE JOB'S REMAINING ROUTE THROUGH THE
                 PROJECTED MACHINE QUEUES: every unscheduled operation of the
                 shop spreads its duration over its eligible machines IN
                 PROPORTION TO HOW LIKELY IT IS TO BE ROUTED THERE - a
                 machine is chosen with probability proportional to 1/p(o,m),
                 so each eligible machine expects the same amount 1/sum(1/p)
                 of work and the total an operation puts on the shop is the
                 HARMONIC mean of its durations, near its fastest machine
                 rather than its average one - giving each machine a residual
                 workload q_m on top of its committed backlog, and the job's
                 next operations are routed greedily, each arriving at machine
                 m no earlier than now + HALF*q_m MINUS ITS OWN SHARE OF THAT
                 QUEUE (a job never waits behind itself), so a job whose route
                 crosses the bottleneck is seen as at risk while one with a
                 clear road ahead is not.  sigma_j = K*sqrt(p_bar*(p_bar +
                 H_j)) is the width of the risk curve, H_j = Chat_j - e_best
                 being the FORECAST HORIZON: the error of a forecast grows
                 like a random walk over the flow time it spans, QUEUEING
                 INCLUDED, not merely over the work content, because waiting
                 is what a forecast is really uncertain about (with no
                 queueing H_j is the remaining work and sigma_j reduces to the
                 classical K*p_bar*sqrt(1+r_j)).  In the far tail
                 log(rho_j) would degrade to log(w_j) + slack/sigma_j, an
                 UNBOUNDED slack rule under which a comfortably safe job is
                 worth e^-40 of a borderline one; the log-probability of
                 lateness is therefore COMPRESSED SMOOTHLY through
                 -LCAP*tanh(-x/LCAP), but x is measured AGAINST THE MOST
                 EXPOSED JOB ON THE FLOOR, x = risk - max risk <= 0.  Only
                 DIFFERENCES of log(rho) matter here - displacement enters as
                 harm/rho_j, contention as a price ratio - so an absolute
                 anchor throws information away: with loose due dates every
                 job sits deep in the tail and all of them flatten onto one
                 floor, killing the risk term exactly where the few
                 endangered jobs must be found.  Anchored at the leader,
                 moderate instances are unchanged (their leader is already at
                 or past its due date) while on loose instances the safe jobs
                 keep their order at full resolution within a band of width
                 LCAP, and the VALUE RATIOS that price displacement below
                 stay finite.  rho saturates at w_j
                 once a job is certainly late, where the index becomes WSPT
                 among the tardy.  p_eff = p +
                 max(0, start - earliest instant any available operation could
                 occupy that machine) is exactly the machine time burned:
                 unavoidable idle is free, self-inflicted idle is paid for.
                 harm is the DISPLACEMENT CHARGE, the value this candidate
                 destroys by seizing the machine: among the rival candidates
                 on m FOR WHICH m IS THEIR BEST REMAINING ROUTING (a rival
                 whose minimum-end machine lies elsewhere simply goes there
                 and loses nothing, so it is not counted), the largest
                 rho' * min(end - start', next-best-routing loss') - the rival
                 is pushed back until this operation finishes, but never by
                 more than it would lose by rerouting.  Only one rival can
                 actually take the machine next, so the MAXIMUM, not the sum,
                 is the value forgone.  It enters the index as machine time
                 converted at the candidate's OWN rate, harm/rho_j: the time
                 this job would need to earn back what it destroys.  The term
                 is exactly zero when nobody is delayed or when every rival
                 has an equally good machine elsewhere, and it is decisive
                 precisely where a long, unhurried operation would otherwise
                 sit on top of an urgent job that has nowhere else to go -
                 the way a greedy appender loses most of its weighted
                 tardiness on congested, moderately-due instances.
                 delay = end - e_best is the completion thrown away relative
                 to the best machine offered for the same operation.  It
                 costs the job rho_j*delay of expected weighted tardiness,
                 i.e. exactly `delay` MORE machine time at its own rate, so
                 it is simply ADDED to the time the candidate asks for - one
                 currency, no exchange rate, no parameter, and no saturation
                 (a completion given away turns into tardiness one for one).
                 price(o,m) is the geometric mean of D_m - the summed rho of
                 the operations contending for m, each discounted by
                 tau/(tau+lag) so only imminent rivals count and shared over
                 its own eligible machines so a rival that can go elsewhere
                 presses only its fair share - and p(o,m) + q_m, how long m
                 stays busy ONCE THIS OPERATION HAS RUN ON IT: the
                 operation's OWN duration there, not a shop average, so a
                 machine that is slow for this particular operation is priced
                 as the extra congestion it really creates.  It is compared
                 against the cheapest machine still eligible for the
                 operation, so a machine is penalised only when an
                 alternative exists.  The last term is the SCARCITY BONUS:
                 alt_end is the best completion this operation could still
                 reach WITHOUT machine m (infinite when m is its only machine,
                 zero bonus when an equally good machine remains), so an
                 operation with nowhere else to go outbids an equally urgent
                 rival that can be served elsewhere - least flexible first.
                 There is NO separate capacity-waste charge.  The machine time
                 an operation throws away by running on a slow machine is
                 already paid for three times over: once in p_eff, once in
                 delay = end - e_best, and once inside price(o,m), which uses
                 the operation's own duration on m.  Because it enters through
                 the price it is charged only in proportion to the urgency
                 D_m actually pressing on that machine - burning an
                 uncontended machine's spare capacity stays free, which is
                 precisely the right move when it spares a bottleneck - and it
                 is charged LOGARITHMICALLY, so a machine ten times slower
                 costs a bounded amount of priority instead of swamping the
                 index on instances where durations vary widely across
                 machines.  An unbounded linear excess charge collapses the
                 rule into "always take the fastest eligible machine", which
                 piles the shop onto a few machines and is exactly the wrong
                 move when routing speeds differ.
                 Ties: indices within BAND of the best (the index being
                 logarithmic, a fixed ratio) are equivalent and settled by
                 earliest completion, then earliest start, then smallest job,
                 operation, machine index.
    PARAMETERS:  HALF = 0.5 - an operation joining the UNCOMMITTED part of a
                 queue expects to wait half of it (the committed backlog is
                 certain and is waited out in full); K = 2.0 - urgency width,
                 in mean operation
                 durations per sqrt(operation of remaining work); THETA = 0.5
                 - price of machine contention, split evenly between rival
                 urgency and projected busy time; PSI = 0.5 - log-priority
                 worth of being irreplaceable on a machine; LCAP = 6.0 -
                 width, in log-probability, of the band into which safe jobs
                 are compressed: a job whose forecast chance of lateness is
                 already far below e^-6 (0.25%) earns almost nothing more for
                 being safer still; BAND = 0.05 - relative index tolerance
                 within which two candidates are indistinguishable.  The
                 displacement charge itself has NO parameter; RIVAL_CAP = 6 is
                 a purely computational cap (only the six most damaging
                 rivals per machine are examined) that keeps the cost linear
                 in the number of candidates.
    """
    HALF = 0.5
    K = 2.0
    THETA = 0.5
    PSI = 0.5
    LCAP = 6.0
    BAND = 0.05
    RIVAL_CAP = 6

    # --- Restriction: who competes ---------------------------------------
    # (a) Active filter: nobody may leave a machine idle past the instant at
    #     which some available operation could already be finished.
    c_star = min(c.end for c in candidates)
    active = [c for c in candidates if c.start < c_star]
    if not active:                          # only if some p(o,m) == 0
        active = list(candidates)
    # Earliest instant each machine could be put to work by SOMEBODY: idle
    # before it is unavoidable, idle after it is a deliberate wait.
    first_use = {}
    for c in candidates:
        f = first_use.get(c.machine)
        if f is None or c.start < f:
            first_use[c.machine] = c.start
    # (b) Profitable waits: a candidate whose machine is still busy - or
    #     whose job is not yet ready - is invisible to (a), yet waiting for a
    #     much faster machine can be a bargain.  Admit it when the completion
    #     it BUYS, gain = (best end reachable without waiting) - end, exceeds
    #     the avoidable machine idle it COSTS.  One currency, machine time,
    #     the same the selection prices in, and no parameter.  Vacuous when
    #     durations do not depend on the machine.
    competing = list(active)
    if len(active) < len(candidates):
        fallback_end = {}
        for c in active:
            b = fallback_end.get(c.op)
            if b is None or c.end < b:
                fallback_end[c.op] = c.end
        for c in candidates:
            if c.start < c_star:            # already among the active
                continue
            fb = fallback_end.get(c.op)
            if fb is None:
                continue
            gain = fb - c.end               # completion bought by waiting
            if gain <= 0.0:
                continue
            idle = c.start - first_use[c.machine]       # capacity it costs
            if idle < 0.0:
                idle = 0.0
            if gain > idle:
                competing.append(c)

    # --- Selection: who wins ---------------------------------------------
    jobs = _fld(instance, 'jobs')

    # Best completion attainable now for each available operation, and the
    # earliest instant each machine could be put to work by anybody.
    e_best = {}
    b1 = {}
    b2 = {}
    for c in candidates:
        e = e_best.get(c.op)
        if e is None or c.end < e:
            e_best[c.op] = c.end
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
        sh = [0.0] * L
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
                if p > 0.0:
                    inv += 1.0 / p
            # Routing-weighted share: the operation lands on machine m with
            # probability proportional to 1/p(o,m), so every eligible machine
            # expects the same amount p_m*P(m) = 1/sum(1/p) of work, and the
            # total load the operation puts on the shop is the HARMONIC mean
            # of its durations - close to its fastest machine, where it will
            # in fact be run, instead of the arithmetic mean, which loads
            # machines that are far too slow ever to be chosen.
            share = (1.0 / inv) if inv > 0.0 else 0.0
            sh[k] = share
            for a in alts:
                load[a[0]] = load.get(a[0], 0.0) + share
            suf[k] = suf[k + 1] + mn
            tot_p += mn
            cnt += 1
        suffix[j] = suf
        shares[j] = sh
    p_bar = (tot_p / cnt) if cnt else 1.0
    if p_bar <= 0.0:
        p_bar = 1.0

    # Queue a newly arriving operation faces on each machine: committed
    # backlog still to run plus the residual workload headed for it.
    q = {}
    back = {}                   # committed backlog alone: certain, not halved
    for m in load:
        b = machine_free.get(m, 0.0) - now
        if b < 0.0:
            b = 0.0
        back[m] = b
        q[m] = b + load[m]
    for c in candidates:
        if c.machine not in q:
            b = machine_free.get(c.machine, 0.0) - now
            if b < 0.0:
                b = 0.0
            back[c.machine] = b
            q[c.machine] = b
    tau = K * p_bar

    # Marginal weighted-tardiness rate of each available operation's job,
    # the risk being read off a forecast completion obtained by routing the
    # job's remaining operations greedily through the projected queues.
    look = alt_cnt <= _LOOKAHEAD_CAP
    lrho = {}
    rho = {}
    raw = {}                # absolute log-probability of lateness
    logw = {}
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
            sh = shares.get(j)
            t = eb
            for k in range(o + 1, L):
                alts = ops_j[k]
                own = sh[k] if sh is not None else 0.0
                best = None
                for a in alts:
                    m = a[0]
                    # the committed backlog is CERTAIN and is waited out in
                    # full; only the uncommitted residual queue is halved,
                    # and a job never queues behind its own projected share
                    lm = load.get(m, 0.0) - own
                    av = now + back.get(m, 0.0)
                    if lm > 0.0:
                        av += HALF * lm
                    st = t if t > av else av
                    e2 = st + a[1]
                    if best is None or e2 < best:
                        best = e2
                if best is not None:
                    t = best
        else:                       # oversized instance: work-limited bound
            t = eb + rem
        # Forecast error: a random walk over the FORECAST HORIZON - the flow
        # time still ahead of this operation, queueing included - rather than
        # over its bare work content, since queueing is the part of the
        # forecast that is actually uncertain.  With an empty shop the
        # horizon is the remaining work and this is the classical form.
        hor = t - eb
        if hor < rem:
            hor = rem
        sigma = K * math.sqrt(p_bar * (p_bar + hor))
        if sigma <= 0.0:
            sigma = 1.0
        raw[op] = _log_sigmoid((t - d) / sigma)
        logw[op] = math.log(w)

    # RELATIVE risk.  Only DIFFERENCES of log(rho) matter in this index -
    # displacement enters as harm/rho_j, contention as a price ratio - so the
    # compression band of width LCAP is anchored at the MOST EXPOSED job on
    # the floor rather than at an absolute zero.  With loose due dates every
    # job sits deep in the tail and an absolute anchor flattens them all onto
    # one floor, destroying exactly the discrimination needed there; anchored
    # at the leader, moderate instances (leader already at or past due) are
    # untouched while safe jobs keep their order at full resolution.
    r_max = max(raw.values())
    for op, r in raw.items():
        lr = logw[op] - LCAP * math.tanh((r_max - r) / LCAP)
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
        # how long m stays busy ONCE THIS OPERATION HAS RUN ON IT - the
        # operation's own duration there, not a shop average - times the
        # urgency pressing on m.  Running an operation on a machine that is
        # slow for it is therefore charged as the extra congestion it really
        # creates: nothing on an uncontended machine, and only
        # logarithmically even on a heavily contended one.
        pr = dem[c.machine] * (c.proc + q.get(c.machine, 0.0) + 1e-9)
        price[(c.op, c.machine)] = pr
        cur = pmin.get(c.op)
        if cur is None or pr < cur:
            pmin[c.op] = pr

    # Displacement: who is pushed aside by seizing a machine?  A rival only
    # suffers if THIS machine is its best remaining routing - otherwise it
    # goes elsewhere at no loss - and it never suffers more than the loss of
    # its next-best routing, since it can always reroute rather than wait.
    by_m = {}
    for c in candidates:
        lst = by_m.get(c.machine)
        if lst is None:
            by_m[c.machine] = [c]
        else:
            lst.append(c)
    blockers = {}
    for m, lst in by_m.items():
        if len(lst) < 2:
            continue
        maxend = max(x.end for x in lst)
        info = []
        for x in lst:
            bb = b1[x.op]
            if bb[1] != m:
                continue            # a machine at least as good remains
            sec = b2.get(x.op)
            # loss if denied m: reroute to the next-best machine, or - when m
            # is its only option - wait, bounded by the horizon of this step
            L = (sec[0] - x.end) if sec is not None else (maxend - x.start)
            if L <= 0.0:
                continue
            info.append((rho[x.op] * L, rho[x.op], x.start, L, x.op))
        if info:
            info.sort(key=lambda t: -t[0])   # most damage first
            blockers[m] = info[:RIVAL_CAP]

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
        # offered for this same operation.  It costs the job rho_j*delay of
        # expected weighted tardiness, i.e. exactly `delay` MORE machine time
        # at its own rate, so it is simply added below to the time the
        # candidate asks for - one currency, no exchange rate, no parameter.
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
        # displacement charge: the largest weighted-tardiness value destroyed
        # among the rivals that were counting on this very machine.  Only one
        # of them can take it next, so the maximum - not the sum - is what is
        # really forgone.
        harm = 0.0
        for (_b, rho_x, s_x, L_x, op_x) in blockers.get(c.machine, ()):
            if op_x == c.op:
                continue
            dd = c.end - s_x            # how far back this candidate pushes it
            if dd <= 0.0:
                continue                # the rival cannot even start yet
            if dd > L_x:
                dd = L_x                # it would reroute rather than wait
            h = rho_x * dd
            if h > harm:
                harm = h
        # the machine time this candidate really costs: what it burns itself,
        # plus the completion it gives up by not taking its best machine,
        # plus the value it destroys to the rivals it displaces, converted at
        # its OWN rate (harm/rho = the time it would need to earn that back)
        p_tot = p_eff + delay + harm / rho[c.op]
        s = (lrho[c.op] - math.log(p_tot)
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
