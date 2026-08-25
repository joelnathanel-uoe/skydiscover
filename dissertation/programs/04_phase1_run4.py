# EVOLVE-BLOCK-START
import math


def _fld(obj, name):
    """Read a field from a dict-like or attribute-like record."""
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    """
    MARGINAL-TARDINESS RAMP dispatcher, evaluated on FULL shop information
    but choosing only inside an ACTIVE-SCHEDULE window, with an ADAPTIVE ramp
    width, a CHAIN-FORECAST slack, a MEASURED delay-propagation factor, a
    MEASURED one-step deferral exposure, an EXACT crossing term, a
    two-operation (WINQ) routing horizon, an ARRIVAL-aware contention charge
    and a near-tie band arbitrated by the EXACT MARGINAL WEIGHTED TARDINESS.
    The ATC exponential is replaced by the exact one-step marginal cost of
    deferring the operation plus a fat saturating tail whose HEIGHT shrinks
    with measured congestion, and every other term is a MEASURED quantity
    rather than a processing-time guess.

    EVALUATE GLOBALLY, CHOOSE LOCALLY (new).  Congestion, propagation,
    deferral exposure, displacement, arrivals and machine loads describe the
    SHOP, so they are measured over EVERY candidate; only the final argmax
    ranges over the window, and the routing-quality reference is the best
    SELECTABLE routing of the same operation.  Restricting the measurements
    to the window biases them exactly when the window bites hardest.

    ACTIVE WINDOW (new).  A candidate whose start is at or beyond
    e_min = min_c e(c) can always be deferred at NO cost: scheduling the
    earliest-completing candidate first cannot push it back (other machine,
    other job), so committing such a far-future machine slot now merely
    discards the information that one more committed operation would give.
    The admissible window is therefore
        t_min + min(THETA*p_bar, DF*(e_min - t_min) + FLOOR*p_bar),
    the classical Giffler-Thompson active-schedule restriction softened by a
    floor (so routing alternatives of very short operations survive) and
    still capped by the hard non-delay cap.  A busy but fast machine is thus
    implicitly RESERVED -- no one may append to it until the rest of the shop
    has caught up in time -- which is exactly the chronological discipline an
    append-only builder needs.

    Routing losses are additionally scaled by the measured propagation factor
    rho_eff: a detour, like any other delay, only hurts to the extent that it
    reaches the completion time.  Finally, candidates whose scores differ by
    less than EPS are treated as tied and resolved by earliest completion:
    hundredths of a log unit are noise in a hand-built priority.

    ADAPTIVE WIDTH (new).  The ramp width is nominally the one-step deferral
    horizon K_eff*p_bar.  On a loose instance every live slack is many times
    that, so all operations sit in the flat tail and the due-date signal
    degenerates into near-WSPT.  The width is therefore blended with the
    OBSERVED median positive slack of the live operations,
    width = clamp(SB*med + (1-SB)*K_eff*p_bar, SBLO*.., SBHI*..), which keeps
    the discriminating linear part of the ramp spanning the actual slack
    distribution while leaving a tight shop essentially unchanged.

    PROPAGATION.  A ramp value w*f(x) assumes a delay of DELTA here delays
    the JOB by DELTA; that is false when the tail must queue anyway.  The
    remaining chain is rolled forward twice over the current machine free
    times, from e_best and from e_best+p_bar; rho in [0,1] is the measured
    share of a delay that survives to the completion time, and
    rho_eff = RHOF+(1-RHOF)*rho multiplies both the value of acting now and
    the rate at which an operation is defended against displacement.

    EXPOSURE.  delta(o) = how much later o can finish if the rival that frees
    its machine soonest seizes it instead -- the deferral horizon o really
    faces; it stretches (small delta) or compresses (large delta) the ramp
    and is rewarded directly through DEF*urg*log1p(delta/p_bar).

    RESTRICTION 1 -- MACHINE-LOCAL DOMINANCE (exact, lossless).  If another
      candidate on the SAME machine m completes at or before s(o,m), then
      committing it first leaves free[m] <= s(o,m), so (o,m) still starts at
      EXACTLY s(o,m) next round (its job is untouched), while taking (o,m)
      first would strictly delay that rival.  (o,m) is dominated this round
      and is dropped at no cost.  The tables it needs are the ones the
      deferral-exposure term already builds, so it is free.
    RESTRICTION 2 -- window + hard non-delay cap: s(o,m) > t_min +
      min(THETA*p_bar, DF*(e_min-t_min)+FLOOR*p_bar) is dropped (append-only
      inserted idle is irreversible); each restriction falls back to the
      previous set if it empties it.  p_bar = mean, over the distinct
      available OPERATIONS, of that operation's mean processing time, so
      routing-rich operations cannot inflate the scale.
    SMITH REGIME.  When the measured congestion is high almost every live
      job is late, so TWT and weighted COMPLETION time differ by a constant
      and the right static rule is Smith's ratio w/(p+R).  Since -log(cost)
      - lam*log1p(R/cost) interpolates exactly between WSPT (lam=0) and
      Smith (lam=1), lam = SRW*(0.25+0.75*cong) + SMW*cong^2 is ramped up
      quadratically with congestion, leaving loose shops untouched.
    EXACT RAMP WIDTH.  Deferring by DELTA moves the completion by rho*DELTA,
      so the marginal ramp is not only LOWER by rho but also NARROWER: the
      width is divided by (1-RHW+RHW*rho_eff), shifting priority to the
      operations whose delay genuinely reaches a due date.

    PRIORITY FUNCTION (the novel core).  Deferring this operation by one
    round costs it about DELTA_t = K_eff*p_bar.  The weighted tardiness that
    decision actually creates is w*[T(C+DELTA_t) - T(C)]/DELTA_t with
    T(x)=max(0,x-d), i.e. a LINEAR RAMP in the normalised slack x:
        f(x) = max(0, 1-x) + TAIL*(1 + x/A)^(-A)
    f = 1+TAIL while the job is already late (asymptotically WSPT, correct),
    falls ~2x more sharply than exp(-x) exactly across the due date where the
    tardiness is decided, and then hands over to a heavy-tailed term whose
    log decays like -A*log1p(x/A): far-from-due jobs are de-emphasised but
    never ignored, which is what keeps loose instances from being lost.
    urg = f/(1+TAIL) in (0,1] drives every regime blend below.

    SLACK (CHAIN FORECAST).  R(j,o) = sum over later ops of
    0.5*min_m p + 0.5*mean_m p, inflated by lead = 1 + QW*min(2, n_live/M)
    because the tail must QUEUE.  The whole remaining chain is additionally
    rolled forward over the CURRENT machine free times, and
    qd = min(cap, max(0, t_end - e_best - sum_t min_m p)) is the forecast
    downstream QUEUE: the part of the job's future that is pure waiting for
    already-booked machines rather than processing.  slack = d_j - e_best -
    lead*R - PSI*qd; positive slack is divided by (1+LAM*(nrem-1)) so this
    decision may spend only its own share, while the undivided value raw is
    what a routing detour may still absorb.

    SCORE (maximised):
        log w_j + log f(x) + log rho_eff         ramp value that propagates
      - log(p + IW*(0.5+pq)*(idle_av + IUN*idle_un))   machine time burnt
      - G*(0.35+0.65*urg)*eff/p_bar              2-op routing quality
      - (s-t_min)/(B_eff*p_bar)                  chronology
      - DBLK*(1.5-urg)*log1p((blk+ARW*arr)/(w_bar*p_bar))  contention
      + REG*(0.4+0.6*urg)*spread(o)              routing regret
      + DEF*urg*log1p(delta/p_bar)               measured deferral exposure
      - BOT*(1-0.7*urg)*detour(o,m)              forecast bottleneck
      - SRW*(0.25+0.75*cong)*urg*log1p(R/p_bar)  WSRPT among late jobs

    qual = (e-e_best) + WQ*winq(o,m) adds the successor's earliest start to
    the routing loss (a "faster" routing whose successor machine is booked
    is not faster at all); eff charges the part of qual exceeding raw in
    full and the absorbable part only at MINF.  arr prices, exactly like
    blk, the push-back this occupation imposes on the successor operations
    that will ARRIVE on m at their predecessors' best completion times.

    idle_av = s(o,m)-s_min(m) is idle THIS routing inserts, idle_un =
    max(0, s_min(m)-free[m]) is idle no candidate on m can avoid and is only
    partly charged.  blk prices the measured deterioration of each rival's
    best reachable completion once m is frozen until e(o,m), weighted by the
    rival's marginal weighted-tardiness rate w*urg, so a rival with a good
    alternative routing is cheap.  spread = min(1,(e_2nd-e_1st)/p_bar) (1 for
    a unique routing) takes rigid work while its machine is free; detour is
    the extra forecast load versus the least loaded routing of the SAME
    operation (0 for a unique routing).  Congestion cong (weighted share of
    live jobs already predicted tardy) sets K_eff = K*(0.55+0.9*cong) and
    B_eff = B*(1.25-0.5*cong): a saturated shop turns idle straight into
    tardiness, so chronology is enforced harder.
    CROSSING: XW*log1p(w*max(0, rho_eff*delta - raw)/(w_bar*p_bar)) is the
    exact weighted tardiness one round of waiting really creates: zero while
    the deferral provably stays on the near side of the due date, w*D once
    the job is late.  The WSRPT pull is normalised by Smith's rule against
    the operation's OWN time burnt, SND*cost + (1-SND)*p_bar, and on a LAST
    operation the absorbable share of a routing loss is not charged at all
    (nothing downstream can absorb or amplify it, so the extra tardiness is
    exactly max(0, qual-raw)).
    NEAR-TIE ARBITRATION: candidates within EPS log units of the best are
    below the modelling accuracy of every term above; they are re-ranked by
    the exact marginal weighted tardiness this commitment creates -- own
    detour w*max(0, rho*(e-e_best)-raw) plus the displacement
    w_r*max(0, rho_r*push_r-raw_r) of the rivals queued on m and of the
    successors certain to arrive on m -- quantised so that differences below
    the forecast's own resolution still fall back on earliest completion.
    """
    n = len(candidates)
    if n == 1:
        return candidates[0]

    K = 2.2        # deferral horizon -> ramp width (scaled by congestion)
    A = 6.0        # tail saturation exponent
    TAIL = 0.35    # height of the saturating tail under the ramp
    TAILC = 0.35   # shrink of that tail height under measured congestion
    SB = 0.45      # share of the ramp width taken from the OBSERVED slacks
    SBLO = 0.85    # lower clamp of the adaptive width, in units of K_eff*p_bar
    SBHI = 2.50    # upper clamp of the adaptive width
    DEF = 0.35     # bonus for measured one-step deferral exposure
    DW = 0.40      # share of the ramp width driven by measured exposure
    DCAP = 2.0     # exposure cap, in units of p_bar
    MINF = 0.45    # share of an ABSORBABLE routing loss still charged
    WQ = 0.5       # weight of the successor-queue (WINQ) part of quality
    ARW = 0.5      # weight of future arrivals in the contention charge
    TOPA = 6       # arrivals examined per machine
    B = 1.5        # chronology / delay aversion
    G = 0.9        # routing-quality penalty
    IW = 0.6       # share of inserted machine idle charged as work
    IUN = 0.35     # share of UNAVOIDABLE machine idle still charged
    THETA = 2.5    # hard non-delay cap, in units of p_bar
    DF = 1.0       # active-schedule (Giffler-Thompson) window factor
    FLOOR = 0.7    # softening floor of that window, in units of p_bar
    RHQ = 0.6      # share of the routing loss governed by propagation
    XW = 0.50      # weight of the EXACT one-deferral crossing term
    SND = 0.6      # share of the Smith normaliser taken from the op's own p
    SMW = 0.55     # extra Smith-ratio pull, gated by squared congestion
    RHW = 0.5      # blend towards the EXACT rho-narrowed marginal ramp
    EPS = 0.03     # near-tie band, arbitrated by exact marginal TWT
    DBLK = 0.9     # one-step displacement-regret charge
    REG = 0.55     # routing-regret bonus
    LAM = 0.25     # how strongly slack is split over remaining operations
    SRW = 0.20     # weighted shortest-remaining-work pull on late jobs
    QW = 0.12      # queueing allowance per live operation per machine
    PSI = 0.6      # weight on the forecast downstream queueing
    QCAP = 2.5     # forecast-queue cap per remaining op, in units of p_bar
    QMAX = 6.0     # absolute cap on the forecast downstream queue
    RHOF = 0.45    # floor of the delay-propagation factor
    RSTEP = 1.0    # probe delay of the propagation roll, in units of p_bar
    BOT = 0.45     # forecast-bottleneck detour charge
    RLX = 1.0      # width of the release discount in the load forecast
    LOOK = 3       # operations of look-ahead in the load forecast
    DECAY = 0.55   # discount per further look-ahead operation
    TOPR = 10      # rivals examined per machine in the regret sum
    INF = float('inf')

    jobs = _fld(instance, 'jobs')
    try:
        M = _fld(instance, 'n_machines')
    except Exception:
        M = 0
    if not M or M < 2:
        M = 2

    # --- per-operation aggregation: scale, best and second-best routing ----
    t_min = candidates[0].start
    e1 = {}
    e2 = {}
    m1 = {}
    psum = {}
    nm = {}
    for c in candidates:
        if c.start < t_min:
            t_min = c.start
        k = c.op
        b = e1.get(k)
        if b is None:
            e1[k] = c.end
            m1[k] = c.machine
        elif c.end < b:
            e2[k] = b
            e1[k] = c.end
            m1[k] = c.machine
        else:
            s = e2.get(k)
            if s is None or c.end < s:
                e2[k] = c.end
        psum[k] = psum.get(k, 0.0) + c.proc
        nm[k] = nm.get(k, 0) + 1

    p_bar = 0.0
    for k, s in psum.items():
        p_bar += s / nm[k]
    p_bar /= len(psum)
    if p_bar <= 0.0:
        p_bar = 1.0

    # --- earliest / second-earliest completion offered on each machine -----
    # (this also identifies the rival that would seize m if a given operation
    # waited one round, which both restrictions and the exposure term need)
    d1 = {}
    d1k = {}
    d2 = {}
    for c in candidates:
        m = c.machine
        e = c.end
        a = d1.get(m)
        if a is None:
            d1[m] = e
            d1k[m] = c.op
        elif e < a:
            d2[m] = a
            d1[m] = e
            d1k[m] = c.op
        else:
            b = d2.get(m)
            if b is None or e < b:
                d2[m] = e

    # --- RESTRICTION 1: MACHINE-LOCAL DOMINANCE (exact, lossless) ----------
    # if another candidate c' on the SAME machine m satisfies e(c') <= s(o,m)
    # then appending c' first leaves free[m] = e(c') <= s(o,m), so this
    # operation STILL starts at exactly s(o,m) next round (its own job is
    # untouched), whereas taking it now would strictly delay c'.  It is
    # therefore dominated this round: dropping it costs nothing and buys one
    # more committed operation of information.  This is the active-schedule
    # argument applied PER MACHINE, where it is actually valid -- the global
    # window below compares against min_c e(c) over ALL machines, which is
    # neither necessary nor sufficient for the same guarantee.
    rival_end = {}
    ndom = []
    for c in candidates:
        m = c.machine
        re_ = d1[m] if d1k[m] != c.op else d2.get(m)
        rival_end[(c.op, m)] = re_
        if re_ is None or c.start < re_:
            ndom.append(c)
    if not ndom:
        ndom = candidates

    # --- RESTRICTION 2: active-schedule window + hard non-delay cap --------
    # nothing is lost by deferring a candidate that starts at or after the
    # earliest completion currently reachable, and one more committed
    # operation of information is gained; a small floor keeps the routing
    # alternatives of very short operations inside the window
    e_min = None
    for v in e1.values():
        if e_min is None or v < e_min:
            e_min = v
    win = DF * (e_min - t_min) + FLOOR * p_bar
    hard = THETA * p_bar
    if win > hard:
        win = hard
    limit = t_min + win
    competing = [c for c in ndom if c.start <= limit]
    if not competing:
        competing = ndom
    if len(competing) == 1:
        return competing[0]

    sel_m = set()
    for c in competing:
        sel_m.add(c.machine)

    # --- FULL-INFORMATION ANALYSIS SET -------------------------------------
    # every MEASURED quantity below (congestion, propagation, exposure,
    # displacement, machine load) describes the SHOP, not the window:
    # restricting them to the selectable subset biases them exactly when the
    # window bites hardest.  Evaluate globally, choose locally.
    e_best = e1
    by_m = {}
    s_min = {}
    for c in candidates:
        lst = by_m.get(c.machine)
        if lst is None:
            by_m[c.machine] = [c]
            s_min[c.machine] = c.start
        else:
            lst.append(c)
            if c.start < s_min[c.machine]:
                s_min[c.machine] = c.start

    # --- MEASURED ONE-STEP DEFERRAL EXPOSURE -------------------------------
    # if this operation waits one round, the machine it wants is seized by
    # the rival that frees it soonest (one candidate per op-machine pair, so
    # that rival is necessarily a different operation); delta is how much
    # later it can then finish -- the horizon its own ramp really faces.
    # The rival table was already built for the dominance restriction, so the
    # exposure costs no extra pass over the candidates.
    delta_of = {}
    for c in candidates:
        k = c.op
        m = c.machine
        re_ = rival_end[(k, m)]
        if re_ is None:
            # no rival on m at all: it stays free, waiting costs nothing here
            pushed = c.end
        else:
            jr = job_ready.get(k[0], 0)
            st = jr if jr > re_ else re_
            pushed = st + c.proc
        cur = delta_of.get(k)
        if cur is None or pushed < cur:
            delta_of[k] = pushed
    for k in delta_of:
        dv = delta_of[k] - e_best[k]
        delta_of[k] = dv if dv > 0.0 else 0.0

    # --- forecast: near-term work each offered machine must absorb ---------
    load = {}
    for j, nx in job_next_op.items():
        if nx is None:
            continue
        ops = _fld(jobs[j], 'operations')
        last = nx + LOOK
        if last > len(ops):
            last = len(ops)
        # a job that is not ready yet cannot contend for a machine in the
        # near term: fade its demand out with its arrival distance, else
        # unreleased work makes every machine look like a bottleneck
        jr = job_ready.get(j, 0)
        if jr > t_min:
            wgt = 1.0 / (1.0 + (jr - t_min) / (RLX * LOOK * p_bar))
        else:
            wgt = 1.0
        for q in range(nx, last):
            alts = ops[q]
            na = len(alts)
            if na:
                share = wgt / na
                for a in alts:
                    load[a[0]] = load.get(a[0], 0.0) + share * a[1]
            wgt *= DECAY

    fore = {}
    fmin = None
    fmax = None
    for m in by_m:
        mf = machine_free.get(m, 0)
        f = (mf if mf > t_min else t_min) + load.get(m, 0.0)
        fore[m] = f
        if fmin is None or f < fmin:
            fmin = f
        if fmax is None or f > fmax:
            fmax = f
    span = (fmax - fmin) if fmax is not None else 0.0
    inv_span = 1.0 / span if span > 0.0 else 0.0
    bott = {}
    for m, f in fore.items():
        bott[m] = (f - fmin) * inv_span
    bmin = {}
    for c in candidates:
        k = c.op
        v = bott.get(c.machine, 0.0)
        b0 = bmin.get(k)
        if b0 is None or v < b0:
            bmin[k] = v

    # --- chain forecast: remaining work, downstream queue, propagation -----
    q_cong = len(e_best) / float(M)
    if q_cong > 2.0:
        q_cong = 2.0
    lead = 1.0 + QW * q_cong

    info = {}
    succ_of = {}
    rho_of = {}
    probe = RSTEP * p_bar
    w_tot = 0.0
    w_late = 0.0
    for k in e_best:
        j, o = k
        job = jobs[j]
        ops = _fld(job, 'operations')
        nops = len(ops)
        eb = e_best[k]
        rem = 0.0
        s_min_p = 0.0
        tcur = eb            # machine-aware forward roll of the whole tail
        tcur2 = eb + probe   # ... and the same roll with a probe delay added
        for t in range(o + 1, nops):
            mn = None
            sm = 0.0
            cnt = 0
            fb = None
            fb2 = None
            live2 = tcur2 > tcur
            for a in ops[t]:
                pt = a[1]
                sm += pt
                cnt += 1
                if mn is None or pt < mn:
                    mn = pt
                fv = machine_free.get(a[0], 0)
                st = fv if fv > tcur else tcur
                v = st + pt
                if fb is None or v < fb:
                    fb = v
                if live2:
                    st2 = fv if fv > tcur2 else tcur2
                    v2 = st2 + pt
                    if fb2 is None or v2 < fb2:
                        fb2 = v2
            if cnt:
                rem += 0.5 * mn + 0.5 * (sm / cnt)
                s_min_p += mn
                tcur = fb
                # the probe track is monotone: once the delay has been
                # absorbed by a booked downstream machine the two rolls
                # coincide for ever, so the second track is simply dropped
                tcur2 = fb2 if (fb2 is not None and fb2 > fb) else fb
        nrem = nops - o
        # DELAY PROPAGATION: measured share of a delay HERE that survives the
        # remaining chain and actually moves the job's completion time
        if nrem > 1:
            rho = (tcur2 - tcur) / probe
            if rho < 0.0:
                rho = 0.0
            elif rho > 1.0:
                rho = 1.0
        else:
            rho = 1.0
        rho_of[k] = RHOF + (1.0 - RHOF) * rho
        w = _fld(job, 'weight')
        succ_of[k] = ops[o + 1] if o + 1 < nops else None
        # FORECAST DOWNSTREAM QUEUE: how much of the tail is pure waiting for
        # machines already booked rather than processing
        qd = (tcur - eb) - s_min_p
        if qd < 0.0:
            qd = 0.0
        else:
            cap = QCAP * p_bar * (nrem - 1 if nrem > 1 else 1)
            if cap > QMAX * p_bar:
                cap = QMAX * p_bar
            if qd > cap:
                qd = cap
        slack = _fld(job, 'due_date') - eb - lead * rem - PSI * qd
        raw = slack          # undivided slack: what a routing loss may eat
        if slack > 0.0 and nrem > 1:
            slack /= (1.0 + LAM * (nrem - 1))
        info[k] = (slack, w, rem, raw, nrem)
        w_tot += w
        if slack <= 0.0:
            w_late += w
    cong = (w_late / w_tot) if w_tot > 0.0 else 0.0
    w_bar = w_tot / len(info)
    if w_bar <= 0.0:
        w_bar = 1.0

    inv_p = 1.0 / p_bar
    width = K * (0.55 + 0.9 * cong) * p_bar

    # --- ADAPTIVE RAMP WIDTH ----------------------------------------------
    # the nominal width is one deferral horizon; when every live slack is far
    # larger than that the ramp saturates and the rule degenerates towards
    # WSPT.  Blend in the OBSERVED median positive slack so the discriminating
    # linear part of the ramp actually spans the slack distribution, clamped
    # so a tight (already-late) shop is left essentially unchanged.
    pos = [v[0] for v in info.values() if v[0] > 0.0]
    if pos:
        pos.sort()
        med = pos[len(pos) >> 1]
        wa = SB * med + (1.0 - SB) * width
        lo = SBLO * width
        hi = SBHI * width
        if wa < lo:
            wa = lo
        elif wa > hi:
            wa = hi
        width = wa
    inv_slack = 1.0 / width
    inv_delay = 1.0 / (B * (1.25 - 0.5 * cong) * p_bar)

    # --- RAMP priority: marginal weighted tardiness of one deferral --------
    # in a saturated shop the flat tail is what lets far-from-due jobs keep
    # stealing capacity from the jobs actually accruing tardiness: shrink the
    # tail height with measured congestion, floored so they are never ignored
    tail_e = TAIL * (1.0 - TAILC * cong)
    if tail_e < 0.10:
        tail_e = 0.10
    inv_wb = 1.0 / (w_bar * p_bar)

    base = 1.0 + tail_e
    lf_of = {}
    urg_of = {}
    rate_of = {}
    dlog_of = {}
    xlog_of = {}
    for k, (slack, w, rem, raw, nrem) in info.items():
        dl = delta_of.get(k, 0.0)
        dn = dl * inv_p
        if dn > DCAP:
            dn = DCAP
        dlog_of[k] = math.log1p(dn)
        rh = rho_of[k]
        if slack <= 0.0:
            f = base
        else:
            # EXACT marginal ramp: one deferral of DELTA moves the JOB's
            # completion by rho*DELTA, not DELTA, so the ramp is not only
            # LOWER by rho (the log rh below) but also NARROWER by rho -- a
            # tail that swallows the delay also postpones the moment the due
            # date is really threatened.  rho is a noisy estimate, so the
            # blend towards that exact form is partial (RHW).  The width is
            # otherwise the deferral horizon THIS operation really faces.
            rw = 1.0 - RHW + RHW * rh
            x = slack * inv_slack / ((1.0 - DW + DW * dn) * rw)
            r = 1.0 - x
            if r < 0.0:
                r = 0.0
            f = r + tail_e * math.exp(-A * math.log1p(x / A))
            if f < 1e-300:
                f = 1e-300
        # value of acting now = ramp * share of the delay that would survive
        lf_of[k] = math.log(f) + math.log(rh)
        u = f / base
        urg_of[k] = u
        # marginal weighted-tardiness rate at which this op is defended
        rate_of[k] = (w if w > 0 else 1.0) * u * rh
        # EXACT CROSSING: T(C+D)-T(C) with T(x)=max(0,x-d) is a ramp in the
        # UNDIVIDED slack whose width D = rho_eff*delta is MEASURED, not
        # assumed -- zero while one round of waiting provably stays on the
        # near side of the due date, w*D once the job is late
        D = rh * dl
        if D > 0.0:
            cx = D if raw <= 0.0 else (D - raw)
            if cx < 0.0:
                cx = 0.0
        else:
            cx = 0.0
        xlog_of[k] = math.log1p((w if w > 0 else 1.0) * cx * inv_wb)

    # --- how costly the queue behind each machine actually is --------------
    queue = {}
    own_q = {}
    for c in candidates:
        k = c.op
        q = own_q.get(k)
        if q is None:
            q = rate_of[k] / nm[k]
            own_q[k] = q
        queue[c.machine] = queue.get(c.machine, 0.0) + q
    q_max = 0.0
    for v in queue.values():
        if v > q_max:
            q_max = v
    inv_q = 1.0 / q_max if q_max > 0.0 else 0.0

    # --- rival tables for the one-step displacement regret -----------------
    rivals = {}
    for m, lst in by_m.items():
        if m not in sel_m:
            continue
        tab = []
        for r in lst:
            k = r.op
            alt = e1[k] if m1[k] != m else e2.get(k, INF)
            tab.append((rate_of[k], job_ready.get(k[0], 0), r.proc, alt,
                        e1[k], k))
        if len(tab) > TOPR:
            tab.sort(key=lambda z: -z[0])
            del tab[TOPR:]
        rivals[m] = tab

    inv_blk = 1.0 / (w_bar * p_bar)
    # SMITH REGIME: in a saturated shop weighted tardiness IS weighted
    # completion time up to a constant, so the remaining-work pull must grow
    # towards the full Smith exponent 1 (where -log(cost) - log1p(R/cost)
    # = -log(cost+R) exactly).  The quadratic gate leaves loose shops, where
    # only a handful of jobs are at risk, essentially untouched.
    cw = SRW * (0.25 + 0.75 * cong) + SMW * cong * cong

    # --- ARRIVALS: successor operations that will want each machine soon ---
    # every live job's next-but-one operation is a demand certain to arrive at
    # a known time (its predecessor's best completion); freezing m until
    # e(o,m) can push it back exactly like a rival queued now
    arrivals = {}
    for k, sa in succ_of.items():
        if not sa:
            continue
        a = e_best[k]
        v1 = INF
        v1m = None
        v2 = INF
        vals = []
        for alt in sa:
            mm = alt[0]
            fv = machine_free.get(mm, 0)
            st = fv if fv > a else a
            v = st + alt[1]
            if mm in sel_m:
                vals.append((mm, alt[1], v))
            if v < v1:
                v2 = v1
                v1 = v
                v1m = mm
            elif v < v2:
                v2 = v
        if not vals:
            continue
        rt = rate_of[k]
        for mm, pm, v in vals:
            excl = v1 if v1m != mm else v2
            rec = (rt, a, pm, excl, v1, k)
            lst = arrivals.get(mm)
            if lst is None:
                arrivals[mm] = [rec]
            else:
                lst.append(rec)
    for m, lst in arrivals.items():
        if len(lst) > TOPA:
            lst.sort(key=lambda z: -z[0])
            del lst[TOPA:]

    # --- WINQ: machine-dependent two-operation horizon ---------------------
    # the routing decides not only e(o,m) but when the SUCCESSOR can start;
    # staying on the same machine is free continuity, a booked successor
    # machine silently swallows the head start of a "faster" routing
    hs = []
    h_best = {}
    eb_sel = {}
    for c in competing:
        k = c.op
        # the routing loss must be measured against the best routing that is
        # actually SELECTABLE now, not against one the window has removed
        v = eb_sel.get(k)
        if v is None or c.end < v:
            eb_sel[k] = c.end
        sa = succ_of.get(k)
        E = c.end
        if not sa:
            h = E
        else:
            h = INF
            cm = c.machine
            for alt in sa:
                mm = alt[0]
                if mm == cm:
                    st = E
                else:
                    fv = machine_free.get(mm, 0)
                    st = fv if fv > E else E
                v = st + alt[1]
                if v < h:
                    h = v
            if h == INF:
                h = E
        hs.append(h)
        b = h_best.get(k)
        if b is None or h < b:
            h_best[k] = h
    wq_cap = 2.0 * p_bar

    scored = []
    best_s = None

    for idx, c in enumerate(competing):
        k = c.op
        slack, w, rem, raw, nrem = info[k]
        eb = eb_sel[k]
        lw = math.log(w) if w > 0 else -30.0
        urg = urg_of[k]

        pq = (queue[c.machine] - own_q[k]) * inv_q
        if pq < 0.0:
            pq = 0.0

        mf = machine_free.get(c.machine, 0)
        sm0 = s_min[c.machine]
        idle_av = c.start - sm0          # idle THIS routing inserts on m
        if idle_av < 0.0:
            idle_av = 0.0
        idle_un = sm0 - mf               # idle no candidate on m can avoid
        if idle_un < 0.0:
            idle_un = 0.0
        cost = c.proc + IW * (0.5 + pq) * (idle_av + IUN * idle_un)
        if cost <= 0.0:
            cost = 1e-6

        # measured deterioration of the rivals queued on m once m is frozen
        E = c.end
        blk = 0.0
        for rt, rr, rp, ralt, rcur, rk in rivals[c.machine]:
            if rk == k:
                continue
            nend = (rr if rr > E else E) + rp
            if nend > ralt:
                nend = ralt
            d = nend - rcur
            if d > 0.0:
                blk += rt * d

        # ... and of the successor operations that will ARRIVE on m while it
        # is busy: a machine may look empty now yet be the only fast routing
        # of an urgent job's very next operation
        for rt, aa, pm, aexcl, acur, rk in arrivals.get(c.machine, ()):
            if rk == k:
                continue
            nend = (aa if aa > E else E) + pm
            if nend > aexcl:
                nend = aexcl
            d = nend - acur
            if d > 0.0:
                blk += ARW * rt * d

        s2 = e2.get(k)
        if s2 is None:
            spread = 1.0
        else:
            spread = (s2 - e1[k]) * inv_p
            if spread > 1.0:
                spread = 1.0

        detour = bott.get(c.machine, 0.0) - bmin[k]
        if detour < 0.0:
            detour = 0.0

        # two-operation horizon lost versus the best routing of this same
        # operation; capped so one congested successor cannot dominate
        wq = hs[idx] - h_best[k]
        if wq > wq_cap:
            wq = wq_cap
        qual = (c.end - eb) + WQ * wq

        # the part of the routing loss that fits inside the job's own
        # undivided slack creates no tardiness and is charged only at MINF
        if qual <= 0.0:
            eff = 0.0
        elif raw > 0.0:
            unab = qual - raw
            if unab < 0.0:
                unab = 0.0
            # on the LAST operation raw = d - e_best exactly and nothing
            # downstream can absorb or amplify the detour, so the extra
            # tardiness is exactly max(0, qual-raw) and charging the MINF
            # share of a demonstrably free detour would be pure noise
            mi = MINF if nrem > 1 else 0.0
            eff = mi * qual + (1.0 - mi) * unab
        else:
            eff = qual
        # a routing detour is a delay like any other: it only hurts to the
        # extent that it PROPAGATES to the completion time, so a job whose
        # tail must queue anyway is charged less for taking a slower routing
        if eff > 0.0:
            eff *= (1.0 - RHQ) + RHQ * rho_of[k]

        score = (lw + lf_of[k] - math.log(cost)
                 - G * (0.35 + 0.65 * urg) * eff * inv_p
                 - (c.start - t_min) * inv_delay
                 - DBLK * (1.5 - urg) * math.log1p(blk * inv_blk)
                 + REG * (0.4 + 0.6 * urg) * spread
                 + DEF * urg * dlog_of[k]
                 - BOT * (1.0 - 0.7 * urg) * detour
                 + XW * xlog_of[k]
                 - cw * urg * math.log1p(
                     rem / (SND * cost + (1.0 - SND) * p_bar)))

        scored.append((score, c))
        if best_s is None or score > best_s:
            best_s = score

    # NEAR-TIE ARBITRATION IN THE OBJECTIVE'S OWN UNITS.  Score gaps of a few
    # hundredths of a log unit are below the modelling accuracy of every term
    # above, so inside that band the candidates are re-ranked by the EXACT
    # marginal weighted tardiness this very commitment creates:
    #   w_j * max(0, rho_j*(e - e_best_j) - raw_j)            own detour
    # + sum over rivals queued on m and successors arriving on m of
    #   w_r * max(0, rho_r*push_r - raw_r)                    displacement,
    # since T(C+D) - T(C) = max(0, D - max(0, slack)) exactly.  The value is
    # quantised (units of w_bar*p_bar/20) so differences below the forecast's
    # own resolution still fall back on earliest completion.
    thr = best_s - EPS
    band = [c for score, c in scored if score >= thr]
    if len(band) == 1:
        return band[0]

    best = None
    best_key = None
    for c in band:
        k = c.op
        slack, w, rem, raw, nrem = info[k]
        rw = raw if raw > 0.0 else 0.0
        d0 = rho_of[k] * (c.end - e1[k]) - rw
        tw = (w if w > 0 else 1.0) * d0 if d0 > 0.0 else 0.0
        E = c.end
        for rt, rr, rp, ralt, rcur, rk in rivals[c.machine]:
            if rk == k:
                continue
            nend = (rr if rr > E else E) + rp
            if nend > ralt:
                nend = ralt
            d = nend - rcur
            if d > 0.0:
                ir = info[rk]
                dc = rho_of[rk] * d - (ir[3] if ir[3] > 0.0 else 0.0)
                if dc > 0.0:
                    tw += (ir[1] if ir[1] > 0 else 1.0) * dc
        for rt, aa, pm, aexcl, acur, rk in arrivals.get(c.machine, ()):
            if rk == k:
                continue
            nend = (aa if aa > E else E) + pm
            if nend > aexcl:
                nend = aexcl
            d = nend - acur
            if d > 0.0:
                ir = info[rk]
                dc = rho_of[rk] * d - (ir[3] if ir[3] > 0.0 else 0.0)
                if dc > 0.0:
                    tw += ARW * (ir[1] if ir[1] > 0 else 1.0) * dc
        k2 = (int(tw * inv_wb * 20.0), c.end, c.proc, c.machine, c.op)
        if best_key is None or k2 < best_key:
            best_key = k2
            best = c

    if best is None:
        best = min(competing, key=lambda c: (c.end, c.proc, c.machine, c.op))
    return best

# EVOLVE-BLOCK-END
