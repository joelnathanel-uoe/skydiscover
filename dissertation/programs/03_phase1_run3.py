# EVOLVE-BLOCK-START
import math


def _f(obj, name):
    """Attribute or mapping access, whichever the container supports."""
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _pt(alt):
    """Processing time of an (machine, time) alternative."""
    try:
        return alt[1]
    except (TypeError, KeyError, IndexError):
        return _f(alt, 'proc_time')


def _mid(alt):
    """Machine id of an (machine, time) alternative."""
    try:
        return alt[0]
    except (TypeError, KeyError, IndexError):
        return _f(alt, 'machine_id')


def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    """
    Alpha-non-delay conflict set + ATC/WSRPT index whose machine
    arbitration is priced by an EXACT ONE-STEP DISPLACEMENT LOOKAHEAD.

    RESTRICTION (alpha-non-delay): with t_min = min start and e_min = min
      completion over all candidates, only candidates starting no later
      than t_min + ALPHA*(e_min - t_min) compete.  Appending welds tail
      idle time onto a machine forever, so near-non-delay behaviour is
      enforced, yet a slightly later but far more urgent operation may
      still win.

    URGENCY, ONCE PER OPERATION (not per candidate).  For available
      operation o of job j let E(o) be its earliest completion over the
      machines offered.  Remaining work after o is
      Rp = sum_k (min_p + THETA*(mean_p - min_p)) and remaining flow adds
      a queue allowance, Rf = Rp + WAIT*n*pbar.  With scale = K*pbar,
        xs = max(0, slack)/scale,   u = exp(-xs) <= 1,
        rho(o) = w_j*u  (marginal weighted-tardiness rate of o).
      Evaluating xs at E(o) keeps the due-date term identical across the
      machines of one operation, so it never biases machine choice.

      MEASURED SUCCESSOR QUEUE (PHI).  The allowance for the *next*
      operation is not guessed: it is read off machine_free as
      min_a(max(E(o), free_a) + p_a) - E(o) - min_a p_a, scaled by PHI
      and clamped into [WAIT*pbar, 4*pbar], then substituted for that
      one blind term in Rf.  Congestion the job is actually about to
      hit therefore tightens its slack immediately.

      OPERATIONAL DUE DATE (BETA).  slack = d_j - (E(o)+Rf) alone makes
      every early operation of a long route look relaxed, so the whole
      route turns urgent at once and too late.  The route allowance is
      also shared out along the job, ODD(o) = r_j + (d_j-r_j)*(o+1)/n,
      and whenever ODD is the tighter signal the slack is pulled BETA of
      the way toward it (only ever increasing urgency, never reducing
      it, and identically zero on the last operation where the two
      measures coincide).

    DISPLACEMENT (the novel term).  Committing c occupies machine m until
      c.end.  Every other available operation o' that may use m then has
      earliest completion
        E'(o') = min( best completion of o' on a machine != m ,
                      max(job_ready(o'), c.end) + p(o',m) ),
      so it is delayed by max(0, E'(o') - E(o')) -- EXACTLY ZERO whenever
      o' owns a comparable alternative machine, and large only for rivals
      genuinely trapped on m.  The charge is
        DELTA * (1-0.3u) * min(4, sum_o' rho(o')*delay(o') / (rho_bar*pbar)),
      i.e. real, urgency-weighted damage rather than an average
      contention count: a long operation that strands an almost-tardy,
      single-machine rival is expensive; the same operation on a machine
      whose rivals can go elsewhere is free.  Only rivals whose BEST
      option is m contribute, so the sum is sparse; it is bucketed by
      machine and guarded by an affordability test that keeps the work
      linear in the candidate count, falling back to an average-pressure
      proxy on very large conflict sets.

    OTHER PENALTIES (units of pbar): LAM late start, IOTA idle frozen
      permanently at that machine's tail (capped at 3), ETA a machine
      slower than the operation's best, GAMMA expected future workload of
      the machine relative to the least loaded machine offered for the
      SAME operation.  Machine terms are modulated by (1 -/+ 0.3u): an
      almost-tardy operation may seize the fast/contested machine, a
      slack one steps aside.  The workload map spreads each unscheduled
      operation over its eligible machines; when the full map is too
      expensive it is replaced by a HOR-deep, 1/(1+depth)-damped horizon
      map costing O(n_jobs*HOR*alts), so GAMMA never silently vanishes
      on the large instances where machine arbitration matters most.

    INDEX: pi = w_j*u/(p + MU*u*Rp) * exp(-x).  The MU*u*Rp term blends
      WSPT (slack jobs, u~0) into WSRPT (doomed jobs, u~1) -- once
      tardiness is certain, nearly finished jobs must go first.  Largest
      pi wins; ties by earliest completion, then (job, op), then machine.

    PARAMETERS: ALPHA=0.35, K=3.0, LAM=1.0, GAMMA=0.4, THETA=0.5,
    WAIT=0.3, MU=0.6, ETA=0.5, IOTA=0.35, DELTA=0.5, PHI=0.5, BETA=0.3,
    HOR=3.  Stateless; any failure falls back to earliest completion.
    """
    try:
        ALPHA, K, LAM = 0.35, 3.0, 1.0
        GAMMA, THETA, WAIT = 0.4, 0.5, 0.3
        MU, ETA, IOTA, DELTA = 0.6, 0.5, 0.35, 0.5
        PHI, BETA, HOR = 0.5, 0.3, 3
        jobs = _f(instance, 'jobs')

        t_min = candidates[0].start
        e_min = candidates[0].end
        for c in candidates:
            if c.start < t_min:
                t_min = c.start
            if c.end < e_min:
                e_min = c.end

        thr = t_min + ALPHA * (e_min - t_min)
        pool = [c for c in candidates if c.start <= thr]
        if not pool:
            pool = candidates
        if len(pool) == 1:
            return pool[0]

        pbar = 0.0
        for c in pool:
            pbar += c.proc
        pbar /= len(pool)
        if pbar <= 0.0:
            pbar = 1.0
        scale = K * pbar

        # --- per-operation summary over ALL candidates -------------------
        # best1[op] = (earliest completion offered, the machine offering it)
        # best2[op] = second-best completion (on a different machine)
        # machops[m] = [(op, proc)] of every available op that may use m
        best1 = {}
        best2 = {}
        nalt = {}
        machops = {}
        for c in candidates:
            op = c.op
            b = best1.get(op)
            if b is None:
                best1[op] = (c.end, c.machine)
                nalt[op] = 1
            else:
                nalt[op] += 1
                if c.end < b[0]:
                    b2 = best2.get(op)
                    if b2 is None or b[0] < b2:
                        best2[op] = b[0]
                    best1[op] = (c.end, c.machine)
                else:
                    b2 = best2.get(op)
                    if b2 is None or c.end < b2:
                        best2[op] = c.end
            lst = machops.get(c.machine)
            if lst is None:
                machops[c.machine] = [(op, c.proc)]
            else:
                lst.append((op, c.proc))

        # earliest start offered on each machine inside the conflict set:
        # datum for the idle that appending would freeze on its tail
        mstart = {}
        for c in pool:
            b = mstart.get(c.machine)
            if b is None or c.start < b:
                mstart[c.machine] = c.start

        # --- remaining work / remaining flow of a job, cached -------------
        prof = {}

        def _rem(j, o):
            v = prof.get(j)
            if v is not None:
                return v
            ops = _f(jobs[j], 'operations')
            nops = len(ops)
            Rp = 0.0
            n = 0
            nxt = None
            for k in range(o + 1, nops):
                alts = ops[k]
                mn = None
                sm = 0.0
                for a in alts:
                    pa = _pt(a)
                    sm += pa
                    if mn is None or pa < mn:
                        mn = pa
                if mn is None:
                    continue
                Rp += mn + THETA * (sm / len(alts) - mn)
                n += 1
                if nxt is None:
                    nxt = (alts, mn)
            v = (Rp, Rp + WAIT * n * pbar, nxt, nops if nops > 0 else 1)
            prof[j] = v
            return v

        # --- urgency, evaluated ONCE PER OPERATION ------------------------
        info = {}
        rho = {}
        rsum = 0.0
        qlo = WAIT * pbar
        qhi = 4.0 * pbar
        for op, b in best1.items():
            j, o = op
            jb = jobs[j]
            Rp, Rf, nxt, nops = _rem(j, o)
            e0 = b[0]
            if nxt is not None:
                # measured, not guessed: the successor's queueing time is
                # visible in machine_free right now.  Clamped, so it can
                # only sharpen the slack estimate, never destabilise it.
                alts, mn1 = nxt
                bf = None
                for a in alts:
                    mf2 = machine_free.get(_mid(a), 0.0)
                    st = e0 if e0 > mf2 else mf2
                    vq = st + _pt(a)
                    if bf is None or vq < bf:
                        bf = vq
                q = PHI * (bf - e0 - mn1)
                if q < qlo:
                    q = qlo
                elif q > qhi:
                    q = qhi
                Rf = Rf - qlo + q
            w = _f(jb, 'weight') + 1e-9
            d = _f(jb, 'due_date')
            sl = d - (e0 + Rf)
            if sl > 0.0:
                # operational due date: share the route allowance along
                # the job so early operations of a long route already
                # feel pressure instead of all turning urgent at once.
                r0 = _f(jb, 'release_time')
                sl2 = r0 + (d - r0) * ((o + 1.0) / nops) - e0
                if sl2 < sl:
                    sl -= BETA * (sl - sl2)
                    if sl < 0.0:
                        sl = 0.0
            xs = sl / scale if sl > 0.0 else 0.0
            if xs > 60.0:
                xs = 60.0
            u = math.exp(-xs)
            info[op] = (u, Rp, w)
            r = w * u
            rho[op] = r
            rsum += r
        rbar = rsum / len(best1)
        if rbar <= 0.0:
            rbar = 1.0

        # --- is the exact displacement lookahead affordable? --------------
        # cost = sum over machines of (#competing candidates on m) x
        # (#available ops that may use m); the guard keeps the per-call
        # work proportional to the number of candidates.
        cnt = {}
        for c in pool:
            k = cnt.get(c.machine)
            cnt[c.machine] = 1 if k is None else k + 1
        cost = 0
        for m, k in cnt.items():
            cost += k * len(machops[m])
        exact = cost <= 60 * len(pool) + 500
        press = None
        if not exact:
            # cheap fallback: average urgency pressure carried by a machine
            press = {}
            for m, lst in machops.items():
                s = 0.0
                for rop, _p in lst:
                    s += rho[rop] / nalt[rop]
                press[m] = s

        # --- expected future load of each machine (bottleneck map) -------
        Ls = None
        work = 0
        for j, nx in job_next_op.items():
            if nx is not None:
                work += len(_f(jobs[j], 'operations')) - nx
        if work > 0:
            # full remaining-work map when affordable, otherwise a
            # HOR-deep damped horizon map: congestion in the next few
            # operations is what an append decision actually feels, and
            # it costs only O(n_jobs*HOR*alts) so GAMMA stays alive on
            # instances where the full map would be too slow.
            full = work <= 6000
            load = {}
            for j, nx in job_next_op.items():
                if nx is None:
                    continue
                ops = _f(jobs[j], 'operations')
                hi = len(ops)
                if not full:
                    h2 = nx + HOR
                    if h2 < hi:
                        hi = h2
                for k in range(nx, hi):
                    alts = ops[k]
                    na = len(alts)
                    if na <= 0:
                        continue
                    dec = 1.0 if full else 1.0 / (1.0 + (k - nx))
                    for a in alts:
                        m = _mid(a)
                        load[m] = load.get(m, 0.0) + dec * _pt(a) / na
            if load:
                Ls = {}
                tot = 0.0
                for m, v in load.items():
                    mf = machine_free.get(m, 0.0)
                    if mf < t_min:
                        mf = t_min
                    L = (mf - t_min) + v
                    Ls[m] = L
                    tot += L
                Lmean = tot / len(Ls)
                if Lmean <= 0.0:
                    Ls = None

        # best (lowest) load among the machines offered for each operation
        opbest = {}
        if Ls is not None:
            for c in pool:
                L = Ls.get(c.machine)
                if L is None:
                    continue
                b = opbest.get(c.op)
                if b is None or L < b:
                    opbest[c.op] = L

        INF = float('inf')
        best = None
        best_key = None
        for c in pool:
            op = c.op
            u, Rp, w = info[op]
            e0 = best1[op][0]
            ug = 1.0 - 0.3 * u

            x = LAM * (c.start - t_min) / pbar
            ms0 = mstart.get(c.machine)
            if ms0 is not None and c.start > ms0:
                # idle welded permanently onto this machine's tail
                gi = (c.start - ms0) / pbar
                x += IOTA * (3.0 if gi > 3.0 else gi)
            if c.end > e0:
                x += ETA * (1.0 + 0.3 * u) * (c.end - e0) / pbar
            if Ls is not None:
                L = Ls.get(c.machine)
                b = opbest.get(op)
                if L is not None and b is not None:
                    rl = (L - b) / Lmean
                    if rl > 0.0:
                        x += GAMMA * ug * (2.0 if rl > 2.0 else rl)

            if exact:
                # exact one-step displacement: how much later would every
                # rival that may use this machine finish, given it is now
                # busy until c.end?  Zero for rivals with a comparable
                # alternative machine, large for those trapped here.
                T = c.end
                dsum = 0.0
                for rop, rp in machops[c.machine]:
                    if rop == op:
                        continue
                    r = rho[rop]
                    if r <= 0.0:
                        continue
                    rb = best1[rop]
                    if rb[1] == c.machine:
                        alt = best2.get(rop, INF)
                    else:
                        alt = rb[0]
                    jr = job_ready[rop[0]]
                    ne = (T if T > jr else jr) + rp
                    if alt < ne:
                        ne = alt
                    dl = ne - rb[0]
                    if dl > 0.0:
                        dsum += r * dl
                if dsum > 0.0:
                    dv = dsum / (rbar * pbar)
                    x += DELTA * ug * (4.0 if dv > 4.0 else dv)
            else:
                pm = press.get(c.machine, 0.0) - rho[op] / nalt[op]
                if pm > 0.0:
                    bl = (pm / rbar) * (c.end - t_min) / pbar
                    x += DELTA * ug * (4.0 if bl > 4.0 else bl)

            if x > 60.0:
                x = 60.0
            p = c.proc if c.proc > 0 else 0.1
            # WSPT while slack (u~0) -> WSRPT once tardiness is certain
            pi = w * u / (p + MU * u * Rp) * math.exp(-x)

            key = (-pi, c.end, c.op, c.machine)
            if best_key is None or key < best_key:
                best_key = key
                best = c
        return best
    except Exception:
        # Never fail: fall back to earliest completion.
        return min(candidates, key=lambda c: (c.end, c.op, c.machine))

# EVOLVE-BLOCK-END
