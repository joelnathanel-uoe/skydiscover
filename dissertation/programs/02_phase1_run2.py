# EVOLVE-BLOCK-START

import math

_INF = float('inf')


def _get(rec, key, default=None):
    """Field access that works for dicts and for attribute-style records."""
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    """
    LANDING-TIME ATC PER UNIT OF SCARCE MACHINE TIME.

    The rule is an apparent-tardiness-cost index in log space, but urgency is
    measured at the JOB LANDING TIME ew = c.end + WINQ (the instant the job can
    actually resume on its successor's fastest free machine) instead of at the
    operation's completion.  Downstream congestion therefore enters the ATC
    kernel itself rather than being a bolted-on correction, so finishing early
    on a machine that drops the job in front of a busy successor earns nothing.

    RESTRICTION (active schedules): C* = min candidate completion; only
        candidates that can START strictly before C* compete (fallback: all).
        Appending behind C* buries an unrecoverable idle gap in a tail.

    ONE LOOK-AHEAD PASS over the unscheduled operations gives
        load[m] : fractional claim of every remaining op on every eligible
                  machine, split ROUTING-AWARE (proportional to 1/p(o,m), since
                  an op mostly lands on a machine that is fast for it);
        dem[m]  : the same claims priced by a preliminary ATC urgency of their
                  job, i.e. how much weighted tardiness still depends on m;
        Rrem[j] : 0.5*min + 0.5*mean tail estimate AFTER the frontier op;
        nrem[j] : number of remaining ops after the frontier one.
      scarcity sc(m) = clamp(.5*load/Lbar + .5*dem/Dbar, .4, 1.5) is the shadow
      price of one unit of machine m; proj[m] = free[m] + load[m] its horizon.

    URGENCY: slack = d_j - (ew + Rrem + ALPHA*nrem*p_bar), kernel width
        kw = K*p_bar*clamp(sqrt(1+nrem)/fbar, .5, 1.8) (S/OPN uncertainty:
        a job's last op has an exact completion, a long tail a guessed one),
        K = clamp(.5*mean positive normalised slack, 1.2, 5).
        pt = exp(-slack+/kw) ~ P(tardy) stays ABSOLUTE (it gates the regrets);
        the ranking urgency is FLOORED, u = w*(exp(-(z-zmin)) + .05), so one
        very urgent candidate cannot crush every rate and collapse the rule
        into pure minimum-slack.

    MACHINE TIME: cost = proc + fill*sc*idle + BETA*ct*Rrem, where idle is only
        the AVOIDABLE part start - max(free[m], smin[m]) (smin[m] = earliest
        start any available op reaches on m; time below it is lost whatever we
        decide), fill = .3+.7*frac with frac the largest fraction of the gap a
        rival could really cover, and ct = (-slack)/((-slack)+kw) is 0 unless
        the job is late beyond recovery -- there the kernel has saturated, the
        local objective is sum w_j C_j and Smith's rule wants the SHORT chain.

    SELECTION (log space, largest wins), every penalty SATURATED so one bad
        attribute discounts instead of annihilating a candidate:
        s = log(u/cost)
            - NU*(dem/Dbar)*min((idle+excess)/p_bar, 2)   shadow-priced waste
            - XI*min((proj[m]-best proj)/(Lbar+p_bar), 2) congestion routing
            - ETA*pt*min((ew - best ew)/p_bar, 3)         landing regret
            - SIG*min((start - t0)/p_bar, 2)              deferred commitment
            - RHO*min(.5*mean rival delay + .5*duel loss, 2)  blocking regret:
              committing c forces every rival queued on m past c.end, capped by
              the rival's escape to another eligible machine, u-weighted and
              normalised by the machine's own urgency mass so it ranks WITHIN a
              machine rather than merely avoiding busy ones
            - OMEGA*min(depth-2 blocking, 2.5)  the term above only sees rivals
              AVAILABLE now; the claims m will really face next are the
              SUCCESSORS of the frontier operations, whose best-case arrival is
              already fixed by the current candidate set.  Appending c pushes
              such a claimant to min(c.end + p', escape) instead of its
              untouched completion, so the charge is exactly 0 when it arrives
              after c.end or can escape elsewhere for free, and it grows with
              c.end -- a machine awaited by urgent traffic therefore prefers
              SHORT operations.  This is the one lever an append-only rule
              keeps on the near future beyond the rivals that happen to be
              ready at this instant.
            + THETA*(.25+.75*pt)*min(adv,p_bar)/p_bar     option value: what
              the op forfeits by yielding this slot to its best OTHER machine
              -- the only term separating a merely-best machine from the ONLY
              good one, which matters because appends are irreversible.
        Ties: earliest end, (job, op), machine.  Every field access is
        defensive and any failure degrades to the earliest-completion
        candidate, so the rule can never crash the run.
    """
    if len(candidates) == 1:
        return candidates[0]
    try:
        return _pick(candidates, machine_free, job_ready, job_next_op,
                     instance)
    except Exception:
        return min(candidates, key=lambda c: (c.end, c.op, c.machine))


def _pick(candidates, machine_free, job_ready, job_next_op, instance):
    n = len(candidates)
    jobs = _get(instance, 'jobs') or []

    # machine free times: snapshot defensively (never index directly)
    mf = {}
    try:
        for _m, _v in machine_free.items():
            mf[_m] = 0.0 if _v is None else float(_v)
    except Exception:
        pass
    try:
        n_mach = int(_get(instance, 'n_machines', 0) or 0)
    except Exception:
        n_mach = 0
    if n_mach <= 0:
        n_mach = max(1, len(mf))

    def jrec(j):
        try:
            return jobs[j]
        except Exception:
            return None

    # ---- restriction: active-schedule filter -----------------------------
    c_star = min(c.end for c in candidates)
    comp = [i for i in range(n) if candidates[i].start < c_star]
    if not comp:
        comp = list(range(n))
    if len(comp) == 1:
        return candidates[comp[0]]

    p_bar = 0.0
    t0 = _INF
    for i in comp:
        cc = candidates[i]
        p_bar += cc.proc
        if cc.start < t0:
            t0 = cc.start
    p_bar /= float(len(comp))
    if p_bar <= 0.0:
        p_bar = 1.0

    # ---- per-operation options, per-machine candidate lists --------------
    pmin = {}
    b1 = {}
    b2 = {}
    by_m = {}
    smin = {}
    for i in range(n):
        c = candidates[i]
        o = c.op
        v = pmin.get(o)
        if v is None or c.proc < v:
            pmin[o] = c.proc
        sv = smin.get(c.machine)
        if sv is None or c.start < sv:
            smin[c.machine] = c.start
        x = b1.get(o)
        if x is None or c.end < x[0]:
            if x is not None:
                b2[o] = x
            b1[o] = (c.end, c.machine)
        else:
            y = b2.get(o)
            if y is None or c.end < y[0]:
                b2[o] = (c.end, c.machine)
        lst = by_m.get(c.machine)
        if lst is None:
            by_m[c.machine] = [i]
        else:
            lst.append(i)

    def alt_end(o, m):
        """Best completion of operation o on a machine OTHER than m."""
        x = b1.get(o)
        if x is not None and x[1] != m:
            return x[0]
        y = b2.get(o)
        return y[0] if y is not None else _INF

    # ---- look-ahead: routing-aware load, urgency demand, remaining work ---
    load = {}
    dem = {}
    urg0 = {}
    Rrem = {}
    nrem = {}
    n_active = 0
    fsum = 0.0
    Kp0 = 2.0 * p_bar
    try:
        items = list(job_next_op.items())
    except Exception:
        items = []
    for j, nx in items:
        if nx is None:
            continue
        n_active += 1
        rec = jrec(j)
        ops = _get(rec, 'operations') or []
        tail = 0.0
        cnt = 0
        first_min = 0.0
        claims = []
        for k in range(nx, len(ops)):
            alts = ops[k]
            if not alts:
                continue
            tmin = _INF
            ssum = 0.0
            for a in alts:
                t = a[1]
                ssum += t
                if t < tmin:
                    tmin = t
            # ROUTING-AWARE CLAIM: an op does not spread uniformly over its
            # eligible machines, it mostly lands on a fast one; uniform
            # splitting inflates slow alternatives and blurs the bottleneck.
            qs = 0.0
            if tmin < _INF and tmin > 0.0:
                for a in alts:
                    ta = a[1]
                    qs += (tmin / ta) if ta > 0.0 else 1.0
            if qs > 0.0:
                inv = 1.0 / qs
                for a in alts:
                    ta = a[1]
                    q = ((tmin / ta) if ta > 0.0 else 1.0) * inv
                    load[a[0]] = load.get(a[0], 0.0) + ta * q
                    claims.append((a[0], q))
            else:
                q = 1.0 / len(alts)
                for a in alts:
                    load[a[0]] = load.get(a[0], 0.0) + a[1] * q
                    claims.append((a[0], q))
            if k > nx:
                tail += 0.5 * tmin + 0.5 * (ssum / len(alts))
                cnt += 1
            elif tmin < _INF:
                first_min = tmin
        Rrem[j] = tail
        nrem[j] = cnt
        fsum += math.sqrt(1.0 + cnt)

        # preliminary urgency prices this job's claims: a busy machine awaited
        # only by slack-rich jobs is cheap, one several near-due heavy jobs
        # must cross is precious even at average load
        w = _get(rec, 'weight', 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0
        if not (w > 0.0):
            w = 1e-9
        d = _get(rec, 'due_date', 0.0)
        try:
            d = float(d)
        except Exception:
            d = 0.0
        try:
            jr = job_ready.get(j, 0.0)
        except Exception:
            jr = 0.0
        if jr is None:
            jr = 0.0
        sl0 = d - (float(jr) + tail + first_min)
        if sl0 <= 0.0:
            ub = w
        else:
            z0 = sl0 / Kp0
            ub = w * (math.exp(-z0) if z0 < 60.0 else 0.0)
        # the same preliminary urgency prices this job's machine claims AND,
        # below, its IMMINENT successor claim on each eligible machine
        urg0[j] = ub
        if ub > 0.0:
            for mm, q in claims:
                dem[mm] = dem.get(mm, 0.0) + ub * q

    Lbar = 0.0
    for v in load.values():
        Lbar += v
    Lbar /= float(n_mach)
    if Lbar <= 0.0:
        Lbar = 1e-9
    Dbar = 0.0
    for v in dem.values():
        Dbar += v
    Dbar /= float(n_mach)
    if Dbar <= 0.0:
        Dbar = 1.0

    fbar = (fsum / float(n_active)) if n_active > 0 else 1.0
    if fbar <= 0.0:
        fbar = 1.0

    scar = {}
    for _m in by_m:
        sv = 0.5 * (load.get(_m, 0.0) / Lbar) + 0.5 * (dem.get(_m, 0.0) / Dbar)
        if sv < 0.4:
            sv = 0.4
        elif sv > 1.5:
            sv = 1.5
        scar[_m] = sv

    cont = n_active / float(n_mach)
    if cont < 0.5:
        cont = 0.5
    elif cont > 3.0:
        cont = 3.0
    ALPHA = 0.3 * cont

    # ---- IMMINENT (depth-2) CLAIMS on each machine ------------------------
    # The blocking regret in the selection loop only sees rivals that are
    # AVAILABLE right now.  The claims a machine will really face next are the
    # SUCCESSORS of the frontier operations: their best-case arrival is already
    # known (the best completion of their frontier op over the current
    # candidates), so this append can be priced against them.  For each such
    # claim we record (job, arrival, proc on m, urgency, best escape completion
    # on another eligible machine, completion it gets if m is left untouched).
    imm = {}
    Uref = 0.0
    utot = 0.0
    for j, nx in items:
        if nx is None:
            continue
        uj = urg0.get(j, 0.0)
        if uj <= 0.0:
            continue
        x = b1.get((j, nx))
        if x is None:
            continue
        ops = _get(jrec(j), 'operations') or []
        if nx + 1 >= len(ops):
            continue
        alts = ops[nx + 1]
        if not alts:
            continue
        e_ref = x[0]
        f1 = _INF
        f2 = _INF
        m1 = None
        vals = []
        for a in alts:
            mm = a[0]
            fr = mf.get(mm, 0.0)
            st = e_ref if e_ref > fr else fr
            v = st + a[1]
            vals.append((mm, a[1], st))
            if v < f1:
                f2 = f1
                f1 = v
                m1 = mm
            elif v < f2:
                f2 = v
        for mm, pm2, st in vals:
            bex = f2 if mm == m1 else f1
            base = st + pm2
            if bex < base:
                base = bex
            lst = imm.get(mm)
            if lst is None:
                imm[mm] = [(j, e_ref, pm2, uj, bex, base)]
            else:
                lst.append((j, e_ref, pm2, uj, bex, base))
            utot += uj
    if utot > 0.0:
        # SYSTEM-average imminent urgency: normalising by the machine's own
        # mass would cancel precisely the contention signal we want to read.
        Uref = utot / float(n_mach)
        # bound the per-candidate cost: only the most urgent claimants can
        # plausibly drive the decision, and truncation keeps the selection
        # loop linear in the number of candidates
        for mm, lst in imm.items():
            if len(lst) > 10:
                lst.sort(key=lambda t: -t[3])
                del lst[10:]

    # ---- projected machine horizons --------------------------------------
    proj = {}
    projbest = {}
    for i in range(n):
        c = candidates[i]
        m = c.machine
        pv = proj.get(m)
        if pv is None:
            pv = mf.get(m, 0.0) + load.get(m, 0.0)
            proj[m] = pv
        bb = projbest.get(c.op)
        if bb is None or pv < bb:
            projbest[c.op] = pv

    # ---- WINQ and the JOB LANDING TIME ew = end + wq ----------------------
    wq = [0.0] * n
    ew = [0.0] * n
    e1 = {}
    e2 = {}
    for i in range(n):
        c = candidates[i]
        j, o = c.op
        ops = _get(jrec(j), 'operations') or []
        dw = 0.0
        if o + 1 < len(ops):
            alts = ops[o + 1]
            if alts:
                e = c.end
                best_next = _INF
                pm = _INF
                for a in alts:
                    t = a[1]
                    if t < pm:
                        pm = t
                    fm = mf.get(a[0], 0.0)
                    st = e if e > fm else fm
                    v = st + t
                    if v < best_next:
                        best_next = v
                dw = best_next - e - pm
                if dw < 0.0:
                    dw = 0.0
        wq[i] = dw
        v = c.end + dw
        ew[i] = v
        oo = c.op
        x = e1.get(oo)
        if x is None or v < x[0]:
            if x is not None:
                e2[oo] = x
            e1[oo] = (v, c.machine)
        else:
            y = e2.get(oo)
            if y is None or v < y[0]:
                e2[oo] = (v, c.machine)

    def alt_ew(o, m):
        """Best job-level resumption time of o on a machine OTHER than m."""
        x = e1.get(o)
        if x is not None and x[1] != m:
            return x[0]
        y = e2.get(o)
        return y[0] if y is not None else _INF

    # ---- per-job weight / due date / remaining allowance ------------------
    jinfo = {}
    for i in range(n):
        j = candidates[i].op[0]
        if j in jinfo:
            continue
        rec = jrec(j)
        w = _get(rec, 'weight', 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0
        if not (w > 0.0):
            w = 1e-9
        d = _get(rec, 'due_date', 0.0)
        try:
            d = float(d)
        except Exception:
            d = 0.0
        R = Rrem.get(j, 0.0) + ALPHA * nrem.get(j, 0) * p_bar
        jinfo[j] = (w, d, R)

    # ---- slack measured at the LANDING time, adaptive kernel --------------
    slack = [0.0] * n
    for i in range(n):
        _, d, R = jinfo[candidates[i].op[0]]
        slack[i] = d - (ew[i] + R)

    tot_sl = 0.0
    for i in comp:
        if slack[i] > 0.0:
            tot_sl += slack[i]
    K = 0.5 * ((tot_sl / float(len(comp))) / p_bar)
    if K < 1.2:
        K = 1.2
    elif K > 5.0:
        K = 5.0
    Kp = K * p_bar

    inv_fbar = 1.0 / fbar
    zs = [0.0] * n
    pt = [1.0] * n
    ct = [0.0] * n
    zmin = _INF
    for i in range(n):
        j = candidates[i].op[0]
        f = math.sqrt(1.0 + nrem.get(j, 0)) * inv_fbar
        if f < 0.5:
            f = 0.5
        elif f > 1.8:
            f = 1.8
        kwi = Kp * f
        s = slack[i]
        if s > 0.0:
            z = s / kwi
        else:
            z = 0.0
            if s < 0.0:
                ct[i] = (-s) / ((-s) + kwi)
        zs[i] = z
        pt[i] = math.exp(-z) if z < 60.0 else 0.0
        if z < zmin:
            zmin = z
    if zmin == _INF:
        zmin = 0.0

    EPS = 0.05
    u = [0.0] * n
    for i in range(n):
        w = jinfo[candidates[i].op[0]][0]
        dz = zs[i] - zmin
        u[i] = w * ((math.exp(-dz) if dz < 60.0 else 0.0) + EPS)

    NU = 0.48
    XI = 0.55
    ETA = 0.40
    SIG = 0.13
    RHO = 0.68
    THETA = 0.28
    BETA = 0.20
    OMEGA = 0.26

    # ---- selection --------------------------------------------------------
    best = None
    best_key = None
    for i in comp:
        c = candidates[i]
        m = c.machine
        sc = scar.get(m, 1.0)

        # AVOIDABLE idle only: below max(free[m], smin[m]) nothing available
        # could have started anyway, so charging it would tax starving machines
        mfm = mf.get(m, 0.0)
        base = smin.get(m, c.start)
        if base < mfm:
            base = mfm
        idle = c.start - base
        if idle < 0.0:
            idle = 0.0

        # one rival pass: gap fillability AND escape-aware blocking regret
        frac = 0.0
        num = 0.0
        ui = u[i]
        den = ui
        mx = 0.0
        for k in by_m.get(m, ()):
            if k == i:
                continue
            c2 = candidates[k]
            uk = u[k]
            den += uk
            if idle > 0.0 and c2.start < c.start:
                st2 = c2.start
                if st2 < base:
                    st2 = base
                cov = st2 + c2.proc
                if cov > c.start:
                    cov = c.start
                f2 = (cov - st2) / idle
                if f2 > 1.0:
                    f2 = 1.0
                if f2 > frac:
                    frac = f2
            if uk <= 0.0:
                continue
            st2 = c.end if c.end > c2.start else c2.start
            ne = st2 + c2.proc
            a = alt_end(c2.op, m)
            if a < ne:
                ne = a
            dly = ne - b1[c2.op][0]
            if dly > 0.0:
                v = uk * dly
                num += v
                v /= (ui + uk)
                if v > mx:
                    mx = v
        if idle > 0.0:
            idle *= 0.3 + 0.7 * frac

        cost = c.proc + idle * sc
        if ct[i] > 0.0:
            cost += BETA * ct[i] * Rrem.get(c.op[0], 0.0)
        if cost <= 0.0:
            cost = 1e-9

        s = math.log(u[i] / cost)

        ex = c.proc - pmin[c.op]
        if ex < 0.0:
            ex = 0.0
        waste = idle + ex
        if waste > 0.0:
            x = waste / p_bar
            if x > 2.0:
                x = 2.0
            s -= NU * (dem.get(m, 0.0) / Dbar) * x

        dc = proj[m] - projbest[c.op]
        if dc > 0.0:
            x = dc / (Lbar + p_bar)
            if x > 2.0:
                x = 2.0
            s -= XI * x

        # landing regret: once the kernel saturates every unit of the job's
        # resumption time still costs w_j, which the ATC term can no longer see
        if pt[i] > 0.0:
            de = ew[i] - e1[c.op][0]
            if de > 0.0:
                x = de / p_bar
                if x > 3.0:
                    x = 3.0
                s -= ETA * pt[i] * x

        # deferred commitment: a start well in the future is an irreversible
        # append judged against a rival set that is only partly known
        if c.start > t0:
            x = (c.start - t0) / p_bar
            if x > 2.0:
                x = 2.0
            s -= SIG * x

        if den > 0.0 and (num > 0.0 or mx > 0.0):
            x = 0.5 * (num / (den * p_bar)) + 0.5 * (mx / p_bar)
            if x > 2.0:
                x = 2.0
            s -= RHO * x

        # DEPTH-2 BLOCKING: incremental delay this append imposes on the
        # successors about to claim m.  Appending c pushes such a claimant to
        # min(c.end + p', escape) instead of its untouched completion, so the
        # charge vanishes when it arrives after c.end or can escape at no cost,
        # and grows with c.end.  The candidate's own job is skipped -- its
        # successor's wait is already inside the landing time ew.
        if Uref > 0.0:
            lst = imm.get(m)
            if lst:
                acc = 0.0
                jc = c.op[0]
                ce = c.end
                for (j2, arr, pm2, uj, bex, base) in lst:
                    if arr >= ce or j2 == jc:
                        continue
                    ne = ce + pm2
                    if bex < ne:
                        ne = bex
                    dly = ne - base
                    if dly > 0.0:
                        acc += uj * dly
                if acc > 0.0:
                    x = acc / (Uref * p_bar)
                    if x > 2.5:
                        x = 2.5
                    s -= OMEGA * x

        # option value: every penalty above is 0 on the op's best machine, so
        # none can tell a merely-best machine from the ONLY good one
        adv = alt_ew(c.op, m) - ew[i]
        if adv > 0.0:
            if adv > p_bar:
                adv = p_bar
            s += THETA * (0.25 + 0.75 * pt[i]) * (adv / p_bar)

        key = (-s, c.end, c.op, m)
        if best_key is None or key < best_key:
            best_key = key
            best = c

    return best if best is not None else candidates[comp[0]]

# EVOLVE-BLOCK-END
