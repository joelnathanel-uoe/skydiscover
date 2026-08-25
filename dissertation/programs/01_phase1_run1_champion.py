# EVOLVE-BLOCK-START
import math


def _fld(obj, key):
    """Field access that works for dicts and for attribute-style objects."""
    try:
        return obj[key]
    except (TypeError, KeyError, IndexError):
        return getattr(obj, key)


def choose_next(candidates, machine_sequences, machine_free, job_ready,
                job_next_op, instance):
    """
    NET-MARGINAL-TARDINESS dispatch with adaptive WSPT<->WSRPT blending,
    evaluated as an additive log index.

    RESTRICTION: t* = min candidate start, e* = min candidate completion.
                 Only candidates with start <= t* + DELTA*(e* - t*) compete
                 (DELTA = 0.35): near-non-delay, since an append-only scheme
                 can never repair a machine tail committed behind an
                 avoidable wait.  Dropping a late-starting candidate is safe
                 (every candidate on m starts at or after machine_free[m], so
                 m is merely left undecided one more step).

    BASE RATE (log space, never underflows into ties):
      1. PROPAGATED chain deadline (multi-stage, resource aware).  From the
         candidate's own completion ce the job's next HZ stages are walked
         through the COMMITTED schedule:
             t <- SM*b1 + (1-SM)*b2 + QIN*q,
         where b1,b2 are the two smallest values of max(machine_free[m'], t)
         + p(m') at that stage.  The soft-min over the two best routes avoids
         staking the whole forecast on one machine a rival will take, and
         propagating rather than pricing a single stage removes the
         systematic optimism that made the kernel fire late for jobs deep in
         their route.  Stages beyond the horizon use a cached flat tail
         (0.7*min + 0.3*mean per stage) plus a LOAD-ADAPTIVE queueing
         allowance q = clip(QB + QW*(J/M - 1), QMIN, QMAX)*p_bar (J = live
         jobs, M = machines): a job queues behind ~J/M - 1 rivals at each
         future stage, so a congested shop recognises urgency early while a
         draining shop relaxes back.
         Chat = forecast completion, dl = d_j - (Chat - ce), slack = dl - end.
         Because Chat depends on ce, the rule sees that finishing early buys
         nothing when the downstream bottleneck is busy anyway.
      2. Saturating urgency kernel with UNCERTAINTY-SCALED width
             f = 1/(1 + (slack^+/(K*p_bar*sqrt(1+n_rem)))^2),
         = 1 exactly when the job is projected late (the true marginal rate
         of w*(C-d)^+ is then w) with a heavy tail that preserves a w/p order
         for far-due jobs, and a width growing like the accumulated error of
         the chain estimate so deep-in-chain decisions fall back on WSPT.
      3. Denominator = machine time actually CONSUMED, p + IDL*hole with
         hole = start - min_start[m]: an unavoidable wait is not punished, a
         self-inflicted hole is charged as if it were processing.
         dens = w*f/(p + IDL*hole).

    NET OBJECTIVE DELTA (the new core).  Let rv be the strongest rival on m
    (highest dens among candidates on m belonging to another operation).
      4. GAIN of taking m now: if we pass, o realistically waits for rv, so
         its completion becomes ne_o = min(max(end_rv, start_o) + p_o, alt_o)
         where alt_o is its best completion on another machine.  The
         weighted tardiness it AVOIDS by being served now,
             w_o * [ (ne_o - dl_o)^+ - (end_o - dl_o)^+ ],
         is a BONUS (negative exponent).  This is what separates jobs that a
         saturated kernel would tie: among equally late jobs it prefers the
         one that would actually lose the most by yielding.
      5. DAMAGE of taking m now: every other candidate v queued on m is
         pushed to ne_v = min(max(end_o, start_v) + p_v, alt_v), costing
             w_v * [ (ne_v - dl_v)^+ - (best_end_v - dl_v)^+ ],
         summed over the MAXV densest rivals.  Victims that are flexible or
         still slack contribute exactly zero.
      Both are normalised by w_bar*p_bar, so 4-5 form the symmetric marginal
      objective delta of monopolising the machine.

    ROUTING DAMPING:
      6. detour  B*(end - best_end[op])/p_bar.
      7. lock/flex veto LAM*flex*rival/(rival+dens) with lock =
         clip((alt-end)/p_bar,0,1) (how badly an op needs THIS machine):
         replaceable operations yield, trapped ones are never starved.
      8. machine criticality GAM*(load[m] - min eligible load)/(mean load +
         p_bar), load[m] = fractional FUTURE workload owed to m (each future
         operation loads every eligible m by 1/sum(1/p_i), i.e. it picks m
         with probability ~1/p_m, so a merely SLOW machine is not declared
         congested).  Computed in the same single walk as the chain tails.

      log S = log dens - clip(ex, -EXCAP, EXCAP); largest wins, ties by
      (end, op, machine).

    REMAINING-WORK BLEND (term 9).  ex += arw*log(1 + W_tail/p) interpolates
    operation-level WSPT (arw = 0) and job-level WSRPT (arw = 1), W_tail being
    the expected work still owed by the job AFTER this operation (taken from
    the same single walk that builds the chain tails, so it is free).  The
    strength is ADAPTIVE, arw = ARW0 + ARW1*frac_late with frac_late the
    fraction of live jobs projected tardy even on their BEST machine: a
    saturated shop (where the kernel saturates and TWT degenerates into
    weighted flow time) adopts the flow-time-correct order, while a loose shop
    stays purely deadline-driven.

    NEW IN THIS VERSION
      (a) CONGESTION-AWARE ROUTE FORECAST: the fractional future demand
          load[m] (free, it is already needed by term 8) enters the HZ-stage
          propagation as a per-machine queue allowance
              wait[m] = clip(qin*load[m]/mean_load, WLO*qin, WHI*qin),
          charged inside max(machine_free[m],t)+p+wait[m] BEFORE the two-best
          soft-min, so the forecast follows the route the job will really
          take instead of the nominally freest one.  Demand decays
          geometrically along the route (DECAY) and the out-of-horizon flat
          allowance is scaled by stage flexibility, 0.6 + 0.8/(1+#eligible).
      (b) MARGINAL DISPLACEMENT DAMAGE: only ONE rival can really take the
          machine in our place, so the k-th densest rival is charged with a
          geometrically decaying weight DEC^k rather than in full, keeping
          term 5 marginal instead of penalising crowded machines twice.
      (c) DETOUR PRICED IN OBJECTIVE UNITS: besides the scale-free
          B*(end-best_end)/p_bar a detour now also pays
          ND*w*[(end-dl)^+ - (best_end-dl)^+]/(w_bar*p_bar) - exactly zero
          while the job still has slack, decisive for late, heavy jobs
          (B relaxed 0.8 -> 0.7 to avoid double counting).
      (d) URGENCY-WEIGHTED MACHINE CRITICALITY: term 8 uses uload[m], the
          future demand weighted by job urgency u_j (= w_j if the job cannot
          meet its due date even with zero waiting, else
          w_j/(1+(slack0/(K*p_bar))^2)) rescaled by the mean urgency, so
          routing yields machines that IMPORTANT jobs need, not merely busy
          ones.  The physical load is still used for the time forecast (a).
      (e) CAPACITY-ADAPTIVE DELAY WINDOW: DELTA -> DELTA + DEX*(M-J)^+/M.
          When machines outnumber live jobs some machine idles anyway, so a
          short wait for a better-matched machine is free and the rule is
          allowed to look further; a congested shop keeps the tight
          near-non-delay window (idle there is unrecoverable).

      log S = log dens - clip(ex, -EXCAP, EXCAP); largest wins, ties by
      (end, op, machine).

    PARAMETERS:  DELTA=0.35, DEX=0.30, K=1.3, B=0.7, ND=0.50, IDL=1.0,
                 LAM=0.40, MU=0.45, DEC=0.75, NU=0.50, GAM=0.45, MAXV=8,
                 EXCAP=8.0, QB=0.30, QW=0.30, QMIN=0.15, QMAX=1.20, HZ=3,
                 QIN=0.75, SM=0.75, WLO=0.5, WHI=2.0, DECAY=0.75, ARW0=0.12,
                 ARW1=0.33.
    """
    n = len(candidates)
    if n == 1:
        return candidates[0]

    try:
        jobs = _fld(instance, 'jobs')

        DELTA = 0.35   # delay-window width (congested shop)
        DEX = 0.30     # extra window when machines outnumber live jobs
        K = 1.3        # base slack scale of the urgency kernel
        B = 0.7        # routing-detour damping (scale-free part)
        ND = 0.50      # routing detour priced in real weighted tardiness
        IDL = 1.0      # avoidable idle charged as consumed machine time
        QB = 0.30      # queueing allowance per remaining stage at J/M = 1
        QW = 0.30      # extra allowance per excess available-op-per-machine
        QMIN = 0.15    # floor of the allowance (drained shop)
        QMAX = 1.20    # cap of the allowance (congested shop)
        HZ = 3         # stages propagated through the committed schedule
        QIN = 0.75     # base allowance fraction inside the propagated horizon
        WLO = 0.5      # floor multiplier of the per-machine queue allowance
        WHI = 2.0      # cap multiplier of the per-machine queue allowance
        DECAY = 0.75   # geometric decay of future demand along the route
        SM = 0.75      # soft-min weight on the better of the two routes
        LAM = 0.40     # lock/flex contention veto
        MU = 0.45      # damage inflicted on displaced rivals
        DEC = 0.75     # rank decay of the displaced-rival damage
        NU = 0.50      # gain from not being deferred behind the top rival
        GAM = 0.45     # machine-criticality (future demand) routing penalty
        MAXV = 8       # rivals examined per machine
        EXCAP = 8.0    # cap on the total (signed) exponent
        ARW0 = 0.12    # WSPT->WSRPT blend, uncongested shop
        ARW1 = 0.33    # extra blend at full projected lateness

        # --- pass 1: global stats, per-op best/second, per-machine groups --
        t_min = candidates[0].start
        e_min = candidates[0].end
        tot = 0.0
        best_end = {}
        best_mach = {}
        second_end = {}
        min_start = {}
        for c in candidates:
            tot += c.proc
            if c.start < t_min:
                t_min = c.start
            if c.end < e_min:
                e_min = c.end
            op = c.op
            be = best_end.get(op)
            if be is None or c.end < be:
                if be is not None:
                    second_end[op] = be
                best_end[op] = c.end
                best_mach[op] = c.machine
            else:
                se = second_end.get(op)
                if se is None or c.end < se:
                    second_end[op] = c.end
            m = c.machine
            ms = min_start.get(m)
            if ms is None or c.start < ms:
                min_start[m] = c.start
        pbar = tot / n
        if pbar <= 0.0:
            pbar = 1.0
        invP = 1.0 / pbar
        invKP = 1.0 / (K * pbar)

        # --- load-adaptive queueing allowance per remaining stage ----------
        try:
            nm = _fld(instance, 'n_machines')
        except Exception:
            nm = 0
        if not nm or nm <= 0:
            nm = len(machine_free)
        if nm <= 0:
            nm = len(min_start)
        if nm <= 0:
            nm = 1
        navail = len(best_end)          # distinct available ops = live jobs
        q = QB + QW * (navail / float(nm) - 1.0)
        if q < QMIN:
            q = QMIN
        elif q > QMAX:
            q = QMAX
        qstage = q * pbar
        qin = QIN * qstage
        SM2 = 1.0 - SM

        # --- job data cache -------------------------------------------------
        jcache = {}

        def jdat(j):
            e = jcache.get(j)
            if e is None:
                jb = jobs[j]
                w = _fld(jb, 'weight')
                if w <= 0.0:
                    w = 1e-9
                e = (w, _fld(jb, 'due_date'), _fld(jb, 'operations'))
                jcache[j] = e
            return e

        # --- ONE walk over every live job's remaining operations: the FLAT
        #     tail beyond the propagated horizon (stages k0+1+HZ ..), the work
        #     still owed after this operation, AND the fractional future
        #     workload owed by each machine.  Stages k0+1 .. k0+HZ are priced
        #     per candidate below from the real machine free times.
        lcache = {}
        load = {}          # physical fractional future demand per machine
        uraw = {}          # the same demand weighted by job urgency
        usum = 0.0
        for op in best_end:
            w0, d0, ops = jdat(op[0])
            nops = len(ops)
            k0 = op[1]
            hstart = k0 + 1 + HZ
            R = 0.0
            W = 0.0
            ew0 = 0.0
            wgt = 1.0
            ent = []
            for k in range(k0, nops):
                alts = ops[k]
                mn = alts[0][1]
                s = 0.0
                S = 0.0
                for a in alts:
                    pt = a[1]
                    s += pt
                    if pt < mn:
                        mn = pt
                    S += (1.0 / pt) if pt > 0.0 else 1e9
                share = (wgt / S) if S > 0.0 else 0.0
                if share > 0.0:
                    for a in alts:
                        mm = a[0]
                        lv = load.get(mm)
                        load[mm] = share if lv is None else lv + share
                        ent.append((mm, share))
                wgt *= DECAY        # far-off demand is soft and re-routable
                na = len(alts)
                ew = 0.7 * mn + 0.3 * (s / na)
                if k > k0:
                    W += ew                     # work still owed AFTER this op
                    if k >= hstart:
                        # flat tail beyond the horizon; an inflexible stage
                        # queues far longer than a highly flexible one
                        R += ew + qstage * (0.6 + 0.8 / (1.0 + na))
                else:
                    ew0 = ew
            # coarse job urgency: could this job still make its due date even
            # with ZERO waiting anywhere?  Used to weight the future demand so
            # that machine criticality protects the machines that matter.
            sl0 = d0 - (job_ready.get(op[0], 0) + ew0 + W)
            if sl0 <= 0.0:
                u = w0
            else:
                x0 = sl0 * invKP
                if x0 > 1e8:
                    x0 = 1e8
                u = w0 / (1.0 + x0 * x0)
            usum += u
            for mm, share in ent:
                lv = uraw.get(mm)
                uraw[mm] = share * u if lv is None else lv + share * u
            nrem = nops - k0 - 1
            lcache[op] = (R, nrem, math.sqrt(1.0 + nrem), W)

        # urgency-weighted demand, rescaled by the mean urgency so that uload
        # keeps the SCALE of load (and GAM keeps its calibrated meaning)
        nlive0 = len(best_end)
        ubar = (usum / nlive0) if nlive0 else 1.0
        if ubar <= 0.0:
            ubar = 1e-9
        uscale = 1.0 / ubar
        uload = {}
        tu = 0.0
        for mm, v in uraw.items():
            vv = v * uscale
            uload[mm] = vv
            tu += vv
        cnorm = 1.0 / (tu / nm + pbar)

        tload = 0.0
        for v in load.values():
            tload += v
        mload = tload / nm

        # per-machine queueing allowance used INSIDE the route forecast:
        # normalised so an AVERAGE-demand machine still costs ~qin per stage,
        # while an oversubscribed machine is forecast to make the job wait
        # longer and an idle one barely at all.  This is what lets the
        # two-best soft-min below pick the realistic downstream route.
        wlo = WLO * qin
        whi = WHI * qin
        wait = {}
        if mload > 0.0:
            for mm, lv in load.items():
                v = qin * lv / mload
                if v < wlo:
                    v = wlo
                elif v > whi:
                    v = whi
                wait[mm] = v

        cmin = {}
        for c in candidates:
            lm = uload.get(c.machine, 0.0)
            v = cmin.get(c.op)
            if v is None or lm < v:
                cmin[c.op] = lm

        # --- pass 2: rate density, lock, chain deadline per candidate ------
        info = []
        by_mach = {}
        maxsl = {}
        wsum = 0.0
        for idx in range(n):
            c = candidates[idx]
            op = c.op
            w, d, ops = jdat(op[0])
            wsum += w
            R, nrem, wfac, Wt = lcache[op]
            ce = c.end

            if nrem > 0:
                # propagate the next HZ stages through the committed schedule
                t = float(ce)
                k1 = op[1] + 1
                kend = k1 + HZ
                nops = len(ops)
                if kend > nops:
                    kend = nops
                for k in range(k1, kend):
                    b1 = None
                    b2 = None
                    for a in ops[k]:
                        mm = a[0]
                        fr = machine_free.get(mm, 0)
                        st = fr if fr > t else t
                        # congestion of the downstream machine is charged
                        # BEFORE the soft-min, so the forecast routes the job
                        # away from oversubscribed machines instead of always
                        # betting on the nominally freest one
                        fin = st + a[1] + wait.get(mm, wlo)
                        if b1 is None or fin < b1:
                            b2 = b1
                            b1 = fin
                        elif b2 is None or fin < b2:
                            b2 = fin
                    if b2 is None:
                        b2 = b1
                    t = SM * b1 + SM2 * b2
                L = (t - ce) + R
            else:
                L = 0.0
            dl = d - L

            slack = dl - ce
            ms = maxsl.get(op)                  # best-machine slack of this op
            if ms is None or slack > ms:
                maxsl[op] = slack
            if slack < 0.0:
                slack = 0.0
            x = slack * invKP / wfac
            if x > 1e8:              # guard: log(dens) must stay finite
                x = 1e8
            f = 1.0 / (1.0 + x * x)

            hole = c.start - min_start[c.machine]
            if hole < 0.0:
                hole = 0.0
            cost = c.proc + IDL * hole
            if cost <= 0.0:
                cost = 1e-9
            dens = w * f / cost
            if dens < 1e-300:
                dens = 1e-300

            if best_mach[op] != c.machine:
                alt = best_end[op]
            else:
                alt = second_end.get(op)
            if alt is None:
                lock = 1.0
            else:
                lock = (alt - ce) * invP
                if lock > 1.0:
                    lock = 1.0
                elif lock < 0.0:
                    lock = 0.0
            pp = c.proc if c.proc > 1e-9 else 1e-9
            info.append((dens, lock, w, dl, alt, math.log1p(Wt / pp)))
            g = by_mach.get(c.machine)
            if g is None:
                by_mach[c.machine] = [idx]
            else:
                g.append(idx)

        wbar = wsum / n
        if wbar <= 0.0:
            wbar = 1.0
        invWP = 1.0 / (wbar * pbar)

        # --- shop lateness -> strength of the WSPT/WSRPT blend --------------
        nlive = len(maxsl)
        if nlive > 0:
            nlate = 0
            for v in maxsl.values():
                if v <= 0.0:
                    nlate += 1
            arw = ARW0 + ARW1 * (nlate / float(nlive))
        else:
            arw = ARW0

        # --- lazy per-machine rival order (densest first, capped) ----------
        order = {}

        def ordm(m):
            o = order.get(m)
            if o is None:
                idxs = by_mach[m]
                if len(idxs) > MAXV:
                    o = sorted(idxs, key=lambda i: -info[i][0])[:MAXV]
                elif len(idxs) > 1:
                    o = sorted(idxs, key=lambda i: -info[i][0])
                else:
                    o = idxs
                order[m] = o
            return o

        # --- Restriction + selection (additive log index) -------------------
        # capacity-adaptive window: with spare machines an idle moment is
        # unavoidable anyway, so waiting briefly for a better-matched
        # operation is free; a congested shop keeps the tight window.
        dw = DELTA
        if navail < nm:
            dw += DEX * (nm - navail) / float(nm)
        thr = t_min + dw * (e_min - t_min) + 1e-9
        best = None
        best_key = None
        for idx in range(n):
            c = candidates[idx]
            if c.start > thr:
                continue
            dens, lock, w, dl, alt, lr = info[idx]
            # WSPT (arw=0) <-> WSRPT (arw=1) interpolation in log space
            ex = arw * lr
            ce = c.end
            op = c.op

            bo = best_end[op]
            dr = ce - bo
            if dr > 0.0:
                ex += B * dr * invP
                # the same detour priced in real objective units: the extra
                # weighted tardiness this slower/later route actually creates
                # (exactly zero while the job still has slack)
                t1 = ce - dl
                if t1 > 0.0:
                    t0 = bo - dl
                    if t0 < 0.0:
                        t0 = 0.0
                    ex += ND * w * (t1 - t0) * invWP

            over = uload.get(c.machine, 0.0) - cmin[op]
            if over > 0.0:
                ex += GAM * over * cnorm

            lst = ordm(c.machine)

            # strongest rival on this machine (drives lock veto and gain)
            rvi = -1
            for i in lst:
                if candidates[i].op != op:
                    rvi = i
                    break

            if rvi >= 0:
                rival = info[rvi][0] * info[rvi][1]
                flex = 1.0 - lock
                if rival > 0.0 and flex > 0.0:
                    ex += LAM * flex * rival / (rival + dens + 1e-12)

                # GAIN: tardiness avoided by not queueing behind that rival
                re = candidates[rvi].end
                st = re if re > c.start else c.start
                ne = st + c.proc
                if alt is not None and alt < ne:
                    ne = alt
                base = ce - dl
                if base < 0.0:
                    base = 0.0
                tv = ne - dl
                if tv > base:
                    ex -= NU * w * (tv - base) * invWP

            # DAMAGE: weighted tardiness pushed onto displaced rivals
            if len(lst) > 1:
                # Only ONE rival can really take the machine in our place, so
                # the k-th densest rival is charged with a geometrically
                # decaying probability DEC^k of being the displaced one; this
                # keeps the term MARGINAL instead of over-counting crowded
                # machines (already priced by the criticality term above).
                dmg = 0.0
                rw = 1.0
                for i in lst:
                    v = candidates[i]
                    vop = v.op
                    if vop == op:
                        continue
                    _vd, _vl, vw, vdl, valt, _vr = info[i]
                    vbt = best_end[vop] - vdl
                    if vbt < 0.0:
                        vbt = 0.0
                    pe = ce if ce > v.start else v.start
                    ne = pe + v.proc
                    if valt is not None and valt < ne:
                        ne = valt
                    tvv = ne - vdl
                    if tvv > vbt:
                        dmg += rw * vw * (tvv - vbt)
                    rw *= DEC
                if dmg > 0.0:
                    ex += MU * dmg * invWP

            if ex > EXCAP:
                ex = EXCAP
            elif ex < -EXCAP:
                ex = -EXCAP
            s = math.log(dens) - ex

            key = (-s, c.end, c.op, c.machine)
            if best_key is None or key < best_key:
                best_key = key
                best = c

        return best if best is not None else min(
            candidates, key=lambda c: (c.end, c.op, c.machine))
    except Exception:
        # Never fail: fall back to a safe earliest-completion rule.
        return min(candidates, key=lambda c: (c.end, c.op, c.machine))

# EVOLVE-BLOCK-END
