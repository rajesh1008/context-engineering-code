"""Replay the full mix in date order with Chapter 21's memory, under a chosen guard."""
from contextlab import memguard as G
from contextlab.memguard import MemoryRecord as M, Origin as O, ReadLog
import lakeshore22 as L

WATER = ("water_backup", "burst_pipe", "mould", "seepage")


def relevant(m, d):
    if m.status != "live" or m.written >= d.day:
        return False
    if m.scope in ("policy", "claimant"):
        return m.subject == d.policy
    if m.scope == "adjuster":
        if m.asserts:                                   # a derived coverage note
            return m.subject == d.adjuster and d.state == "WI" and d.peril in WATER[:2]
        return m.subject == d.adjuster and m.mid.endswith("-0")
    if m.mid == "M02116":                               # scope HO/WI/water, lesson floor
        return d.state == "WI" and d.peril in WATER[:3] and (
            d.did in _PASSES)
    if m.mid == "M-T0419":
        return d.state == "IL" and d.peril == "hail"
    if m.asserts:
        return m.asserts[0] == f"{d.state.lower()}_{d.peril}"
    return False


def order(m):
    return (0 if m.asserts and m.scope != "policy" else 1 if m.scope == "policy"
            else 2 if m.scope == "adjuster" else 3, m.mid)


def answer(d, mems):
    if d.residual:
        return "wrong:residual"
    truth = d.truth()
    for m in mems:
        if not m.asserts:
            continue
        topic, val = m.asserts
        if topic in ("wi_water_cause", "wi_basement_water") and d.peril in ("water_backup", "mould"):
            if d.did in L.FOLLOWED:                     # the adjuster accepted the named cause
                return "freezing:HO3-freezing"
        if topic == "il_hail" and truth in ("covered:HO3-hail", "covered:hail"):
            corroborated = len({o.ref for o in m.origins}) >= 3     # looks like consensus
            if L.u(d.did, m.mid) < (0.60 if corroborated else L.P_FOLLOW[m.scope]):
                return "excluded:CDE"
        if topic == "ho0495" and d.peril == "water_backup" and val != d.ho0495:
            if L.u(d.did, m.mid) < L.P_FOLLOW["policy"]:
                return "covered:HO0495" if val else "excluded:HO3-2.c"
    if len(mems) > 6 and L.u(d.did, "distract") < L.P_DISTRACT * (len(mems) - 6):
        return "wrong:distracted"
    return truth


_PASSES = set()


def run(dets=None, second=False, gate=False, monitor=False, k_max=L.K_MAX,
        quarantine_on=None, rollback_on=None, memory=True):
    """quarantine_on: (day, root) quarantines the root's family from that day.
    rollback_on: (day, before) restores the store to `before` from `day`."""
    dets = dets or L.make_mix()
    _PASSES.clear(); _PASSES.update(L.lesson_passes(dets))
    store = L.seed_store(dets)
    for m in L.correct_positions():
        store[m.mid] = m
    store["M02116"] = L.poison()                         # written by the go-live back-fill
    if gate:
        for m in list(store.values()):
            G.write_gate(m)
    log, out, diverge, events = ReadLog(), {}, {}, []
    by_day = {}
    for d in dets:
        by_day.setdefault(d.day, []).append(d)
    for day in L.DAYS:
        if quarantine_on and day == quarantine_on[0]:
            fam = G.family(store, quarantine_on[1])
            events.append((day, "quarantine", sorted(fam), G.quarantine(store, fam, "incident 22-01")))
        if rollback_on and day == rollback_on[0]:
            events.append((day, "rollback", rollback_on[1], G.rollback(store, rollback_on[1])))
        for d in by_day.get(day, []):
            mems = sorted((m for m in store.values() if memory and relevant(m, d)), key=order)[:k_max]
            log.log(d.did, d.day, [m.mid for m in mems], d.keys())
            a = answer(d, mems)
            if monitor:
                for m in mems:
                    if m.asserts and m.authority() == "unverified" and m.scope != "policy":
                        if answer(d, [x for x in mems if x is not m]) != a:
                            a = "review:" + d.truth()            # held; a person decides
                            diverge[m.mid] = diverge.get(m.mid, 0) + 1
                            if diverge[m.mid] >= 2 and m.status == "live":
                                fam = G.family(store, G.root_of(store, m.mid))
                                G.quarantine(store, fam, "read-time divergence")
                                events.append((day, "monitor", sorted(fam), d.did))
            out[d.did] = a
        _consolidate(store, dets, by_day.get(day, []), out, log, day, second, gate)
    return dets, store, log, out, events


def _consolidate(store, dets, today, out, log, day, second, gate):
    new = [M(f"M-E{d.did}", "claimant", d.policy,
             f"claim {d.did} ({d.peril.replace('_', ' ')}): {out[d.did]}", day,
             (O("determination", d.did),)) for d in today]
    if second and day == L.SECOND_NIGHT:
        new.append(L.second_poison())
    if day == L.DAYS[4]:                     # Friday 9 October: adjuster notes
        for adj in ("A01", "A02"):
            mid = f"M-A{adj[1:]}-D"
            fed = [d for d in today if d.adjuster == adj and "M02116" in log.read_by(d.did)
                   and d.peril in WATER[:3]]
            if fed and mid not in store:
                new.append(M(mid, "adjuster", adj,
                             "Wisconsin basement water: check the supply lines in the north wall before you code back-up.",
                             day, tuple(O("determination", d.did) for d in fed), parents=("M02116",),
                             asserts=("wi_basement_water", "excluded")))
    for m in new:
        if gate:
            G.write_gate(m)
        store[m.mid] = m


def final(a):
    return a.split(":", 1)[1] if a.startswith("review:") else a


def wrong(dets, out):
    return [d.did for d in dets if final(out[d.did]) != d.truth()]
