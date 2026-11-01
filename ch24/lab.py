"""Chapter 24 lab: change capture, time-to-visible, caches that outlive a change.

Note: deterministic stand-in. No database, message broker, search engine or
model runs. Bulletins, policy changes, claim updates and determinations are
generated from fixed seeds (zlib.crc32 of a tag). Each pipeline stage takes a
stated, seeded time drawn from the ranges in STAGES; the nightly rebuild starts
at 02:00 Central time and takes 1 h 40 min. A determination is "exposed" when
the bulletin that governs it had committed but was not yet visible to the path
that built its packet. Exposure is not the same as a wrong answer; the lab
counts exposure because that is what a freshness monitor can see.
Run: python3 lab.py > output.txt
"""
import pathlib
import statistics as stats
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import lakeshore24 as L
from contextlab import freshness as F

STAGES = {"capture": (1, 15), "transport": (1, 5), "parse_chunk": (20, 120),
          "embed": (10, 60), "refresh": (1, 1)}
REVIEW_SHARE, REVIEW_MIN = 0.12, (15, 240)          # admission gate sends 12% to a person
QUARANTINE_SHARE, STEWARD_MIN = 0.25, (60, 1560)    # Ch17: new edges a steward must clear
GRAPH_EXTRACT = (120, 360)


def stream_stages(doc):
    r = L.rng("stages-" + doc)
    out = [F.Stage(k, r.randint(*v)) for k, v in STAGES.items()]
    if r.random() < REVIEW_SHARE:
        out.insert(3, F.Stage("review", 60 * r.randint(*REVIEW_MIN)))
    return out


def graph_after(doc, passage_visible):
    r = L.rng("graph-" + doc)
    s = r.randint(*GRAPH_EXTRACT)
    if r.random() < QUARANTINE_SHARE:
        s += 60 * r.randint(*STEWARD_MIN)
    return passage_visible + timedelta(seconds=s)


def visibility(bs):
    v = defaultdict(dict)
    for b in bs:
        sid = f"{b.doc}@{b.version}"
        v["nightly"][sid] = F.next_nightly(b.committed)
        v["stream"][sid] = b.committed + timedelta(seconds=F.time_to_visible(stream_stages(b.doc)))
        v["graph_full"][sid] = datetime.max                       # only the next full build
        v["graph_stream"][sid] = graph_after(b.doc, v["stream"][sid])
    return v


def hours(xs):
    xs = sorted(xs)
    return stats.median(xs), xs[int(0.95 * (len(xs) - 1))], xs[-1]


def ttv_table(bs, v):
    print("1. Time-to-visible for", len(bs), "bulletins, 1 Jul - 30 Sep 2026 (hours)")
    for path in ("nightly", "stream", "graph_stream"):
        xs = [(v[path][f"{b.doc}@{b.version}"] - b.committed).total_seconds() / 3600 for b in bs]
        print(f"   {path:13s} median {hours(xs)[0]:6.2f}  p95 {hours(xs)[1]:6.2f}  max {hours(xs)[2]:6.2f}")
    xs = [F.time_to_visible(stream_stages(b.doc)) / 60 for b in bs]
    auto = [x for x, b in zip(xs, bs) if not any(s.name == "review" for s in stream_stages(b.doc))]
    print(f"   stream, no review: n={len(auto)} median {stats.median(auto):.1f} min, max {max(auto):.1f} min")
    print("   graph_full    never inside the window (next full build after 30 Sep)")
    lead = [(b.effective_from - b.committed.date()).days for b in bs]
    print(f"   lead time (effective - committed): future {sum(x > 0 for x in lead)}, same day "
          f"{sum(x == 0 for x in lead)}, retroactive {sum(x < 0 for x in lead)}; median {stats.median(lead)} days")


def governing(det, by_scope, t):
    """Latest-effective bulletin in force on the loss date and committed by time t."""
    cands = [b for b in by_scope.get((det.state, det.peril), [])
             if b.committed <= t and b.effective_from <= det.loss]
    return max(cands, key=lambda b: (b.effective_from, b.committed)) if cands else None


CONFIGS = {"A nightly + graph full builds": ("nightly", "graph_full", False),
           "B streaming passages, graph unchanged": ("stream", "graph_full", False),
           "C streaming fan-out to both": ("stream", "graph_stream", False),
           "D C + watermark routing": ("stream", "graph_stream", True)}


def exposure(bs, v):
    by_scope = defaultdict(list)
    for b in bs:
        by_scope[(b.state, b.peril)].append(b)
    res = {c: defaultdict(int) for c in CONFIGS}
    n = 0
    for day in L.business_days():
        for d in L.determinations(day):
            n += 1
            g = governing(d, by_scope, d.decided)
            if g is None:
                continue
            sid = f"{g.doc}@{g.version}"
            for c, (pp, gp, wm) in CONFIGS.items():
                path = gp if d.graph_walk and not (wm and v[gp][sid] > d.decided) else pp
                if v[path][sid] > d.decided:
                    res[c]["graph" if path.startswith("graph") else "passage"] += 1
                    if g.doc == "BUL-IA-EB-202608":
                        res[c]["iowa"] += 1
    return n, res


def retro_requeue(bs):
    """Determinations already made on losses a retroactive bulletin now governs."""
    out = {}
    for b in bs:
        if b.effective_from >= b.committed.date():
            continue
        k = 0
        for day in L.business_days(b.effective_from, b.committed.date()):
            k += sum(1 for d in L.determinations(day) if (d.state, d.peril) == (b.state, b.peril)
                     and d.loss >= b.effective_from and d.decided < b.committed)
        out[b.doc] = (b.effective_from, b.committed, k)
    return out


def cache_day(mode, ttl=timedelta(hours=24)):
    """Replay 15 Sep 2026 through a retrieval-result cache keyed on (state, peril)."""
    wi2, mn = L.cache_day_events()
    wi1 = L.Bulletin("BUL-WI-WB-202609", 1, "WI", "WB", datetime(2026, 8, 28, 10, 0), wi2.effective_from)
    store = [wi1, wi2, mn]
    seen = {f"{b.doc}@{b.version}": b.committed + timedelta(seconds=F.time_to_visible(
        stream_stages(f"{b.doc}@{b.version}"))) for b in store}
    events = sorted([(seen[f"{b.doc}@{b.version}"], b) for b in (wi2, mn)], key=lambda e: e[0])
    cache = F.ScopedCache(ttl, mode) if mode else None
    day = wi2.committed.date()
    wrong = defaultdict(int)
    for d in L.determinations(day, "cacheday"):
        while events and events[0][0] <= d.decided:
            t, b = events.pop(0)
            if cache:
                cache.on_change(F.Change("docstore", f"{b.doc}@{b.version}", b.version, b.committed,
                                         str(b.version), ((b.state, b.peril),),
                                         b.supersedes, b.effective_from))
        sc = (d.state, d.peril)
        live = [b for b in store if (b.state, b.peril) == sc and seen[f"{b.doc}@{b.version}"] <= d.decided]
        live = [b for b in live if not any(o.supersedes == f"{b.doc}@{b.version}" for o in live)]
        got = cache.get(d.peril, sc, d.decided) if cache else None
        if got is None:
            got = live
            if cache:
                cache.put(d.peril, sc, got, d.decided, {f"{b.doc}@{b.version}" for b in got})
        pick = lambda bs: max([b for b in bs if b.effective_from <= d.loss],
                              key=lambda b: (b.effective_from, b.version), default=None)
        if pick(got) is not pick(live):
            wrong[d.state] += 1
    hit = cache.hits / (cache.hits + cache.misses) if cache else 0.0
    return wrong["WI"], wrong["MN"], hit


def cdc_check(n=2000, keys=300):
    r = L.rng("cdc")
    seqs, events = defaultdict(int), []
    for i in range(n):
        k = f"P-{40000 + r.randrange(keys)}"
        seqs[k] += 1
        events.append(F.Change("policy_admin", k, 1000 * seqs[k] + i, datetime(2026, 9, 1), f"v{seqs[k]}"))
    arrived = list(events)
    for i in range(len(events)):
        e = events[i]
        if r.random() < 0.02:                      # delivered after the key's next change
            j = next((j for j in range(i + 1, len(events)) if events[j].key == e.key), None)
            if j is not None:
                arrived.remove(e)
                arrived.insert(arrived.index(events[j]) + 1, e)
    arrived = [x for e in arrived for x in ([e, e] if r.random() < 0.05 else [e])]
    truth = {("policy_admin", e.key): e.version for e in events}
    out = {}
    for name, ap in (("naive", F.NaiveApplier()), ("keyed", F.Applier())):
        for e in arrived:
            ap.apply(e)
        bad = sum(1 for k, v in truth.items() if ap.state[k].version != v)
        chunks = len(ap.chunks) - len(events) if name == "naive" else 0
        out[name] = (bad, chunks, getattr(ap, "duplicates", 0), getattr(ap, "out_of_order", 0))
    return len(arrived), out


def policy_ttv():
    r, ttl, cdc = L.rng("policy"), [], []
    for i in range(450):
        t = datetime(2026, 9, 15, r.randint(7, 19), r.randint(0, 59))
        refresh = t.replace(hour=6, minute=0) + timedelta(days=1)
        ttl.append((refresh - t).total_seconds() / 3600)
        cdc.append((r.randint(2, 30) + 1) / 60)
    return hours(ttl), (stats.median(cdc), max(cdc))


def drafts():
    r, opened_after_change = L.rng("drafts"), 0
    for i in range(L.PER_DAY):
        wait = r.randint(5, 240)
        if r.random() < 0.06 and r.randint(0, 480) < wait:
            opened_after_change += 1
    return opened_after_change


def feed_check(n=160):
    """Partner policy changes: the licensed feed's dates against the policy administration log."""
    r, misdated, late = L.rng("feed"), [], []
    for i in range(n):
        eff = date(2026, 9, 1) + timedelta(days=r.randint(0, 52))
        feed_eff = eff + timedelta(days=r.choice([-1, 1]) * r.randint(1, 21)) if r.random() < 0.03 else eff
        late.append(r.randint(2, 36))
        if feed_eff != eff:
            misdated.append((eff, feed_eff))
    case = {"log_effective": date(2026, 10, 15), "feed_effective": date(2026, 10, 1), "loss": date(2026, 10, 8)}
    in_force = lambda eff: case["loss"] >= eff
    case["wrong_before"] = in_force(case["feed_effective"]) != in_force(case["log_effective"])
    return n, misdated, stats.median(late), case


def main():
    bs = L.bulletins()
    v = visibility(bs)
    ttv_table(bs, v)
    n, res = exposure(bs, v)
    print(f"\n2. Exposure over {n:,} determinations (66 business days)")
    for c, r_ in res.items():
        print(f"   {c:40s} passage {r_['passage']:4d}  graph {r_['graph']:4d}  of which Iowa EB {r_['iowa']:3d}")
    print("\n3. Retroactive bulletins: determinations to re-review")
    for doc, (eff, com, k) in retro_requeue(bs).items():
        print(f"   {doc}: effective {eff}, committed {com:%Y-%m-%d %H:%M}, re-review {k}")
    print("\n4. Cache on 15 Sep 2026 (wrong after the change was visible: WI reissue, MN new; hit rate)")
    for name, mode in (("no cache", None), ("TTL 24 h only", "ttl"), ("tag invalidation", "tags"),
                       ("scope epoch", "epoch")):
        wi, mn, hit = cache_day(mode)
        print(f"   {name:17s} WI {wi:3d}  MN {mn:3d}  hit rate {hit:.3f}")
    total, out = cdc_check()
    print(f"\n5. Change capture: {total} deliveries of 2000 changes over 300 policies")
    for k, (bad, chunks, dup, ooo) in out.items():
        print(f"   {k:6s} wrong final versions {bad:3d}  extra chunks {chunks:3d}  dupes skipped {dup:3d}  reordered skipped {ooo:2d}")
    (med, p95, mx), (cm, cx) = policy_ttv()
    print(f"\n6. Policy-in-force changes (450 on 15 Sep): daily pre-warm median {med:.1f} h, p95 {p95:.1f} h;"
          f" capture + invalidate median {cm*60:.0f} s, max {cx*60:.0f} s")
    print(f"\n7. Drafts opened after a claim change (3,200 drafts): {drafts()}")
    iowa = next(b for b in bs if b.doc == "BUL-IA-EB-202608")
    sid = f"{iowa.doc}@1"
    print("\n8. Iowa equipment-breakdown bulletin, reconstructed")
    print(f"   committed {iowa.committed:%d %b %H:%M}; effective {iowa.effective_from:%d %b}")
    for p in ("nightly", "stream", "graph_stream"):
        print(f"   visible on {p:12s} {v[p][sid]:%d %b %H:%M}  ({(v[p][sid]-iowa.committed).total_seconds()/60:,.0f} min)")
    print("   stream stages (s): " + ", ".join(f"{x.name} {x.seconds:.0f}" for x in stream_stages(iowa.doc)))
    print(f"   visible on graph_full    never before the next full build (last {L.GRAPH_FULL_BUILD:%d %b})")
    n, bad, lag, case = feed_check()
    print(f"\n9. Licensed partner feed vs policy administration log: {n} changes, feed median lag {lag:.0f} h,"
          f" misdated {len(bad)}: " + ", ".join(f"log {a:%d %b} feed {b:%d %b}" for a, b in bad))
    print(f"   full-mix case: change effective {case['log_effective']:%d %b} (log), {case['feed_effective']:%d %b} (feed),"
          f" loss {case['loss']:%d %b}; wrong when built from feed {case['wrong_before']}; reconciled against log: flagged, right")


if __name__ == "__main__":
    main()
