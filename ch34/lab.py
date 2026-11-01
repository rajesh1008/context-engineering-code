"""Chapter 34 lab: lay out the packet for the cache and measure what it saves.

Runs on deterministic stand-ins (seed 34): synthetic traffic, a stand-in
prefix cache and a stand-in tariff. Writes output.txt and out/*.json.
"""
import json
import math
import os
import random
from collections import Counter, defaultdict
from contextlab.caching import (PrefixCache, Prices, ResultCache, CallResult,
                                lint_packet, hit_rate)
from lakeshore34 import (Builder, run_day, FIX_STEPS, route_orders, traffic,
                         TYPES, STATES, FORMS, hhmm)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
LINES = []
FINAL = ("stamp_end", "canonical", "reorder")


def say(*a):
    LINES.append(" ".join(str(x) for x in a))


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1)


class NoCache(PrefixCache):
    def call(self, t, segments, output_tokens, model="m", routing_key=None):
        n = sum(s.tokens for s in segments)
        return CallResult(n, 0, 0, output_tokens,
                          self.prices.cost(n, 0, 0, output_tokens), 0)


def breakdown(res, prices):
    u = sum(r.input_tokens - r.cached - r.written for r in res)
    c = sum(r.cached for r in res)
    w = sum(r.written for r in res)
    o = sum(r.output for r in res)
    dets = len({r.det for r in res})
    return {"uncached_input": u * prices.input / 1e6 / dets,
            "cached_reads": c * prices.read * prices.input / 1e6 / dets,
            "write_premium": w * prices.write * prices.input / 1e6 / dets,
            "output": o * prices.output / 1e6 / dets}


def section_fixes():
    say("== 1. The day as found, then one fix at a time ==")
    base = run_day(Builder(()), cache=NoCache())
    say(f"no cache at all: cost/det US${base['cost_per_det']:.4f}")
    rows = []
    for name, fx in FIX_STEPS:
        r = run_day(Builder(fx))
        rows.append(dict(step=name, hit_rate=r["hit_rate"],
                         cost_per_det=r["cost_per_det"]))
        say(f"{name:28s} hit {r['hit_rate']:.1%}  cost/det US${r['cost_per_det']:.4f}")
    k = run_day(Builder(FIX_STEPS[-1][1]), routing=lambda d: d["state"] + d["form"])
    say(f"{'all 12 tools + routing key':28s} hit {k['hit_rate']:.1%}  "
        f"cost/det US${k['cost_per_det']:.4f}")
    rows.append(dict(step="all 12 tools + routing key", hit_rate=k["hit_rate"],
                     cost_per_det=k["cost_per_det"]))
    first, best = rows[0], min(rows, key=lambda x: x["cost_per_det"])
    fall = 1 - best["cost_per_det"] / first["cost_per_det"]
    say(f"chosen: {best['step']}; cost/det falls {fall:.1%}")
    p = Prices()
    b0 = breakdown(run_day(Builder(()))["results"], p)
    b1 = breakdown(run_day(Builder(FINAL))["results"], p)
    say("breakdown v7.4:", {k: round(v, 4) for k, v in b0.items()})
    say("breakdown v7.5:", {k: round(v, 4) for k, v in b1.items()})
    dump("fixes.json", dict(no_cache=base["cost_per_det"], steps=rows,
                            fall=fall, breakdown_v74=b0, breakdown_v75=b1))
    return rows


def section_timestamp():
    say("== 2. Failure injection: a timestamp in the system prompt ==")
    good = run_day(Builder(FINAL))

    class Stamped(Builder):
        def packet(self, d, t, wm, adjuster="T. Brennan"):
            segs = super().packet(d, t, wm, adjuster)
            i = [s.name for s in segs].index("instructions")
            s = segs[i]
            segs[i] = type(s)(s.name, s.text + f" now {hhmm(t)}", s.tokens, s.volatility)
            return segs
    stamped = Stamped(FINAL)
    stamped.rerender = True                 # rendered on every call
    bad = run_day(stamped)
    lint = lint_packet(Stamped(FINAL).packet(traffic(1)[0], 36000.0, "wm"))
    say(f"clean v7.5: hit {good['hit_rate']:.1%} cost/det US${good['cost_per_det']:.4f}")
    say(f"stamped:    hit {bad['hit_rate']:.1%} cost/det US${bad['cost_per_det']:.4f}")
    say(f"lint on stamped packet: {lint}")
    dump("timestamp.json", dict(clean=good["hit_rate"], stamped=bad["hit_rate"],
                                clean_cost=good["cost_per_det"],
                                stamped_cost=bad["cost_per_det"]))


def section_lint():
    say("== 3. Cache lint on the v7.4 builder ==")
    b = Builder(())
    d = traffic(1)[0]
    segs = b.packet(d, 36000.0, "wm-0930")
    orders = len({Builder(()).tools(dict(d, worker=w)).text for w in range(4)})
    f = lint_packet(segs, tool_orders_seen=orders,
                    memo_keys={"render_state_rules": "worker,state"})
    for rule, where in f:
        say(f"  {rule}: {where}")
    f2 = lint_packet(Builder(FINAL).packet(d, 36000.0, "wm-0930"),
                     memo_keys={"render_state_rules": "worker,state,watermark"})
    say(f"v7.5 findings: {len(f2)}")
    dump("lint.json", dict(v74=f, v75=f2))


def section_route_order():
    say("== 4. Ordering the route-level segments ==")
    rows = []
    for order in route_orders():
        r = run_day(Builder(FINAL, route_order=order), n=1600)
        rows.append(dict(order=list(order), hit_rate=r["hit_rate"],
                         cost_per_det=r["cost_per_det"]))
    rows.sort(key=lambda x: x["cost_per_det"])
    for x in rows[:3] + rows[-2:]:
        say(f"  {'>'.join(x['order']):34s} hit {x['hit_rate']:.1%} "
            f"cost/det US${x['cost_per_det']:.4f}")
    dump("route_order.json", rows)
    return rows


def section_invalidation():
    say("== 5. Invalidation events and their cost ==")
    base = run_day(Builder(FINAL, memo_watermark=True))
    cache_holder = {}

    def mk_cache():
        c = PrefixCache(shards=4)
        cache_holder["c"] = c
        return c

    def model_switch(b, wm):
        for s in cache_holder["c"].store:
            s.clear()
    events = {
        "Rulebook deploy 12:30": [(45000, lambda b, wm: setattr(b, "rulebook", "rb-42"))],
        "tool description edit 12:30": [(45000, lambda b, wm: setattr(b, "tool_rev", "t-8"))],
        "WI bulletin 10:00": [(36000, lambda b, wm: wm.__setitem__("WI", "wm-1000"))],
        "model change 12:30": [(45000, model_switch)],
        "deploy every 30 min": [(25200 + 1800 * i, (lambda i: lambda b, wm:
                                 setattr(b, "rulebook", f"rb-x{i}"))(i)) for i in range(24)],
    }
    rows = []
    for name, ev in events.items():
        r = run_day(Builder(FINAL, memo_watermark=True), cache=mk_cache(), events=ev)
        extra = r["cost_day"] - base["cost_day"]
        rows.append(dict(event=name, extra_usd=extra, hit_rate=r["hit_rate"]))
        say(f"  {name:28s} extra US${extra:7.2f}/day  hit {r['hit_rate']:.1%}")
    c = run_day(Builder(FINAL, memo_watermark=True), compact_share=0.3)
    extra = c["cost_day"] - base["cost_day"]
    rows.append(dict(event="compaction on 30% of call 3", extra_usd=extra,
                     hit_rate=c["hit_rate"]))
    say(f"  {'compaction on 30% of call 3':28s} extra US${extra:7.2f}/day  hit {c['hit_rate']:.1%}")
    dump("invalidation.json", dict(base_day=base["cost_day"], rows=rows))


def section_ttl():
    say("== 6. Implicit, 5-minute and 1-hour caching; low-volume routes ==")
    rows = []
    for name, pr, ttl in (("implicit, no write premium", Prices(write=1.0), 300),
                          ("explicit 5m, writes 1.25x", Prices(write=1.25), 300),
                          ("explicit 1h, writes 2x", Prices(write=2.0), 3600)):
        r = run_day(Builder(FINAL), cache=PrefixCache(prices=pr, ttl_s=ttl, shards=4))
        rows.append(dict(mode=name, hit_rate=r["hit_rate"], cost_per_det=r["cost_per_det"]))
        say(f"  {name:28s} hit {r['hit_rate']:.1%} cost/det US${r['cost_per_det']:.4f}")
    r = run_day(Builder(FINAL))
    by = defaultdict(list)
    for x in r["results"]:
        if x.k == 0:
            by[x.state].append(x)
    st = {s: hit_rate(v) for s, v in sorted(by.items())}
    say("  first-call hit rate by state:", {k: f"{v:.1%}" for k, v in st.items()})
    hours = defaultdict(list)
    for x in r["results"]:
        hours[int(x.t // 3600)].append(x)
    hr = {h: hit_rate(v) for h, v in sorted(hours.items())}
    say("  hit rate by hour:", {h: f"{v:.0%}" for h, v in hr.items()})
    dump("ttl.json", dict(modes=rows, first_call_by_state=st, by_hour=hr))


def section_freshness():
    say("== 7. The freshness bug the cache metrics found ==")
    out = {}
    for label, fixed in (("v7.5 as first built", False), ("memo keyed on watermark", True)):
        b = Builder(FINAL, memo_watermark=fixed)
        ev = [(36000, lambda b, wm: wm.__setitem__("WI", "wm-1000")),
              (50400, lambda b, wm: b.memo.clear())]      # 14:00 restart
        r = run_day(b, events=ev)
        first_seen, stale = {}, []
        for x in r["results"]:
            if x.state != "WI" or x.k:
                continue
            st = next(s for s in x.segs if s.name == "state").text
            first_seen.setdefault(st, x.t)
            if x.t >= 36000 and "wm-1000" not in st:
                stale.append(x.det)
        new_keys = sorted((v, k) for k, v in first_seen.items())
        seen = [f"{int(v // 3600):02d}:{int(v % 3600 // 60):02d} {k}" for v, k in new_keys]
        alert = not any(36000 <= v <= 36600 for v, _ in new_keys)
        dets = {d["id"]: d for d in traffic()}
        backup = sum(dets[i]["ctype"] == "water_backup" for i in stale)
        out[label] = dict(new_state_prefixes=seen, alert_fired=alert,
                          stale_wi_determinations=len(stale), stale_backup=backup)
        say(f"  {label}: WI state prefixes first seen {seen}; alert={alert}; "
            f"stale WI determinations {len(stale)} (water back-up {backup})")
    dump("freshness.json", out)


def words(q):
    v = Counter(q.lower().replace("?", "").replace(",", "").split())
    n = math.sqrt(sum(c * c for c in v.values()))
    return v, n


def cosine(a, b):
    (va, na), (vb, nb) = a, b
    return sum(c * vb.get(w, 0) for w, c in va.items()) / (na * nb)


def section_semantic():
    say("== 8. Result cache and a semantic-cache audit ==")
    rng = random.Random(34)
    names = {"WI": "Wisconsin", "MN": "Minnesota", "IA": "Iowa", "IL": "Illinois",
             "MI": "Michigan"}
    peril = {"water_backup": "water back-up from a sewer", "frozen_pipe": "a frozen pipe",
             "mould": "hidden mould", "hail": "hail damage to the roof",
             "wind": "wind damage to siding", "theft": "theft from a garage"}
    frames = ["Is {p} covered under {f} in {s}, claim of ${a}?",
              "Does {f} cover {p} for a {s} policyholder, loss ${a}?",
              "For a {s} claim on {f} of ${a}, is {p} covered?"]
    qs = []
    for i in range(1500):
        s, t, f = rng.choice(list(STATES)), rng.choice(list(TYPES)), rng.choice(list(FORMS))
        q = rng.choice(frames).format(p=peril[t], f=f.replace("@", " edition "), s=names[s],
                                    a=f"{rng.randrange(20, 400) * 100:,}")
        qs.append(((s, t, f), q, words(q)))
    rows = []
    for th in (0.70, 0.80, 0.85, 0.90, 0.95):
        cache, exact, para, wrong = [], 0, 0, 0
        for facts, q, v in qs:
            best = max(cache, key=lambda c: cosine(c[2], v), default=None)
            if best and cosine(best[2], v) >= th:
                if best[1] == q:
                    exact += 1
                elif best[0] == facts:
                    para += 1
                else:
                    wrong += 1
            else:
                cache.append((facts, q, v))
        hits = exact + para + wrong
        rows.append(dict(threshold=th, exact=exact / len(qs), paraphrase=para / len(qs),
                         wrong=wrong / len(qs), wrong_among_hits=wrong / max(hits, 1)))
        say(f"  threshold {th:.2f}: exact repeats {exact / len(qs):.1%}, "
            f"paraphrase hits {para / len(qs):.1%}, wrong-fact hits {wrong / len(qs):.1%} "
            f"({wrong / max(hits, 1):.1%} of hits)")
    rc, hits, stale_blocked = ResultCache(), 0, 0
    for d in traffic():
        wm = "wm-1000" if d["t"] >= 36000 and d["state"] == "WI" else "wm-0930"
        k = rc.key(d["id"], "determine", "snap-1", wm, "fp-v7.5", "sim-large")
        rc.put(k, "det")
        if d["reopen"]:
            wm2 = "wm-1000" if d["state"] == "WI" else wm
            k2 = rc.key(d["id"], "Determine ", "snap-1", wm2, "fp-v7.5", "sim-large")
            if rc.get(k2):
                hits += 1
            else:
                stale_blocked += 1
    say(f"  exact-key result cache: {hits} re-opened determinations served, "
        f"{stale_blocked} recomputed because the watermark moved")
    dump("semantic.json", dict(semantic=rows, result_cache_hits=hits,
                               result_cache_recomputed=stale_blocked))


if __name__ == "__main__":
    section_fixes()
    section_timestamp()
    section_lint()
    section_route_order()
    section_invalidation()
    section_ttl()
    section_freshness()
    section_semantic()
    text = "\n".join(LINES) + "\n"
    open(os.path.join(os.path.dirname(OUT), "output.txt"), "w").write(text)
    print(text)
