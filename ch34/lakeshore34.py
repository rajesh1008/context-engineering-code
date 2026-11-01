"""Chapter 34 stand-in: one replayed day of Lakeshore determinations.

Deterministic synthetic traffic (seed 34). Token counts are stand-ins chosen
to match earlier chapters (tool loadout 1,661 tokens from Chapter 26); prices
are the stand-in tariff in contextlab.caching.Prices, not a vendor list.
"""
import random
from itertools import permutations
from contextlab.caching import Segment, PrefixCache, Prices, cache_layout, hit_rate

STATES = {"WI": .62, "MN": .14, "IA": .10, "IL": .09, "MI": .05}
TYPES = {"water_backup": .25, "frozen_pipe": .15, "mould": .10,
         "hail": .20, "wind": .18, "theft": .12}
FORMS = {"HO-3@2024-06": .6, "HO-3@2022-01": .4}
TOOLS = ["get_policy", "get_declarations", "search_clauses", "get_endorsement",
         "get_bulletins", "get_claim_history", "get_photos", "get_estimate",
         "check_coverage_limits", "get_weather", "get_police_report",
         "draft_determination"]
LOADOUT = {"water_backup": [3, 4, 7], "frozen_pipe": [9, 7, 5], "mould": [5, 6, 4],
           "hail": [9, 6, 7], "wind": [9, 6, 8], "theft": [10, 5, 7]}
WORKERS = 4
DAY0, DAY1 = 7 * 3600, 19 * 3600
CALLS = [(0, 400), (1300, 500), (1100, 2300)]   # (tool tokens in, output incl. reasoning)


def hash_share(text):
    return (sum(map(ord, text)) * 2654435761 % 1000) / 1000


def pick(rng, table):
    return rng.choices(list(table), weights=list(table.values()))[0]


def traffic(n=3200, seed=34):
    rng, out = random.Random(seed), []
    for i in range(n):
        t = DAY0 + (DAY1 - DAY0) * rng.betavariate(2.2, 2.6)
        out.append(dict(id=f"C{i:04d}", t=t, state=pick(rng, STATES),
                        ctype=pick(rng, TYPES), form=pick(rng, FORMS),
                        endorse=rng.random() < .3, worker=rng.randrange(WORKERS),
                        gaps=(rng.uniform(5, 15), rng.uniform(8, 25)),
                        reopen=rng.random() < .06))
    return sorted(out, key=lambda d: d["t"])


def hhmm(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


class Builder:
    """Renders packets. `layout` is v7.4 (as found) or a list of fixes."""

    def __init__(self, fixes=(), rulebook="rb-41", tool_rev="t-7",
                 route_order=("examples", "state", "form", "endorse"),
                 memo_watermark=False):
        self.fixes, self.rulebook, self.tool_rev = set(fixes), rulebook, tool_rev
        self.route_order, self.memo_watermark = route_order, memo_watermark
        self.memo = {}                     # per-worker memo of state rules
        self.rerender = "stamp_end" not in self.fixes

    def state_rules(self, d, watermark):
        key = (d["worker"], d["state"]) + ((watermark,) if self.memo_watermark else ())
        if key not in self.memo:
            self.memo[key] = f"rules {d['state']} as of {watermark}"
        return self.memo[key]

    def tools(self, d):
        if "all_tools" in self.fixes:
            return Segment("tools", f"12 tools canonical {self.tool_rev}", 2840)
        idx = [0, 1, 2, 11] + LOADOUT[d["ctype"]]
        if "canonical" not in self.fixes:      # set-ordered JSON per worker
            rng = random.Random(f"{d['worker']}{d['ctype']}")
            rng.shuffle(idx)
        return Segment("tools", f"loadout {idx} {self.tool_rev}", 1661)

    def packet(self, d, t, watermark, adjuster="T. Brennan"):
        f = self.fixes
        stamp = (f"date {int(t // 86400)}" if "stamp_end" in f
                 else f"Requested {hhmm(t)} by {adjuster}")
        segs = [self.tools(d),
                Segment("header", stamp, 45, "call" if "stamp_end" not in f else "daily"),
                Segment("instructions", f"rulebook {self.rulebook}", 2150),
                Segment("examples", f"examples {d['ctype']}", 1100, "route"),
                Segment("facts", f"facts {d['id']}", 1200, "claim"),
                Segment("form", f"form {d['form']}", 1900, "route"),
                Segment("state", self.state_rules(d, watermark), 650, "route")]
        if d["endorse"]:
            segs.append(Segment("endorse", f"HO 04 95 {d['form']}", 700, "route"))
        segs += [Segment("evidence", f"evidence {d['id']}", 1400, "claim"),
                 Segment("question", f"question {d['id']}", 60, "claim")]
        if "reorder" in f:
            segs = cache_layout(segs, self.route_order)
        if "stamp_end" in f:                    # date line after everything
            segs = [s for s in segs if s.name != "header"] + [
                Segment("header", stamp, 12, "call")]
        return segs


def run_day(builder, cache=None, n=3200, events=None, routing=None, seed=34,
            compact_share=0.0):
    """Replay a day. events: list of (time, callable(builder, state_dict))."""
    cache = cache or PrefixCache(shards=4)
    calls, wm = [], {"WI": "wm-0930", "MN": "wm-0700", "IA": "wm-0700",
                     "IL": "wm-0700", "MI": "wm-0700"}
    for d in traffic(n, seed):
        t = d["t"]
        for k, (tool_in, out) in enumerate(CALLS):
            calls.append((t, k, d, tool_in, out))
            if k < 2:
                t += d["gaps"][k]
    calls.sort(key=lambda c: c[0])
    events = sorted(events or [], key=lambda e: e[0])
    res, per_det, ev = [], {}, 0
    hist = {}
    for t, k, d, tool_in, out in calls:
        while ev < len(events) and events[ev][0] <= t:
            events[ev][1](builder, wm)
            ev += 1
        if k == 0:
            hist[d["id"]] = builder.packet(d, t, wm[d["state"]])
        elif builder.rerender:                  # v7.4 re-renders the header
            fresh = builder.packet(d, t, wm[d["state"]])
            hist[d["id"]] = fresh + hist[d["id"]][len(fresh):]
        segs = hist[d["id"]]
        if k > 0:                               # append, never rewrite
            prev_out = CALLS[k - 1][1]
            segs = segs + [Segment(f"turn{k}", f"{d['id']} out{k}", prev_out, "call"),
                           Segment(f"tools{k}", f"{d['id']} res{k}", tool_in, "call")]
            if k == 2 and compact_share and hash_share(d["id"]) < compact_share:
                segs = [s if s.name != "tools1" else
                        Segment("tools1", f"{d['id']} summary", 400, "call")
                        for s in segs]          # Chapter 33 compaction rewrite
            hist[d["id"]] = segs
        key = routing(d) if routing else None
        r = cache.call(t, segs, out, routing_key=key)
        r.det, r.t, r.k, r.state = d["id"], t, k, d["state"]
        r.segs = segs
        res.append(r)
        per_det[d["id"]] = per_det.get(d["id"], 0) + r.cost
    cost = sum(per_det.values()) / len(per_det)
    return dict(hit_rate=hit_rate(res), cost_per_det=cost,
                cost_day=sum(per_det.values()), results=res)


FIX_STEPS = [("v7.4 as found", ()),
             ("timestamp to end, date only", ("stamp_end",)),
             ("canonical tool JSON", ("stamp_end", "canonical")),
             ("stable before volatile", ("stamp_end", "canonical", "reorder")),
             ("all 12 tools, allowed list", ("stamp_end", "canonical", "reorder",
                                             "all_tools"))]


def route_orders():
    return list(permutations(("examples", "state", "form", "endorse")))
