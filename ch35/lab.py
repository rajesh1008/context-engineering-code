"""Chapter 35 lab: the cost-quality frontier for Lakeshore's determinations.

Note: deterministic stand-in; see sim35.py for the tier behaviour and its
stated parameters. Run from 04-code/: python3 ch35/lab.py > ch35/output.txt
"""
import csv
import json
import pathlib
import sys
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import lakeshore35 as L                                                  # noqa: E402
import sim35 as S                                                        # noqa: E402
from contextlab.costs import Router, Route, pareto, BUDGETS              # noqa: E402

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
DAY, TRAIN = L.day(), L.day(seed=36)
REV = [d for d in DAY if d.reviewed]
FEATS = list(DAY[0].feats())
ROUTES = (Route("small", "low", True), Route("mid", "medium", False), Route("large", "high", False))
CUTS = (0.30, 0.65)


def section(t):
    print(f"\n== {t}")


def fit(keys):
    X = np.array([[1.0] + [d.feats()[k] for k in keys] for d in TRAIN])
    w, *_ = np.linalg.lstsq(X, np.array([d.h for d in TRAIN]), rcond=None)
    return {k: round(float(v), 3) for k, v in zip(keys, w[1:])}, round(float(w[0]), 3)


def line(name, s, extra=""):
    print(f"{name:34s} US${s['cost']:.4f}  {s['tokens']:>6,.0f} tok  wrong {s['wrong']:>3}  "
          f"flagged {s['flagged']:>3}  clean {s['clean']:>3}{extra}")


def go(cfg):
    outs, rc, tc = S.run(cfg, DAY)
    return outs, S.summary(outs), rc, tc


section("1. The day and the reviewed full mix")
print(f"determinations today: {len(DAY):,}; reviewed full mix: {len(REV):,}; record-layer errors in it: "
      f"{sum(d.record_error for d in REV)}")
print("true difficulty (h<0.35 / 0.35-0.70 / >=0.70):", dict(sorted(Counter(d.band for d in DAY).items())))
print("outcome fixed by rule (under deductible, one cause, no bulletin, one state):", sum(d.mechanical for d in DAY))

section("2. Where the tokens go and where the dollars go (Chapter 34 close: every call on the large tier)")
base_o, base, _, _ = go(S.Config("Chapter 34 close"))
led = base["ledger"]
tok, usd = led.by("stage", "tokens"), led.by("stage")
rows = []
for st in S.PROFILE:
    rows.append([st, round(tok[st]), round(tok[st] / base["tokens"] * 100, 1), round(usd[st], 5),
                 round(usd[st] / base["cost"] * 100, 1)])
    print(f"{st:8s} {tok[st]:>7,.0f} tokens ({rows[-1][2]:>4}%)   US${usd[st]:.5f} ({rows[-1][4]:>4}%)")
comp = led.by_component()
think_tok = sum(c.thinking for c in led.calls) / led.n
print("by component, US$ per determination:", {k: round(v, 5) for k, v in comp.items()})
print(f"thinking tokens: {think_tok:,.0f} per determination = {think_tok / base['tokens'] * 100:.1f}% of tokens, "
      f"{comp['thinking'] / base['cost'] * 100:.1f}% of dollars")
line("Chapter 34 close", base, f"   per day US${base['day']:,.2f}")
with open(OUT / "where.csv", "w", newline="") as f:
    csv.writer(f).writerows([["stage", "tokens", "token_pct", "usd", "usd_pct"]] + rows)

section("3. Difficulty: what the router can see, and when it can see it")
W_pre, b_pre = fit([k for k in FEATS if k != "ambiguity"])
W, b = fit(FEATS)
json.dump({"weights": W, "bias": b, "cuts": CUTS}, open(OUT / "router.json", "w"), indent=1)
for name, (w, bb) in {"before extraction (no notes signal)": (W_pre, b_pre), "after extraction": (W, b)}.items():
    r = Router(w, CUTS, bb, strict=False)
    err = np.std([d.h - r.score(d.feats()) for d in DAY])
    conf = Counter((d.band, r.band(d.feats())) for d in DAY)
    print(f"{name}: residual sd {err:.3f}; true-hard sent to band 0: {conf[(2, 0)]}; "
          f"true-medium sent to band 0: {conf[(1, 0)]}")
print("fitted weights (after extraction):", W, "bias", b)
ROUTER = Router(W, CUTS, b, ROUTES)
band_n = Counter(ROUTER.band(d.feats()) for d in DAY)
print("router bands today:", dict(sorted(band_n.items())), "routes:", [f"{r.tier}-{r.budget}" for r in ROUTES])

section("4. Reasoning budgets: thinking tokens against quality, true-hard reviewed determinations")
hard = [d for d in REV if d.band == 2]
sweep = []
for tier in ("small", "mid", "large"):
    for bud in BUDGETS:
        mis = sum(d.h > S.CAP[(tier, bud)] for d in hard)
        fl = sum(S.flagged(tier, bud, d.h, d.u_flag) for d in hard)
        th = np.mean([S.think(bud, d.h) for d in hard])
        sweep.append([tier, bud, len(hard), mis, fl, round(float(th))])
        print(f"{tier:5s} {bud:6s} misread {mis:>2}/{len(hard)}  flagged {fl:>2}  thinking {th:>6,.0f}")
with open(OUT / "budgets.csv", "w", newline="") as f:
    csv.writer(f).writerows([["tier", "budget", "n", "misread", "flagged", "thinking"]] + sweep)
easy = [d for d in REV if d.band == 0]
print(f"true-easy reviewed ({len(easy)}): large-medium thinks {np.mean([S.think('medium', d.h) for d in easy]):,.0f} "
      f"tokens; misreads at small-low: {sum(d.h > S.CAP[('small', 'low')] for d in easy)}")

section("5. Routing by difficulty (draft step only; everything else as Chapter 34)")
routed = S.Config("routed", router=ROUTER, escalate=Route("mid", "medium", False))
o5, s5, _, _ = go(routed)
line("Chapter 34 close", base)
line("routed by difficulty", s5, f"   exits {sum(o.exited for o in o5)}, escalated {sum(o.escalated for o in o5)}")
print("routes taken:", dict(Counter(o.route for o in o5)))
o5p, s5p, _, _ = go(S.Config("pre", router=Router(W_pre, CUTS, b_pre, ROUTES, strict=False), escalate=routed.escalate))
line("routed before extraction", s5p, f"   misread causes across all 3,200: {sum(o.wrong and o.d.band > 0 for o in o5p)}")

section("6. Smaller models for retrieval-side work")
ret = []
for rw, rr, ex in [("large", "large", "large"), ("small", "large", "large"), ("small", "small", "large"),
                   ("small", "mid", "large"), ("small", "mid", "small"), ("small", "mid", "mid")]:
    o, s, _, _ = go(S.Config("r", router=ROUTER, escalate=routed.escalate, retrieval=(rw, rr), extract=ex))
    miss = sum(d.u_rerank < S.RERANK_MISS[rr] for d in REV)
    retry = sum(1 for d in DAY if ex != "large" and d.u_extract < S.EXTRACT_ERR[ex])
    ret.append([rw, rr, ex, round(s["cost"], 5), s["wrong"], s["flagged"], miss, retry])
    line(f"rewrite {rw}, rerank {rr}, extract {ex}", s, f"   support misses {miss}, extract retries {retry}")
with open(OUT / "retrieval_side.csv", "w", newline="") as f:
    csv.writer(f).writerows([["rewrite", "rerank", "extract", "usd", "wrong", "flagged", "support_miss", "retries"]] + ret)

section("7. Caching at three layers")
c7 = dict(router=ROUTER, escalate=routed.escalate, retrieval=("small", "mid"), extract="small")
_, s7a, _, _ = go(S.Config("prefix only", **c7))
_, s7b, rc, _ = go(S.Config("+ retrieval", retrieval_cache="versioned", **c7))
_, s7c, rc, tc = go(S.Config("+ tool", retrieval_cache="versioned", tool_cache=True, **c7))
o7u, s7u, rcu, _ = go(S.Config("unversioned", retrieval_cache="unversioned", tool_cache=True, **c7))
print(f"prefix layer (Chapter 34 layout): {S.PREFIX_HIT:.0%} of every stable prefix read from cache")
line("prefix layer only", s7a)
line("+ retrieval-result layer", s7b, f"   hit rate {rc['hit_rate']:.1%} ({rc['hits']:,} of {rc['hits'] + rc['misses']:,})")
line("+ tool-result layer", s7c, f"   hit rate {tc['hit_rate']:.1%} ({tc['hits']:,} of {tc['hits'] + tc['misses']:,})")
stale_all = sum(o.stale for o in o7u)
stale_rev = sum(o.stale for o in o7u if o.d.reviewed)
line("retrieval key without bulletin version", s7u,
     f"   hit rate {rcu['hit_rate']:.1%}; stale serves after {L.BULLETIN_CHANGE_AT}: {stale_all} today, {stale_rev} reviewed")

section("8. Early exit")
c8 = dict(c7, retrieval_cache="versioned", tool_cache=True)
o8, s8, _, _ = go(S.Config("exit", mechanical_exit=True, **c8))
line("+ exit before the model (rule)", s8, f"   rule exits {sum(o.route == 'rule' for o in o8)}")
o8b, s8b, _, _ = go(S.Config("exit+redraft", mechanical_exit=True, hard_redraft=True, **c8))
line("+ one redraft for flagged hard drafts", s8b)
ex = Counter("rule" if o.route == "rule" else ("small exit" if o.exited else
             ("escalated" if o.escalated else o.route)) for o in o8b)
print("where each determination stopped:", dict(ex))
hard_flag = {n: sum(o.flag for o in oo if o.d.reviewed and o.d.band == 2) for n, oo in
             (("Chapter 34", base_o), ("after", o8b))}
print("flagged true-hard reviewed determinations:", hard_flag, f"of {len(hard)}")

section("9. The chosen configuration, with a 5% shadow audit")
CHOSEN = S.Config("chosen", mechanical_exit=True, hard_redraft=True, shadow=0.05, **c8)
o9, s9, _, _ = go(CHOSEN)
sh = [o for o in o9 if o.exited and (o.d.i * 7919) % 1000 < 50]
line("chosen", s9, f"   per day US${s9['day']:,.2f}")
print(f"shadow audit: {len(sh)} small exits re-run on mid-medium; disagreements {sum(o.shadow_disagree for o in sh)}")
fall = 1 - s9["cost"] / base["cost"]
print(f"cost per determination US${base['cost']:.4f} -> US${s9['cost']:.4f}: falls {fall:.1%}; "
      f"tokens {base['tokens']:,.0f} -> {s9['tokens']:,.0f} ({s9['tokens'] / base['tokens'] - 1:+.1%})")
print(f"per day US${base['day']:,.2f} -> US${s9['day']:,.2f}; wrong {base['wrong']} -> {s9['wrong']}; "
      f"clean {base['clean']} -> {s9['clean']}; flagged {base['flagged']} -> {s9['flagged']}")
for nm, oo in (("Chapter 34 close", base_o), ("chosen", o9)):
    cs = np.array([sum(c.cost() for c in o.calls) for o in oo])
    print(f"{nm:17s} per-determination cost: median US${np.median(cs):.4f}, 90th pct US${np.percentile(cs, 90):.4f}, "
          f"max US${cs.max():.4f}; top 10% of determinations = {np.sort(cs)[-len(cs) // 10:].sum() / cs.sum():.1%} of spend")

section("10. Failure injection: the router goes blind to the hard claims")
BLIND = S.Config("blind", mechanical_exit=True, hard_redraft=True, shadow=0.05,
                 router=Router(W, CUTS, b, ROUTES, strict=False),
                 rename={"ambiguity": "ambiguity_score"}, **{k: v for k, v in c8.items() if k != "router"})
o10, s10, _, _ = go(BLIND)
line("feature renamed upstream, router lax", s10)
wb = Counter(o.d.band for o in o10 if o.d.reviewed and o.wrong and not o.d.record_error)
print("extra wrong by true difficulty:", dict(sorted(wb.items())))
sh10 = [o for o in o10 if o.exited and (o.d.i * 7919) % 1000 < 50]
print(f"check passes on {sum(o.exited and o.wrong for o in o10 if o.d.reviewed)} of the wrong small drafts; "
      f"shadow disagreements {sum(o.shadow_disagree for o in sh10)} of {len(sh10)}")
try:
    S.run(S.Config("strict", router=ROUTER, rename={"ambiguity": "ambiguity_score"}), DAY[:5])
except KeyError as e:
    print("strict router refuses:", e)

section("11. The frontier (Signature 8)")
pts = []
for tier in ("small", "mid", "large"):
    for bud in BUDGETS:
        _, s, _, _ = go(S.Config(f"all {tier}-{bud}", fixed=Route(tier, bud, False)))
        pts.append(("fixed", f"{tier}-{bud}", s))
for cuts in ((0.20, 0.55), (0.25, 0.60), (0.30, 0.65), (0.35, 0.70), (0.40, 0.75)):
    _, s, _, _ = go(S.Config("r", router=Router(W, cuts, b, ROUTES), escalate=routed.escalate))
    pts.append(("routed", f"routed {cuts[0]:.2f}/{cuts[1]:.2f}", s))
    _, s, _, _ = go(S.Config("a", router=Router(W, cuts, b, ROUTES), mechanical_exit=True,
                             hard_redraft=True, shadow=0.05, **{k: v for k, v in c8.items() if k != "router"}))
    pts.append(("all levers", f"all levers {cuts[0]:.2f}/{cuts[1]:.2f}", s))
pts.append(("failure", "blind router", s10))
front = pareto([(n, s["cost"], s["clean"]) for _, n, s in pts if s["wrong"] <= base["wrong"]])
with open(OUT / "frontier.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["family", "name", "usd", "tokens", "wrong", "flagged", "clean", "on_frontier"])
    fn = {p[0] for p in front}
    for fam, n, s in pts:
        wr.writerow([fam, n, round(s["cost"], 5), round(s["tokens"]), s["wrong"], s["flagged"], s["clean"], n in fn])
for fam, n, s in pts:
    line(f"{n}", s, "   *" if n in fn else "")
print("frontier (wrong no worse than 4):", [p[0] for p in front])

section("12. Lakeshore, before and after")
steps = [("Chapter 34 close", base), ("route the draft", s5), ("retrieval side to small/mid", s7a),
         ("retrieval-result cache", s7b), ("tool-result cache", s7c), ("exit before the model", s8),
         ("redraft flagged hard", s8b), ("shadow audit 5%", s9)]
with open(OUT / "waterfall.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["step", "usd", "tokens", "wrong", "clean"])
    for n, s in steps:
        wr.writerow([n, round(s["cost"], 5), round(s["tokens"]), s["wrong"], s["clean"]])
        print(f"{n:30s} US${s['cost']:.5f}  wrong {s['wrong']}  clean {s['clean']}")
json.dump({"base": {k: v for k, v in base.items() if k != "ledger"},
           "chosen": {k: v for k, v in s9.items() if k != "ledger"},
           "blind": {k: v for k, v in s10.items() if k != "ledger"}, "fall": fall},
          open(OUT / "results.json", "w"), indent=1)

# data for the figures
with open(OUT / "components.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["config", "uncached", "cached", "output", "thinking"])
    for n, s in (("Chapter 34 close", base), ("chosen", s9)):
        c = s["ledger"].by_component()
        wr.writerow([n] + [round(c[k], 5) for k in ("uncached", "cached", "output", "thinking")])
pre = Router(W_pre, CUTS, b_pre, strict=False)
with open(OUT / "dets.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["i", "h", "band", "score_pre", "score_post", "reviewed",
                                     "route_chosen", "wrong_chosen", "route_blind", "wrong_blind"])
    for o, ob in zip(o9, o10):
        d = o.d
        wr.writerow([d.i, round(d.h, 3), d.band, round(pre.score(d.feats()), 3), round(ROUTER.score(d.feats()), 3),
                     int(d.reviewed), o.route, int(o.wrong), ob.route, int(ob.wrong)])
with open(OUT / "caches.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["layer", "hit_rate", "usd_after", "wrong", "stale_today"])
    wr.writerow(["prefix", S.PREFIX_HIT, round(s7a["cost"], 5), s7a["wrong"], 0])
    wr.writerow(["retrieval", round(rc["hit_rate"], 3), round(s7b["cost"], 5), s7b["wrong"], 0])
    wr.writerow(["tool", round(tc["hit_rate"], 3), round(s7c["cost"], 5), s7c["wrong"], 0])
    wr.writerow(["retrieval unversioned", round(rcu["hit_rate"], 3), round(s7u["cost"], 5), s7u["wrong"], stale_all])
with open(OUT / "stops.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["stop", "n"])
    for k in ("rule", "small exit", "escalated", "mid-medium", "large-high"):
        wr.writerow([k, ex[k]])
