"""Chapter 32 lab: budgeting and selection for Lakeshore's packets.

Run: python3 lab.py   (writes out/results.json; prints the tables in output.txt)
Deterministic stand-in: constructed candidate pools, a rule-based reader, a
prefix-cache simulator.  No model runs and no provider is called.
"""
from __future__ import annotations
import json, os, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from contextlab import budgeter as B
import lakeshore32 as L

T = L.TODAY
S = L.SIZES
OUT = os.path.join(os.path.dirname(__file__), "out")


def fit_calibrator(mix):
    xs, ys = [], []
    for d in mix[:200]:                       # the labelled evaluation set
        f = L.needed(d)
        for c in d.cands:
            xs.append(B.score(c, T)); ys.append(1 if f(c) else 0)
    return B.Calibrator().fit(xs, ys)


def packers(cal):
    key = lambda c: B.score(c, T)
    val = lambda c: cal(B.score(c, T))
    return {
        "top_k": lambda d, b: (B.top_k(d.cands, 10, key), B.cost_flat),
        "fill": lambda d, b: (B.fill_by_score(d.cands, b, key), B.cost_flat),
        "greedy": lambda d, b: (B.greedy_density(d.cands, b, val, 0.10), B.cost_flat),
        "knapsack": lambda d, b: (B.knapsack(d.cands, b, val, 0.10), B.cost_flat),
        "budgeter": lambda d, b: (B.Budgeter(budget=b, floor=0.60).pack(d.cands)[0], B.cost_hier),
    }


def evaluate(mix, fn, budget):
    wrong, faith, toks, dup, orphan, miss_b = 0, [], [], 0, 0, 0
    for d in mix:
        p, cost = fn(d, budget)
        r = L.read(d, p)
        wrong += r["wrong"]; faith.append(r["faithful"]); toks.append(cost(p))
        g = [s.group for s in p if s.group]
        dup += len(g) != len(set(g))
        ids = {s.span_id for s in p}
        orphan += any(e in ids and df not in ids for e, df in d.dependent.items())
        miss_b += any(m.startswith("g:") for m in r["missing"])
    return {"wrong": wrong, "faithful": round(st.mean(faith), 4), "tokens": round(st.mean(toks)),
            "p95_tokens": sorted(toks)[int(0.95 * len(toks))], "duplicates": dup,
            "orphans": orphan, "base_clause_missing": miss_b}


def per_step_tokens(ev_flat, ev_hier, n_spans, per_step):
    """Input and output tokens for Lakeshore's five-call determination (Chapter 26)."""
    fixed = S["tools"] + S["instructions"] + S["claim"] + S["memory"]
    hist = S["tool_step"] * sum(range(5))
    out = 4 * S["out_step"] + S["out_draft"]
    if not per_step:
        return 5 * (fixed + ev_flat) + hist + out
    index = 20 + 8 * n_spans
    return 5 * (fixed + index) + ev_hier + hist + out   # the draft keeps the index so its prefix matches


def cache_run(mix_packets, layout, prices=B.CachePrices()):
    cache = B.PrefixCache(prices)
    offsets = [0, 7, 14, 21, 30]
    for i, (det, ev, n) in enumerate(mix_packets):
        t0 = 8 * 3600 + i * 11.25
        index = 20 + 8 * n
        for k in range(5):
            hist = [(f"{det}:h{j}", S["tool_step"]) for j in range(k)]
            tools = ("tools", S["tools"])
            instr = ("instr", S["instructions"]) if layout != "trim_rulebook" else (f"{det}:instr", 380)
            claim = (f"{det}:claim", S["claim"] + S["memory"])
            if layout == "one_packet":
                blocks = [tools, instr, claim, (f"{det}:ev", ev)] + hist
                bps = [1, len(blocks) - 1]
            else:
                blocks = [tools, instr, claim, (f"{det}:idx", index)] + hist
                bps = [1, len(blocks) - 1]
                if k == 4:
                    if layout == "drop_tools_on_draft":
                        blocks = blocks[1:]; bps = [0, len(blocks) - 1]
                    blocks = blocks + [(f"{det}:ev", ev)]
                    if layout == "breakpoint_after_evidence":
                        bps = bps[:-1] + [len(blocks) - 1]
            cache.call(t0 + offsets[k], blocks, bps)
    n = len(mix_packets)
    return {"hit_rate": round(cache.hit_rate(), 4),
            "usd_per_1k": round(cache.dollars() / n * 1000, 2),
            "list_usd_per_1k": round(sum(cache.tokens.values()) * prices.per_million / 1e6 / n * 1000, 2),
            "input_tokens_per_det": round(sum(cache.tokens.values()) / n)}


def injection(cal, full):
    week = L.this_week()
    configs = {
        "usage_scorer_no_pin": B.Budgeter(budget=1_000, floor=0.60, pin_kinds=(), scorer=B.score_usage),
        "rule_scorer_no_pin": B.Budgeter(budget=1_000, floor=0.60, pin_kinds=()),
        "rule_scorer_pinned": B.Budgeter(budget=1_000, floor=0.60),
    }
    res = {}
    for name, bud in configs.items():
        inc = wrong = 0
        for d in week:
            p, _ = bud.pack(d.cands)
            inc += any(s.kind == "bulletin" for s in p)
            wrong += L.read(d, p)["wrong"]
        fm = sum(L.read(d, bud.pack(d.cands)[0])["wrong"] for d in full)
        res[name] = {"newest_included": inc, "of": len(week), "wrong_this_week": wrong,
                     "withheld_if_checked_after_pack": len(week) - inc, "full_mix_wrong": fm}
    return res


def ablation(full):
    base = dict(budget=1_000, floor=0.60)
    variants = {
        "Budgeter (all parts)": B.Budgeter(**base),
        "no document slots": B.Budgeter(**base, slot_tier=0),
        "no dependency bundles": None,
        "no redundancy (second copy at full value)": B.Budgeter(**base, redundancy=0.0),
        "no whole small sections": B.Budgeter(**base, whole_section=0),
    }
    out = {}
    for name, bud in variants.items():
        if bud is None:
            class NoBundle(B.Budgeter):
                def bundles(self, cands):
                    return [(c, [c]) for c in cands]
            bud = NoBundle(**base)
        r = evaluate(full, lambda d, b: (bud.pack(d.cands)[0], B.cost_hier), 0)
        out[name] = r
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    full = L.full_mix()
    cal = fit_calibrator(full)
    P = packers(cal)
    R = {"calibrator": [round(cal.a, 3), round(cal.b, 3)]}

    print("== 1. Four budgeters on the full mix (1,000 determinations)")
    R["entry"] = evaluate(full, P["fill"], 3_000)
    R["top_k"] = evaluate(full, P["top_k"], 0)
    print(f"  top-10 by score (Ch 2), no budget     {R['top_k']}")
    print(f"  fill by score to 3,000 (entry, Ch 4)   {R['entry']}")
    R["matched"] = {}
    for name in ("fill", "greedy", "knapsack"):
        R["matched"][name] = evaluate(full, P[name], 1_500)
        print(f"  {name:9s} at 1,500                     {R['matched'][name]}")
    R["matched"]["budgeter"] = evaluate(full, P["budgeter"], 1_000)
    print(f"  Budgeter, target 1,000, cap 2,400      {R['matched']['budgeter']}")

    print("\n== 2. Budget sweep: wrong per 1,000 and mean evidence tokens")
    R["sweep"] = {}
    for name in ("fill", "greedy", "knapsack", "budgeter"):
        R["sweep"][name] = []
        for b in (600, 800, 1_000, 1_200, 1_500, 2_000, 2_500, 3_000):
            r = evaluate(full, P[name], b)
            R["sweep"][name].append([b, r["wrong"], r["tokens"], r["faithful"]])
        print(f"  {name:9s}", " ".join(f"{b}:{w}/{t}" for b, w, t, _ in R["sweep"][name]))

    print("\n== 3. Ablation of the Budgeter")
    R["ablation"] = ablation(full)
    for k, v in R["ablation"].items():
        print(f"  {k:42s} {v}")

    print("\n== 4. Tokens per determination, five calls (Ch 26 loop)")
    bud = B.Budgeter(budget=1_000, floor=0.60)
    packs = []
    for d in full:
        p, _ = bud.pack(d.cands)
        packs.append((d.det_id, B.cost_hier(p), len(p)))
    ev_flat = R["entry"]["tokens"]
    ev_h = st.mean(x[1] for x in packs); n_sp = st.mean(x[2] for x in packs)
    ent_n = st.mean(len(B.fill_by_score(d.cands, 3_000, lambda c: B.score(c, T))) for d in full)
    R["tokens_per_det"] = {
        "entry": per_step_tokens(ev_flat, ev_flat, ent_n, False),
        "budgeter_only": per_step_tokens(ev_h, ev_h, n_sp, False),
        "budgeter_and_per_step": round(per_step_tokens(ev_flat, ev_h, n_sp, True)),
        "mean_spans_entry": round(ent_n, 1), "mean_spans_after": round(n_sp, 1),
        "evidence_entry": ev_flat, "evidence_after": round(ev_h)}
    R["tokens_per_det"]["budgeter_only"] = round(R["tokens_per_det"]["budgeter_only"])
    R["tokens_per_det"]["entry"] = round(R["tokens_per_det"]["entry"])
    for k, v in R["tokens_per_det"].items():
        print(f"  {k:24s} {v:,}")
    e, a = R["tokens_per_det"]["entry"], R["tokens_per_det"]["budgeter_and_per_step"]
    print(f"  change {100 * (a - e) / e:.1f}%   list-price input US$ per day at 3,200: "
          f"entry {3200 * (e - 660) * 3 / 1e6:.2f}  after {3200 * (a - 660) * 3 / 1e6:.2f}")

    print("\n== 5. Prefix cache: one day, 3,200 determinations, five calls each")
    day = (packs * 4)[:3_200]                     # one day: the mix, replayed to 3,200 claims
    after_packs = [(f"{p[0]}#{i}", p[1], p[2]) for i, p in enumerate(day)]
    entry_packs = [(p[0], ev_flat, p[2]) for p in after_packs]
    R["cache"] = {
        "one_packet_entry": cache_run(entry_packs, "one_packet"),
        "one_packet_budgeted": cache_run(after_packs, "one_packet"),
        "per_step_layout": cache_run(after_packs, "per_step"),
        "breakpoint_after_evidence": cache_run(after_packs, "breakpoint_after_evidence"),
        "drop_tools_on_draft": cache_run(after_packs, "drop_tools_on_draft"),
        "trim_rulebook": cache_run(after_packs, "trim_rulebook"),
    }
    for k, v in R["cache"].items():
        print(f"  {k:28s} {v}")

    print("\n== 6. Failure injection: the newest bulletin (BUL-WI-WB-202610, 40 claims this week)")
    R["injection"] = injection(cal, full)
    for k, v in R["injection"].items():
        print(f"  {k:22s} {v}")

    print("\n== 7. Packing time per determination (ms, this machine)")
    import time
    d0 = full[:200]
    for name, fn in (("fill", lambda d: B.fill_by_score(d.cands, 1500, lambda c: B.score(c, T))),
                     ("knapsack", lambda d: B.knapsack(d.cands, 1500, lambda c: cal(B.score(c, T)), 0.1)),
                     ("budgeter", lambda d: bud.pack(d.cands))):
        t = time.perf_counter()
        for d in d0:
            fn(d)
        print(f"  {name:9s} {1000 * (time.perf_counter() - t) / len(d0):.2f}")

    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
