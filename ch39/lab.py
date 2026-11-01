"""Chapter 39 lab: evaluate retrieval, assembly and grounding separately; build a gold
set with two adjusters and an adjudicator; validate judges; gate releases; watch
production. Run from 04-code: python3 ch39/lab.py
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import lakeshore39 as L  # noqa: E402
from contextlab.evaluator import (Evaluator, GoldSet, ReleaseGate, agreement, attribute,  # noqa: E402
                                  cohen_kappa, corrected_ci, judged_at_k, mrr, ndcg_at_k,
                                  paired_bootstrap, recall_at_k, hit_at_k, wilson)

OUT = os.path.join(HERE, "out")
R = {}
PROD, CAND = "rerank", "v9"
SYSTEMS = ["title-only", "bm25", "dense", "hybrid", "rerank", "edition-aware", "sched-boost", "v9"]
POOL_FROM = ["bm25", "dense", "hybrid", "rerank"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def kendall(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += (1 if (a[i] - a[j]) * (b[i] - b[j]) > 0 else -1)
    return s / (n * (n - 1) / 2)


def main():
    os.makedirs(OUT, exist_ok=True)
    claims = L.make_claims()
    by_id = {c.id: c for c in claims}
    gov = {c.id: c.governing for c in claims}
    truth = {c.id: c.grades for c in claims}

    # 1 -------------------------------------------------------------------------------
    print("1. Gold set v1: pooled spans, two adjusters, one adjudicator")
    runs = {s: {c.id: L.ranked(s, c) for c in claims} for s in SYSTEMS}
    pool = {c.id: sorted({x for s in POOL_FROM for x in runs[s][c.id][:10]}) for c in claims}
    pairs = [(c, sid) for c in claims for sid in pool[c.id]]
    a = [L.adjuster("A", c, sid, truth[c.id][sid]) for c, sid in pairs]
    b = [L.adjuster("B", c, sid, truth[c.id][sid]) for c, sid in pairs]
    t = [truth[c.id][sid] for c, sid in pairs]
    disagree = sum(1 for x, y in zip(a, b) if x != y)
    both_wrong = sum(1 for x, y, z in zip(a, b, t) if x == y != z)
    a_wrong = sum(1 for x, z in zip(a, t) if x != z)
    hours = len(pairs) * 2 * L.SECONDS_PER_LABEL / 3600
    adj_hours = disagree * 60 / 3600
    kw = cohen_kappa(a, b, weighted=True)
    print(f"   claims {len(claims)}  pooled spans {len(pairs)}  "
          f"({len(pairs) / len(claims):.1f} per claim, top-10 of {len(POOL_FROM)} retrievers)")
    print(f"   adjuster hours {hours:.0f}  adjudicator hours {adj_hours:.0f}")
    print(f"   raw agreement {pct(1 - disagree / len(pairs))}  weighted kappa {kw:.3f}  "
          f"sent to adjudicator {disagree}")
    print(f"   one adjuster alone: {a_wrong} wrong grades; two agreeing on the same wrong grade: {both_wrong}")
    risky = [i for i, (c, sid) in enumerate(pairs) if c.kind(sid) in ("lookalike", "schedule")
             or c.case in ("edition", "conflict")]
    rs = set(risky)
    left = sum(1 for i, (x, y, z) in enumerate(zip(a, b, t)) if (x == y != z) if i in rs) + \
        sum(1 for i, (x, z) in enumerate(zip(a, t)) if i not in rs and x != z)
    print(f"   targeted second label (look-alikes, schedules, edition and conflict claims): "
          f"{len(risky)} spans ({pct(len(risky) / len(pairs))}), wrong grades left {left}")
    kap = {}
    for case, _ in L.CASES:
        idx = [i for i, (c, _) in enumerate(pairs) if c.case == case]
        kap[case] = cohen_kappa([a[i] for i in idx], [b[i] for i in idx], weighted=True)
    look = [i for i, (c, sid) in enumerate(pairs) if c.kind(sid) == "lookalike"]
    look_a3 = sum(1 for i in look if a[i] == 3)
    print("   weighted kappa by case: " + "  ".join(f"{k} {v:.2f}" for k, v in kap.items()))
    print(f"   superseded-edition look-alikes in pool {len(look)}; adjuster A graded {look_a3} as governing")
    single = {c.id: {} for c in claims}
    for (c, sid), g in zip(pairs, a):
        single[c.id][sid] = g
    adjud = {c.id: {sid: truth[c.id][sid] for sid in pool[c.id]} for c in claims}
    gold = GoldSet("gold-v1", tuple(c.id for c in claims),
                   tuple((c.id, s, g) for c in claims for s, g in sorted(adjud[c.id].items())))
    print(f"   gold set key {gold.key}  labels {len(gold.labels)}")
    R["gold"] = dict(pairs=len(pairs), hours=hours, adj_hours=adj_hours, kappa=kw,
                     disagree=disagree, a_wrong=a_wrong, targeted=len(risky), targeted_left=left, both_wrong=both_wrong, by_case=kap,
                     look=len(look), look_a3=look_a3, key=gold.key)

    # 2 -------------------------------------------------------------------------------
    print("2. Retrieval metrics on the 914 claims with a governing span (full labels)")
    ev = Evaluator(gold)
    table = {}
    for s in SYSTEMS:
        table[s] = ev.retrieval(runs[s], truth, gov)
        m = table[s]
        print(f"   {s:12s} hit@10 {m['hit@10']:.3f}  recall@10 {m['recall@10']:.3f}  "
              f"MRR {m['mrr']:.3f}  nDCG@10 {m['ndcg@10']:.3f}")
    ex = next(c for c in claims if c.case == "limit" and runs[PROD][c.id].index(c.governing) == 1)
    lst = runs[PROD][ex.id][:5]
    print(f"   worked example {ex.id} ({ex.case}, {ex.state}, {ex.form}): top 5 grades "
          f"{[truth[ex.id][x] for x in lst]}  gold(>=2) {sum(1 for g in truth[ex.id].values() if g >= 2)}")
    g2 = {x for x, g in truth[ex.id].items() if g >= 2}
    print(f"   recall@5 {recall_at_k(lst, g2, 5):.3f}  MRR(governing) {mrr(lst, {ex.governing}):.3f}  "
          f"nDCG@5 {ndcg_at_k(lst, truth[ex.id], 5):.3f}")
    R["retrieval"] = table
    R["example"] = dict(id=ex.id, case=ex.case, state=ex.state, form=ex.form,
                        top=[(x, ex.kind(x), truth[ex.id][x]) for x in lst],
                        recall5=recall_at_k(lst, g2, 5), mrr=mrr(lst, {ex.governing}),
                        ndcg5=ndcg_at_k(lst, truth[ex.id], 5), n_gold=len(g2))
    curves = {s: [sum(recall_at_k(runs[s][c], {x for x, g in truth[c].items() if g >= 2}, k)
                      for c in runs[s] if gov[c]) / 914 for k in range(1, 21)] for s in
              ["rerank", "sched-boost", "v9", "bm25"]}
    R["curves"] = curves

    # 3 -------------------------------------------------------------------------------
    print("3. Pooling bias: unjudged spans scored as irrelevant")
    pool_grades = {c: {x: g for x, g in adjud[c].items()} for c in adjud}
    pb = {}
    for s in ["rerank", "sched-boost", "v9"]:
        mp = ev.retrieval(runs[s], pool_grades, gov)
        jd = sum(judged_at_k(runs[s][c], set(pool[c]), 10) for c in runs[s]) / len(claims)
        pb[s] = dict(pool=mp, full=table[s], judged=jd)
        print(f"   {s:12s} judged@10 {jd:.3f}  nDCG@10 pool {mp['ndcg@10']:.3f} -> full {table[s]['ndcg@10']:.3f}  "
              f"recall@10 pool {mp['recall@10']:.3f} -> full {table[s]['recall@10']:.3f}")
    holes = sum(1 for c in claims for x in runs[CAND][c.id][:10] if x not in pool[c.id])
    print(f"   holes in v9 top-10 to label: {holes}  ({holes * 2 * L.SECONDS_PER_LABEL / 3600:.1f} adjuster hours)")
    R["pooling"] = pb
    R["holes"] = holes

    # 4 -------------------------------------------------------------------------------
    print("4. End-to-end on the full mix: where the governing span was lost")
    runs_p, recs_p = L.pipeline(claims, PROD)
    runs_c, recs_c = L.pipeline(claims, CAND)
    e_p, e_c = ev.run(recs_p), Evaluator(gold).run(recs_c)
    for name, e in (("v8 (rerank)", e_p), ("v9 candidate", e_c)):
        print(f"   {name:13s} faithfulness {pct(e['faithfulness'])}  wrong-clause {e['wrong_clause']}/1,000  "
              f"packet sufficiency {pct(e['packet_sufficiency'])}  by stage {e['by_stage']}")
    stage_case = {}
    for cid, r in recs_p.items():
        st = attribute(r)
        stage_case.setdefault(r["case"], {}).setdefault(st, 0)
        stage_case[r["case"]][st] += 1
    unsup_lim = sum(1 for cid, r in recs_p.items() if attribute(r) == "grounding: unsupported"
                    and r["case"] in ("limit", "deductible"))
    sch_missing = sum(1 for cid, r in recs_p.items() if attribute(r) == "grounding: unsupported"
                      and r["case"] in ("limit", "deductible")
                      and f"{cid}-sch" not in L.assemble(by_id[cid], runs_p[cid]))
    print(f"   unsupported limit/deductible determinations {unsup_lim}, of which schedule not in packet {sch_missing}")
    pos = {}
    for name, rn in (("v8", runs_p), ("v9", runs_c)):
        ps = [L.assemble(by_id[c], rn[c]).index(gov[c]) + 1 for c in rn
              if gov[c] and gov[c] in L.assemble(by_id[c], rn[c])]
        pos[name] = dict(first=sum(1 for p in ps if p == 1) / len(ps), mean=sum(ps) / len(ps),
                         dist=[sum(1 for p in ps if p == i) for i in range(1, L.PACKET + 1)])
        print(f"   {name}: governing span first in packet {pct(pos[name]['first'])}  mean position "
              f"{pos[name]['mean']:.2f}  by position {pos[name]['dist']}")
    cited_g = [r for r in recs_p.values() if r["governing"] and r["cited"] == r["governing"]]
    s_all = sum(len(r["sents"]) for r in cited_g)
    s_ok = sum(sum(r["sents"]) for r in cited_g)
    print(f"   sentence-level support on determinations citing the governing span: {pct(s_ok / s_all)} "
          f"of {s_all} sentences; determination-level (all supported) {pct(sum(1 for r in cited_g if r['supported']) / len(cited_g))}")
    R["e2e"] = dict(prod=e_p, cand=e_c, by_case=stage_case, unsup_lim=unsup_lim, sch_missing=sch_missing,
                    position=pos, sent_level=s_ok / s_all, n_sent=s_all)

    # 5 -------------------------------------------------------------------------------
    print("5. How big must the gold set be?")
    fp = e_p["faithfulness"]
    widths = {}
    for n in (100, 200, 500, 1000, 2000, 5000):
        lo, hi = wilson(round(fp * n), n)
        widths[n] = (hi - lo) / 2
    wlo, whi = wilson(4, 1000)
    w3lo, w3hi = wilson(3, 1000)
    print("   faithfulness 95% half-width: " + "  ".join(f"n={n} +/-{100 * w:.1f}" for n, w in widths.items()))
    print(f"   wrong-clause 4/1,000 interval {pct(wlo, 2)} to {pct(whi, 2)};  3/1,000 {pct(w3lo, 2)} to {pct(w3hi, 2)}")
    ids = sorted(recs_p)
    fa = [1.0 if attribute(recs_p[i]) in ("ok", "correct referral") else 0.0 for i in ids]
    fb = [1.0 if attribute(recs_c[i]) in ("ok", "correct referral") else 0.0 for i in ids]
    sub = {}
    for n in (100, 200, 500, 1000):
        rng = random.Random(n)
        pick = rng.sample(range(len(ids)), n)
        sub[n] = paired_bootstrap([fa[i] for i in pick], [fb[i] for i in pick])
        print(f"   v9 minus v8 faithfulness on {n:4d} claims: {100 * sub[n][0]:+.1f} points "
              f"[{100 * sub[n][1]:+.1f}, {100 * sub[n][2]:+.1f}]")
    R["size"] = dict(widths=widths, wc=(wlo, whi), wc3=(w3lo, w3hi), paired=sub)

    # 6 -------------------------------------------------------------------------------
    print("6. Validating the faithfulness judge on a calibration set")
    cal_claims = L.make_claims("K", 3000, seed=7)
    _, cal_recs = L.pipeline(cal_claims, PROD)
    cited = [(cid, r) for cid, r in cal_recs.items() if r["cited"] == r["governing"] and r["governing"]]
    neg = [x for x in cited if not x[1]["supported"]][:100]
    pos = [x for x in cited if x[1]["supported"]][:200]
    cal = sorted(neg + pos)
    rng = random.Random(39)
    rng.shuffle(cal)
    dev, test = cal[:100], cal[100:]

    def verdict(j, cid, r, ask=0):
        return all(L.judge_sentence(j, f"{cid}-s{i}", ok, L.difficulty(f"{cid}-s{i}"), ask)
                   for i, ok in enumerate(r["sents"]))

    human = [r["supported"] for _, r in test]
    jr = {}
    for j in ("strict", "lenient"):
        jv = [verdict(j, cid, r) for cid, r in test]
        jr[j] = Evaluator.validate_judge(jv, human)
        a_ = jr[j]
        print(f"   {j:8s} test n={a_['n']} (negatives {human.count(False)})  agreement {pct(a_['agreement'])}  "
              f"kappa {a_['kappa']:.2f}  TPR {a_['tpr']:.2f}  TNR {a_['tnr']:.2f}  valid {a_['valid']}")
    # base-rate trap: same judge on a random (unenriched) calibration sample
    rnd = random.Random(5).sample(cited, 200)
    hr = [r["supported"] for _, r in rnd]
    lr = Evaluator.validate_judge([verdict("lenient", cid, r) for cid, r in rnd], hr)
    print(f"   lenient on a random 200 (negatives {hr.count(False)}): agreement {pct(lr['agreement'])}  "
          f"kappa {lr['kappa']:.2f}  TNR {lr['tnr']:.2f}")
    R["judge"] = dict(strict=jr["strict"], lenient=jr["lenient"], random=lr,
                      negatives=human.count(False), random_neg=hr.count(False))

    # observed and corrected faithfulness on the full mix
    obs = {}
    for j in ("strict", "lenient"):
        target = [verdict(j, cid, r) for cid, r in recs_p.items() if r["cited"] == r["governing"] and r["governing"]]
        n_cite = len(target)
        others = sum(1 for r in recs_p.values() if attribute(r) == "correct referral")
        observed = (sum(target) + others) / len(recs_p)
        jv_test = [verdict(j, cid, r) for cid, r in test]
        point, lo, hi = corrected_ci(jv_test, human, target)
        corr = (point * n_cite + others) / len(recs_p)
        clo, chi = (lo * n_cite + others) / len(recs_p), (hi * n_cite + others) / len(recs_p)
        obs[j] = dict(observed=observed, corrected=corr, lo=clo, hi=chi)
        print(f"   {j:8s} judge reports full-mix faithfulness {pct(observed)}; corrected {pct(corr)} "
              f"[{pct(clo)}, {pct(chi)}]; adjudicated truth {pct(e_p['faithfulness'])}")
    R["observed"] = obs

    # 7 -------------------------------------------------------------------------------
    print("7. Instance-level reliability: five paraphrased re-asks per calibration sentence")
    bins = {"0": [0, 0], "1": [0, 0], "2": [0, 0]}
    for cid, r in test:
        for i, ok in enumerate(r["sents"]):
            sid = f"{cid}-s{i}"
            vs = [L.judge_sentence("strict", sid, ok, L.difficulty(sid), ask) for ask in range(5)]
            minority = min(sum(vs), 5 - sum(vs))
            key = str(minority)
            bins[key][0] += 1
            bins[key][1] += int(vs[0] != ok)
    for k, (n, wrong) in bins.items():
        print(f"   re-asks disagreeing with the majority: {k} of 5  sentences {n}  first verdict wrong {wrong} ({pct(wrong / n if n else 0)})")
    R["flips"] = bins

    # 8 -------------------------------------------------------------------------------
    print("8. A model judge for relevance: system rankings against item agreement")
    rel = {}
    for j in ("strict", "lenient"):
        qj = {c.id: {sid: L.judge_relevance(j, c, sid, truth[c.id][sid]) for sid, _, _ in c.spans}
              for c in claims}
        items_h = [truth[c.id][sid] for c, sid in pairs]
        items_j = [qj[c.id][sid] for c, sid in pairs]
        sc = {s: Evaluator(gold).retrieval(runs[s], qj, gov)["ndcg@10"] for s in SYSTEMS}
        tau = kendall([table[s]["ndcg@10"] for s in SYSTEMS], [sc[s] for s in SYSTEMS])
        rel[j] = dict(kappa=cohen_kappa(items_h, items_j, weighted=True), tau=tau, ndcg=sc,
                      look3=sum(1 for c, sid in pairs if c.kind(sid) == "lookalike" and qj[c.id][sid] == 3))
        print(f"   {j:8s} item weighted kappa {rel[j]['kappa']:.2f}  Kendall tau over {len(SYSTEMS)} systems {tau:.2f}  "
              f"look-alikes graded governing {rel[j]['look3']}  "
              f"v9 minus rerank nDCG@10 {sc['v9'] - sc['rerank']:+.3f} (humans {table['v9']['ndcg@10'] - table['rerank']['ndcg@10']:+.3f})  "
              f"edition-aware minus rerank {sc['edition-aware'] - sc['rerank']:+.3f} (humans {table['edition-aware']['ndcg@10'] - table['rerank']['ndcg@10']:+.3f})")
    R["relevance_judge"] = rel

    # 9 -------------------------------------------------------------------------------
    print("9. Regression gate on three releases")
    gate = ReleaseGate()
    base_rec = recs_p
    r_ = {c: recall_at_k(runs[PROD][c], {x for x, g in truth[c].items() if g >= 2}, 10) for c in ids if gov[c]}
    releases = {}
    _, recs_dd = L.pipeline(claims, PROD, dedupe_by_form=True)
    for name, recs, rr, jname in (("R1 schedule-aware reranker (v9)", recs_c, runs[CAND], "strict"),
                                  ("R2 dedupe packet by form", recs_dd, runs[PROD], "strict"),
                                  ("R3 cheaper judge model", recs_p, runs[PROD], "lenient")):
        cand = Evaluator(gold).run(recs)
        f_b = [1.0 if attribute(base_rec[i]) in ("ok", "correct referral") else 0.0 for i in ids]
        f_c = [1.0 if attribute(recs[i]) in ("ok", "correct referral") else 0.0 for i in ids]
        rc = {c: recall_at_k(rr[c], {x for x, g in truth[c].items() if g >= 2}, 10) for c in r_}
        rci = paired_bootstrap([r_[c] for c in r_], [rc[c] for c in r_])
        reported = obs[jname]["observed"] if name.startswith("R3") else cand["faithfulness"]
        fci = paired_bootstrap(f_b, f_c)
        ok, why = gate.check(e_p, cand, jr[jname], fci, rci)
        releases[name] = dict(adopt=ok, why=why, faith=cand["faithfulness"], reported=reported,
                              wrong=cand["wrong_clause"], suff=cand["packet_sufficiency"],
                              recall=sum(rc.values()) / len(rc), fci=fci, rci=rci)
        print(f"   {name:32s} adopt={ok}  reported faithfulness {pct(reported)}  "
              f"wrong-clause {cand['wrong_clause']}  sufficiency {pct(cand['packet_sufficiency'])}  {why}")
    good = lambda r: attribute(r) in ("ok", "correct referral")  # noqa: E731
    moved = sum(1 for i in ids if good(recs_p[i]) and not good(recs_dd[i]) and recs_dd[i]["cited"] is None)
    print(f"   R2: faithful determinations turned into referrals {moved}")
    R["releases"] = releases
    R["r2_referrals"] = moved

    # 10 ------------------------------------------------------------------------------
    print("10. Twelve weeks of production: weekly sample, overrides, frozen gold set")
    weeks = []
    for w in range(1, 13):
        wk = L.make_claims(f"W{w:02d}-", 2240, seed=1000 + w)
        rngw = random.Random(f"week{w}")
        fails, overrides, sample = 0, 0, []
        lag = w in (7, 8)
        for c in wk:
            run = L.ranked(PROD, c, seed=w)
            if lag and c.state == "IL" and c.governing and rngw.random() < 0.5:
                run = [x for x in run if x != c.governing]
            pk = L.assemble(c, run)
            cited, sents = L.determine(c, pk, seed=w)
            r = L.record(c, run, pk, cited, sents)
            good = attribute(r) in ("ok", "correct referral")
            if rngw.random() < (0.22 if not good else 0.002):
                overrides += 1
            sample.append(good)
        smp = random.Random(f"sample{w}").sample(sample, 150)
        k = sum(smp)
        lo, hi = wilson(k, 150)
        weeks.append(dict(week=w, sample=k / 150, lo=lo, hi=hi, override=overrides / len(wk),
                          truth=sum(sample) / len(sample), gold=e_p["faithfulness"]))
        print(f"   week {w:2d}  sample faithfulness {pct(k / 150)} [{pct(lo)}, {pct(hi)}]  "
              f"override rate {pct(overrides / len(wk), 2)}  all-week truth {pct(sum(sample) / len(sample))}  "
              f"frozen gold {pct(e_p['faithfulness'])}")
    R["weeks"] = weeks
    added = 40
    print(f"   overrides in weeks 7-8 triaged; {added} claims adjudicated into gold-v1.1 "
          f"(new Illinois edition), gold set {len(claims)} -> {len(claims) + added}")
    R["gold_v11"] = len(claims) + added

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)


if __name__ == "__main__":
    main()
