"""Fit the stand-in's fixtures to the chapter's illustrative numbers; writes data/calibration.json.

Three fits, in order:
  1. keyword scores per question (the retriever's fixture), chosen from random candidate pools by a local
     search so the ranks, drop reasons and packet statistics come out as printed;
  2. the model's misreadings and embellishments per determination, chosen the same way so every
     configuration's error count comes out as printed;
  3. span lengths (a least-squares solve with integer repair) so the token means come out as printed.
Run once: python3 calibrate.py  (deterministic; takes a minute or two).
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import standin as S
import world

HERE = Path(__file__).resolve().parent
from contextlab.models import Kind

R = random.Random(2026)
TAILS = ["", " The insured reported it the same day.", " Photos are in the file.", " The contractor's estimate is attached.",
         " This is the first claim on the policy.", " The mortgage company has asked for a status update.",
         " There is no prior damage noted.", " The loss was reported by the agent.", " The insured wants an answer this week.",
         " The adjuster has inspected the property.", " A neighbour reported the same problem.",
         " The insured has receipts for the repairs.", " The broker is asking on the insured's behalf."]
KEEP_Q = {"DET-0001", "DET-0008", "DET-0022"}


def unique_questions(rows):
    used, out = set(), {}
    for r in rows:
        if r["determination_id"] in KEEP_Q:
            used.add(r["question"])
    for r in rows:
        if r["determination_id"] in KEEP_Q:
            continue
        for t in TAILS + [a + b for a in TAILS[1:] for b in TAILS[1:] if a != b]:
            q = r["question"] + t
            if q not in used:
                used.add(q)
                if t:
                    out[r["determination_id"]] = q
                break
        else:
            raise RuntimeError(r["determination_id"])
    return out


# ------------------------------------------------------------------ stage 1: retrieval pools
STALE = {"HO-3 (2024-06)": "HO-3 (2022-01)"}


def facts(form, ends, state):
    return f" {form} {' '.join(ends)} {state}"


class DetCtx:
    def __init__(self, corpus, det):
        self.det = det
        self.c = corpus
        pol = corpus.policy()
        self.gold = det.gold_span_id
        line = det.line
        U = []
        for s in corpus.spans:
            ln = world.ENDORSEMENTS.get(s.provenance.doc_id, {}).get("line") or world.FORM_LINE.get(s.provenance.doc_id)
            if s.topic == det.topic or s.kind in (Kind.UW_NOTE, Kind.GUIDANCE, Kind.BULLETIN) or ln == line:
                U.append(s.span_id)
        self.U = U
        self.f_true = facts(det.form, det.endorsements, det.state)
        self.stale = det.form in STALE
        self.det_stale = det.model_copy(update={"form": STALE.get(det.form, det.form)})
        self.f_stale = facts(self.det_stale.form, det.endorsements, det.state)
        self.det_blank = det.model_copy(update={"endorsements": []})
        self.f_blank = facts(det.form, [], det.state)
        sp = corpus.span
        self.b_true = {x: S.bonus(sp(x), self.f_true) for x in U}
        self.b_stale = {x: S.bonus(sp(x), self.f_stale) for x in U}
        self.b_blank = {x: S.bonus(sp(x), self.f_blank) for x in U}
        self.app = {x: pol.applies(sp(x), det) for x in U}
        self.app_stale = {x: pol.applies(sp(x), self.det_stale).ok for x in U}
        self.app_blank = {x: pol.applies(sp(x), self.det_blank).ok for x in U}
        self.sup = {x: sp(x).provenance.superseded_by is not None for x in U}
        self.note = {x: sp(x).kind == Kind.UW_NOTE for x in U}
        self.ontopic = {x: sp(x).topic == det.topic for x in U}
        self.kind = {x: sp(x).kind for x in U}
        self.jur = {x: sp(x).provenance.jurisdiction for x in U}
        self.own = {x for x in U if sp(x).provenance.label in det.endorsements}


def rank_list(base, bon):
    sc = {x: round(b + bon[x], 2) for x, b in base.items()}
    return sorted(sc, key=lambda x: (-sc[x], x))[:30]


def stats(ctx: DetCtx, base: dict) -> dict:
    g = ctx.gold
    top = rank_list(base, ctx.b_true)
    e = top.index(g) + 1 if g in top else 0
    kept = [x for x in top if ctx.app[x].ok]
    fk = kept.index(g) + 1 if g in kept else 0
    drops = Counter()
    for x in top:
        a = ctx.app[x]
        if not a.ok:
            drops[a.reason] += 1
            if a.reason == "wrong state":
                drops["ws:" + ctx.jur[x]] += 1
    p10 = top[:10]
    f10 = kept[:10]
    bare = sorted(base, key=lambda x: (-round(base[x], 2), x))[:10]
    br = bare.index(g) + 1 if g in bare else 0
    if ctx.stale:
        st = rank_list(base, ctx.b_stale)
        s10 = st[:10]
        sr = s10.index(g) + 1 if g in s10 else 0
        sk = [x for x in st if ctx.app_stale[x]][:10]
        sfr = sk.index(g) + 1 if g in sk else 0
        ssup = int(any(ctx.sup[x] and ctx.ontopic[x] for x in s10))
    else:
        ssup = 0
        sr = e if e <= 10 else 0
        sfr = fk if fk <= 10 else 0
    if ctx.det.endorsements:
        bl = [x for x in rank_list(base, ctx.b_blank) if ctx.app_blank[x]][:10]
        bkr = bl.index(g) + 1 if g in bl else 0
    else:
        bkr = fk if fk <= 10 else 0
    return dict(e=e, fk=fk, br=br, sr=sr, sfr=sfr, bkr=bkr, ssup=ssup, drops=dict(drops),
                bsup=sum(ctx.sup[x] for x in p10), binapp=sum(not ctx.app[x].ok for x in p10),
                bnote=int(any(ctx.note[x] for x in p10)), fsup=sum(ctx.sup[x] for x in f10),
                fon=any(ctx.ontopic[x] for x in f10), top=top, p10=p10, f10=f10)


def sample_pool(ctx: DetCtx, r: random.Random, force: dict | None = None, gold_lo=4.0, gold_hi=17.0) -> dict:
    base = {}
    p_on = r.uniform(0.3, 1.0)
    p_note = r.uniform(0.0, 0.2)
    p_guid = r.uniform(0.0, 0.6)
    p_bul = r.uniform(0.05, 0.4)
    p_end = r.uniform(0.02, 0.35)
    p_own = r.uniform(0.3, 1.0)
    p_form = r.uniform(0.2, 0.7)
    lvl = r.uniform(-3, 2)
    for x in ctx.U:
        k = ctx.kind[x]
        if x == ctx.gold:
            continue
        if ctx.ontopic[x]:
            if r.random() < p_on:
                base[x] = r.uniform(5, 15) + lvl
        elif k == Kind.UW_NOTE:
            if r.random() < p_note:
                base[x] = r.uniform(6, 17)
        elif k == Kind.GUIDANCE:
            if r.random() < p_guid:
                base[x] = r.uniform(2, 10)
        elif k == Kind.BULLETIN:
            if r.random() < p_bul:
                base[x] = r.uniform(3, 12) + lvl
        elif x in ctx.own:
            if r.random() < p_own:
                base[x] = r.uniform(4, 14) + lvl
        elif k == Kind.ENDORSEMENT:
            if r.random() < p_end:
                base[x] = r.uniform(3, 13) + lvl
        elif ctx.app[x].ok:
            if r.random() < p_form + 0.15:
                base[x] = r.uniform(1, 12) + lvl
        elif r.random() < p_form * 0.7:
            base[x] = r.uniform(1, 12) + lvl
    boost = r.uniform(-1, 5)
    gboost = r.uniform(0, 2)
    for x in base:
        if not ctx.app[x].ok:
            base[x] += boost
        elif ctx.kind[x] in (Kind.GUIDANCE, Kind.ENDORSEMENT) and gboost > 1:
            base[x] += 2 * gboost
    rest = [x for x in ctx.U if x not in base and x != ctx.gold and not ctx.note[x]]
    r.shuffle(rest)
    while len(base) < 36 and rest:
        base[rest.pop()] = r.uniform(-1, 4) + lvl
    base[ctx.gold] = r.uniform(gold_lo, gold_hi)
    for x, (lo, hi) in (force or {}).items():
        base[x] = r.uniform(lo, hi)
    if ctx.stale and r.random() < 0.8:         # a clear margin that survives the stale pointer
        for x in list(base):
            if x.startswith("HO-3@2022-01#") and ctx.ontopic[x]:
                base[x] -= r.uniform(0, 6)
        base[ctx.gold] += r.uniform(0, 5)
    if r.random() < 0.6:                       # place the gold span at a chosen rank of the bare ranking
        br = r.choice([1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 8])
        others = sorted((v for x, v in base.items() if x != ctx.gold), reverse=True)
        if br == 1:
            base[ctx.gold] = others[0] + r.uniform(0.1, 1.5)
        elif others[br - 2] - others[br - 1] > 0.05:
            base[ctx.gold] = r.uniform(others[br - 1] + 0.02, others[br - 2] - 0.02)
    return {x: round(v, 2) for x, v in base.items()}


# per-determination pins: gold rank (of 30) and kept rank; spans that must sit high in the pool
PIN_E = {"DET-0003": 10, "DET-0004": 16, "DET-0134": 10, "DET-0053": 20, "DET-0083": 20, "DET-0075": 2,
         "DET-0049": 3, "DET-0069": 1, "DET-0071": 5, "DET-0113": 7, "DET-0131": 8, "DET-0145": 2,
         "DET-0022": 0, "DET-0063": 0, "DET-0181": 0, "DET-0163": 0, "DET-0178": 0, "DET-0185": 2, "DET-0188": 2}
PIN_F = {"DET-0049": 3, "DET-0071": 5, "DET-0113": 7, "DET-0131": 8, "DET-0145": 2, "DET-0185": 2, "DET-0188": 2,
         "DET-0022": 0, "DET-0063": 0, "DET-0181": 0, "DET-0163": 0, "DET-0178": 0}
FROZEN = {"DET-0022", "DET-0063", "DET-0181", "DET-0163", "DET-0178"}
HI = (13.0, 18.0)
FORCE = {
    "DET-0003": {"BP 04 50@2025-01#employee_dishonesty": HI}, "DET-0004": {"BP 04 50@2025-01#employee_dishonesty": HI},
    "DET-0134": {"BP 04 50@2025-01#employee_dishonesty": HI},
    "DET-0075": {"WI Bulletin 25-03@2025-03#roof_matching": HI},
    "DET-0069": {"HO-3@2022-01#water_backup": (8, 12)},
    "DET-0049": {"BOAT@2025-03#navigation_limits": (8, 12)},
    "DET-0071": {"HO-3@2024-06#earth_movement": (8, 12)}, "DET-0131": {"HO-3@2024-06#earth_movement": (8, 12)},
    "DET-0113": {"HO-3@2022-01#earth_movement": (8, 12)},
    "DET-0145": {"HO-3@2024-06#food_spoilage": (8, 12)}, "DET-0185": {"HO-3@2024-06#food_spoilage": (8, 12)},
    "DET-0188": {"HO-3@2024-06#roof_settlement": (8, 12)},
}
for d in FROZEN:
    FORCE[d] = {"HO-3@2024-06#freezing": (10, 14), "HO-3@2019-03#freezing": (10, 14)}
for d in ("DET-0022", "DET-0063", "DET-0181"):
    FORCE[d]["IL Bulletin 25-05@2025-05#freezing"] = (12, 16)


def place_gold(ctx: DetCtx, base: dict, rank: int) -> dict:
    """Move the gold span's keyword score so it lands at this rank of the enriched list (0: below 30)."""
    g = ctx.gold
    others = sorted((round(b + ctx.b_true[x], 2) for x, b in base.items() if x != g), reverse=True)
    if rank == 0:
        sc = others[min(30, len(others)) - 1] - 1.0
    elif rank == 1:
        sc = others[0] + 0.3
    else:
        hi, lo = others[rank - 2], others[rank - 1]
        if hi - lo < 0.02:
            return base
        sc = round((hi + lo) / 2, 2)
    out = dict(base)
    out[g] = round(sc - ctx.b_true[g], 2)
    return out


def pin_ok(did, st, ctx):
    if did in PIN_E and st["e"] != PIN_E[did]:
        return False
    if did in PIN_F and st["fk"] != PIN_F[did]:
        return False
    if did in FROZEN and st["fon"]:
        return False
    for x in FORCE.get(did, {}):
        if x not in st["p10"]:
            return False
    if did not in FROZEN and not (1 <= st["fk"] <= 10):
        return False
    if did not in PIN_E and not (1 <= st["e"] <= 10):
        return False
    return True


def hand_pool(ctx: DetCtx, top: list[tuple[str, float]], fill: list[str], start: float, step: float = 0.2) -> dict:
    """A pool given as enriched scores for its top ranks; other spans fill in below at falling scores."""
    base = {x: round(sc - ctx.b_true[x], 2) for x, sc in top}
    sc = start
    for x in fill:
        if x in base:
            continue
        base[x] = round(sc - ctx.b_true[x], 2)
        sc -= step
    return base


def pool_det1(ctx):
    top = [("HO 04 95@2024-06#schedule", 22.61), ("HO 04 95@2024-06#water_backup", 22.55),
           ("HO 04 95@2019-03#water_backup", 22.37), ("HO-3@2024-06#water_backup", 18.80),
           ("HO 04 95@2019-03#schedule", 18.21), ("UW-NOTE-0002", 16.86), ("HO-3@2019-03#water_backup", 16.75),
           ("HO-3@2022-01#water_backup", 16.75), ("HO-3@2024-06#flood", 16.69), ("HO-3@2019-03#flood", 12.34),
           ("HO-3@2024-06#hail_cosmetic", 12.2), ("HO-3@2019-03#hail_cosmetic", 12.1), ("HO-3@2024-06#cond-11", 11.9),
           ("UW-NOTE-0001", 11.7), ("HO-3@2024-06#ice_dam", 11.5), ("MN Bulletin 24-09@2024-09#hail_cosmetic", 11.4),
           ("HO-3@2024-06#wind_hail", 11.3), ("HO-3@2024-06#tree_debris", 11.1), ("HO-3@2019-03#ice_dam", 11.0),
           ("HO-3@2024-06#mold", 10.9)]
    fill = [x for x in ctx.U if not x.startswith(("HO-3@2024-06", "HO-3@2022-01"))]
    return hand_pool(ctx, [t for t in top if t[0] in ctx.U], fill, 10.6, 0.15)


def pool_det8(ctx):
    top = [("LM HO 17@2024-06#roof_settlement", 24.0), ("HO-3@2024-06#roof_settlement", 23.0),
           ("HO-3@2024-06#hail_cosmetic", 19.9), ("HO-3@2022-01#roof_settlement", 19.8),
           ("HO 04 54@2022-01#schedule", 19.0), ("LM HO 17@2024-06#schedule", 18.5), ("HO-3@2024-06#wind_hail", 18.0),
           ("HO 04 54@2022-01#earth_movement", 17.5), ("HO-3@2024-06#ice_dam", 17.0),
           ("HO-3@2019-03#roof_settlement", 16.45), ("HO-3@2024-06#roof_matching", 16.0),
           ("HO-3@2022-01#hail_cosmetic", 15.5)]
    rest = [x for x in ctx.U if not ctx.ontopic[x] and x not in dict(top)]
    ok = [x for x in rest if ctx.app[x].ok]
    sup = [x for x in rest if not ctx.app[x].ok and ctx.app[x].reason == "superseded"]
    fill = ok[:3] + sup + [x for x in rest if x not in ok[:3] and x not in sup]   # ten applicable, then old editions
    return hand_pool(ctx, top, fill, 15.3, 0.2)


HAND = {"DET-0001": pool_det1, "DET-0008": pool_det8}


def iv(lo, hi):
    return (lo, hi)


TARGET = {  # aggregate over the 200 working determinations: (low, high) inclusive
    "e1": iv(119, 119), "e2_5": iv(64, 64), "e6_10": iv(9, 9), "e11_30": iv(3, 3), "e0": iv(5, 5),
    "e_mrr": iv(146.5, 146.69), "e_le3": iv(177, 178),
    "f1": iv(159, 160), "f5": iv(187, 189), "f_le3": iv(181, 182), "f_mrr": iv(172.1, 172.29),
    "b1": iv(42, 43), "b5": iv(172, 173), "b_mrr": iv(94.3, 94.49),
    "s1": iv(111, 113), "s5": iv(183, 184), "sf1": iv(126, 126), "sf5": iv(146, 147),
    "bk5": iv(60, 61), "fend5": iv(100, 100),
    "superseded": iv(1079, 1079), "endorsement not on policy": iv(656, 656), "not on policy": iv(381, 381),
    "not entitled (claims)": iv(246, 246), "wrong state": iv(407, 407), "bulletin not in force": iv(26, 26),
    "ws:WI": iv(95, 95), "ws:MN": iv(75, 75), "ws:IA": iv(63, 63), "ws:IN": iv(62, 62), "ws:MI": iv(57, 57),
    "ws:IL": iv(55, 55),
    "ssup": iv(42, 62),
    "bsup": iv(768, 769), "binapp": iv(915, 917), "bnote": iv(91, 91), "fsup": iv(573, 574),
}
WEIGHT = {"e_mrr": 20, "f_mrr": 20, "b_mrr": 20}


def contrib(st, has_end):
    e, f, b = st["e"], st["fk"], st["br"]
    v = Counter()
    v["e1"] += e == 1
    v["e2_5"] += 2 <= e <= 5
    v["e6_10"] += 6 <= e <= 10
    v["e11_30"] += e > 10
    v["e0"] += e == 0
    v["e_mrr"] += 1 / e if 1 <= e <= 10 else 0
    v["e_le3"] += 1 <= e <= 3
    f10 = f if f <= 10 else 0
    v["f1"] += f10 == 1
    v["f5"] += 1 <= f10 <= 5
    v["f_le3"] += 1 <= f10 <= 3
    v["f_mrr"] += 1 / f10 if f10 else 0
    v["b1"] += b == 1
    v["b5"] += 1 <= b <= 5
    v["b_mrr"] += 1 / b if b else 0
    v["s1"] += st["sr"] == 1
    v["s5"] += 1 <= st["sr"] <= 5
    v["sf1"] += st["sfr"] == 1
    v["sf5"] += 1 <= st["sfr"] <= 5
    if has_end:
        v["bk5"] += 1 <= st["bkr"] <= 5
        v["fend5"] += 1 <= f10 <= 5
    v["ssup"] += st["ssup"]
    for k, n in st["drops"].items():
        v[k] += n
    for k in ("bsup", "binapp", "bnote", "fsup"):
        v[k] += st[k]
    return v


def loss(tot):
    out = 0.0
    for k, (lo, hi) in TARGET.items():
        x = tot.get(k, 0)
        out += WEIGHT.get(k, 1) * (max(0, lo - x) + max(0, x - hi))
    return out


def search(ids, vecs, target, weight, sweeps=40, seed=7):
    """Choose one candidate per determination so the summed vectors land in the target intervals:
    a small integer programme (HiGHS via scipy), warm-started by nothing; slack keeps it feasible."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    from scipy.sparse import coo_matrix
    keys = list(target)
    col, blocks = [], []
    for a, i in enumerate(ids):
        Mi = np.array([[v.get(k, 0) for k in keys] for v in vecs[i]], float)
        _, first = np.unique(np.round(Mi, 6), axis=0, return_index=True)
        for j in sorted(first):
            col.append((a, int(j)))
            blocks.append(Mi[j])
    nv, nk = len(col), len(keys)
    n = nv + 2 * nk
    V = np.array(blocks)                       # nv x nk
    r_, c_, d_ = [], [], []
    for c, (a, j) in enumerate(col):
        r_.append(a); c_.append(c); d_.append(1.0)
    nz = np.nonzero(V)
    for c, t in zip(*nz):
        x = V[c, t]
        r_ += [len(ids) + 2 * t, len(ids) + 2 * t + 1]; c_ += [c, c]; d_ += [x, x]
    for t in range(nk):
        r_ += [len(ids) + 2 * t, len(ids) + 2 * t + 1]; c_ += [nv + 2 * t, nv + 2 * t + 1]; d_ += [1.0, -1.0]
    A = coo_matrix((d_, (r_, c_)), shape=(len(ids) + 2 * nk, n))
    lo = [1] * len(ids)
    hi = [1] * len(ids)
    for k in keys:
        lo += [target[k][0], -np.inf]
        hi += [np.inf, target[k][1]]
    cost = np.zeros(n)
    for t, k in enumerate(keys):
        cost[nv + 2 * t] = cost[nv + 2 * t + 1] = weight.get(k, 1) * 100
    integrality = np.zeros(n)
    integrality[:nv] = 1
    ub = np.full(n, np.inf)
    ub[:nv] = 1
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), np.array(lo, float), np.array(hi, float)), integrality=integrality,
               bounds=Bounds(np.zeros(n), ub), options={"time_limit": 600, "disp": False})
    if res.x is None:
        print(f"milp: no incumbent ({res.message}); falling back to annealing", file=sys.stderr)
        return search_anneal(ids, vecs, target, weight, sweeps, seed)
    x = res.x
    choice = {}
    for c, (a, j) in enumerate(col):
        if x[c] > 0.5:
            choice[ids[a]] = j
    print(f"milp: status {res.status} slack {x[nv:].sum():.2f}", file=sys.stderr)
    return choice


def search_anneal(ids, vecs, target, weight, sweeps=40, seed=7):
    """Choose one candidate per determination so the summed vectors land in the target intervals.
    Greedy coordinate descent with random restarts of single coordinates (a small annealing)."""
    import numpy as np
    keys = list(target)
    lo = np.array([target[k][0] for k in keys], float)
    hi = np.array([target[k][1] for k in keys], float)
    w = np.array([weight.get(k, 1) for k in keys], float)
    M = {i: np.array([[v.get(k, 0) for k in keys] for v in vecs[i]], float) for i in ids}
    rr = random.Random(seed)
    choice = {i: rr.randrange(len(vecs[i])) for i in ids}
    tot = sum(M[i][choice[i]] for i in ids)

    def L(t):
        return (w * (np.maximum(0, lo - t) + np.maximum(0, t - hi))).sum(axis=-1)

    cur = float(L(tot))
    best_choice, best = dict(choice), cur
    import math
    temp = 3.0
    for sw in range(sweeps):
        order = ids[:]
        rr.shuffle(order)
        for i in order:
            T = tot - M[i][choice[i]] + M[i]
            ls = L(T)
            if temp > 0.02:
                p = np.exp(-(ls - ls.min()) / temp)
                j = int(rr.choices(range(len(ls)), weights=p.tolist())[0])
            else:
                j = int(np.argmin(ls))
            tot = T[j]
            choice[i] = j
            cur = float(ls[j])
            if cur < best:
                best, best_choice = cur, dict(choice)
        temp *= 0.985
        if sw % 20 == 0 or best == 0:
            print(f"search sweep {sw}: loss {cur:.2f} best {best:.2f} temp {temp:.3f}", file=sys.stderr)
        if best == 0:
            break
    return best_choice


def fit_pools(corpus, K=120, iters=40):
    dets = corpus.eval_set(200)
    cands, ctxs = {}, {}
    for d in dets:
        ctx = DetCtx(corpus, d)
        ctxs[d.determination_id] = ctx
        r = random.Random(f"pool-{d.determination_id}")
        has_end = bool(d.endorsements)
        if d.determination_id in HAND:
            b = HAND[d.determination_id](ctx)
            st = stats(ctx, b)
            cands[d.determination_id] = [(b, st, contrib(st, has_end))]
            continue
        lst, tries = [], 0
        pinned = d.determination_id in PIN_E or d.determination_id in FORCE
        need = K if not pinned else 12
        while len(lst) < need and tries < (K * 40 if pinned else K * 4):
            tries += 1
            glo, ghi = (r.uniform(4, 12), 19.0)
            if d.determination_id in FROZEN:
                glo, ghi = (-6.0, -5.0)
            b = sample_pool(ctx, r, FORCE.get(d.determination_id), glo, ghi)
            if d.determination_id in PIN_E:
                b = place_gold(ctx, b, PIN_E[d.determination_id])
            st = stats(ctx, b)
            if pin_ok(d.determination_id, st, ctx):
                lst.append((b, st, contrib(st, has_end)))
        if not lst:
            raise RuntimeError(f"no candidate pool satisfies the pins for {d.determination_id}")
        cands[d.determination_id] = lst
    ids = [d.determination_id for d in dets]
    choice = search(ids, {i: [v for _, _, v in cands[i]] for i in ids}, TARGET, WEIGHT, iters)
    tot = Counter()
    for i in ids:
        tot.update(cands[i][choice[i]][2])
    return {i: cands[i][choice[i]][0] for i in ids}, {i: cands[i][choice[i]][1] for i in ids}, tot, ctxs


def pin2_ok(did, ctx, base):
    """Stage-2 feasibility in stage 1: the span the chapter says the reader cites must be in the packet."""
    top = rank_list(base, ctx.b_true)
    b10 = top[:10]
    f10 = [x for x in top if ctx.app[x].ok][:10]
    want = BASE_WRONG.get(did, ctx.gold)
    if want not in b10:
        return False
    fw = FIX_WRONG.get(did, ctx.gold)
    if fw is not None and fw not in f10:
        return False
    if did not in FIX_WRONG and ctx.gold not in f10:
        return False
    return True


def repair_pools(corpus, pools, K=80, rounds=6, seed=3):
    """Local repair of a pool choice: per determination, try fresh candidate pools and keep any
    swap that lowers the aggregate loss (pins still enforced)."""
    dets = corpus.eval_set(200)
    ctxs = {d.determination_id: DetCtx(corpus, d) for d in dets}
    cur = {d.determination_id: contrib(stats(ctxs[d.determination_id], pools[d.determination_id]),
                                       bool(d.endorsements)) for d in dets}
    tot = Counter()
    for v in cur.values():
        tot.update(v)
    rr = random.Random(seed)
    for d in dets:
        i = d.determination_id
        if i in HAND or pin2_ok(i, ctxs[i], pools[i]):
            continue
        ctx, got = ctxs[i], None
        for _ in range(K * 60):
            glo, ghi = (rr.uniform(4, 12), 19.0)
            if i in FROZEN:
                glo, ghi = (-6.0, -5.0)
            b = sample_pool(ctx, rr, FORCE.get(i), glo, ghi)
            if i in PIN_E:
                b = place_gold(ctx, b, PIN_E[i])
            st = stats(ctx, b)
            if not (pin_ok(i, st, ctx) and pin2_ok(i, ctx, b)):
                continue
            v = contrib(st, bool(d.endorsements))
            t2 = tot.copy(); t2.subtract(cur[i]); t2.update(v)
            if got is None or loss(t2) < got[0]:
                got = (loss(t2), b, v)
        if got is None:
            print(f"repair: no pool for {i} puts the cited span in the packet", file=sys.stderr)
            continue
        tot.subtract(cur[i]); tot.update(got[2]); cur[i], pools[i] = got[2], got[1]
        print(f"repair: re-pooled {i}", file=sys.stderr)
    best = loss(tot)
    for rnd in range(rounds):
        if best == 0:
            break
        order = [d for d in dets if d.determination_id not in HAND]
        rr.shuffle(order)
        for d in order:
            i = d.determination_id
            ctx = ctxs[i]
            for _ in range(K):
                glo, ghi = (rr.uniform(4, 12), 19.0)
                if i in FROZEN:
                    glo, ghi = (-6.0, -5.0)
                b = sample_pool(ctx, rr, FORCE.get(i), glo, ghi)
                if i in PIN_E:
                    b = place_gold(ctx, b, PIN_E[i])
                st = stats(ctx, b)
                if not (pin_ok(i, st, ctx) and pin2_ok(i, ctx, b)):
                    continue
                v = contrib(st, bool(d.endorsements))
                t2 = tot.copy()
                t2.subtract(cur[i])
                t2.update(v)
                l2 = loss(t2)
                if l2 < best - 1e-9:
                    best, tot, cur[i], pools[i] = l2, t2, v, b
            if best == 0:
                break
        print(f"repair round {rnd}: loss {best:.3f}", file=sys.stderr)
    return pools, tot


def refit_pools(corpus, init, K=150, seed=5):
    """Stage 1 again, warm: the current pool plus K fresh candidates per determination, chosen by the MILP."""
    dets = corpus.eval_set(200)
    cands = {}
    rr = random.Random(seed)
    for d in dets:
        i = d.determination_id
        ctx = DetCtx(corpus, d)
        lst = [(init[i], contrib(stats(ctx, init[i]), bool(d.endorsements)))]
        if i not in HAND:
            tries = 0
            while len(lst) < K + 1 and tries < K * 20:
                tries += 1
                glo, ghi = (rr.uniform(4, 12), 19.0)
                if i in FROZEN:
                    glo, ghi = (-6.0, -5.0)
                b = sample_pool(ctx, rr, FORCE.get(i), glo, ghi)
                if i in PIN_E:
                    b = place_gold(ctx, b, PIN_E[i])
                st = stats(ctx, b)
                if pin_ok(i, st, ctx) and pin2_ok(i, ctx, b):
                    lst.append((b, contrib(st, bool(d.endorsements))))
        cands[i] = lst
    ids = [d.determination_id for d in dets]
    choice = search(ids, {i: [v for _, v in cands[i]] for i in ids}, TARGET, WEIGHT)
    tot = Counter()
    for i in ids:
        tot.update(cands[i][choice[i]][1])
    return {i: cands[i][choice[i]][0] for i in ids}, tot


# ------------------------------------------------------------------ stage 2: the model's misreadings
def holdout_pools(corpus, dets):
    out = {}
    for d in dets:
        ctx = DetCtx(corpus, d)
        r = random.Random(f"pool-{d.determination_id}")
        for _ in range(400):
            b = sample_pool(ctx, r, None, r.uniform(6, 12), 19.0)
            st = stats(ctx, b)
            if 1 <= st["e"] <= 10 and 1 <= st["fk"] <= 10:
                break
        out[d.determination_id] = b
    return out


def stale_view(det):
    return det.model_copy(update={"form": STALE.get(det.form, det.form)})


def blank_view(det):
    return det.model_copy(update={"endorsements": []})


WCONF = {  # name: (config changes, view)
    "base10": ({}, None), "base3": ({"k": 3}, None), "base5": ({"k": 5}, None), "base20": ({"k": 20}, None),
    "fix3": ({"policy_filter": True, "k": 3}, None), "fix5": ({"policy_filter": True, "k": 5}, None),
    "fix10": ({"policy_filter": True}, None), "fix20": ({"policy_filter": True, "k": 20}, None),
    "bare": ({"query_facts": False}, None), "plain": ({"fmt": "plain"}, None),
    "edges": ({"policy_filter": True, "order": "edges"}, None),
    "prec": ({"policy_filter": True, "order": "precedence"}, None),
    "stale": ({}, stale_view), "stalef": ({"policy_filter": True}, stale_view),
    "blankf": ({"policy_filter": True}, blank_view), "blankb": ({}, blank_view),
}
WTARGET = {"base10": 17, "base3": 28, "base5": 23, "base20": 15, "fix3": 25, "fix5": 18, "fix10": 12, "fix20": 13,
           "bare": 27, "plain": 23, "edges": 13, "prec": 10, "stale": 57, "stalef": 53, "blankf": 46, "blankb": 50,
           "stale_ho24": 44, "stale_sup": 40, "blankf_cx": 41}
BASE_WRONG = {"DET-0003": "BP 04 50@2025-01#employee_dishonesty", "DET-0004": "BP 04 50@2025-01#employee_dishonesty",
              "DET-0134": "BP 04 50@2025-01#employee_dishonesty",
              "DET-0022": "IL Bulletin 25-05@2025-05#freezing", "DET-0063": "IL Bulletin 25-05@2025-05#freezing",
              "DET-0181": "IL Bulletin 25-05@2025-05#freezing",
              "DET-0053": "HO-3@2024-06#freezing", "DET-0083": "HO-3@2024-06#freezing",
              "DET-0163": "HO-3@2024-06#freezing", "DET-0178": "HO-3@2024-06#freezing",
              "DET-0075": "WI Bulletin 25-03@2025-03#roof_matching", "DET-0049": "BOAT@2025-03#navigation_limits",
              "DET-0069": "HO-3@2022-01#water_backup", "DET-0071": "HO-3@2024-06#earth_movement",
              "DET-0113": "HO-3@2022-01#earth_movement", "DET-0131": "HO-3@2024-06#earth_movement",
              "DET-0145": "HO-3@2024-06#food_spoilage"}
FIX_WRONG = {"DET-0049": "BOAT@2025-03#navigation_limits", "DET-0071": "HO-3@2024-06#earth_movement",
             "DET-0113": "HO-3@2022-01#earth_movement", "DET-0131": "HO-3@2024-06#earth_movement",
             "DET-0145": "HO-3@2024-06#food_spoilage", "DET-0185": "HO-3@2024-06#food_spoilage",
             "DET-0188": "HO-3@2024-06#roof_settlement", **{d: None for d in FROZEN}}
GRID = [-3.0, -2.0, -1.5, -1.0, -0.6, -0.3, -0.15, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]


def renders(corpus, dets, confs):
    from contextlab.pipeline import Assembler, Pipeline, claim_facts
    base = Pipeline(corpus=corpus)
    out = {}
    for name, (chg, view) in confs.items():
        pipe = base.with_config(**chg)
        asm = Assembler(pipe.config.fmt, pipe.config.order)
        for d in dets:
            dv = view(d) if view else d
            if name.startswith("blank") and not d.endorsements:
                continue
            packet, _, _ = pipe.build_packet(dv)
            out[(name, d.determination_id)] = asm.render(packet, claim_facts(dv))
    return out


def outcome(model, rendered, d, mis):
    model.misreads = {d.determination_id: mis}
    e = model.choose(rendered, d.determination_id, d.topic)
    return None if e is None else e.span.span_id


def det_candidates(corpus, model, d, R_, confs, rr, pairs=250, mixed=600):
    on = sorted({e.span.span_id for (c, i), r in R_.items() if i == d.determination_id
                 for e in r.entries if e.span.topic == d.topic} | {d.gold_span_id})
    cands = [{}]
    for x in on:
        for g in GRID:
            cands.append({x: g})
    for _ in range(pairs if len(on) > 1 else 0):
        a, b = rr.sample(on, 2)
        cands.append({a: rr.choice(GRID), b: rr.choice(GRID)})
    # interference candidates: a span read differently when a span that only some runs keep is present
    sets = [frozenset(e.span.span_id for e in r.entries) for (c, i), r in R_.items() if i == d.determination_id]
    varying = sorted(set().union(*sets) - frozenset.intersection(*sets)) if sets else []
    for x in on:
        for y in varying:
            if y != x:
                for g in (-3.0, -1.5, -0.6, 0.6, 1.5):
                    cands.append({f"{x}|{y}": g})
    # mixed candidates: one plain misreading together with one interference
    for _ in range(mixed if len(on) > 1 and varying else 0):
        a, b = rr.choice(on), rr.choice(on)
        y = rr.choice([v for v in varying if v != b] or [None])
        if y is None:
            continue
        cands.append({a: rr.choice(GRID), f"{b}|{y}": rr.choice((-3.0, -1.5, -0.6, 0.6, 1.5, 3.0))})
    res = []
    for mis in cands:
        cited = {c: outcome(model, R_[(c, d.determination_id)], d, mis) for c in confs
                 if (c, d.determination_id) in R_}
        res.append((mis, cited))
    return res


def wvec(d, cited, corpus):
    v = Counter()
    for c, sid in cited.items():
        wrong = sid != d.gold_span_id
        v[c] += wrong
        if c == "stale" and d.form == "HO-3 (2024-06)" and wrong:
            v["stale_ho24"] += 1
            v["stale_sup"] += sid is not None and corpus.span(sid).provenance.superseded_by is not None
        if c == "blankf" and wrong and d.difficulty == "complex":
            v["blankf_cx"] += 1
    return v


def wpin_ok(d, cited):
    i = d.determination_id
    want_b = BASE_WRONG.get(i, d.gold_span_id)
    if cited["base10"] != want_b:
        return False
    if i in FIX_WRONG:
        w = FIX_WRONG[i]
        if cited["fix10"] == d.gold_span_id or (w is not None and cited["fix10"] != w):
            return False
    elif cited["fix10"] != d.gold_span_id:
        return False
    if i == "DET-0008" and cited["stale"] != "HO-3@2022-01#roof_settlement":
        return False
    if i == "DET-0008" and cited["stalef"] != "HO-3@2022-01#roof_settlement":
        return False
    return True


def fit_model(corpus, name, dets, confs, target, vec_fn, pin_fn=None, sweeps=300, seed=11):
    model = corpus.model(name)
    saved = model.misreads
    R_ = renders(corpus, dets, confs)
    rr = random.Random(f"{name}-{seed}")
    cands, vecs = {}, {}
    for d in dets:
        res = det_candidates(corpus, model, d, R_, confs, rr)
        if pin_fn:
            res = [x for x in res if pin_fn(d, x[1])]
            if not res:
                raise RuntimeError(f"{name}: no misreading satisfies the pins for {d.determination_id}")
        # deduplicate by outcome so the search sees distinct choices
        seen, uniq = set(), []
        for mis, cited in res:
            key = tuple(sorted(cited.items()))
            if key not in seen:
                seen.add(key)
                uniq.append((mis, cited))
        uniq.sort(key=lambda x: len(x[0]))
        cands[d.determination_id] = uniq
        vecs[d.determination_id] = [vec_fn(d, c) for _, c in uniq]
    ids = [d.determination_id for d in dets]
    tgt = {k: (v, v) if not isinstance(v, tuple) else v for k, v in target.items()}
    choice = search(ids, vecs, tgt, {}, sweeps, seed)
    tot = Counter()
    for i in ids:
        tot.update(vecs[i][choice[i]])
    model.misreads = saved
    miss = {k: (tot.get(k, 0), v) for k, v in tgt.items() if not v[0] <= tot.get(k, 0) <= v[1]}
    mis = {i: cands[i][choice[i]][0] for i in ids if cands[i][choice[i]][0]}
    return mis, miss


HTARGET = {"hb_noend": 5, "hb_end": 13, "hb_simple": 8, "hb_complex": 10, "hf_noend": 0, "hf_end": 12,
           "hf_simple": 2, "hf_complex": 10, "TT": 8, "TF": 10, "FT": 4}


def hvec(d, cited):
    v = Counter()
    b = cited["base10"] != d.gold_span_id
    f = cited["fix10"] != d.gold_span_id
    for p, w in (("hb", b), ("hf", f)):
        v[f"{p}_{'end' if d.endorsements else 'noend'}"] += w
        v[f"{p}_{d.difficulty}"] += w
    v["TT"] += b and f
    v["TF"] += b and not f
    v["FT"] += f and not b
    return v


def two_vec(d, cited):
    return Counter({"base10": cited["base10"] != d.gold_span_id, "fix10": cited["fix10"] != d.gold_span_id})


def pin_correct_printed(d, cited):
    if d.determination_id in ("DET-0001", "DET-0008"):
        return cited["base10"] == d.gold_span_id and cited["fix10"] == d.gold_span_id
    return True


MEDIAN_TARGET = {"bare": (3.55, 1.2), "plain": (7.10, 4.5), "edges": (3.49, 1.2)}   # printed lookup median, minutes per right answer


def jitter(did):
    import hashlib
    rng = random.Random(int(hashlib.sha256(did.encode()).hexdigest()[:12], 16))
    return rng.lognormvariate(0, 0.35)


def lookup_values(d, kr):
    j = jitter(d.determination_id)
    return kr * j + 2.2, 11.0 * j + 0.5          # right (exact or not), wrong


def median_pairs(dets, conf, m, kr):
    """Pairs (100th, 101st) of the sorted lookup values whose mean prints as m."""
    vals = sorted((v, d.determination_id, w) for d in dets for w, v in zip((False, True), lookup_values(d, kr)))
    out = []
    for i, (a, da, wa) in enumerate(vals):
        for b, db, wb in vals[i + 1:i + 40]:
            if db != da and f"{(a + b) / 2:.2f}" == f"{m:.2f}":
                out.append(((a, da, wa), (b, db, wb)))
    out.sort(key=lambda p: p[1][0] - p[0][0])
    return out or [None]


def median_vec(d, cited, chosen):
    v = Counter()
    for c, pair in chosen.items():
        if pair is None or c not in cited:
            continue
        (a, da, wa), (b, db, wb) = pair
        wrong = cited[c] != d.gold_span_id
        x = lookup_values(d, MEDIAN_TARGET[c][1])[wrong]
        i = d.determination_id
        v[f"{c}_A"] += i == da and wrong == wa
        v[f"{c}_B"] += i == db and wrong == wb
        if i not in (da, db):
            v[f"{c}_lt"] += x < a
            v[f"{c}_mid"] += a < x < b
        elif i == da and wrong != wa:
            v[f"{c}_lt"] += x < a
        elif i == db and wrong != wb:
            v[f"{c}_lt"] += x < a
            v[f"{c}_mid"] += a < x < b
    for (c, _) in list(v.items()):
        pass
    return v


def stale_exact(cited, corpus):
    """The stale run must still cite an edition and a page (the stale smoke test prints 200/200 exact)."""
    sid = cited.get("stale")
    if sid is None:
        return False
    c = corpus.span(sid).citation()
    return "page" in c and "(" in c


def make_cal(corpus, pools, questions):
    tables = {}
    for d in corpus.dets:
        q = questions.get(d.determination_id, d.question)
        tables[q] = pools[d.determination_id]
    return {"questions": questions, "tables": {"bm25": tables}, "lengths": {},
            "models": {m: {"misreads": {}, "embellish": []} for m in ("sim-large", "sim-mid", "sim-small")}}


# ------------------------------------------------------------------ stage 3: span lengths
TOK_TARGET = {"base3": 440, "base5": 621, "base10": 1088.15, "base20": 2044, "fix3": 439, "fix5": 633,
              "fix10": 1103, "fix20": 1673, "bare": 1089, "stale": 1091, "stalef": 1095}
FMT_TARGET = {"xml": 1154, "json": 1184, "plain": 746}
KIND_TOTAL = {"form": 11186, "endorsement": 8655, "bulletin": 1513, "guidance": 519, "uw_note": 453}
NOMINAL = {"form": 55, "endorsement": 73, "bulletin": 50, "guidance": 40, "uw_note": 32}


def min_len(spec):
    if spec.get("fixed_text"):
        return S.count_tokens(spec["fixed_text"])
    return S.count_tokens(S.span_text(spec, None))


def fit_lengths(cal, n_work=200):
    import numpy as np
    specs = {s["span_id"]: s for s in world.all_span_specs()}
    fixed = {x for x, s in specs.items() if s.get("fixed_text") and s["kind"] != "uw_note"}
    var = [x for x in specs if x not in fixed]
    idx = {x: i for i, x in enumerate(var)}
    lo_b = np.array([min_len(specs[x]) for x in var], float)
    x0 = np.array([max(NOMINAL[specs[x]["kind"]], lo_b[i] + 3) for i, x in enumerate(var)], float)
    cal["lengths"] = {x: int(x0[idx[x]]) for x in var}
    corpus = S.build_corpus(cal)
    dets = corpus.eval_set(n_work)
    confs = {k: WCONF[k] for k in TOK_TARGET}
    R_ = renders(corpus, dets, confs)
    cur = {s.span_id: s.tokens for s in corpus.spans}

    def row_of(rendered):
        v = np.zeros(len(var))
        const = rendered.tokens
        for e in rendered.entries:
            sid = e.span.span_id
            if sid in idx:
                v[idx[sid]] += 1
                const -= cur[sid]
        return v, const

    rows, rhs, kinds = [], [], []            # equality rows (centre targets), with tolerance
    per = {}
    for (c, i), r in R_.items():
        per[(c, i)] = row_of(r)
    for c, t in TOK_TARGET.items():
        v = sum(per[(c, d.determination_id)][0] for d in dets)
        k = sum(per[(c, d.determination_id)][1] for d in dets)
        rows.append(v); rhs.append(t * len(dets) - k)
        kinds.append(("mean", c, len(dets) * (0.3 if c == "base10" else 0.4)))   # base10: plain must round to 746
    for (c, i, t) in (("base10", "DET-0001", 1141), ("fix10", "DET-0001", 1182), ("fix10", "DET-0008", 1102)):
        v, k = per[(c, i)]
        rows.append(v); rhs.append(t - k); kinds.append(("exact", f"{c}:{i}", 0))
    for kd, t in KIND_TOTAL.items():
        v = np.array([1.0 if specs[x]["kind"] == kd else 0.0 for x in var])
        k = sum(cur[x] for x in fixed if specs[x]["kind"] == kd)
        rows.append(v); rhs.append(t - k); kinds.append(("exact", f"kind:{kd}", 0))
    maxdet = None
    for attempt in range(12):
        A = np.array(rows); b = np.array(rhs, float)
        free = np.ones(len(var), bool)
        x = x0.copy()
        for _ in range(30):
            Af = A[:, free]
            bf = b - A[:, ~free] @ x[~free]
            xf0 = x0[free]
            lam = np.linalg.lstsq(Af @ Af.T, bf - Af @ xf0, rcond=None)[0]
            x[free] = xf0 + Af.T @ lam
            low = x < lo_b
            if not low.any():
                break
            x[low] = lo_b[low]
            free &= ~low
        xi = np.round(x).astype(int)
        xi = np.maximum(xi, lo_b.astype(int))
        for _rep in range(4):
            # integer repair: exact rows first, by nudging spans with the smallest footprint elsewhere
            for _ in range(600):
                res = b - A @ xi
                bad = [r for r in range(len(rows)) if abs(res[r]) > (kinds[r][2] if kinds[r][0] == "mean" else 0.0)]
                if not bad:
                    break
                r = bad[0]
                step = 1 if res[r] > 0 else -1
                cand = np.where(A[r] > 0)[0]
                others = np.abs(A).sum(axis=0) - A[r]
                best = None
                for j in cand:
                    if step < 0 and xi[j] - 1 < lo_b[j]:
                        continue
                    if 0 < xi[j] - lo_b[j] + step < 3 and specs[var[j]]["kind"] != "uw_note" and _rep < 3:
                        continue                      # would leave a one- or two-token filler
                    sc = others[j] / A[r, j]
                    if best is None or sc < best[0]:
                        best = (sc, j)
                if best is None:
                    break
                j = best[1]
                xi[j] += step * max(1, int(abs(res[r]) // max(A[r, j], 1)) if kinds[r][0] == "exact" else 1)
            # no filler of one or two tokens: lift such spans to three
            for j in range(len(var)):
                gap = xi[j] - lo_b[j]
                if 0 < gap < 3 and specs[var[j]]["kind"] != "uw_note":
                    xi[j] = lo_b[j] + 3
        cal["lengths"] = {x: int(xi[idx[x]]) for x in var}
        corpus = S.build_corpus(cal)
        R1 = renders(corpus, dets, {"base10": WCONF["base10"], "stale": WCONF["stale"]})
        (cmax, imax), rmax = max(R1.items(), key=lambda kv: kv[1].tokens)
        mx = rmax.tokens
        print(f"lengths attempt {attempt}: largest packet {mx} ({cmax} {imax})", file=sys.stderr)
        if mx == 1259:
            break
        cur = {s.span_id: s.tokens for s in corpus.spans}
        if maxdet is None or mx > 1259:
            maxdet = (cmax, imax)
        for (c2, i2), r2 in R1.items():          # every packet over the line is capped below it
            if r2.tokens > 1259 or (c2, i2) == maxdet:
                v, k = row_of(r2)
                rows.append(v)
                rhs.append((1259 if (c2, i2) == maxdet else 1245) - k)
                kinds.append(("exact" if (c2, i2) == maxdet else "mean", f"max:{c2}:{i2}", 0 if (c2, i2) == maxdet else 14))
    return cal


# ------------------------------------------------------------------ main
EMBELLISH = {"sim-large": 6, "sim-mid": 5, "sim-small": 18}
TWO_TARGET = {"sim-mid": {"base10": 20, "fix10": 15}, "sim-small": {"base10": 16, "fix10": 16}}


def main(pool_pickle: str | None = None) -> None:
    import json
    import pickle
    corpus0 = S.build_corpus({"lengths": {}, "tables": {"bm25": {}},
                              "models": {m: {"misreads": {}, "embellish": []} for m in EMBELLISH}})
    if pool_pickle and Path(pool_pickle).exists():
        pools, _ = pickle.load(open(pool_pickle, "rb"))
        tot = Counter()
        for d in corpus0.eval_set(200):
            ctx = DetCtx(corpus0, d)
            if d.determination_id in HAND:
                pools[d.determination_id] = HAND[d.determination_id](ctx)
            tot.update(contrib(stats(ctx, pools[d.determination_id]), bool(d.endorsements)))
        print("stage 1: pools from", pool_pickle, file=sys.stderr)
        if loss(tot) > 0 or not all(pin2_ok(d.determination_id, DetCtx(corpus0, d), pools[d.determination_id])
                                    for d in corpus0.eval_set(200) if d.determination_id not in HAND):
            pools, tot = repair_pools(corpus0, pools)
            pickle.dump((pools, None), open(pool_pickle, "wb"))
    else:
        pools, _, tot, _ = fit_pools(corpus0, K=1000, iters=500)
    for k, (lo, hi) in TARGET.items():
        if not lo <= tot.get(k, 0) <= hi:
            print("stage 1 MISS", k, round(tot.get(k, 0), 2), lo, hi, file=sys.stderr)
    pools.update(holdout_pools(corpus0, corpus0.holdout()))
    questions = unique_questions(S.build_rows())
    cal = make_cal(corpus0, pools, questions)
    corpus = S.build_corpus(cal)
    work, hold = corpus.eval_set(200), corpus.holdout()

    pairs = {c: median_pairs(work, c, *MEDIAN_TARGET[c]) for c in MEDIAN_TARGET}
    best = None
    for attempt in range(10):
        # bare and plain keep their first pair; edges walks its list; the last try drops edges
        chosen = {"bare": pairs["bare"][0], "plain": pairs["plain"][0],
                  "edges": pairs["edges"][attempt % len(pairs["edges"])] if attempt < 9 else None}
        tgt = dict(WTARGET)
        for c, p in chosen.items():
            if p is not None:
                tgt.update({f"{c}_lt": 99, f"{c}_mid": 0, f"{c}_A": 1, f"{c}_B": 1})
        mis, miss = fit_model(corpus, "sim-large", work, WCONF, tgt,
                              lambda d, c: wvec(d, c, corpus) + median_vec(d, c, chosen),
                              pin_fn=lambda d, c: wpin_ok(d, c) and pin_correct_printed(d, c)
                              and stale_exact(c, corpus))
        print(f"stage 2 sim-large work miss (median pair {attempt}):", miss, file=sys.stderr)
        if best is None or len(miss) < len(best[1]):
            best = (mis, miss)
        if not miss:
            break
    mis, miss = best
    cal["models"]["sim-large"]["misreads"].update(mis)
    hconf = {k: WCONF[k] for k in ("base10", "fix10")}
    mis, miss = fit_model(corpus, "sim-large", hold, hconf, HTARGET, hvec)
    print("stage 2 sim-large holdout miss:", miss, file=sys.stderr)
    cal["models"]["sim-large"]["misreads"].update(mis)
    for name, tgt in TWO_TARGET.items():
        mis, miss = fit_model(corpus, name, work, hconf, tgt, two_vec)
        print(f"stage 2 {name} miss:", miss, file=sys.stderr)
        cal["models"][name]["misreads"].update(mis)
    for name, n in EMBELLISH.items():
        cal["models"][name]["embellish"] = [d.determination_id for d in work[3::31]][:n] if n <= 6 else \
            [d.determination_id for d in work[3::11]][:n]
    cal = fit_lengths(cal)
    out = HERE / "data" / "calibration.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(cal, indent=1, sort_keys=True))
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
