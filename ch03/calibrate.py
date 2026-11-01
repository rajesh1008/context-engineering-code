"""Build Chapter 3's fixtures (calibration.json) and fit them to the chapter's profile.

python calibrate.py   (deterministic; a few minutes). lab.py and the other scripts only read
calibration.json. Steps:
  1. 200 reviewed determinations (DET-0001..0200) and 15 roof probes, with their questions;
  2. span sizes, scaled per product family and edition so that the corpus totals match;
  3. the retriever's candidate lists (12 per determination) and passage lengths;
  4. the misread fixtures: for each experiment cell, which packets each stand-in model
     misreads and which rival it then cites. Each is checked by running the real pipeline.
"""
from __future__ import annotations

import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "calibration.json"
if not OUT.exists():
    OUT.write_text("{}")
import standin as S  # noqa: E402

R = random.Random(2026)
CAL: dict = {}

TOPIC_N = {"water_backup": 13, "food_spoilage": 13, "frozen_pipes": 13, "rental_reimbursement": 12,
           "navigation_limits": 11, "roof_settlement": 11, "rideshare": 11, "roof_matching": 10, "deer_swerve": 9,
           "tree_debris": 8, "hail_cosmetic": 8, "deer_collision": 7, "employee_dishonesty": 7,
           "equipment_breakdown": 7, "spoilage": 7, "personal_property_rc": 7, "wind_hail": 7, "hull_damage": 7,
           "earth_movement": 7, "ice_dam": 6, "flood": 6, "mold": 6, "snow_load": 3, "dog_bite": 2, "wear_tear": 2}
QUESTIONS = {
    "water_backup": ["Does the policy cover water that backs up through a floor drain, and up to what limit?",
                     "Sewage came up through the basement drain. Is the backup covered?"],
    "food_spoilage": ["A power cut spoiled the food in the freezer. Is the spoilage covered?",
                      "Is spoiled food covered after a two-day outage?"],
    "frozen_pipes": ["Pipes froze and burst while the house was empty for a week. Is the damage covered?",
                     "A frozen pipe split in the crawl space. Is it covered?"],
    "rental_reimbursement": ["The car is in the shop after a covered loss. Is a rental covered, and for how long?",
                             "Is a rental car covered while the car is repaired?"],
    "navigation_limits": ["The boat was damaged outside the stated navigation area. Is it covered?",
                          "Was the boat inside its navigation limits, and is the loss covered?"],
    "roof_settlement": ["Wind lifted shingles on a 19-year-old roof. Is the roof paid at replacement cost?",
                        "How is the roof settled after this wind loss?"],
    "rideshare": ["The insured was driving for a rideshare app when the crash happened. Is it covered?",
                  "Is a crash during a rideshare trip covered?"],
    "roof_matching": ["Must we replace undamaged roofing so that the repair matches?",
                      "The new shingles will not match. What do we owe for matching?"],
    "deer_swerve": ["The insured swerved to miss a deer and hit a tree. Is this collision or comprehensive?",
                    "Swerving to avoid a deer, the car left the road. Which coverage applies?"],
    "tree_debris": ["A neighbour's tree fell in the yard without hitting the house. Is debris removal covered?",
                    "Is removal of a fallen tree covered?"],
    "hail_cosmetic": ["Hail dented the metal roof but it does not leak. Is the cosmetic damage covered?",
                      "Is cosmetic hail damage to the roof covered?"],
}
FIRST = ["James", "Mary", "John", "Patricia", "Michael", "Linda", "David", "Barbara", "William", "Elizabeth",
         "Richard", "Jennifer", "Joseph", "Maria", "Thomas", "Karen", "Daniel", "Nancy", "Paul", "Lisa"]
LAST = ["Smith", "Johnson", "Brown", "Miller", "Davis", "Wilson", "Moore", "Taylor", "Thomas", "Jackson",
        "White", "Harris", "Martin", "Thompson", "Garcia", "Clark", "Lewis", "Walker", "Hall", "Young"]
PREFIX = {"HO-3": "LMH", "PAP": "LMA", "BOAT": "LMB", "BOP": "LMC"}


def question(t: str, r: random.Random) -> str:
    q = QUESTIONS.get(t)
    if q:
        return r.choice(q)
    return r.choice([f"Is the {t.replace('_', ' ')} loss covered under this policy?",
                     f"Is the {t.replace('_', ' ')} loss covered?"])


def ed_window(fam: str, ed: str) -> tuple[date, date]:
    eds = S.EDITIONS[fam]
    i = eds.index(ed)
    lo = S._ed_date(ed)
    hi = S._ed_date(eds[i + 1]) - timedelta(days=1) if i + 1 < len(eds) else date(2026, 10, 15)
    return lo, hi


def endorsements_for(fam: str, ed: str, loss: date, topic: str, r: random.Random) -> list[str]:
    out = []
    for code, f, t, eds in S.ENDORSEMENTS:
        if f != fam:
            continue
        ok = [e for e in eds if S._ed_date(e) <= loss]
        if not ok:
            continue
        p = 0.75 if t == topic else 0.2
        if r.random() < p:
            out.append(f"{code} ({ok[-1]})")
    return out


def make_det(i: int, topic: str, r: random.Random, **fix) -> dict:
    fam = S.TOPIC_FAMILY[topic]
    ed = fix.get("ed") or r.choice(S.EDITIONS[fam])
    lo, hi = ed_window(fam, ed)
    loss = fix.get("loss") or lo + timedelta(days=r.randrange((hi - lo).days + 1))
    state = fix.get("state") or r.choice(S.STATES)
    ends = fix["ends"] if "ends" in fix else endorsements_for(fam, ed, loss, topic, r)
    d = dict(determination_id=f"DET-{i:04d}", holder=fix.get("holder") or f"{r.choice(FIRST)} {r.choice(LAST)}",
             state=state, family=fam, form=f"{fam} ({ed})", endorsements=ends,
             policy_number=fix.get("pn") or f"{PREFIX[fam]}-{state}-{r.randrange(10**7):07d}",
             date_of_loss=loss.isoformat(), topic=topic, question=fix.get("q") or question(topic, r), gold_span_id="")
    det = S.Determination(**d)
    d["gold_span_id"] = S.CORPUS.governing(det).span_id
    return d


def build_dets() -> list[dict]:
    topics = [t for t, n in TOPIC_N.items() for _ in range(n)]
    fixed = {1: "water_backup", 2: "navigation_limits", 6: "navigation_limits"}
    for t in fixed.values():
        topics.remove(t)
    R.shuffle(topics)
    out = []
    for i in range(1, 201):
        if i == 1:
            out.append(make_det(1, "water_backup", R, ed="2024-06", loss=date(2026, 3, 14), state="WI",
                                ends=["HO 04 95 (2024-06)"], holder="Emily Carter", pn="LMH-WI-0048213",
                                q=QUESTIONS["water_backup"][0]))
        elif i == 2:
            out.append(make_det(2, "navigation_limits", R, ed="2021-05", loss=date(2024, 5, 15), state="MI",
                                ends=[], holder="Robert Weber"))
        elif i == 6:
            out.append(make_det(6, "navigation_limits", R, ed="2025-03", loss=date(2025, 8, 9), state="IL",
                                ends=["BT 05 (2025-03)"], holder="Susan Peterson"))
        else:
            out.append(make_det(i, topics.pop(), R))
    return out


def post_cut(d: dict) -> bool:
    return S.CORPUS.by_id[d["gold_span_id"]].provenance.effective_from > S.CUT


def fit_post_cut(ds: list[dict], target: int = 79) -> list[dict]:
    r = random.Random(7)
    while (n := sum(map(post_cut, ds))) != target:
        i = r.randrange(3, 200)
        if i == 5:
            continue
        d = ds[i]
        if (n > target) == post_cut(d):
            nd = make_det(i + 1, d["topic"], r)
            if post_cut(nd) != post_cut(d):
                ds[i] = nd
    return ds


def a_tokens(d: dict) -> int:
    det = S.Determination(**d)
    return (sum(S.count_tokens(i.text) for i in S.instructions_for(det)) + S.count_tokens(S.claim_facts(det))
            + S.count_tokens("Question: " + det.question))


def header(s) -> int:
    e = S.ev_of(s)
    return S.count_tokens(S.Assembler().span_text(1, e)) - S.count_tokens(e.text)


def fit_questions(ds: list[dict], lo: float, hi: float) -> None:
    """Pick between each topic's question wordings so that mean fixed tokens fall in [lo, hi)."""
    r = random.Random(11)
    tot = sum(map(a_tokens, ds))
    while not lo * 200 <= tot < hi * 200:
        i = r.randrange(3, 200)
        d = ds[i]
        old = a_tokens(d)
        d2 = dict(d, question=question(d["topic"], r))
        new = a_tokens(d2)
        if (tot < lo * 200 and new > old) or (tot >= hi * 200 and new < old):
            ds[i], tot = d2, tot - old + new


# Tokens moved between the families' pre-cut text, keeping the corpus and the family-packet mean:
# it puts each family's shared prefix where Listing 3.10's 1,024-token blocks give a 92.1% cached share.
FAM_SHIFT = {"HO-3": 315, "PAP": -252, "BOP": 101, "BOAT": -12, "BUL": -152}


def fit_sizes(ds: list[dict]) -> dict:
    """Span sizes: 22,326 text tokens in all; the 99 spans that took effect after the cut and the
    product families sized so that the stale snapshot and the family packets come out as in the chapter."""
    sk = S.corpus_skeleton()
    hand = {sid: S.count_tokens(t) for sid, t in S.HAND_TEXT.items()}
    spans = {s.span_id: s for s in S.CORPUS.spans}
    H = {sid: header(spans[sid]) for sid, *_ in sk}
    post = {sid: p.effective_from > S.CUT for sid, p, *_ in sk}
    fam = {sid: "BUL" if p.kind == "bulletin" else p.family for sid, p, *_ in sk}   # bulletins are in every family
    A = sum(map(a_tokens, ds)) / 200
    mean_all = A + sum(H.values()) + 22_326
    w = {f: sum(d["family"] == f for d in ds) / 200 for f in S.EDITIONS} | {"BUL": 1.0}
    FAMS = (*S.EDITIONS, "BUL")
    P = round(mean_all - 25_047 - sum(H[s] for s in post if post[s]))          # post-cut text tokens
    fam_need = 14_463 - A - sum(w[fam[s]] * H[s] for s in H)                    # weighted family text tokens
    r = random.Random(5)
    base = {sid: max(8, int(r.lognormvariate(math.log(56), 0.45))) for sid, *_ in sk if sid not in hand}

    def group_scale(sel, total):
        ids = [s for s in base if sel(s)]
        k = total / sum(base[s] for s in ids)
        for s in ids:
            base[s] = max(8, round(base[s] * k))
        base[max(ids, key=lambda s: base[s])] += total - sum(base[s] for s in ids)

    hp = sum(hand[s] for s in hand if post[s])
    totals = {True: P - hp, False: 22_326 - P - (sum(hand.values()) - hp)}
    hand_w = sum(w[fam[s]] * hand[s] for s in hand)

    def groups(m):
        g = {}
        for flag in (True, False):
            wt = {f: m if f == "HO-3" else 1.0 for f in FAMS}
            raw = {f: wt[f] * sum(base[s] for s in base if fam[s] == f and post[s] == flag) for f in FAMS}
            for f in FAMS:
                g[f, flag] = totals[flag] * raw[f] / sum(raw.values())
        return g

    lo, hi = 0.2, 5.0
    for _ in range(60):
        m = (lo + hi) / 2
        val = sum(w[f] * v for (f, _), v in groups(m).items()) + hand_w
        lo, hi = (m, hi) if val < fam_need else (lo, m)
    g = groups(m)
    gi = {k: int(v) for k, v in g.items()}
    for flag in (True, False):                     # integer group totals that keep the flag totals
        ks = sorted((k for k in g if k[1] == flag), key=lambda k: g[k] - gi[k], reverse=True)
        for k in ks[:totals[flag] - sum(gi[k] for k in ks)]:
            gi[k] += 1
    for f, n in FAM_SHIFT.items():
        gi[f, False] += n
    for (f, flag), total in gi.items():
        ids = [s for s in base if fam[s] == f and post[s] == flag]
        if not ids:
            continue
        k = total / sum(base[s] for s in ids)
        for s in ids:
            base[s] = max(8, round(base[s] * k))
        diff = total - sum(base[s] for s in ids)
        for i in range(abs(diff)):
            s = sorted(ids, key=lambda s: -base[s])[i % len(ids)]
            base[s] += 1 if diff > 0 else -1
    return base


def check_tokens(ds):
    S.CAL.update(CAL)
    S.CORPUS = S.Corpus(CAL["sizes"])
    dets = [S.Determination(**d) for d in ds]
    out = {}
    for key, cfg, corpus in (("all", dict(stuff="all"), None), ("family", dict(stuff="family"), None)):
        p = S.Pipeline(S.PipelineConfig(**cfg))
        out[key] = sum(p.render(d).tokens for d in dets) / 200
    stale = S.Corpus(CAL["sizes"])
    stale.spans = [s for s in stale.spans if s.provenance.effective_from <= S.CUT]
    p = S.Pipeline(S.PipelineConfig(stuff="all"), corpus=stale)
    out["stale"] = sum(p.render(d).tokens for d in dets) / 200
    out["text"] = sum(s.tokens for s in S.CORPUS.spans)
    return out


PAIR = ("DET-0094", "DET-0130")


def build_retrieval(ds: list[dict], u_adjust: dict | None = None) -> dict:
    """Twelve candidates per determination (the retriever's list, best first) with passage
    lengths fitted so that mean packet tokens match: filtered 1,046 on the 60 and the 200,
    unfiltered 1,018 on the 60 and 1,031 on the 200, DET-0001's filtered packet 1,182."""
    r = random.Random(23)
    by = S.CORPUS.by_id
    dets = [S.Determination(**d) for d in ds]
    nine = set(r.sample(range(3, 200), 40))                 # 40 determinations with 9 applicable candidates
    out = {}
    for i, d in enumerate(dets):
        gold = by[d.gold_span_id]
        topic = [s for s in S.CORPUS.spans if s.topic == d.topic and s is not gold]
        fam = [s for s in S.CORPUS.spans if s.provenance.family == d.family and s.topic != d.topic]
        app_t = [s for s in topic if S.applies(s, d)]
        ina_t = [s for s in topic if not S.applies(s, d)]
        app_f = [s for s in fam if S.applies(s, d)]
        ina_f = [s for s in fam if not S.applies(s, d)]
        r.shuffle(app_t); r.shuffle(ina_t); r.shuffle(app_f); r.shuffle(ina_f)
        n_in = 3 if i in nine else r.choice([1, 2])
        ina = (ina_t + ina_f)[:n_in]
        app = (app_t + app_f)[:11 - n_in]
        if d.determination_id == "DET-0006":
            first = by["BOAT (2025-03)|navigation_limits"]
            app = [first] + [s for s in app if s is not first][:10 - n_in]
            cands = [first, gold] + [s for s in app[1:]]
            cands = cands[:2] + ina[:1] + cands[2:] + ina[1:]
        else:
            others = app + ina
            r.shuffle(others)
            others.sort(key=lambda s: -(s.topic == d.topic))
            g = 0 if r.random() < 0.8 else r.randint(1, 3)
            cands = others[:g] + [gold] + others[g:]
        if d.determination_id == "DET-0002":
            first = by["BOAT (2025-03)|navigation_limits"]
            cands = [first] + [s for s in cands if s is not first]
        cands = cands[:12]
        top10 = cands[:10]
        if not any(not S.applies(s, d) for s in top10):          # at least one look-alike the filter removes
            j = next(k for k in range(10, 12) if not S.applies(cands[k], d))
            cands[9], cands[j] = cands[j], cands[9]
        out[d.determination_id] = [s.span_id for s in cands]
    # one pair of claims on the same policy form and state retrieves the same spans (Listing 3.10's 0.5%)
    a, b = PAIR
    out[b] = list(out[a])
    # passage lengths
    ids = [d.determination_id for d in dets]
    T = {k: 1046 + r.randint(-60, 60) for k in ids}
    U = {k: T[k] - 15 + r.randint(-30, 30) for k in ids}
    T["DET-0001"] = 1182
    da, db = (S.Determination(**next(x for x in ds if x["determination_id"] == k)) for k in PAIR)
    T[a], U[a] = 1130, 1120
    T[b], U[b] = T[a] - a_tokens(da.model_dump()) + a_tokens(db.model_dump()), \
        U[a] - a_tokens(da.model_dump()) + a_tokens(db.model_dump())

    def fix_mean(tgt, keys, val, frozen=()):
        free = [k for k in keys if k not in frozen]
        diff = val * len(keys) - sum(tgt[k] for k in keys)
        for n in range(abs(diff)):
            tgt[free[n % len(free)]] += 1 if diff > 0 else -1

    s60, rest = ids[:60], ids[60:]
    fix_mean(T, s60, 1046, ("DET-0001",)); fix_mean(T, rest, 1046, PAIR)
    fix_mean(U, s60, 1018); fix_mean(U, rest, round((1031 * 200 - 1018 * 60) / 140), PAIR)
    for k, v in (u_adjust or {}).items():
        U[k] += v
    build_retrieval.U = U
    res = {}
    for d in dets:
        k = d.determination_id
        if k == b:
            res[k] = [list(x) for x in res[a]]
            continue
        cands = [by[s] for s in out[k]]
        base = sum(S.count_tokens(i.text) for i in S.instructions_for(d)) + S.count_tokens(S.claim_facts(d)) \
            + S.count_tokens("Question: " + d.question)
        filt = [s for s in cands if S.applies(s, d)][:10]
        unf = cands[:10]
        shared = [s for s in filt if s in unf]
        fo = [s for s in filt if s not in unf]
        uo = [s for s in unf if s not in filt]
        n = {}
        budget_f = T[k] - base - sum(header(s) for s in filt)
        budget_u = U[k] - base - sum(header(s) for s in unf)
        # shared passages: an even share of the filtered budget, capped by span length
        for s in shared:
            n[s.span_id] = max(6, min(s.tokens, budget_f // len(filt)))
        for group, budget in ((fo, budget_f), (uo, budget_u)):
            left = budget - sum(n[s.span_id] for s in shared)
            for j, s in enumerate(group):
                share = left // (len(group) - j)
                n[s.span_id] = max(1, min(s.tokens, share))
                left -= n[s.span_id]
            if left:                                          # push the remainder onto shared spans
                for s in shared:
                    add = max(-n[s.span_id] + 6, min(s.tokens - n[s.span_id], left)) if left > 0 else \
                        max(-(n[s.span_id] - 6), left)
                    n[s.span_id] += add
                    left -= add
                    if not left:
                        break
        res[k] = [[s.span_id, n.get(s.span_id, min(s.tokens, 30))] for s in cands]
        S.CAL["retrieval"] = res
        pf, pu = S.Pipeline(S.PipelineConfig(policy_filter=True)), S.Pipeline(S.PipelineConfig())
        for _ in range(80):                                   # refine on the real renders
            df, du = T[k] - pf.render(d).tokens, U[k] - pu.render(d).tokens
            if not df and not du:
                break
            row = {sid: m for sid, m in res[k]}
            pool, delta = ((filt, df) if df else (uo or unf, du))
            room = [(s.tokens - row[s.span_id] if delta > 0 else row[s.span_id] - 4, s) for s in pool]
            gap, s = max(room, key=lambda x: x[0])
            if gap <= 0 and not df and fo:
                # trade text from a shared passage into a filtered-only one: same filtered size, smaller unfiltered
                sh = max(shared, key=lambda s: row[s.span_id] if du < 0 else s.tokens - row[s.span_id])
                f_ = max(fo, key=lambda s: s.tokens - row[s.span_id] if du < 0 else row[s.span_id])
                step = min(abs(du), (row[sh.span_id] - 4) if du < 0 else sh.tokens - row[sh.span_id],
                           (f_.tokens - row[f_.span_id]) if du < 0 else row[f_.span_id] - 4)
                if step <= 0:
                    break
                sign = 1 if du < 0 else -1
                row[sh.span_id] -= sign * step
                row[f_.span_id] += sign * step
                res[k] = [[sid, row[sid]] for sid, _ in res[k]]
                S.CAL["retrieval"] = res
                continue
            if gap <= 0:
                break
            row[s.span_id] += (1 if delta > 0 else -1) * min(abs(delta), gap)
            res[k] = [[sid, row[sid]] for sid, _ in res[k]]
            S.CAL["retrieval"] = res
    return res


# ---------------------------------------------------------------- step 4: misread fixtures
SWEEP = {  # Listing 3.6, share of the 60 citing the governing clause, depth 0%..100%
    ("sim-large", 2_000): (95, 97, 97, 93, 95, 93, 95, 93, 93, 93, 95),
    ("sim-large", 15_000): (95, 97, 93, 87, 53, 37, 40, 62, 88, 95, 93),
    ("sim-large", 30_000): (93, 95, 87, 57, 27, 18, 22, 40, 73, 88, 97),
    ("sim-large", 60_000): (58, 55, 45, 30, 13, 15, 13, 28, 40, 58, 67),
    ("sim-small", 2_000): (90, 97, 97, 93, 92, 90, 90, 85, 92, 95, 93),
    ("sim-small", 15_000): (65, 67, 53, 45, 35, 10, 17, 33, 43, 62, 65),
}
ON_200 = {(2_000, 0.0): 190, (2_000, 0.5): 186, (15_000, 0.0): 189, (15_000, 0.5): 65,
          (60_000, 0.0): 118, (60_000, 0.5): 29, (60_000, 1.0): 129}
TOPIC_15K = {"water_backup": (12, 1), "food_spoilage": (13, 1), "frozen_pipes": (12, 3),
             "rental_reimbursement": (12, 7), "navigation_limits": (9, 1), "roof_settlement": (11, 0),
             "rideshare": (11, 1), "roof_matching": (8, 0), "deer_swerve": (9, 2), "tree_debris": (8, 0),
             "hail_cosmetic": (8, 4)}
BOOT = {2_000: [(-10, 10), (-5, 10), (-3.3, 6.7), (-1.7, 5.0), (0.0, 4.2)],
        15_000: [(40, 90), (45, 80), (50, 76.7), (53.3, 70), (57.5, 66.7)]}


def n_of(pct: int, n: int = 60) -> int:
    return next(c for c in range(n + 1) if round(100 * c / n) == pct)


def boot_ok(size: int, gaps, idx) -> bool:
    import numpy as np
    g = np.asarray(gaps)
    for (lo, hi), n in zip(BOOT[size], (10, 20, 30, 60, 120)):
        v = np.sort(100 * g[idx[size, n]].sum(1) / n)
        if abs(round(v[100], 1) - lo) > 0.05 or abs(round(v[1899], 1) - hi) > 0.05:
            return False
    return True


def boot_index():
    import numpy as np
    rng, idx = random.Random(2026), {}
    for size in (2_000, 15_000):
        for n in (10, 20, 30, 60, 120):
            idx[size, n] = np.array([rng.sample(range(200), n) for _ in range(2_000)])
    return idx


def pick(r, pool, k):
    pool = list(pool)
    return set(r.sample(pool, k))


def hit_vectors(dets) -> dict:
    """Which determinations cite the governing clause in each padded cell."""
    ids = [d.determination_id for d in dets]
    first, rest = ids[:60], ids[60:]
    r = random.Random(31)
    idx = boot_index()
    hits = {}
    import experiments as X
    groups = {}                                 # at 2,000 tokens neighbouring depths can render the same packet
    for d in dets[:60] if False else dets:
        sg = {}
        for dp in S_DEPTHS:
            sg.setdefault(S.signature("", X.render(X.padded_packet(d, 2_000, dp), d)), []).append(dp)
        groups[d.determination_id] = {dp: tuple(v) for v in sg.values() for dp in v}
    hit_vectors.groups = groups
    # 2,000 tokens, start and middle, all 200 (bootstrap must match Listing 3.8)
    while True:
        m = r.randint(0, 8)                     # misread at the start only (m) and at the middle only (m + 4)
        both = 10 - m
        sel = r.sample(ids, both + m + m + 4)
        miss0 = set(sel[:both + m])
        miss5 = set(sel[:both]) | set(sel[both + m:])
        if len(miss0 & set(first)) != 3 or len(miss5 & set(first)) != 4:
            continue
        if sum(len(groups[i][0.5]) > 1 for i in miss5 & set(first)) > 1:
            continue
        h0, h5 = set(ids) - miss0, set(ids) - miss5
        if boot_ok(2_000, [(i in h0) - (i in h5) for i in ids], idx):
            break
    hits["sim-large", 2_000, 0.0], hits["sim-large", 2_000, 0.5] = h0, h5
    # 15,000 tokens: per-topic shares (Listing 3.8's table) and the bootstrap
    by_topic = {}
    for d in dets:
        by_topic.setdefault(d.topic if d.topic in TOPIC_15K else "_other", []).append(d.determination_id)
    TOPIC_15K["_other"] = (76, 45)
    while True:
        h0, h5 = set(), set()
        for t, (a, b) in TOPIC_15K.items():
            h0 |= pick(r, by_topic[t], a)
            h5 |= pick(r, by_topic[t], b)
        if len(h0 & set(first)) != 57 or len(h5 & set(first)) != 22:
            continue
        if boot_ok(15_000, [(i in h0) - (i in h5) for i in ids], idx):
            break
    del TOPIC_15K["_other"]
    hits["sim-large", 15_000, 0.0], hits["sim-large", 15_000, 0.5] = h0, h5
    # 60,000 tokens on all 200 (Listing 3.7); DET-0001 holds, DET-0002 is the misread shown in Listing 3.6's text
    for depth, pct in ((0.0, 58), (0.5, 15), (1.0, 67)):
        k60 = n_of(pct)
        while True:
            h = pick(r, first, k60) | pick(r, rest, ON_200[60_000, depth] - k60)
            if depth != 0.5 or ("DET-0001" in h and "DET-0002" not in h):
                break
        hits["sim-large", 60_000, depth] = h
    # the rest of the sweep, on the 60: a latent difficulty per determination
    z = {i: r.random() for i in first}
    for (model, size), row in SWEEP.items():
        for depth, pct in zip(S_DEPTHS, row):
            if (model, size, depth) in hits:
                continue
            k = n_of(pct)
            order = sorted(first, key=lambda i: z[i] + 0.6 * r.random())
            if size == 2_000:                   # a packet already seen at a neighbouring depth keeps its outcome
                forced = {}
                for i in first:
                    for dp in groups[i][depth]:
                        if dp != depth and (model, size, dp) in hits:
                            forced[i] = i in hits[model, size, dp]
                fh = {i for i, v in forced.items() if v}
                free = [i for i in order if i not in forced]
                assert len(fh) <= k <= len(fh) + len(free), (model, depth)
                hits[model, size, depth] = fh | set(free[:k - len(fh)])
                continue
            hits[model, size, depth] = set(order[:k])
    return hits


S_DEPTHS = tuple(round(0.1 * i, 1) for i in range(11))


class Fixtures:
    """Records which packet (model, determination, signature) is misread and the rival cited."""

    def __init__(self):
        self.m: dict = {}
        self.models = {}

    def model(self, name):
        if name not in self.models:
            self.models[name] = S.SimulatedModel(name)
        return self.models[name]

    def default(self, name, r, d):
        """What the stand-in cites with no fixture entry."""
        return self.cite(name, r, d)

    def cite(self, name, r, d):
        m = self.model(name)
        m.misreads = self.m.get(name, {})
        return m.generate(r, key=d.determination_id, topic=d.topic).cited_span_id

    def set_wrong(self, name, r, d, avoid=(), rival=None):
        m = self.model(name)
        gold = S.CORPUS.governing(d).span_id
        sc = m.scores(r, d)
        if rival is None:
            cands = [s for s in sorted(sc, key=lambda s: -sc[s]) if s != gold and s not in avoid
                     and not s.startswith("PAGE")]
            rival = cands[0]
        sig = S.signature(name, r)
        old = self.m.get(name, {}).get(d.determination_id, {}).get(sig)
        if old is not None and rival is None and old not in avoid:
            return old                            # two cells render the same packet
        assert old in (None, rival), (name, d.determination_id, sig)
        self.m.setdefault(name, {}).setdefault(d.determination_id, {})[sig] = rival
        return rival

    def set_right(self, name, r, d):
        sig = S.signature(name, r)
        assert sig not in self.m.get(name, {}).get(d.determination_id, {}), (name, d.determination_id)


def fit_misreads(dets) -> dict:
    import experiments as X
    fx = Fixtures()
    by = {d.determination_id: d for d in dets}
    first = dets[:60]
    # padded sweep and the 200-determination cells
    hits = hit_vectors(dets)
    for (model, size, depth), h in sorted(hits.items(), key=lambda kv: kv[0][1:]):
        pool = dets if (model, size, depth) in {("sim-large",) + k for k in ON_200} else first
        if model == "sim-small" and size > 32_000:
            continue
        for d in pool:
            r = X.render(X.padded_packet(d, size, depth), d)
            if d.determination_id not in h:
                rival = "BOAT (2025-03)|navigation_limits" if (d.determination_id == "DET-0002"
                                                              and (size, depth) == (60_000, 0.5)) else None
                fx.set_wrong(model, r, d, rival=rival)
    # retrieved packets: five orders, with and without the filter (Listings 3.7, 3.12, 3.14, 3.15)
    r_ = random.Random(41)
    ids = [d.determination_id for d in dets]
    for filt, plan in ((False, dict(rank=(5, 12), reversed=(1, 1, 2), edges=(2, 0, 2), precedence=(3, 3, 2),
                                    cache=(1, 3, 2), union=11, edges60=(5,))),
                       (True, dict(rank=(2, 10), reversed=(2, 4, 1), edges=(1, 2, 2), precedence=(3, 1, 1),
                                   cache=(2, 2, 2), union=12, edges60=(1,)))):
        packets = {d.determination_id: X.retrieved(d, filt) for d in dets}
        rend = {o: {i: X.render(packets[i], by[i], o) for i in ids} for o in S.ORDERS}
        first60 = set(ids[:60])
        if filt:   # Listing 3.15: the seven flagged, and which topics they come from
            tp = {}
            for d in dets:
                tp.setdefault(d.topic, []).append(d.determination_id)
        for _ in range(200000):
            if filt:
                chain, last = [], "DET-0006"
                for t in ("personal_property_rc", "food_spoilage", "roof_matching", "frozen_pipes"):
                    later = [i for i in tp[t] if i > last]
                    if not later:
                        break
                    last = r_.choice(later)
                    chain.append(last)
                if len(chain) < 4:
                    continue
                flag = {"DET-0006", *chain, *r_.sample(tp["roof_settlement"], 2)}
                fw = set(r_.sample(sorted(flag - {"DET-0006"}), 3))
                n60 = len(fw & first60)
                if n60 > plan["rank"][0]:
                    continue
                w_rank = fw | pick(r_, first60 - flag, plan["rank"][0] - n60) \
                    | pick(r_, set(ids[60:]) - flag, sum(plan["rank"]) - 3 - (plan["rank"][0] - n60) - len(fw - first60) + len(fw - first60))
                if len(w_rank) != sum(plan["rank"]):
                    continue
                f_ = set(r_.sample(sorted(fw), 2))
                sets = {"reversed": (f_, flag - fw, fw - f_)}
                U = set(flag)
            else:
                w_rank = pick(r_, ids[:60], plan["rank"][0]) | pick(r_, ids[60:], plan["rank"][1])
                sets, U = {}, set()
            right = sorted(set(ids) - w_rank)
            extra = plan["union"] - len(U)
            nw = r_.randint(0, extra)
            U |= pick(r_, sorted(w_rank - U), min(nw, len(w_rank - U))) | pick(r_, [i for i in right if i not in U], extra - min(nw, len(w_rank - U)))
            if len(U) != plan["union"]:
                continue
            Uw, Ur = sorted(U & w_rank), sorted(U - w_rank)
            bad = False
            for o in ("reversed", "edges", "precedence", "cache"):
                if o in sets:
                    continue
                fix, brk, oth = plan[o]
                if fix + oth > len(Uw) or brk > len(Ur):
                    bad = True
                    break
                fo = r_.sample(Uw, fix + oth)
                sets[o] = (set(fo[:fix]), set(r_.sample(Ur, brk)), set(fo[fix:]))
            if bad:
                continue
            union = set().union(*(a | b | c for a, b, c in sets.values()))
            if union != U:
                continue

            def status(o, i):
                if o == "rank":
                    return "W" if i in w_rank else "R"
                f_, b_, o_ = sets[o]
                return "R" if i in f_ else "B" if i in b_ else "O" if i in o_ else status("rank", i)
            clash = False
            for i in U:
                seen = {}
                for o in S.ORDERS:
                    sg = S.signature("sim-large", rend[o][i])
                    if seen.setdefault(sg, status(o, i)) != status(o, i):
                        clash = True
            if clash:
                continue
            f_, b_, o_ = sets["edges"]
            w_edges = (w_rank - f_) | b_
            if len(w_edges & first60) != plan["edges60"][0]:
                continue
            break
        else:
            raise RuntimeError("no order plan found")
        rank_cite = {}
        for i in ids:
            d = by[i]
            if i in w_rank:
                rank_cite[i] = fx.set_wrong("sim-large", rend["rank"][i], d)
        for o, (f_, b_, o_) in sets.items():
            for i in ids:
                d = by[i]
                wrong_now = (i in w_rank and i not in f_) or i in b_
                if not wrong_now:
                    continue
                avoid = (rank_cite[i],) if i in o_ else ()
                keep = rank_cite.get(i) if (i in w_rank and i not in o_) else None
                rival = keep
                if i == "DET-0006" and o == "reversed":
                    rival = "BOAT (2025-03)|navigation_limits"
                fx.set_wrong("sim-large", rend[o][i], d, avoid=avoid, rival=rival)
    # stuffed and whole-corpus strategies (Listing 3.12) and the stale snapshot (Listing 3.13)
    for cfg, n_wrong in ((dict(stuff="family"), 37), (dict(stuff="family", order="cache"), 33),
                         (dict(stuff="all", order="cache"), 55)):
        p = S.Pipeline(S.PipelineConfig(**cfg))
        for i in pick(r_, ids, n_wrong):
            fx.set_wrong("sim-large", p.render(by[i]), by[i])
    stale = S.Corpus(S.CAL["sizes"])
    stale.spans = [s for s in stale.spans if s.provenance.effective_from <= S.CUT]
    p = S.Pipeline(S.PipelineConfig(stuff="all", order="cache"), corpus=stale)
    present = [i for i in ids if S.CORPUS.governing(by[i]).span_id in stale.by_id]
    for i in pick(r_, present, 5):
        fx.set_wrong("sim-large", p.render(by[i]), by[i])
    return fx.m


def build_probes() -> tuple[list[dict], list[float]]:
    """Listing 3.11: 15 roof-topic determinations where a state bulletin in force governs, and one
    uniform draw per probe for the clash (the skill wins below the recency prior, 0.75, without a
    hierarchy rule, and at or above 0.96 with one)."""
    r = random.Random(77)
    out = []
    plan = [("roof_matching", "WI", date(2025, 3, 15), date(2026, 9, 30))] * 5 + \
        [("roof_settlement", "MI", date(2025, 8, 1), date(2026, 9, 30))] * 4 + \
        [("hail_cosmetic", "MN", date(2026, 1, 1), date(2026, 9, 30))] * 3 + \
        [("wind_hail", "IN", date(2025, 2, 1), date(2026, 9, 30))] * 3
    for n, (topic, st, lo, hi) in enumerate(plan, 1):
        while True:
            loss = lo + timedelta(days=r.randrange((hi - lo).days))
            fam = S.TOPIC_FAMILY[topic]
            eds = [e for e in S.EDITIONS[fam] if ed_window(fam, e)[0] <= loss <= ed_window(fam, e)[1]]
            if not eds:
                continue
            d = make_det(n, topic, r, state=st, loss=loss, ed=eds[-1])
            if S.CORPUS.by_id[d["gold_span_id"]].provenance.kind == "bulletin":
                break
        d["determination_id"] = f"PRB-{n:02d}"
        out.append(d)
    u = [round(r.uniform(0.02, 0.74), 3) for _ in range(3)] + [round(r.uniform(0.75, 0.95), 3) for _ in range(12)]
    r.shuffle(u)
    return out, u


if __name__ == "__main__":
    import sys
    ds = fit_post_cut(build_dets())
    H = sum(header(s) for s in S.CORPUS.spans)
    fit_questions(ds, 34_053.5 - 22_326 - H, 34_054.5 - 22_326 - H)
    CAL["determinations"] = ds
    CAL["sizes"] = fit_sizes(ds)
    print(check_tokens(ds))
    CAL["retrieval"] = build_retrieval(ds)
    for _ in range(3):                      # move unfiltered surplus from saturated rows to free ones
        S.CAL.update(CAL)
        pu = S.Pipeline(S.PipelineConfig())
        dets = [S.Determination(**d) for d in ds]
        got = {d.determination_id: pu.render(d).tokens for d in dets}
        tgt = getattr(build_retrieval, "U")
        adj = getattr(build_retrieval, "adj", {})
        for grp in (list(got)[:60], list(got)[60:]):
            surplus = sum(got[k] - tgt[k] + adj.get(k, 0) for k in grp)
            free = [k for k in grp if got[k] == tgt[k] and k not in PAIR]
            for n in range(abs(surplus)):
                k = free[n % len(free)]
                adj[k] = adj.get(k, 0) + (-1 if surplus > 0 else 1)
        build_retrieval.adj = adj
        CAL["retrieval"] = build_retrieval(ds, adj)
    json.dump(CAL, open("/tmp/ch03_stage1.json", "w"))
    CAL["probes"], CAL["probe_u"] = build_probes()
    S.CAL.update(CAL)
    CAL["misreads"] = fit_misreads(S.load_dets())
    json.dump(CAL, open(OUT, "w"), indent=0)
