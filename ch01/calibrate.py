"""Build Chapter 1's fixtures and fit them to the chapter's illustrative profile.

Run once (python calibrate.py, about a minute); lab.py only reads calibration.json.
The script is deterministic (seed 2026). It:
  1. generates 500 determinations, 13 claim records and the span sizes;
  2. allocates the profile's cases to determinations (which ones the stand-in
     misreads, which ones retrieval misses, which ones need a bigger budget, which
     ones have a look-alike that reads closer than the governing span, ...);
  3. turns each case into fixture details (coverage values, rank adjustments);
  4. picks a jitter seed per determination so that mean rendered tokens match;
  5. fits the model's length-noise slope so that assembly E's count matches.
Every case is checked by running the real pipeline and model, not assumed.
"""
from __future__ import annotations

import json, math, random, statistics
from datetime import date, timedelta

import standin as S

R = S.rng("calibrate")
corpus = S.build_corpus()
BY = {s.span_id: s for s in corpus}
BUDGETS = [150, 300, 450, 600, 900, 1200, 2000]

# ---------------------------------------------------------------- targets (the chapter's profile)
T_TOK = {"A": 109, "B": 848, "C": [420, 660, 896, 1133, 1579, 1908, 2137], "D": 1587}
N_E_WRONG = 79
CITIES = {"WI": ["Green Bay", "Madison", "Appleton"], "IL": ["Rockford", "Peoria", "Naperville"],
          "IA": ["Cedar Rapids", "Davenport", "Ames"], "MN": ["Duluth", "Rochester", "St. Cloud"],
          "MI": ["Grand Rapids", "Lansing", "Kalamazoo"], "OH": ["Toledo", "Dayton", "Akron"],
          "IN": ["Fort Wayne", "South Bend", "Evansville"]}
Q = {"sewer_backup": "Basement in {c} flooded when the city sewer backed up during heavy rain. Is the water damage covered?",
     "surface_water": "Spring melt in {c} sent surface water under the garage door. Is the damage covered?",
     "burst_pipe": "A supply pipe burst in an upstairs bathroom in {c} while the house was occupied. Is the damage covered?",
     "wind_hail": "Hail in {c} dented the siding and cracked two windows. Is the damage covered?",
     "fire": "A kitchen fire in {c} damaged cabinets and the ceiling. Is the damage covered?",
     "theft": "Tools were stolen from a locked shed in {c}. Is the theft covered?",
     "roof": "Wind lifted shingles on a 19-year-old roof in {c}. Is the roof covered, and at what value?",
     "mold": "Mold appeared behind a wall in {c} after a slow leak. Is the remediation covered?",
     "ice_dam": "An ice dam in {c} forced meltwater into a bedroom ceiling. Is the damage covered?",
     "liability": "A guest slipped on the front steps in {c} and is claiming medical costs. Is liability covered?",
     "collision": "The insured backed into a pillar in a parking garage in {c}. Is the collision damage covered?",
     "comprehensive": "A deer ran into the car near {c}. Is it covered under comprehensive?",
     "rental": "The car is in the shop in {c} after a covered loss. Is a rental car covered, and for how long?",
     "towing": "The car broke down outside {c} and was towed 30 miles. Is the towing covered?",
     "glass": "A stone cracked the windscreen on the highway near {c}. Is the glass covered without a deductible?",
     "uninsured_motorist": "An uninsured driver hit the insured's car in {c}. Is the injury claim covered?",
     "hull": "The boat's hull was holed on a submerged log near {c}. Is the hull damage covered?",
     "ice_damage": "Lake ice near {c} crushed the boat's hull at its winter mooring. Is the ice damage covered?",
     "navigation": "The boat was damaged on a trip outside the stated navigation area near {c}. Is it covered?"}
FIRST = ["Grace", "Tom", "Ana", "Wei", "Priya", "Samuel", "Ines", "Kofi", "Marta", "Omar", "Lena", "Diego",
         "Hannah", "Ravi", "Chloe", "Yusuf", "Elena", "Jonah", "Mei", "Felix"]
LAST = ["Murphy", "Novak", "Okafor", "Lindqvist", "Reyes", "Chen", "Patel", "Kowalski", "Haddad", "Brennan",
        "Silva", "Andersen", "Moreau", "Ibrahim", "Kaur", "Fischer", "Rossi", "Tanaka", "Walsh", "Duarte"]


def current(fam):
    f = S.FAMILIES[fam]
    return f"{f['form']} ({f['editions'][-1][0]})"


def end_labels(fam):
    return [(f"{c} ({eds[-1]})", t) for c, t, eds in S.FAMILIES[fam]["endorsements"]]


# ---------------------------------------------------------------- 1. determinations
def make_dets():
    plan = []                               # (family, governs, endorsed)
    for fam, n_end_gov, n_form_end, n_total in (("HO-3", 66, 109, 299), ("PAP", 20, 50, 140), ("BOAT", 10, 20, 60)):
        n_bull = 18 if fam == "HO-3" else 0
        plan += [(fam, "endorsement", True)] * n_end_gov
        plan += [(fam, "bulletin", i < 6) for i in range(n_bull)]
        plan += [(fam, "form", True)] * n_form_end
        plan += [(fam, "form", False)] * (n_total - n_end_gov - n_bull - n_form_end)
    R.shuffle(plan)
    dets = []
    for i, (fam, gov, endorsed) in enumerate(plan):
        did = f"DET-{i + 2:04d}"
        state = R.choice(S.STATES)
        form = current(fam)
        ends = end_labels(fam)
        if gov == "endorsement":
            if fam == "HO-3" and R.random() < 39 / 66:
                lab, topic = ends[0]
            else:
                lab, topic = R.choice(ends[1:] if fam == "HO-3" else ends)
            el = [lab]
            gold = f"{lab}|Coverage"
        elif gov == "bulletin":
            topic = R.choice(S.FAMILIES[fam]["bulletins"])
            el = []
            gold = f"Bulletin {state}-2025-{topic}|Guidance"
        else:
            avoid = {"wind_hail", "roof"} if fam == "HO-3" else set()
            el = []
            topic = R.choice([t for t in S.FAMILIES[fam]["topics"] if t not in avoid])
            if endorsed:
                pool = [l for l, t in ends if t != topic]
                el = R.sample(pool, 1 if R.random() < 0.7 or len(pool) < 2 else 2)
            effect = "exclusion" if topic in [t for _, t in ends] and not any(t == topic for l, t in ends if l in el) \
                and R.random() < 0.6 else R.choice(["grant", "exclusion"])
            gold = f"{form}|{topic}|{effect}"
        if gov == "bulletin" and endorsed:
            el = R.sample([l for l, t in ends if t != topic], 1)
        if gov == "endorsement" and R.random() < 0.3:
            el += R.sample([l for l, t in ends if l not in el], 1)
        start = date(2025, 3, 1) if gov == "bulletin" else date(2024, 7, 1)
        dol = start + timedelta(days=R.randrange((date(2026, 9, 30) - start).days))
        city = R.choice(CITIES[state])
        g = BY[gold]
        dets.append(dict(determination_id=did, holder=f"{R.choice(FIRST)} {R.choice(LAST)}", state=state,
                         family=fam, form=form, endorsements=sorted(el),
                         policy_number=f"LK-{fam[:2]}-{100000 + R.randrange(900000)}", date_of_loss=dol.isoformat(),
                         topic=topic, question=Q[topic].format(c=city), q_tokens=0, governs=gov, gold_span=gold,
                         gold=f"{g.provenance.label} -> {g.conclusion}"))
    carter = dict(determination_id="DET-0001", holder="Emily Carter", state="WI", family="HO-3",
                  form="HO-3 (2024-06)", endorsements=["HO 04 95 (2024-06)"], policy_number="LK-HO-482113",
                  date_of_loss="2026-04-18", topic="sewer_backup", question=Q["sewer_backup"].format(c="Green Bay"),
                  q_tokens=28, governs="endorsement", gold_span="HO 04 95 (2024-06)|Coverage",
                  gold="HO 04 95 (2024-06) -> covered up to $10,000")
    return [carter] + dets


DETS = make_dets()
assert len(DETS) == 500
assert sum(d["governs"] == "form" for d in DETS) == 385 and sum(d["governs"] == "endorsement" for d in DETS) == 97
assert sum(bool(d["endorsements"]) for d in DETS) == 282, sum(bool(d["endorsements"]) for d in DETS)

# question tokens: mean exactly 24 (claim header + question = 56)
qs = [28] + [R.randint(16, 32) for _ in range(499)]
diff = 500 * 24 - sum(qs)
i = 1
while diff:
    step = 1 if diff > 0 else -1
    if 14 <= qs[i] + step <= 34:
        qs[i] += step
        diff -= step
    i = i % 499 + 1
for d, q in zip(DETS, qs):
    d["q_tokens"] = q

# ---------------------------------------------------------------- 2. claim records
REC = [dict(claim_id="CLM-2023-0117", policy_number="LK-HO-482113", family="HO-3", topic="sewer_backup",
            text="water back-up loss, closed and paid $4,200", tokens=45, header_tokens=34)]
for d in R.sample(DETS[1:], 12):
    REC.append(dict(claim_id=f"CLM-{R.choice([2022, 2023, 2024])}-{R.randrange(1000, 9999)}",
                    policy_number=d["policy_number"], family=d["family"], topic=d["topic"],
                    text=f"prior {d['topic'].replace('_', ' ')} loss, closed and paid ${R.randrange(8, 90) * 100:,}",
                    tokens=R.randint(36, 58), header_tokens=34))

# ---------------------------------------------------------------- 3. sizes and provenance headers
C95 = [f"HO 04 95 (2024-06)|{x}" for x in ["Coverage", "Limit of Liability", "Exclusions", "Conditions", "Schedule", "Definitions"]]
H3 = "HO-3 (2024-06)"
CC = [f"{H3}|sewer_backup|exclusion", f"{H3}|sewer_backup|grant", f"{H3}|surface_water|exclusion",
      f"{H3}|burst_pipe|exclusion", f"{H3}|ice_dam|exclusion", f"{H3}|Definitions"]
CX = [f"{H3}|surface_water|grant", f"{H3}|burst_pipe|grant", f"{H3}|ice_dam|grant", f"{H3}|Conditions",
      f"{H3}|Duties After Loss", f"{H3}|Loss Settlement", f"{H3}|Other Insurance"]
CB = ["UW-01|sewer_backup", "HO 04 95 (2019-03)|Coverage", "HO-3 (2019-03)|sewer_backup|exclusion",
      "HO-3 (2022-06)|sewer_backup|exclusion", "BT 00 01 (2019-01)|ice_damage|exclusion",
      "BT 00 01 (2023-01)|ice_damage|exclusion"]
FIX_SIZE = dict(zip(C95, [78, 70, 68, 76, 66, 72])) | dict(zip(CC, [72, 66, 70, 64, 68, 69])) \
    | dict(zip(CX, [62, 63, 64, 65, 63, 64, 64])) | dict(zip(CB, [60, 64, 62, 58, 62, 62]))
FIX_HEAD = {k: 41 for k in C95} | {k: 40 for k in CC + CX} | dict(zip(CB, [30, 37, 36, 36, 36, 36]))
for k in list(FIX_SIZE):
    assert k in BY, k
assert sum(FIX_SIZE[k] for k in C95 + CC) + 45 == 884 and sum(FIX_SIZE[k] for k in CC + CX) + 45 == 899
BASE_HEAD = {S.Kind.FORM: 36, S.Kind.ENDORSEMENT: 37, S.Kind.BULLETIN: 35, S.Kind.GUIDANCE: 33, S.Kind.NOTE: 30}


import os


def make_sizes(sigma=0.42, gen_scale=1.25):
    r = S.rng("sizes", sigma, gen_scale)
    free = [s for s in corpus if s.span_id not in FIX_SIZE]
    raw = {s.span_id: r.lognormvariate(0, sigma) * (gen_scale if s.topic == "general" else 1.0)
           * (float(os.environ.get("NOTE", 1.6)) if s.kind in (S.Kind.NOTE, S.Kind.BULLETIN, S.Kind.GUIDANCE) else 1.0) for s in free}
    left = S.CORPUS_TOKENS - sum(FIX_SIZE.values()) - sum(x["tokens"] for x in REC)
    k = left / sum(raw.values())
    sizes = {sid: min(140, max(20, round(v * k))) if BY[sid].topic != "general" else max(20, round(v * k))
             for sid, v in raw.items()}
    gen = [sid for sid in sizes if BY[sid].topic == "general"]
    d = left - sum(sizes.values())
    j = 0
    while d:
        sid = gen[j % len(gen)]
        st = 1 if d > 0 else -1
        if sizes[sid] + st >= 20:
            sizes[sid] += st
            d -= st
        j += 1
    sizes.update(FIX_SIZE)
    heads = {s.span_id: FIX_HEAD.get(s.span_id, BASE_HEAD[s.kind]) for s in corpus}
    d = S.PROV_TOTAL - sum(heads.values()) - 34 * len(REC)
    adj = [s.span_id for s in corpus if s.span_id not in FIX_HEAD]
    j = 0
    while d:
        st = 1 if d > 0 else -1
        heads[adj[j % len(adj)]] += st
        d -= st
        j += 1
    assert sum(sizes.values()) + sum(x["tokens"] for x in REC) == S.CORPUS_TOKENS
    return sizes, heads


import os
SIZES, HEADS = make_sizes(float(os.environ.get('SIGMA', 0.55)), float(os.environ.get('GEN', 1.25)))
S.Retriever.ENRICH = float(os.environ.get('ENRICH', 0.13))


# ---------------------------------------------------------------- 4. allocate the profile's cases
def pick(pool, n, used, tag):
    pool = [d for d in pool if d["determination_id"] not in used]
    out = S.rng("pick", tag).sample(pool, n)
    used.update(d["determination_id"] for d in out)
    return [d["determination_id"] for d in out]


rest = DETS[1:]
E_FORM = [d for d in rest if d["governs"] == "form" and d["endorsements"]]
E_END = [d for d in rest if d["governs"] == "endorsement"]
used: set = set()
MIS = pick(E_FORM, 3, used, "mis-f") + pick(E_END, 2, used, "mis-e")
CMISS = pick(E_FORM, 1, used, "cm-f") + pick(E_END, 4, used, "cm-e")
NORULES = pick(E_END, 14, used, "nr")
c_right = [d for d in rest if d["determination_id"] not in set(MIS + CMISS)]
bused = set(NORULES)
LEVEL = {}
for lvl, n in ((300, 26), (450, 2), (600, 8), (900, 1)):
    for did in pick(c_right, n, bused, f"lvl{lvl}"):
        LEVEL[did] = lvl
cwrong = set(MIS + CMISS)
bm_used = set(cwrong)
BMISS = list(cwrong)
for gv, n in (("form", 39), ("endorsement", 11), ("bulletin", 5)):
    BMISS += pick([d for d in rest if d["governs"] == gv], n, bm_used, f"bm-{gv}")
SUP = pick([d for d in rest if d["determination_id"] in set(BMISS) - set(MIS)
            and (d["governs"] != "endorsement" or d["gold_span"].startswith("HO 04 95"))], 17, set(), "sup")
bhit = [d for d in rest if d["determination_id"] not in set(BMISS)]
TRAP = pick([d for d in bhit if d["governs"] == "form" and d["family"] == "HO-3"], 6, set(), "trap")
NH = set(TRAP) | set(pick(bhit, 259, set(TRAP), "nh"))
assert len(NH) == 265 and len(BMISS) == 65

BYD = {d["determination_id"]: d for d in DETS}


def det_obj(d):
    return S.det_from(d)


def named_in(det, s):
    return S.Packet(det, [], True).named(s)


from functools import lru_cache


def supcopies(det, g):
    return _sup(g.span_id)


@lru_cache(maxsize=None)
def _sup(gid):
    g = BY[gid]
    return [s for s in corpus if s.span_id != g.span_id and s.kind == g.kind and s.topic == g.topic
            and s.provenance.doc_id == g.provenance.doc_id and s.section == g.section]


def lookalike(det, g, allow_sup=True, cov=None):
    if allow_sup:
        sc = supcopies(det, g)
        if sc:
            return max(sc, key=lambda s: s.provenance.version)
    cand = [s for s in corpus if s.span_id != g.span_id and not named_in(det, s) and s not in supcopies(det, g)]
    return max(cand, key=lambda s: (S.default_cov(det, s, BY), s.span_id))


def fixture(d):
    did = d["determination_id"]
    det = det_obj(d)
    g = BY[det.gold_span]
    r = S.rng("fx", did)
    kw, cov, rel = {}, {}, {}
    if did in set(BMISS):
        kw[g.span_id] = -2.0
        same_named = [s.span_id for s in corpus if s.topic == det.topic and named_in(det, s) and s.span_id != g.span_id]
        for sid in same_named:
            kw[sid] = -2.0
        if did in set(SUP):
            kw[lookalike(det, g).span_id] = 2.0
        else:
            for s in supcopies(det, g):
                kw[s.span_id] = -2.0
            kw[lookalike(det, g, allow_sup=False).span_id] = 2.0
    else:
        kw[g.span_id] = 2.0
        if did in NH:
            L = lookalike(det, g)
            kw[L.span_id] = 2.0
            cov[L.span_id] = round(r.uniform(1.02, 1.25), 3)
        if did in set(TRAP):
            T = r.choice([s for s in corpus if s.kind == S.Kind.ENDORSEMENT and s.family == det.family
                          and s.section == "Coverage" and s.provenance.label not in det.endorsements])
            kw[T.span_id] = 2.0
            cov[T.span_id] = round(r.uniform(1.08, 1.22), 3)
    if did in set(NORULES):
        x = f"{det.form}|{det.topic}|exclusion"
        cov[x] = round(r.uniform(1.05, 1.15), 3)
        if did not in set(BMISS):
            kw.setdefault(x, -2.0)                # B has no rules either: keep the look-alike out of its eight
    if did in set(CMISS):
        rel[g.span_id] = -5.0
    return dict(seed=0, kw=kw, cov=cov, rel=rel)


for d in DETS[1:]:
    d["fx"] = fixture(d)

# Carter: fixed design (Figure 1.8)
CARTER_ORDER = [C95[0], CC[0], C95[1], CC[1], C95[2], CC[2], C95[3], CC[3], C95[4], CC[4], C95[5], CC[5]] + CX
DETS[0]["fx"] = dict(seed=0, kw={k: 2.0 for k in [C95[0], CC[0]] + CB}, cov={},
                     rel={k: 10.0 - 0.1 * i for i, k in enumerate(CARTER_ORDER)})

CAL = dict(sizes=SIZES, heads=HEADS, records=REC, dets=DETS, misread=MIS, model=dict(slope=0.0, dip=0.0))
W = S.build_world(CAL)
import lab as L
L.setup(W)
POL = S.Policy()


def kept_scores(det):
    hits = L.pipe.retriever.search(L.pipe.query(det), k=40)
    return [(s, v) for s, v in hits if POL.applies(s, det).ok]


def level_of(det):
    for b in BUDGETS:
        if any(s.span_id == det.gold_span for s in L.c_engineered(det, b)[0].spans):
            return b
    return None


# Carter: drop any small candidate that would squeeze into his C or D packet
c = W.dets[0]
fxc = W.fx["DET-0001"]
for _ in range(10):
    small = [s for s, _ in kept_scores(c) + kept_scores(c.model_copy(update={"endorsements": []}))
             if s.span_id not in CARTER_ORDER and s.tokens <= 61]
    if not small:
        break
    for s in small:
        fxc["rel"][s.span_id] = -5.0
pkC, pkD, pkB = L.c_engineered(c)[0], L.d_no_declarations(c)[0], L.b_top8(c)[0]
assert (pkB.tokens, pkC.tokens, len(pkC.spans), pkD.tokens, len(pkD.spans)) == (923, 1567, 13, 1616, 14), \
    (pkB.tokens, pkC.tokens, len(pkC.spans), pkD.tokens, len(pkD.spans))


def place_gold(det, fx):
    """Put the governing span where its case needs it in C's ranking (level = smallest budget that packs it)."""
    did = det.determination_id
    if did in set(CMISS):
        return True
    want = LEVEL.get(did, 150)
    gid = det.gold_span
    fx["rel"].pop(gid, None)
    fx["rel"][gid] = -9.0
    ks = [(s, v) for s, v in kept_scores(det)]
    for p in range(len(ks) + 1):
        hi = ks[p - 1][1] if p > 0 else ks[0][1] + 0.02 if ks else 1.0
        lo = ks[p][1] if p < len(ks) else hi - 0.02
        fx["rel"][gid] = (hi + lo) / 2
        if did in set(NORULES):
            fx["rel"][f"{det.form}|{det.topic}|exclusion"] = fx["rel"][gid] - 1e-4
        if level_of(det) == want:
            return True
    return False


def outcomes(det):
    """Run the real assemblies for one determination; return results and token sums."""
    out = {}
    jd = lambda pk: L.judge(det, pk)[0]
    out["B"] = jd(L.b_top8(det)[0])
    out["Bnh"] = jd(S.Assembler().render(L.b_top8(det)[0], ""))
    out["Br"] = jd(L.b_top8(det, rules=True)[0])
    for b in BUDGETS:
        out[f"C{b}"] = jd(L.c_engineered(det, b)[0])
    pk = L.c_engineered(det)[0]
    pk.rules = False
    out["Cnr"] = jd(pk)
    out["D"] = jd(L.d_no_declarations(det)[0])
    toks = [L.b_top8(det)[0].tokens] + [L.c_engineered(det, b)[0].tokens for b in BUDGETS] \
        + [L.d_no_declarations(det)[0].tokens]
    return out, toks


def expected(det):
    did = det.determination_id
    ok = lambda x: "governing" if x else "wrong"
    bhit_ = did not in set(BMISS)
    e = {"B": ok(bhit_), "Bnh": ok(bhit_ and did not in NH), "Br": ok(bhit_ and did not in set(TRAP))}
    cok = did not in set(MIS + CMISS)
    for b in BUDGETS:
        e[f"C{b}"] = ok(cok and LEVEL.get(did, 150) <= b)
    e["Cnr"] = ok(cok and did not in set(NORULES))
    e["D"] = ok(cok and det.governs != "endorsement")
    return e


if os.environ.get("QUICK"):
    import numpy as np
    acc = []
    for det in W.dets[1:]:
        fx = W.fx[det.determination_id]
        fx["seed"] = 0
        place_gold(det, fx)
        acc.append([L.b_top8(det)[0].tokens] + [L.c_engineered(det, b)[0].tokens for b in BUDGETS]
                   + [L.d_no_declarations(det)[0].tokens])
    m = np.mean(acc, 0)
    tg = np.array([T_TOK["B"]] + T_TOK["C"] + [T_TOK["D"]])
    print("QUICK", os.environ.get('SIGMA'), os.environ.get('GEN'), os.environ.get('ENRICH'), np.round(m - tg).astype(int).tolist())
    raise SystemExit
if os.environ.get("DEBUG_IDS"):
    for det in W.dets:
        if det.determination_id in os.environ["DEBUG_IDS"].split(","):
            fx = W.fx[det.determination_id]
            print(det.determination_id, det.governs, {k: v for k, v in fx.items() if k not in ("cov", "rel", "kw")}, "kw", fx.get("kw"))
            print("  expected", expected(det))
            for seed in range(6):
                fx["seed"] = seed
                pg = place_gold(det, fx)
                print("  seed", seed, "placed", pg, "got", outcomes(det)[0] if pg else None)
    raise SystemExit
K = int(__import__("os").environ.get("K", 24))
TABLE = {}
bad = []
for det in W.dets[1:int(__import__("os").environ.get("N", 500))]:
    did = det.determination_id
    fx = W.fx[did]
    opts = []
    for seed in range(K):
        fx["seed"] = seed
        if not place_gold(det, fx):
            continue
        o, t = outcomes(det)
        if o == expected(det):
            opts.append((seed, dict(fx["rel"]), t))
    extra = K
    while not opts and extra < K + 600:            # a few determinations need a longer search
        fx["seed"] = extra
        extra += 1
        if place_gold(det, fx):
            o, t = outcomes(det)
            if o == expected(det):
                opts.append((fx["seed"], dict(fx["rel"]), t))
    if not opts:
        bad.append(did)
    TABLE[did] = opts
print("determinations with no valid seed:", bad)
assert not bad, bad

# ---------------------------------------------------------------- 5. pick seeds so the mean tokens match
import numpy as np
dids = [d.determination_id for d in W.dets[1:]]
carter_t = np.array([L.b_top8(c)[0].tokens] + [L.c_engineered(c, b)[0].tokens for b in BUDGETS]
                    + [L.d_no_declarations(c)[0].tokens], float)
target = np.array([T_TOK["B"]] + T_TOK["C"] + [T_TOK["D"]], float) * 500
M = {did: np.array([t for _, _, t in TABLE[did]], float) for did in dids if TABLE[did]}
choice = {did: 0 for did in M}
cur = carter_t + sum(M[did][0] for did in M)
print("mean tokens with seed 0:", np.round(cur / 500, 1).tolist())
print("targets:               ", (target / 500).tolist())
rs = S.rng("search")
for it in range(60):
    order = list(M)
    rs.shuffle(order)
    for did in order:
        base = cur - M[did][choice[did]]
        errs = ((base[None, :] + M[did] - target[None, :]) ** 2).sum(1)
        k = int(errs.argmin())
        cur = base + M[did][k]
        choice[did] = k
    if np.all(np.abs(cur - target) < 200):
        break
print("after search:", np.round(cur / 500, 2).tolist(), "iterations", it + 1)
assert np.all(np.abs(cur - target) < 200), cur - target
for did in M:
    seed, rel, _ = TABLE[did][choice[did]]
    W.fx[did]["seed"], W.fx[did]["rel"] = seed, rel
for d in DETS:
    d["fx"] = W.fx[d["determination_id"]]

# ---------------------------------------------------------------- 6. fit E's length-noise slope
E_PK = {det.determination_id: L.e_everything(det)[0] for det in W.dets}


def e_wrong(slope, dip):
    W.model.slope, W.model.dip = slope, dip
    return [det.determination_id for det in W.dets if L.judge(det, E_PK[det.determination_id])[0] != "governing"]


DIP = 0.12
lo, hi = 0.0, 0.05
for _ in range(60):
    mid = (lo + hi) / 2
    n = len(e_wrong(mid, DIP))
    if n < N_E_WRONG:
        lo = mid
    elif n > N_E_WRONG:
        hi = mid
    else:
        break
slope = round(mid, 6)
ew = e_wrong(slope, DIP)
print("E wrong", len(ew), "slope", slope, "Carter right in E:", "DET-0001" not in ew)
assert len(ew) == N_E_WRONG and "DET-0001" not in ew
CAL["model"] = dict(slope=slope, dip=DIP)
CAL["enrich"] = S.Retriever.ENRICH
S.CAL_PATH.write_text(json.dumps(CAL, indent=1, default=str))
print("wrote", S.CAL_PATH)
