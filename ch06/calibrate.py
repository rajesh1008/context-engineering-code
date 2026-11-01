"""Build calibration.json for Chapter 6.

The determinations start from Part I's shared set (04-code/ch08/calibration.json, built by
ch08/calibrate.py), so every claim the chapters name has the same facts. Chapter 6 then
makes a few documented adjustments to claims no chapter names:
  - five evaluation claims are retopicked to the three homeowners topics no skill covers in
    the chapter's account (dog bite 2, wear and tear 2, snow load collapse 1);
  - DET-0042's policy is on PAP (2018-09), as Listing 6.4 shows;
  - in the hold-out (claims 201-500): the sources of Examples A and B get the facts the
    examples state; 28 deer-swerve claims (4 bulletin-governed) and 25 roof-matching claims
    (6 governed by the Wisconsin bulletin) form the two injection slices; and the hold-out
    has 66 bulletin- or endorsement-governed claims, the release gate's hard set.
Model fixtures (which claims each stand-in gets wrong without a rule, resistance to an
example's pull, unsupported additions) and the pool of 34 reviewed examples are chosen
here, with assertions on the facts they rely on. The retrieved-passage length is fitted so
that the mean evidence per evaluation claim is 1,075.6 tokens.
Run: python3 calibrate.py   (a few seconds; rewrites calibration.json)
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = json.loads((HERE.parent / "ch08" / "calibration.json").read_text())
rng = random.Random(606)
E_TARGET_SUM = 215_120          # 1,075.6 x 200
LOW, WEAK = 0.30, 0.10      # resistance to an example's pull (imitation is 0.45)

dets = {d["id"]: dict(d) for d in SHARED["determinations"]}
D = lambda n: dets["DET-%04d" % n]  # noqa: E731

# ---------------------------------------------------------------- world adjustments
D(42)["form_version"] = "2018-09"
for n, topic in [(56, "dog_bite"), (112, "dog_bite"), (11, "wear_tear"), (29, "wear_tear"), (10, "snow_load")]:
    d = D(n)
    d.update(topic=topic, form="HO-3", form_version=d["form_version"] if d["form"] == "HO-3" else "2022-01",
             endorsements=[])
D(236).update(name=D(236)["name"], state="IL", topic="frozen_pipes", form="HO-3", form_version="2024-06",
              endorsements=[["HO 04 90", "2022-01"]], loss="2024-11-25")
D(288).update(state="MN", topic="water_backup", form="HO-3", form_version="2024-06",
              endorsements=[["HO 04 95", "2024-06"]], loss="2025-06-19")

NAMED = {int(x) for x in (
    "1 2 3 4 5 6 8 9 12 14 16 21 22 26 27 30 31 37 42 46 47 48 49 53 54 58 61 63 68 69 70 71 74 75 81 83 "
    "94 97 113 116 118 121 125 128 131 133 134 142 143 145 156 160 163 164 169 172 176 178 181 183 184 185 "
    "186 188 189 191 194 198 210 217 236 252 259 279 288 307 326 393 415 483").split()}
BULL = {"WI-OCI-2023-07": ("WI", "roof_matching", date(2023, 7, 1)),
        "WI-OCI-2025-03": ("WI", "roof_matching", date(2025, 3, 15)),
        "WI-OCI-2024-11": ("WI", "deer_swerve", date(2024, 11, 1)),
        "IA-ID-2024-08": ("IA", "tree_debris", date(2024, 8, 15)),
        "IL-DOI-2025-05": ("IL", "frozen_pipes", date(2025, 5, 1)),
        "MN-COMM-2026-01": ("MN", "hail_cosmetic", date(2026, 1, 1))}
ETOPIC = {"HO 04 95": "water_backup", "HO 04 98": "food_spoilage", "HO 04 54": "earth_movement",
          "HO 04 90": "personal_property_rc", "LM HO 17": "roof_settlement", "LM PP 21": "rental_reimbursement",
          "LM PP 34": "rideshare", "LM BP 12": "spoilage", "BP 04 50": "employee_dishonesty", "BT 05": "navigation_limits"}


def gov(d) -> tuple[str, str]:
    loss = date.fromisoformat(d["loss"])
    live = [(eff, b) for b, (st, tp, eff) in BULL.items() if st == d["state"] and tp == d["topic"] and eff <= loss]
    if live:
        return max(live)[1], "bulletin"
    for e, v in d["endorsements"]:
        if ETOPIC.get(e) == d["topic"]:
            return e, "endorsement"
    return d["form"], "form"


hold = lambda: [d for d in dets.values() if int(d["id"][4:]) > 200]  # noqa: E731
free = lambda d: int(d["id"][4:]) not in NAMED  # noqa: E731

# ---------------------------------------------------------------- example pool (part D)
# 34 reviewed examples drawn from the hold-out. Each pool determination shares its facts
# (topic, state, form edition, endorsements) with an evaluation claim; its date of loss is
# its own. The search below fixes which facts the pool covers, deterministically.
ev = sorted((d for d in dets.values() if int(d["id"][4:]) <= 200), key=lambda d: d["id"])
eval_topics = sorted({d["topic"] for d in ev})
LAST = date(2026, 9, 30)


def facts(d):
    return (d["topic"], d["state"], d["form"], d["form_version"], tuple(sorted(map(tuple, d["endorsements"]))))


def nearest(d, cands):
    return min(cands, key=lambda p: (abs((date.fromisoformat(p["loss"]) - date.fromisoformat(d["loss"])).days),
                                     p["id"]))


def select_same_topic(d, pool):
    return nearest(d, [p for p in pool if p["topic"] == d["topic"]])


def select_facts(d, pool):
    same = [p for p in pool if facts(p) == facts(d)]
    return nearest(d, same) if same else None


def clone(src, delta, new_id="POOL"):
    c = dict(src)
    c["id"] = new_id
    c["loss"] = str(date.fromisoformat(src["loss"]) + timedelta(days=delta))
    return c


def pool_score(sel):
    pool = [clone(dets[i], dl, f"P{k:02d}") for k, (i, dl) in enumerate(sel)]
    if any(date.fromisoformat(p["loss"]) > LAST for p in pool):
        return (99, 0, 0, {})
    mism, roofbad = defaultdict(int), 0
    fx = {facts(p) for p in pool}
    for d in ev:
        p = select_same_topic(d, pool)
        if gov(p)[0] != gov(d)[0]:
            mism[d["topic"]] += 1
            if d["topic"] == "roof_matching" and not (gov(d)[1] == "bulletin" and gov(p)[1] == "form"):
                roofbad += 1
    matched = sum(facts(d) in fx for d in ev)
    m = sum(mism.values())
    pen = (abs(m - 47) + abs(matched - 65) + 3 * roofbad + 2 * abs(mism["roof_matching"] - 3)
           + 3 * abs(mism["rental_reimbursement"] - 9) + 3 * abs(mism["tree_debris"] - 7)
           + 3 * (mism["frozen_pipes"] > 1))
    return pen, m, matched, dict(mism)


by_topic_ev = defaultdict(list)
for d in ev:
    by_topic_ev[d["topic"]].append(d["id"])
iowa = [d["id"] for d in ev if gov(d)[0] == "IA-ID-2024-08"]
sel = [(iowa[0], 1)] + [(rng.choice(by_topic_ev[t]), 1) for t in eval_topics if t != "tree_debris"]
cands = [d["id"] for d in ev if d["topic"] != "tree_debris"]
while len(sel) < 34:
    c = rng.choice(cands)
    if c not in [x for x, _ in sel]:
        sel.append((c, 1))
cur, T = pool_score(sel), 2.0
for it in range(60000):
    if cur[0] == 0:
        break
    new = list(sel)
    k = rng.randrange(1, 34)
    if rng.random() < 0.3:
        new[k] = (new[k][0], rng.choice([1, -1, 30, -30, 90, -90, 200, -200, 400, -400]))
    else:
        c = rng.choice(cands)
        if c in [x for x, _ in new]:
            continue
        new[k] = (c, 1)
    if not all(any(dets[i]["topic"] == tp for i, _ in new) for tp in eval_topics):
        continue
    nc = pool_score(new)
    if nc[0] <= cur[0] or rng.random() < math.exp(-(nc[0] - cur[0]) / T):
        sel, cur = new, nc
    T = max(0.05, T * 0.9997)
assert cur[0] == 0, cur
print("pool search:", it, "steps; mismatched", cur[1], "facts-matched", cur[2], cur[3])

# give 34 unnamed hold-out determinations the pool facts
sources = [d for d in hold() if free(d) and d["id"] not in ("DET-0236", "DET-0288")
           and d["topic"] not in ("deer_swerve", "roof_matching", "water_backup")
           and gov(d)[1] == "form"]
POOL = []
used = set()
for (i, dl), s in zip(sel, sources[::3]):
    c = clone(dets[i], dl, s["id"])
    c["name"] = s["name"]
    dets[s["id"]] = c
    POOL.append(s["id"])
    used.add(s["id"])
POOL.sort()

# hold-out claims for the topics no skill covers, so the example pool can cover them
spare = [d for d in hold() if free(d) and d["id"] not in used and d["topic"] in ("flood", "mold", "wind_hail", "ice_dam")
         and gov(d)[1] == "form"]
for d, topic in zip(spare[:4], ["dog_bite", "dog_bite", "wear_tear", "snow_load"]):
    d.update(topic=topic, endorsements=[])
used |= {d["id"] for d in spare[:4]}

# deer-swerve slice: 28 hold-out claims, 4 governed by the Wisconsin bulletin
deer = [d for d in hold() if d["topic"] == "deer_swerve"]
cands = [d for d in hold() if free(d) and d["id"] not in used and d["form"] == "PAP"
         and d["topic"] in ("deer_collision", "rental_reimbursement", "rideshare") and gov(d)[1] == "form"]
for d in cands[:28 - len(deer)]:
    d.update(topic="deer_swerve", state=d["state"] if d["state"] != "WI" else "MN")
    used.add(d["id"])
deer = [d for d in hold() if d["topic"] == "deer_swerve"]
assert len(deer) == 28 and sum(gov(d)[1] == "bulletin" for d in deer) == 4

# roof-matching slice: 25 hold-out claims, 6 governed by the Wisconsin bulletin of March 2025
roof = [d for d in hold() if d["topic"] == "roof_matching"]
cands = [d for d in hold() if free(d) and d["id"] not in used and d["form"] == "HO-3"
         and d["topic"] in ("flood", "mold", "hail_cosmetic", "wind_hail", "ice_dam") and gov(d)[1] == "form"]
need_b = 6 - sum(gov(d)[0] == "WI-OCI-2025-03" for d in roof)
for i, d in enumerate(cands[:25 - len(roof)]):
    if i < need_b:
        d.update(topic="roof_matching", state="WI", loss=f"2025-{4 + i:02d}-{10 + 3 * i:02d}")
    else:
        d.update(topic="roof_matching", state=d["state"] if d["state"] != "WI" else "MI")
    used.add(d["id"])
roof = [d for d in hold() if d["topic"] == "roof_matching"]
assert len(roof) == 25 and sum(gov(d)[0] == "WI-OCI-2025-03" for d in roof) == 6, \
    (len(roof), [gov(d) for d in roof])

# the gate's hard set: 66 hold-out claims governed by a bulletin or an endorsement
hard = [d for d in hold() if gov(d)[1] != "form"]
drop = [d for d in hard if free(d) and d["id"] not in used and gov(d)[1] == "endorsement"
        and d["topic"] not in ("water_backup", "deer_swerve", "roof_matching")]
for d in drop[:len(hard) - 66]:
    d["endorsements"] = [e for e in d["endorsements"] if ETOPIC.get(e[0]) != d["topic"]]
hard = [d for d in hold() if gov(d)[1] != "form"]
assert len(hard) == 66, len(hard)

evb = [d["id"] for d in ev if gov(d)[1] == "bulletin"]
print("eval bulletin-governed:", len(evb), evb)

# ---------------------------------------------------------------- retrieval misses
# The governing section was not retrieved; the packet had a neighbouring section of the form.
RETRIEVAL_MISS = {
    "DET-0022": ["Conditions 4", 13], "DET-0063": ["Conditions 4", 13], "DET-0163": ["Conditions 1", 12],
    "DET-0178": ["Conditions 1", 12], "DET-0181": ["Conditions 11", 15]}
ho_form = [d for d in hold() if gov(d)[1] == "form" and free(d) and d["id"] not in used
           and d["form"] == "HO-3" and d["topic"] in ("frozen_pipes", "wind_hail")]
for d in ho_form[:2]:
    RETRIEVAL_MISS[d["id"]] = ["Conditions 4", 13]
    used.add(d["id"])

# ---------------------------------------------------------------- model fixtures
best = [dets[i] for i in POOL]
pool_sel = {d["id"]: select_same_topic(d, best) for d in ev}
pool_fx = {d["id"]: select_facts(d, best) for d in ev}
FIXED_EX_TOPICS = ("frozen_pipes", "water_backup", "roof_matching")
unnamed = lambda i: int(i[4:]) not in NAMED  # noqa: E731

# X, Y: water back-up claims governed by HO 04 95 that Example B fixes
water_e = [d["id"] for d in ev if gov(d)[0] == "HO 04 95" and unnamed(d["id"])]
# (DET-0071 follows a correct example in part D, so it is right with a facts-matched example when
# one exists; the large model's facts-matched errors are then the five retrieval misses, X and Y.)
if pool_fx["DET-0071"] is None:
    X = [i for i in water_e if pool_fx[i] is not None][0]
    Y = [i for i in water_e if pool_fx[i] is None][0]
else:
    X, Y = [i for i in water_e if pool_fx[i] is None][:2]
# W1, W2: weak pull towards the form when examples are present without the precedence rule
W1, W2 = [d["id"] for d in ev if gov(d)[1] == "endorsement" and unnamed(d["id"])
          and d["topic"] not in FIXED_EX_TOPICS + ("deer_swerve",) and pool_fx[d["id"]] is None
          and gov(pool_sel[d["id"]])[0] == gov(d)[0]][:2]
R2 = "DET-0156"
assert gov(dets[R2])[0] == gov(dets["DET-0074"])[0] == "WI-OCI-2025-03"

roof_b = sorted(d["id"] for d in roof if gov(d)[0] == "WI-OCI-2025-03")
deer_b = sorted(d["id"] for d in deer if gov(d)[1] == "bulletin")
water_e_ho = [d["id"] for d in hard if gov(d)[0] == "HO 04 95" and d["id"] != "DET-0288" and d["id"] not in POOL]
other_hard = [d["id"] for d in hard if d["topic"] not in ("water_backup", "roof_matching", "deer_swerve",
                                                         "frozen_pipes")]

# D-mode pulls: mismatched claims on topics without a fixed example
mism_free = sorted(i for i, p in pool_sel.items()
                   if gov(p)[0] != gov(dets[i])[0] and dets[i]["topic"] not in FIXED_EX_TOPICS + ("deer_swerve",)
                   and unnamed(i) and i not in RETRIEVAL_MISS)
print("mismatched, free topics:", len(mism_free))


def models():
    L = dict(noprec=["DET-0071", X, Y, "DET-0049", "DET-0145", "DET-0185", "DET-0109"],
             prec=["DET-0071", X, Y],
             resistance={"DET-0061": LOW, X: LOW, Y: LOW, "DET-0074": LOW, R2: LOW, W1: WEAK, W2: WEAK})
    S = dict(noprec=[], prec=[], noskill=["DET-0113"],
             resistance={"DET-0061": LOW, "DET-0074": LOW, R2: LOW})
    M = dict(noprec=[], prec=[], resistance={"DET-0061": LOW})
    return L, S, M


L, S, M = models()
# hold-out: roof slice (v2.1) and deer slice (bad Example D)
for i in roof_b[:4]:
    L["resistance"][i] = LOW
for i in roof_b[:5]:
    S["resistance"][i] = LOW
for i in deer_b:
    L["resistance"][i] = LOW
    S["resistance"][i] = LOW
# hold-out misses without the precedence rule (v1.0, v2.3); two are water claims Example B fixes
L["noprec"] += water_e_ho[:2] + other_hard[:8]
for i in water_e_ho[:2]:
    L["resistance"][i] = LOW
S["noprec"] += other_hard[8:20]
S["prec"] += other_hard[8:10]
open_eval = [d["id"] for d in ev if gov(d)[1] != "form" and unnamed(d["id"])
             and d["topic"] not in FIXED_EX_TOPICS + ("deer_swerve",) and d["id"] not in (W1, W2, "DET-0109")]
rng.shuffle(open_eval)
P1, P2 = None, None
for i in open_eval:  # P1: a small-model precedence miss that a facts-matched example fixes
    if pool_fx[i] is not None and gov(pool_fx[i])[0] == gov(dets[i])[0] and P1 is None:
        P1 = i
P2 = [i for i in open_eval if i != P1 and pool_fx[i] is None][0]
S["prec"] += [P1, P2]
S["resistance"][P1] = LOW
water_e_ev = [i for i in water_e if i not in (X, Y)]
rest = [i for i in open_eval if i not in (P1, P2)]
assert len(rest) >= 7, rest
S["noprec"] += [P1, P2, water_e_ev[0]] + rest[:7]
S["resistance"][water_e_ev[0]] = LOW
M["prec"] = [P1, P2, "DET-0071"]
M["noprec"] = M["prec"] + rest[:7]
print("P1, P2:", P1, P2)

# filter off: misled by an inapplicable document
pool_ok = [d["id"] for d in ev if d["id"] not in RETRIEVAL_MISS and unnamed(d["id"]) and d["id"] not in mism_free
           and d["id"] not in set(L["noprec"]) | set(S["noprec"]) | set(L["resistance"]) | set(S["resistance"])]
rng.shuffle(pool_ok)
L["offapp_norule"] = pool_ok[:49]
L["offapp_rule"] = pool_ok[:9]
S["offapp_norule"] = pool_ok[49:79]
S["offapp_rule"] = pool_ok[49:51]
M["offapp_norule"], M["offapp_rule"] = [], []

ADD = ["Coverage is limited to losses reported within 30 days.",
       "A separate $2,500 deductible also applies under the declarations.",
       "The insured must also provide receipts within 60 days.",
       "Payment is limited to $5,000 for this type of loss.",
       "A separate $1,000 deductible applies to this coverage.",
       "The claim must be reported to the police within 24 hours.",
       "This coverage ends after the policy's first renewal."]
S_ADD_KEEP = ["DET-0012", "DET-0046", "DET-0061", "DET-0128", "DET-0191"]
others = [d["id"] for d in ev if int(d["id"][4:]) not in NAMED and d["id"] not in RETRIEVAL_MISS]
rng.shuffle(others)
S["additions"] = {"DET-0042": [ADD[0], 0.40]}
S["additions"]["DET-0012"] = [ADD[1], 0.90]
for i in S_ADD_KEEP[1:]:
    S["additions"][i] = [ADD[2 + len(S["additions"]) % 5], 0.85]
for i in others[:12]:
    S["additions"][i] = [ADD[2 + len(S["additions"]) % 5], 0.30]
L["additions"] = {i: [ADD[2 + k % 5], 0.90 if k == 0 else 0.35] for k, i in enumerate(others[12:18])}
M_KEEP = ["The insured must also provide receipts and photographs of the damaged property within 60 days.",
          "Payment for this type of loss is limited to $5,000 unless the declarations show a higher limit."]
M["additions"] = {i: [M_KEEP[k] if k < 2 else ADD[2 + k % 5], 0.80 if k < 2 else 0.35]
                  for k, i in enumerate(others[18:23])}


# ---------------------------------------------------------------- evidence tokens
evt = dict(SHARED["evidence_tokens"])
cal = {"note": ("Chapter 6 stand-in calibration: determinations (from Part I's shared set, with the "
                "adjustments in calibrate.py), model fixtures and the example pool. Illustrative."),
       "determinations": sorted(dets.values(), key=lambda d: d["id"]),
       "retrieval_miss": RETRIEVAL_MISS,
       "pool": POOL,
       "models": {"sim-large": L, "sim-mid": M, "sim-small": S},
       "evidence_tokens": evt}
(HERE / "calibration.json").write_text(json.dumps(cal, indent=1))

# fit passages so that labels + passages average 1,075.6 tokens over the evaluation set
import importlib  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
import standin  # noqa: E402
importlib.reload(standin)
from contextlab.instructions import SKILLS, V10, V12  # noqa: E402
p = standin.Pipeline(standin.PipelineConfig(instructions=V10))
cur = 0
labels = {}
for d in standin.CORPUS.determinations:
    pk = p.packet(d)
    labels[d.determination_id] = sum(s.tokens for s in pk.spans if s.meta.get("kind") == "evidence")
for d in standin.CORPUS.determinations:
    evt[d.determination_id] = max(200, evt[d.determination_id])
tot = sum(labels[d["id"]] + evt[d["id"]] for d in ev)
shift = (E_TARGET_SUM - tot) // 200
for d in dets.values():
    evt[d["id"]] += shift
tot = sum(labels[d["id"]] + evt[d["id"]] for d in ev)
for d in sorted(ev, key=lambda d: d["id"])[:E_TARGET_SUM - tot]:
    evt[d["id"]] += 1
assert sum(labels[d["id"]] + evt[d["id"]] for d in ev) == E_TARGET_SUM
cal["evidence_tokens"] = evt
(HERE / "calibration.json").write_text(json.dumps(cal, indent=1))

# part D, same-topic selection: the claims that follow a mismatched example rather than the
# documents. DET-0071, which the large model misreads without an example, follows a correct one.
L["resistance"]["DET-0071"] = LOW
order = list(mism_free)
rng.shuffle(order)
TARGET_SELECTED = {"sim-large": 25, "sim-small": 27}
for name, prof in (("sim-large", L), ("sim-small", S)):
    for i in [None] + order:
        if i is not None:
            prof["resistance"][i] = LOW
        (HERE / "calibration.json").write_text(json.dumps(cal, indent=1))
        importlib.reload(standin)
        exs = standin.pool()
        sel = {d.determination_id: standin.select_same_topic(d, exs) for d in standin.CORPUS.eval}
        t = standin.run_set(standin.PipelineConfig(model=name, instructions=V12.bump("v1.2+sk", skills=SKILLS),
                                                   examples=sel), standin.CORPUS.eval)
        if standin.wrong(t) >= TARGET_SELECTED[name]:
            break
    assert standin.wrong(t) == TARGET_SELECTED[name], (name, standin.wrong(t))
print("wrote calibration.json:", len(dets), "determinations")
