"""Build calibration.json for Chapter 7.

Starts from Chapter 6's calibration (../ch06/calibration.json, built by ch06/calibrate.py), so
seed 2026 is Chapter 6's run, and records every change. Each change touches only claims and
fixtures that no Chapter 6 output shows, and the test suite re-checks Chapter 6's printed
results on this world (test_ch07.py::test_chapter6_unchanged).

World adjustments (hold-out claims unless stated):
  - DET-0142 (evaluation set) carries the rideshare endorsement LM PP 34 and its loss date
    moves to 2023-02-27, so the precedence probes number 115;
  - DET-0387 and DET-0402 become earth-movement claims under HO 04 54, and DET-0417 a food
    spoilage claim under HO 04 98, so the roof probes number 15 and the water probes 20;
  - eight form-governed water claims and four form-governed roof claims are retopicked to
    homeowners topics with no endorsement on the policy (77 water claims in 1-500);
  - form editions of some unnamed hold-out claims change, so the applicability probes number
    208 (36 on roof topics, 40 on water topics).
Model fixtures:
  - sim-large: DET-0307 (Example C's source) instead of DET-0282 resists the bad example
    weakly, and DET-0282 instead of DET-0324 is misread without the precedence rule;
    DET-0219 falls back to the form when no skill is loaded (as sim-small's DET-0113 does);
  - sim-mid and sim-small: DET-0111 instead of DET-0045 is misread with the precedence rule;
  - skill_pull: claims a skill preferring the form tips over once precedence is dropped;
  - draws: the recorded clash draws, per model, skill, seed and claim, for the claims where a
    draw changes an answer (elsewhere a hash of the four is used, and no answer depends on it);
  - seed_additions: unsupported additions on water claims at seeds 2027-2035, and hold-out
    additions at seed 2026;
  - gold_rank: the governing span's rank in the retriever's list, for hit@5;
  - hold-out passage lengths, fitted to a mean of 1,522 prompt tokens under v2.2, and
    DET-0150's passages 8 tokens longer than in Chapter 6 (v3.2's evaluation mean is 1,549.5).
The draws and additions are not sampled: they are recorded values, chosen so the lab
reproduces the chapter's counts. They are illustrative, not measurements.
Run: python3 calibrate.py   (a few seconds; rewrites calibration.json)
"""
from __future__ import annotations

import copy
import importlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
BASE = json.loads((HERE.parent / "ch06" / "calibration.json").read_text())
rng = random.Random(707)
cal = copy.deepcopy(BASE)
dets = {d["id"]: d for d in cal["determinations"]}
D = lambda n: dets["DET-%04d" % n]  # noqa: E731
ROOF = ("roof_matching", "roof_settlement", "hail_cosmetic", "wind_hail", "ice_dam")
WATER = ("water_backup", "frozen_pipes", "flood", "mold")
NEWEST = {"HO-3": "2024-06", "PAP": "2023-01", "BOAT": "2025-03", "BOP": "2025-01"}
ENDT = {"water_backup": "HO 04 95", "food_spoilage": "HO 04 98", "earth_movement": "HO 04 54",
        "personal_property_rc": "HO 04 90"}

# ---------------------------------------------------------------- world adjustments
D(142)["endorsements"] = [["LM PP 34", "2023-01"]]
D(142)["loss"] = "2023-02-27"
for n in (387, 402):
    D(n)["topic"] = "earth_movement"
    D(n)["endorsements"] = [e for e in D(n)["endorsements"] if e[0] not in ("LM HO 17", "HO 04 54")] + \
        [["HO 04 54", "2022-01"]]
D(417)["topic"] = "food_spoilage"
D(417)["endorsements"] = [["HO 04 98", "2024-06"]]

# claims no Chapter 6 output depends on
models = cal["models"]
used = set(cal["retrieval_miss"]) | set(cal["pool"]) | {"DET-0236", "DET-0288", "DET-0307"}
for prof in models.values():
    for v in prof.values():
        used |= set(v)
NAMED = {"DET-0259", "DET-0116", "DET-0188", "DET-0217", "DET-0387", "DET-0402", "DET-0417"}
used |= NAMED


def form_governed(d) -> bool:
    if d["topic"] in ("roof_matching", "deer_swerve", "tree_debris", "frozen_pipes", "hail_cosmetic"):
        return False      # a bulletin may govern; leave these alone
    if d["topic"] == "roof_settlement":
        return not any(e[0] == "LM HO 17" for e in d["endorsements"])
    return not any(e[0] == ENDT.get(d["topic"]) for e in d["endorsements"])


free = [d for d in sorted(dets.values(), key=lambda d: d["id"])
        if int(d["id"][4:]) > 200 and d["id"] not in used and d["form"] == "HO-3" and form_governed(d)]


def retopic(d, targets):
    for t in targets:
        if not any(e[0] == ENDT.get(t) for e in d["endorsements"]):
            d["topic"] = t
            return
    raise AssertionError(d["id"])


OTHER = ["personal_property_rc", "food_spoilage", "earth_movement", "dog_bite", "wear_tear", "snow_load"]
fw = [d for d in free if d["topic"] in WATER]
fr = [d for d in free if d["topic"] in ROOF]
fw.sort(key=lambda d: (d["form_version"] == NEWEST["HO-3"], d["id"]))     # older editions first
fr.sort(key=lambda d: (d["form_version"] == NEWEST["HO-3"], d["id"]))
for k, d in enumerate(fw[:8]):
    retopic(d, OTHER[k % len(OTHER):] + OTHER)
for k, d in enumerate(fr[:4]):
    retopic(d, OTHER[(k + 3) % len(OTHER):] + OTHER)
changed = {d["id"] for d in fw[:8] + fr[:4]}


def app_counts():
    probes = [d for d in dets.values() if d["id"] not in ("DET-0236", "DET-0288", "DET-0307")
              and d["form_version"] != NEWEST[d["form"]]]
    return (len(probes), sum(d["topic"] in ROOF for d in probes), sum(d["topic"] in WATER for d in probes))


TARGET_APP = (208, 36, 40)
edit = [d for d in sorted(dets.values(), key=lambda d: d["id"])
        if int(d["id"][4:]) > 200 and d["id"] not in used]
for cat, topics in (("roof", ROOF), ("water", WATER)):
    want = TARGET_APP[1] if cat == "roof" else TARGET_APP[2]
    for d in edit:           # move older editions to the newest until the scoped count is right
        cur = app_counts()[1 if cat == "roof" else 2]
        if cur <= want:
            break
        if d["topic"] in topics and d["form_version"] != NEWEST[d["form"]]:
            d["form_version"] = NEWEST[d["form"]]
            changed.add(d["id"])
    assert app_counts()[1 if cat == "roof" else 2] == want, (cat, app_counts())
for d in edit:               # then adjust other topics to the total
    tot = app_counts()[0]
    if tot == TARGET_APP[0]:
        break
    if d["topic"] in ROOF + WATER:
        continue
    if tot > TARGET_APP[0] and d["form_version"] != NEWEST[d["form"]]:
        d["form_version"] = NEWEST[d["form"]]
        changed.add(d["id"])
    elif tot < TARGET_APP[0] and d["form_version"] == NEWEST[d["form"]] and d["form"] == "HO-3":
        d["form_version"] = "2022-01"
        changed.add(d["id"])
assert app_counts() == TARGET_APP, app_counts()

# ---------------------------------------------------------------- model fixtures (Chapter 6 compatible)
L, M, S = models["sim-large"], models["sim-mid"], models["sim-small"]
del L["resistance"]["DET-0282"]
L["resistance"]["DET-0307"] = 0.30
L["noprec"] = [i for i in L["noprec"] if i != "DET-0324"] + ["DET-0282"]
L["noskill"] = ["DET-0219"]
for prof in (M, S):
    prof["prec"] = [i for i in prof["prec"] if i != "DET-0045"] + ["DET-0111"]
L["skill_pull"] = ["DET-0116", "DET-0188", "DET-0133"]
M["skill_pull"] = ["DET-0116", "DET-0133", "DET-0188", "DET-0217"]
S["skill_pull"] = ["DET-0116", "DET-0133", "DET-0194", "DET-0217"]

# ---------------------------------------------------------------- roof clash draws
# correct of 15 under v3.0 / v3.1 / v3.2 at seeds 2026-2035 (Listing 7.3)
A = {"sim-large": ([12, 13, 10, 11, 11, 11, 12, 13, 11, 12], [15] * 8 + [14, 14], [15] * 10),
     "sim-mid": ([11, 12, 11, 12, 12, 12, 12, 12, 11, 13], [15] * 7 + [14, 14, 15], [15] * 10),
     "sim-small": ([12, 11, 12, 12, 12, 11, 14, 11, 13, 11], [14, 14, 15, 14, 15, 14, 15, 14, 15, 15],
                   [15, 15, 15, 14, 15, 15, 15, 15, 15, 15])}
FIRST = {"sim-large": ["DET-0116", "DET-0188", "DET-0217"],          # wrong at seed 2026 (section 4)
         "sim-mid": ["DET-0045", "DET-0116", "DET-0188", "DET-0217"],
         "sim-small": ["DET-0045", "DET-0133", "DET-0194"]}
SEEDS = list(range(2026, 2036))
cal["draws"] = {}
for name, prof in models.items():
    susceptible = sorted(set(prof["skill_pull"]) | {i for i in prof["noprec"] if dets[i]["topic"] in ROOF})
    assert len(susceptible) == 5, (name, susceptible)
    v30, v31, v32 = A[name]
    rec = {}
    for k, seed in enumerate(SEEDS):
        n30, n31, n32 = 15 - v30[k], 15 - v31[k], 15 - v32[k]
        order = list(susceptible)
        rng.shuffle(order)
        if seed == 2026:
            order = FIRST[name] + [i for i in order if i not in FIRST[name]]
            assert len(FIRST[name]) == n30
        u = {}
        for j, i in enumerate(order):
            if j < n30:
                u[i] = round(rng.uniform(0.02, 0.74), 3)
            elif j < n30 + n32:
                u[i] = round(rng.uniform(0.991, 0.999), 3)
            elif j < n30 + n31:
                u[i] = round(rng.uniform(0.961, 0.989), 3)
            else:
                u[i] = round(rng.uniform(0.76, 0.95), 3)
        rec[str(seed)] = dict(sorted(u.items()))
    cal["draws"][name] = {"roof": rec}

# ---------------------------------------------------------------- water additions and draws
water = sorted(i for i, d in dets.items() if d["topic"] in WATER)
assert len(water) == 77, len(water)
cal["water"] = water
SENT = {"prior": "The insured's prior claim history voids coverage for repeat water losses.",
        "2500": "A separate $2,500 deductible also applies under the declarations.",
        "30d": "Coverage is limited to losses reported within 30 days.",
        "5000": "Payment is limited to $5,000 for this type of loss.",
        "1000": "A separate $1,000 deductible applies to this coverage.",
        "60d": "The insured must also provide receipts within 60 days."}
hold_water = [i for i in water if int(i[4:]) > 200 and i not in used and i not in changed]
# seed 2026: Chapter 6's additions stand; hold-out water additions are added here
SEED26 = {"sim-large": {hold_water[0]: [SENT["60d"], 0.9]},
          "sim-mid": {hold_water[1]: [SENT["5000"], 0.8], hold_water[2]: [SENT["1000"], 0.35]},
          "sim-small": {"DET-0259": [SENT["2500"], 0.35], hold_water[3]: [SENT["prior"], 0.35]}}
LOW26 = {"sim-large": [], "sim-mid": [hold_water[2]], "sim-small": ["DET-0259", hold_water[3]]}
# seeds 2027-2035: (strong, weak that leaks under v3.0-water, weak that leaks under v3.1-water, weak held)
REST = {"sim-large": (4, 6, 0, 2), "sim-mid": (3, 13, 0, 4), "sim-small": (11, 24, 2, 6)}
SMALL_SENT = ["prior"] * 13 + ["2500"] * 9 + ["30d"] * 4 + ["5000"] * 3 + ["1000"] * 3 + ["60d"] * 3
cal["seed_additions"] = {}
for name, prof in models.items():
    prof["additions"].update(SEED26[name])
    wd = {}
    rec26 = {}
    for i, a in prof["additions"].items():          # seed 2026's weak water additions
        if i in water and a[1] <= 0.75:
            rec26[i] = round(rng.uniform(0.02, 0.74), 3) if i in LOW26[name] else round(rng.uniform(0.76, 0.95), 3)
    wd["2026"] = rec26
    strong, low, high, held = REST[name]
    kinds = ["strong"] * strong + ["low"] * low + ["high"] * high + ["held"] * held
    rng.shuffle(kinds)
    slots = [(SEEDS[1 + k % 9], k) for k in range(len(kinds))]
    sents = list(SMALL_SENT) if name == "sim-small" else []
    rng.shuffle(sents)
    leaking_small = [k for k, kind in enumerate(kinds) if kind in ("strong", "low")]
    per_seed = {}
    for k, kind in enumerate(kinds):
        seed = SEEDS[1 + k % 9]
        i = rng.choice([w for w in water if w not in per_seed.get(seed, {})])
        if name == "sim-small" and k in leaking_small:
            s = SENT[sents.pop()]
        else:
            s = SENT[rng.choice(["30d", "5000", "1000", "60d"])]
        per_seed.setdefault(seed, {})[i] = [s, 0.85 if kind == "strong" else 0.35]
        if kind != "strong":
            u = {"low": rng.uniform(0.02, 0.74), "high": rng.uniform(0.961, 0.99),
                 "held": rng.uniform(0.76, 0.95)}[kind]
            wd.setdefault(str(seed), {})[i] = round(u, 3)
    cal["seed_additions"][name] = {str(s): dict(sorted(v.items())) for s, v in sorted(per_seed.items())}
    cal["draws"][name]["water"] = {s: dict(sorted(v.items())) for s, v in sorted(wd.items())}

# ---------------------------------------------------------------- hit@5: the gold span's rank
miss = set(cal["retrieval_miss"])
rank = {}
for lo, hi, n_out in ((1, 200, 7), (201, 500, 13)):
    ids = [i for i in sorted(dets) if lo <= int(i[4:]) <= hi]
    rest = [i for i in ids if i not in miss]
    out = set(rng.sample(rest, n_out))
    for i in ids:
        rank[i] = 14 if i in miss else rng.randint(6, 10) if i in out else rng.choice([1, 1, 1, 2, 2, 3, 4, 5])
cal["gold_rank"] = rank
cal["note"] = ("Chapter 7 stand-in calibration: Chapter 6's calibration with the adjustments listed in "
               "ch07/calibrate.py, recorded clash draws, per-seed additions and gold ranks. Illustrative.")
cal["determinations"] = sorted(dets.values(), key=lambda d: d["id"])
(HERE / "calibration.json").write_text(json.dumps(cal, indent=1))

# ---------------------------------------------------------------- passages
import importlib.util  # noqa: E402
import standin  # noqa: E402
from contextlab.instructions import V10  # noqa: E402
spec = importlib.util.spec_from_file_location("standin6", HERE.parent / "ch06" / "standin.py")
standin6 = importlib.util.module_from_spec(spec)
sys.modules["standin6"] = standin6
spec.loader.exec_module(standin6)
# evaluation set: DET-0142's new endorsement span replaces a neighbouring section; keep each
# evaluation packet the length it had in Chapter 6
for ins in (V10,):
    old = {d.determination_id: standin6.Pipeline(standin6.PipelineConfig(instructions=ins)).packet(d).tokens
           for d in standin6.CORPUS.eval}
    new = {d.determination_id: standin.Pipeline(standin.PipelineConfig(instructions=ins)).packet(d).tokens
           for d in standin.CORPUS.eval}
    for i in old:
        cal["evidence_tokens"][i] += old[i] - new[i]
# one evaluation claim's passages run 8 tokens longer than in Chapter 6, so that v3.2's mean
# (1,549.5 before this) rounds as printed; no Chapter 6 figure moves
cal["evidence_tokens"]["DET-0150"] += 8
(HERE / "calibration.json").write_text(json.dumps(cal, indent=1))
# hold-out: 1,522 tokens under v2.2 on the large model
TARGET_HOLD = 1522 * 300
for _ in range(3):
    importlib.reload(standin)
    t = standin.run_set(standin.PipelineConfig(), standin.CORPUS.holdout)
    tot = sum(x.packet.tokens for x in t)
    gap = TARGET_HOLD - tot
    if gap == 0:
        break
    ids = [x.det.determination_id for x in t]
    for k, i in enumerate(ids):
        cal["evidence_tokens"][i] += gap // 300 + (1 if k < gap % 300 else 0)
    (HERE / "calibration.json").write_text(json.dumps(cal, indent=1))
importlib.reload(standin)
t = standin.run_set(standin.PipelineConfig(), standin.CORPUS.holdout)
assert sum(x.packet.tokens for x in t) == TARGET_HOLD
print("wrote calibration.json:", len(dets), "determinations;", len(changed), "hold-out claims adjusted")
