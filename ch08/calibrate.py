"""Build calibration.json: the 500 reviewed determinations and the model fixtures.

Determinations are drawn from a seeded generator, with the claims the chapter names
pinned and the topic mix fixed so that the evaluation set (DET-0001..0200) has the
chapter's counts. The constraints that make a rule fire (an endorsement on the policy,
a state and a window of loss dates) are placed by construction, then verified.
Model fixtures are the per-model misses, example pulls and other wrong-clause errors.
Run: python3 calibrate.py   (under a second; rewrites calibration.json)
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
rng = random.Random(808)
TARGET = 1517.3   # mean prompt tokens, v2.2 on sim-large, evaluation set

COVERED = ["water_backup", "food_spoilage", "earth_movement", "personal_property_rc", "roof_settlement",
           "roof_matching", "frozen_pipes", "tree_debris", "rental_reimbursement", "rideshare",
           "deer_swerve", "spoilage"]
EVAL_COVERED = dict(zip(COVERED, [13, 13, 6, 6, 11, 10, 13, 8, 12, 11, 9, 7]))
REST_COVERED = dict(zip(COVERED, [14, 15, 16, 14, 16, 14, 18, 11, 16, 15, 20, 8]))
UNCOVERED = ["flood", "mold", "hail_cosmetic", "wind_hail", "ice_dam", "deer_collision",
             "employee_dishonesty", "navigation_limits", "hull_damage", "equipment_breakdown"]
EVAL_UNCOVERED = dict(zip(UNCOVERED, [5, 5, 7, 7, 5, 13, 12, 10, 8, 9]))
REST_UNCOVERED = dict(zip(UNCOVERED, [9, 10, 14, 13, 8, 18, 16, 14, 10, 11]))
LINE = {**{t: "HO-3" for t in ["water_backup", "food_spoilage", "earth_movement", "personal_property_rc",
                                 "roof_settlement", "roof_matching", "frozen_pipes", "tree_debris", "flood",
                                 "mold", "hail_cosmetic", "wind_hail", "ice_dam"]},
        **{t: "PAP" for t in ["rental_reimbursement", "rideshare", "deer_swerve", "deer_collision"]},
        **{t: "BOP" for t in ["spoilage", "employee_dishonesty", "equipment_breakdown"]},
        **{t: "BOAT" for t in ["navigation_limits", "hull_damage"]}}
TOPIC_ENDORSEMENT = {"water_backup": "HO 04 95", "food_spoilage": "HO 04 98", "earth_movement": "HO 04 54",
                     "personal_property_rc": "HO 04 90", "roof_settlement": "LM HO 17",
                     "rental_reimbursement": "LM PP 21", "rideshare": "LM PP 34", "spoilage": "LM BP 12",
                     "employee_dishonesty": "BP 04 50", "navigation_limits": "BT 05"}
EDITION = {"HO 04 95": "2024-06", "HO 04 98": "2024-06", "HO 04 54": "2022-01", "HO 04 90": "2022-01",
           "LM HO 17": "2024-06", "LM PP 21": "2023-01", "LM PP 34": "2023-01", "LM BP 12": "2025-01",
           "BP 04 50": "2025-01", "BT 05": "2025-03"}
STATES = ["WI", "MN", "IA", "IL", "MI"]
FIRST = ["Emily", "Robert", "Grace", "Eric", "David", "Karen", "Laura", "Olivia", "Daniel", "Linda", "Hannah",
         "Michael", "Sarah", "James", "Anna", "Thomas", "Megan", "Paul", "Rachel", "Mark", "Julie", "Steven",
         "Amy", "Brian", "Kelly", "Jason", "Nicole", "Kevin", "Erin", "Scott"]
LAST = ["Carter", "Weber", "Murphy", "Reid", "Larson", "Novak", "Anderson", "Hoffman", "Becker", "Nelson",
        "Morgan", "Schmidt", "Olson", "Johnson", "Peterson", "Miller", "Kowalski", "Jensen", "Wagner",
        "Hansen", "Fischer", "Lund", "Brennan", "Walsh", "Dahl", "Klein", "Meyer", "Sorensen", "Berg", "Kraus"]

# The claims the chapter (and Part I) names. Fields left out are generated.
PINS = {
    1: dict(name="Emily Carter", state="WI", topic="water_backup", form_version="2024-06",
            endorsements=[("HO 04 95", "2024-06")], loss="2026-03-14"),
    2: dict(name="Robert Weber", state="MI", topic="navigation_limits", form_version="2021-05"),
    3: dict(state="IL", topic="employee_dishonesty", form_version="2020-04", endorsements=[]),
    4: dict(name="Grace Murphy", state="IA", topic="employee_dishonesty", form_version="2025-01", endorsements=[]),
    8: dict(name="Eric Reid", state="MN", topic="roof_settlement", form_version="2024-06", loss="2025-08-14",
            endorsements=[]),
    12: dict(state="MN", topic="deer_collision"),
    22: dict(state="WI", topic="frozen_pipes", form_version="2022-01"),
    26: dict(name="David Larson", state="IA", topic="tree_debris", loss="2026-06-02"),
    30: dict(name="Grace Morgan", state="WI", topic="deer_swerve", loss="2025-11-17"),
    42: dict(state="WI", topic="deer_swerve", loss="2024-05-19"),
    49: dict(state="WI", topic="navigation_limits", form_version="2025-03", endorsements=[("BT 05", "2025-03")],
             loss="2026-08-08"),
    61: dict(name="Linda Nelson", state="IL", topic="frozen_pipes", form_version="2024-06", loss="2025-08-26"),
    63: dict(state="WI", topic="frozen_pipes", form_version="2022-01"),
    69: dict(state="MN", topic="water_backup", loss="2023-09-18", form_version="2022-01",
             endorsements=[("HO 04 95", "2019-03")]),
    71: dict(name="Hannah Morgan", state="IL", topic="earth_movement", form_version="2024-06",
             endorsements=[("HO 04 54", "2022-01")]),
    74: dict(name="Olivia Hoffman", state="WI", topic="roof_matching", form_version="2024-06", loss="2025-04-30"),
    94: dict(state="MI", topic="water_backup", loss="2024-01-14", form_version="2022-01",
             endorsements=[("HO 04 95", "2019-03")]),
    113: dict(state="IL", topic="earth_movement", form_version="2022-01", endorsements=[("HO 04 54", "2022-01")]),
    116: dict(name="Robert Nelson", state="MI", topic="roof_settlement", form_version="2024-06", loss="2026-08-14",
              endorsements=[("LM HO 17", "2024-06")]),
    145: dict(state="WI", topic="food_spoilage", form_version="2024-06", endorsements=[("HO 04 98", "2024-06")],
              loss="2026-03-04"),
    156: dict(state="WI", topic="roof_matching", loss="2026-02-07"),
    163: dict(state="WI", topic="frozen_pipes", form_version="2022-01"),
    172: dict(state="WI", topic="roof_matching", loss="2026-09-06"),
    178: dict(state="WI", topic="frozen_pipes", form_version="2022-01"),
    181: dict(state="IL", topic="frozen_pipes", form_version="2022-01"),
    185: dict(state="WI", topic="food_spoilage", form_version="2024-06",
              endorsements=[("HO 04 95", "2024-06"), ("HO 04 98", "2024-06")], loss="2026-06-11"),
    188: dict(name="Daniel Becker", state="WI", topic="roof_settlement", form_version="2024-06", loss="2026-04-28",
              endorsements=[("LM HO 17", "2024-06")]),
    217: dict(name="David Larson", state="MN", topic="hail_cosmetic", form_version="2024-06", loss="2026-06-25"),
    252: dict(name="Laura Anderson", state="WI", topic="deer_swerve", form_version="2023-01", loss="2025-01-11"),
    259: dict(name="Karen Novak", state="WI", topic="mold", form_version="2022-01"),
    307: dict(state="WI", topic="roof_matching", form_version="2024-06", loss="2025-07-11",
              endorsements=[("HO 04 98", "2024-06")]),
    326: dict(state="WI", topic="deer_swerve", loss="2025-12-06"),
    393: dict(state="WI", topic="deer_swerve", loss="2025-12-28"),
    415: dict(state="WI", topic="deer_swerve", loss="2026-04-04"),
    47: dict(state="WI", topic="roof_matching", loss="2024-02-21"),
    133: dict(state="WI", topic="roof_matching", loss="2024-10-08"),
    194: dict(state="WI", topic="roof_matching", loss="2025-11-03"),
    483: dict(state="IA", topic="tree_debris", form_version="2022-01", loss="2024-04-11"),
}
# Claims that fire a rule's row (adherence probes), beyond the endorsement-carrying ones chosen below.
PROBES = {"water_backup": 18, "food_spoilage": 5, "earth_movement": 16, "personal_property_rc": 11,
          "roof_settlement": 7, "roof_matching": 7, "frozen_pipes": 1, "tree_debris": 3,
          "rental_reimbursement": 13, "rideshare": 16, "deer_swerve": 5, "spoilage": 5}
PINNED_PROBES = {"water_backup": [1, 69, 94], "food_spoilage": [145, 185], "earth_movement": [71, 113],
                 "roof_settlement": [116, 188], "roof_matching": [74, 156, 172, 307],
                 "frozen_pipes": [61], "tree_debris": [26], "deer_swerve": [30, 252, 326, 393, 415]}


def rand_date(lo: date, hi: date) -> date:
    return lo + timedelta(days=rng.randrange((hi - lo).days + 1))


D0, D1 = date(2023, 1, 9), date(2026, 10, 20)


def assign_topics() -> dict[int, str]:
    topic = {n: p["topic"] for n, p in PINS.items()}
    for lo, hi, quotas in ((1, 200, {**EVAL_COVERED, **EVAL_UNCOVERED}), (201, 500, {**REST_COVERED, **REST_UNCOVERED})):
        need = dict(quotas)
        for n in range(lo, hi + 1):
            if n in topic:
                need[topic[n]] -= 1
        pool = [t for t, k in need.items() for _ in range(k)]
        assert all(k >= 0 for k in need.values()), need
        free = [n for n in range(lo, hi + 1) if n not in topic]
        assert len(pool) == len(free), (lo, len(pool), len(free))
        rng.shuffle(pool)
        topic.update(zip(free, pool))
    return topic


def build() -> list[dict]:
    topic = assign_topics()
    by_topic: dict[str, list[int]] = {}
    for n in sorted(topic):
        by_topic.setdefault(topic[n], []).append(n)
    # choose probes per covered topic
    probe = set()
    for t, k in PROBES.items():
        chosen = list(PINNED_PROBES.get(t, []))
        cands = [n for n in by_topic[t] if n not in chosen and n not in PINS]
        if t == "roof_matching":           # four more in the evaluation set: one after 2025-03-15 (id > 172), two before
            chosen += [194, 47, 133]
        elif t not in ("frozen_pipes", "deer_swerve"):
            if t == "tree_debris":
                cands = [n for n in cands if n > 200]
            rng.shuffle(cands)
            chosen += cands[:k - len(chosen)]
        assert len(chosen) == k, (t, chosen)
        probe |= set(chosen)
    for n in (194, 47, 133):
        assert topic[n] == "roof_matching", (n, topic[n])
    wb19 = {69, 94} | set(sorted(n for n in by_topic["water_backup"] if n in probe and n > 94 and n not in PINS)[:8])

    dets = []
    for n in range(1, 501):
        t, p = topic[n], PINS.get(n, {})
        line = LINE[t]
        is_probe = n in probe
        state = p.get("state") or rng.choice(STATES)
        # place or avoid the rule conditions
        if not p.get("state"):
            if t == "tree_debris":
                state = "IA" if is_probe else rng.choice(["WI", "MN", "IL", "MI"])
            if t == "roof_matching" and is_probe:
                state = "WI"
            if t == "frozen_pipes" and state == "IL":
                state = rng.choice(["WI", "MN", "IA", "MI", "IL"])
        lo, hi = D0, D1
        if t == "tree_debris" and state == "IA":
            lo = date(2024, 8, 15)
        if t == "deer_swerve" and state == "WI":
            hi = date(2024, 10, 31)
        if t == "frozen_pipes" and state == "IL":
            hi = date(2025, 4, 30)
        if t == "roof_matching" and state == "WI":
            hi = date(2023, 6, 30)
        if n in wb19 and n not in PINS:
            hi = date(2024, 5, 31)
        loss = date.fromisoformat(p["loss"]) if "loss" in p else rand_date(lo, hi)
        if line == "HO-3":
            fv = p.get("form_version") or ("2024-06" if loss >= date(2024, 9, 1) and rng.random() < 0.8 else "2022-01")
        else:
            fv = p.get("form_version") or {"PAP": "2023-01", "BOP": rng.choice(["2020-04", "2025-01"]),
                                            "BOAT": rng.choice(["2021-05", "2025-03"])}[line]
        if "endorsements" in p:
            ends = list(p["endorsements"])
        else:
            own = TOPIC_ENDORSEMENT.get(t)
            ends = []
            if own and (is_probe or (t in ("employee_dishonesty", "navigation_limits") and rng.random() < 0.4)):
                ed = "2019-03" if n in wb19 else EDITION[own]
                ends.append((own, ed))
            extras = [e for tt, e in TOPIC_ENDORSEMENT.items()
                      if LINE[tt] == line and e != own and EDITION[e] <= fv[:7]]
            if line == "HO-3":
                extras = [e for e in extras if e != "LM HO 17" or fv == "2024-06"]
            for e in extras:
                if rng.random() < 0.2:
                    ends.append((e, EDITION[e]))
            if t == "roof_settlement" and not is_probe:
                ends = [e for e in ends if e[0] != "LM HO 17"]
        name = p.get("name") or f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        dets.append(dict(id=f"DET-{n:04d}", name=name, state=state, form=line, form_version=fv,
                         endorsements=sorted(ends), topic=t, loss=loss.isoformat()))
    return dets


def models(dets) -> dict:
    ev_probes = []   # filled by the caller after rule checks; fixtures name ids directly
    other = {  # the five frozen-pipe wrong-clause errors every model makes (Chapter 6, part B)
        "DET-0022": ["HO-3", "2022-01", "Conditions 4", 13], "DET-0063": ["HO-3", "2022-01", "Conditions 4", 13],
        "DET-0163": ["HO-3", "2022-01", "Conditions 1", 12], "DET-0178": ["HO-3", "2022-01", "Conditions 1", 12],
        "DET-0181": ["HO-3", "2022-01", "Conditions 11", 15]}
    return {"sim-large": {"miss": ["DET-0071"], "pull": ["DET-0061"], "other": other, "resist": ["DET-0393"]},
            "sim-mid": {"miss": ["DET-0071"], "pull": ["DET-0061"], "other": dict(other), "resist": []},
            "sim-small": {"miss": ["DET-0071"], "pull": ["DET-0061"], "other": dict(other), "resist": []}}


def main():
    dets = build()
    cal = {"note": "Calibrated fixture for the Chapter 8 stand-in; illustrative, not measured.",
           "determinations": dets, "models": models(dets)}
    # extra misses for the smaller models: endorsement-governed probes
    ids = {d["id"]: d for d in dets}
    own = {"personal_property_rc": "HO 04 90", "rental_reimbursement": "LM PP 21", "rideshare": "LM PP 34"}
    ev = [d["id"] for d in dets if int(d["id"][4:]) <= 200 and d["topic"] in own
          and any(e[0] == own[d["topic"]] for e in d["endorsements"])]
    late = [d["id"] for d in dets if int(d["id"][4:]) > 200 and d["topic"] == "rideshare"
            and any(e[0] == "LM PP 34" for e in d["endorsements"])]
    cal["models"]["sim-mid"]["miss"] += ev[:2]
    cal["models"]["sim-small"]["miss"] += ev[2:3] + late[:2]
    unc = [d for d in dets if int(d["id"][4:]) <= 200 and d["topic"] == "navigation_limits" and d["endorsements"]]
    u = unc[0]
    cal["models"]["sim-small"]["other"][u["id"]] = [u["form"], u["form_version"], "Declarations Navigation Warranty", 3]
    # retrieved-passage tokens per claim: the packet's evidence text, fitted to the v2.2 prompt size
    evid = {d["id"]: rng.randint(980, 1160) for d in dets}
    cal["evidence_tokens"] = evid
    (HERE / "calibration.json").write_text(json.dumps(cal, indent=1))
    # second pass: shift the evidence text so the v2.2 packet on sim-large averages TARGET tokens
    import importlib
    import standin
    importlib.reload(standin)
    pipe = standin.Pipeline(standin.PipelineConfig(model="sim-large"))
    ev200 = standin.CORPUS.eval
    mean = sum(pipe.packet(d).tokens for d in ev200) / len(ev200)
    need = round((TARGET - mean) * len(ev200))
    base, extra = divmod(need, len(ev200))
    for i, d in enumerate(dets):
        evid[d["id"]] += base + (1 if int(d["id"][4:]) <= extra else 0)
    (HERE / "calibration.json").write_text(json.dumps(cal, indent=1))
    print("wrote calibration.json:", len(dets), "determinations")


if __name__ == "__main__":
    main()
