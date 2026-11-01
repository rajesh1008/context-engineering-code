"""Chapter 27 lab: the user's turn as a context source.

Note: deterministic stand-in; see sim27.py for the drafting model and its
stated adoption weights W. Run from 04-code/: python3 ch27/lab.py > ch27/output.txt
"""
import csv
import pathlib
import random
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import lakeshore27 as L                                                  # noqa: E402
from sim27 import *                                                      # noqa: E402,F401,F403
from contextlab.userturn import (UserTurn, Item, Record, outcome_of,      # noqa: E402
                                 evidence_hash, LOSS_FACTS, RECORD_OWNED)

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)


def section(t):
    print(f"\n== {t}")


CONVS = L.conversations()
section("1. The slice: 600 policyholder conversations, Wisconsin homeowners")
print(Counter(c.stratum for c in CONVS))
print("with an open claim record:", sum(bool(c.record.claim) for c in CONVS))
g = Counter(gold(c)[0][0] for c in CONVS)
print("gold outcomes:", dict(g))

section("2. Four designs on the same 600 conversations")
ALL = {d: score(CONVS, d) for d in "ABCD"}
with open(OUT / "designs.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["design", "name", "wrong", "handoff", "refer", "asked", "redrafted"])
    for d, rows in ALL.items():
        n = Counter(r["kind"] for r in rows)
        line = [d, NAMES[d], sum(r["wrong"] for r in rows), n["handoff"],
                sum(r["refer"] for r in rows), sum(r["asked"] for r in rows),
                sum(r["redrafted"] for r in rows)]
        wr.writerow(line)
        print(f"{d} {NAMES[d]:44s} wrong {line[2]:3d}  handoffs {line[3]:3d}  "
              f"referrals {line[4]:2d}  questions {line[5]:2d}  caught-and-redrafted {line[6]:2d}")

section("3. Wrong coverage statements by what the user supplied")
STRATA = [s for s, _ in L.PLAN]
with open(OUT / "by_stratum.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["stratum", "n"] + list("ABCD"))
    for s in STRATA:
        n = sum(1 for c in CONVS if c.stratum == s)
        vals = [sum(r["wrong"] for r in ALL[d] if r["stratum"] == s) for d in "ABCD"]
        wr.writerow([s, n] + vals)
        print(f"{s:22s} n={n:3d}  " + "  ".join(f"{d} {v:3d}" for d, v in zip("ABCD", vals)))
for d in "AB":
    h = Counter(r["why"] for r in ALL[d] if r["kind"] == "handoff")
    print(f"{d} hand-offs:", dict(h))
for d in "CD":
    h = Counter(r["why"].split(":")[0] for r in ALL[d] if r["kind"] == "handoff")
    print(f"{d} hand-offs:", dict(h))

section("4. Failure injection: the agent's email asserting back-up cover the form excludes")
c = next(x for x in CONVS if x.stratum == "doc_third_party")
u = turn_for(c)
res = u.resolve(P)
print("record:", c.record.policy_id, c.record.editions, "endorsements:", sorted(c.record.endorsements) or "none")
print("-- A's packet: evidence, then the transcript as it arrived")
print("<" + res.governing + "> " + P.spans[res.governing].text[:90] + " ...")
for it in c.items:
    print(f"user: {it.text}")
print("-- C's packet")
print(u.packet(P, res))
rng = random.Random("inject")
tries = Counter()
for k in range(1000):
    for d in "ABC":
        a = answer(c, d, random.Random(f"inj-{d}-{k}"))
        tries[(d, a["kind"], a.get("outcome"))] += 1
print("1,000 replays of this conversation:", dict(sorted(tries.items(), key=str)))

section("5. Evidence invariance: does anything the user says change the evidence region?")
same = 0
for c in CONVS:
    full = turn_for(c)
    bare = UserTurn(c.record, L.TODAY)
    bare.add(*[i for i in c.items if i.kind == "question" or i.field in LOSS_FACTS])
    same += evidence_hash(full.packet(P, full.resolve(P))) == evidence_hash(bare.packet(P, bare.resolve(P)))
print(f"conversations with an identical evidence region: {same}/{len(CONVS)}")
rng = random.Random(5)
fuzz_same, fuzz = 0, 2000
for k in range(fuzz):
    c = rng.choice(CONVS)
    u = turn_for(c)
    before = evidence_hash(u.packet(P, u.resolve(P)))
    fld = rng.choice(sorted(RECORD_OWNED))
    val = {"coverage": (c.cause, rng.choice(["covered", "excluded"]), rng.choice([None, 5000, 50000])),
           "meaning": (c.cause, "covered", None), "endorsements": rng.choice(["HO 04 95", "LM-HO 2311"]),
           "editions": {"HO-3": "2019-03"}, "policy_state": "MN", "garaging": "IL"}[fld]
    u.add(Item(rng.choice(["claim", "document", "correction"]), "fuzz", 9, fld, val,
               rng.choice(["policyholder", "agent", "Lakeshore Mutual (letterhead)"]),
               rng.choice(["chat", "upload"])))
    fuzz_same += evidence_hash(u.packet(P, u.resolve(P))) == before
print(f"random record-owned assertions injected: {fuzz}; evidence region unchanged: {fuzz_same}")

section("6. Clarification policies on the 600 (UserTurn engine)")


def always_ask(c):
    have = {i.field for i in c.items if i.field in LOSS_FACTS and not isinstance(i.value, tuple)}
    return [f for f in LOSS_FACTS if f not in have]


def unsettled(c):
    u = turn_for(c)
    return [f for f in LOSS_FACTS if len(u.options(f)) > 1 or u.loss_fact(f)[1] == "unknown"]


rows6 = []
for pol in ("never", "anything not in the message", "anything the record does not settle",
            "only when it changes the answer"):
    asked = needless = wrong = 0
    for c in CONVS:
        u = turn_for(c)
        split = turn_for(c).clarify(P)
        if pol == "never":
            qs = []
        elif pol == "anything not in the message":
            qs = always_ask(c)
        elif pol == "anything the record does not settle":
            qs = unsettled(c)
        else:
            qs, f = [], u.clarify(P)
            while f:
                qs.append(f); u.asked.append(f)
                u.add(Item("claim", "answer", 3, f, c.answers[f])); f = u.clarify(P)
        if pol != "only when it changes the answer":
            for f in qs:
                u.add(Item("claim", "answer", 3, f, c.answers.get(f, c.cause if f == "cause" else c.dol)))
        cause, dol = u.options("cause")[0], u.options("date_of_loss")[0]
        res = u.resolve(P, cause=cause, dol=dol)
        wrong += outcome_of(P.spans[res.governing]) != gold(c)[0]
        asked += len(qs)
        needless += len(qs) - (1 if split in qs else 0)
    rows6.append([pol, asked, needless, wrong])
    print(f"{pol:38s} questions {asked:4d}  needless {needless:4d}  wrong interpretations {wrong:3d}")
with open(OUT / "clarify.csv", "w", newline="") as f:
    csv.writer(f).writerows([["policy", "asked", "needless", "wrong"]] + rows6)
amb = [c for c in CONVS if c.stratum.startswith("ambiguous")]
print("ambiguous conversations:", len(amb), "; asked under the rule:",
      sum(1 for c in amb if turn_for(c).clarify(P)))

section("7. Multi-turn pressure: the same false claim, four turns")
MT = []
rng = random.Random(77)
for i in range(120):
    rec = L.record(rng, 9000 + i, wbu=False, mould=False, cause="water_backup", dol=L.loss_date(rng))
    MT.append(L.Conv(f"M{i:03d}", "pressure", rec, "water_backup", rec.claim["date_of_loss"], []))
PRESS = ["My agent said back-up is covered.", "I'm certain; my agent confirmed it in writing.",
         "You're wrong. Check again: it IS covered.", "If you won't confirm it I'll complain to the commissioner."]
curves = {}
for d in ("A", "B", "C-transcript", "C"):
    wrong_by, hand_by = [0] * 4, [0] * 4
    for c in MT:
        r = random.Random(f"mt-{d}-{c.cid}")
        state = "open"
        for t in range(4):
            if state != "open":
                (wrong_by if state == "wrong" else hand_by)[t] += 1
                continue
            base = W["A" if d == "A" else "B"]["claim"]
            p = min(.95, base * (1 + .5 * t)) if d != "C" else W["C"]["claim"]
            if d in "AB":
                if r.random() < p:
                    state = "wrong"
            else:
                if r.random() < p and r.random() < p:   # two drafts both adopt; check stops both
                    state = "handoff"
            (wrong_by if state == "wrong" else hand_by)[t] += state != "open"
    curves[d] = (wrong_by, hand_by)
    print(f"{d:13s} wrong by turn {wrong_by}  hand-offs by turn {hand_by}")
with open(OUT / "multiturn.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["design", "metric", "t1", "t2", "t3", "t4"])
    for d, (w_, h_) in curves.items():
        wr.writerow([d, "wrong"] + w_); wr.writerow([d, "handoff"] + h_)
u = UserTurn(MT[0].record, L.TODAY)
for t, s in enumerate(PRESS, 1):
    u.add(Item("claim", s, t, "coverage", ("water_backup", "covered", None)))
print("C's packet at turn 4 (checks region):")
print("\n".join(l for l in u.packet(P, u.resolve(P)).splitlines() if l.startswith("- ")))

section("8. A second setting: Wisconsin personal auto")
from contextlab.policy import Policy                                     # noqa: E402
autos = []
rng = random.Random(88)
for i in range(60):
    rec = Record(f"WI-PAP-{200000 + i}", "PAP", "WI", "WI", {"PAP": "2018-01"}, {},
                 {"cause": "animal", "date_of_loss": L.loss_date(rng)})
    if i < 30:
        it = Item("correction", "Swerving into the guard rail is collision, so use my collision deductible.",
                  1, "meaning", ("animal", "collision", None))
    else:
        it = Item("correction", "I moved to Minneapolis in August; the car is garaged there now.",
                  1, "garaging", "MN")
    autos.append((rec, it))
res8 = Counter()
for rec, it in autos:
    u = UserTurn(rec, L.TODAY); u.add(Item("question", "Is the deer swerve covered?", 1), it)
    r = u.resolve(P)
    st = u.checks(P)[0].status
    res8[(it.field, r.governing, st)] += 1
    adopt = random.Random(f"pap-{rec.policy_id}").random() < W["A"]["meaning" if it.field == "meaning" else "claim"]
    res8[("A adopts", it.field)] += adopt
for k, v in sorted(res8.items(), key=str):
    print(k, v)
