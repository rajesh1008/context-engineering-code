"""Chapter 50 lab: four case layers, one transplant, and what Lakeshore borrowed.

Run: python3 lab.py  (a few seconds; writes output.txt and out/results.json)
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from contextlab.casestudies import (ROWS, ablate, compare, fit_band, gate_curve,  # noqa: E402
                                    rare_event_exit, score, wilson)
import cases50 as C  # noqa: E402

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
R: dict = {}
LINES: list = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LINES.append(s)


def show(name, t):
    lo, hi = wilson(t.wrong, t.n)
    say(f"  {name:<22} wrong {t.wrong:>3}/{t.n} ({100*t.rate():.1f}%, 95% CI {100*lo:.1f}-{100*hi:.1f})"
        f"  deferred {t.deferred:>3}  by cause {dict(sorted(t.wrong_by.items()))}")
    return {"n": t.n, "wrong": t.wrong, "deferred": t.deferred, "right": t.right,
            "wrong_by": dict(t.wrong_by), "ci": [round(lo, 4), round(hi, 4)]}


def s1_cards():
    say("== 1. Layer cards: which rows each layer declares ==")
    cards = C.cards()
    m = compare(cards)
    say("  " + "row".ljust(11) + "".join(c.name[:10].ljust(12) for c in cards))
    for r in ROWS:
        say("  " + r.ljust(11) + "".join(("yes" if m[r][c.name] else "--").ljust(12) for c in cards))
    R["cards"] = {c.name: c.missing() for c in cards}
    for c in cards:
        say(f"  {c.name}: missing {c.missing() or 'none'}")


def s2_dutch():
    say("\n== 2. Dutch bank: three releases on 600 compliance questions ==")
    topics = C.dutch_corpus(); qs = C.dutch_questions(topics)
    say(f"  topics {len(topics)}; intents " + str({i: sum(q.intent == i for q in qs) for i in ('now', 'on-date', 'what-changed')}))
    R["dutch"] = {}
    for name, cfg in (("v1 publication date", C.DUTCH_V1), ("v2 applies-from", C.DUTCH_V2),
                      ("v3 layers + dated rank", C.DUTCH_V3)):
        R["dutch"][name] = show(name, score(qs, C.dutch_answer(topics, cfg)))
    return topics, qs


def s3_transplant(topics, qs):
    say("\n== 3. Failure injection: Lakeshore's rules transplanted into the Dutch layer ==")
    run = lambda cfg: score(qs, C.dutch_answer(topics, cfg))  # noqa: E731
    R["transplant"] = {"lakeshore": show("Lakeshore transplant", run(C.LAKESHORE))}
    say("  one Lakeshore rule at a time, everything else native:")
    for k, t in ablate(C.DUTCH_V3, C.LAKESHORE, run).items():
        R["transplant"][k] = show(f"  {k}={C.LAKESHORE[k]}", t)
    gl = [t for t in topics if t["kind"] == "gl"]
    say(f"  guidelines {len(gl)}; ever not complied with in NL: {sum(any(s != 'comply' for _, s in t['status']) for t in gl)}")


def s4_exit():
    say("\n== 4. Exit rule in the unit failures happen in: rule changes ==")
    first = [True] * (C.PAST_CHANGES - 1) + [False]
    for label, rep, live in (("first replay", first, 0), ("after fix", [True] * C.PAST_CHANGES, 2),
                             ("six changes later", [True] * C.PAST_CHANGES, 6)):
        state, why = rare_event_exit(rep, live, 6)
        say(f"  {label:<16} {state:<14} {why}")
    R["exit"] = {"past_changes": C.PAST_CHANGES, "need_live": 6}


def s5_boston():
    say("\n== 5. Boston hospital: 1,000 guideline questions ==")
    gl, qs = C.boston_questions()
    R["boston"] = {}
    for name, cfg in (("v1 document, today", C.BOSTON_V1), ("v2 section, purpose", C.BOSTON_V2)):
        R["boston"][name] = show(name, score(qs, C.boston_answer(gl, cfg)))
    say("  one v2 rule at a time on v1:")
    for k, t in ablate(C.BOSTON_V1, C.BOSTON_V2, lambda c: score(qs, C.boston_answer(gl, c))).items():
        R["boston"]["fix " + k] = show(f"  fix {k}", t)


def s6_dublin():
    say("\n== 6. Dublin firm: 800 clause questions over 1,400 contracts ==")
    room = C.dublin_room(); qs = C.dublin_questions(room)
    sides = sum(1 for c in room for x in c["inst"] if x[0] == "side")
    unindexed = sum(1 for c in room for x in c["inst"] if x[0] == "side" and not x[3])
    say(f"  side letters {sides}; not in the data-room index {unindexed}; counterparties 310")
    R["dublin"] = {"side_letters": sides, "unindexed": unindexed}
    for name, cfg in (("v1 similarity", C.DUBLIN_V1), ("v2 key + register", C.DUBLIN_V2)):
        R["dublin"][name] = show(name, score(qs, C.dublin_answer(room, cfg)))
    for k, t in ablate(C.DUBLIN_V1, C.DUBLIN_V2, lambda c: score(qs, C.dublin_answer(room, c))).items():
        R["dublin"]["fix " + k] = show(f"  fix {k}", t)
    docs = C.dublin_scans()
    his = [0.80, 0.85, 0.90, 0.95, 0.98]
    say(f"  document quality gate: {len(docs)} documents, {sum(q < 1 for q, _ in docs)} scanned; escalate below 0.55")
    say(f"  no gate: expected extraction errors {sum(e for _, e in docs):.1f}")
    curve = gate_curve(docs, 0.55, his)
    for hi, load, esc in curve:
        say(f"    auto at >= {hi:.2f}: review load {load:>3}, escaped errors {esc}")
    best = fit_band(docs, 0.55, his, capacity=200)
    esc_n = sum(1 for q, _ in docs if q < 0.55)
    say(f"  capacity 200 reviews: upper edge {best[0]:.2f}, load {best[1]}, escaped {best[2]}; escalated {esc_n}")
    R["dublin"]["gate"] = {"curve": curve, "chosen": best, "escalated": esc_n,
                           "no_gate": round(sum(e for _, e in docs), 1)}


def s7_austin():
    say("\n== 7. Austin company: 900 questions ==")
    custs, qs = C.austin_questions()
    say(f"  customers 600; unknown to the router {sum(not c['router_knows'] for c in custs)}")
    R["austin"] = {}
    for name, cfg in (("v1 latest, nightly", C.AUSTIN_V1), ("v2 record, live, graph", C.AUSTIN_V2)):
        R["austin"][name] = show(name, score(qs, C.austin_answer(custs, cfg)))


def s8_lakeshore():
    say("\n== 8. What Lakeshore borrowed: judgement rows from the hospital ==")
    j = C.lakeshore_judgement()
    tok = j["judgement_conditions"] * j["extra_tokens_each"] / j["claims"]
    say(f"  {j['claims']} gold-sized claims; judgement conditions {j['judgement_conditions']}")
    say(f"  drafts deciding a judgement condition: {j['decided_before']} -> {j['decided_after']}")
    say(f"  extra tokens per determination on average: {tok:.1f}; wrong-clause unchanged at 3/1,000 (not touched)")
    R["lakeshore"] = dict(j, avg_extra_tokens=round(tok, 1))


if __name__ == "__main__":
    s1_cards()
    topics, qs = s2_dutch()
    s3_transplant(topics, qs)
    s4_exit(); s5_boston(); s6_dublin(); s7_austin(); s8_lakeshore()
    (HERE / "output.txt").write_text("\n".join(LINES) + "\n")
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=str))
