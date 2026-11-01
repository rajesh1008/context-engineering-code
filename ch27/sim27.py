"""Chapter 27 stand-in: the designs, the drafting model and the scorer.

Note: deterministic stand-in. No language model runs. The 'model' reads a
packet and either states what the governing span says or adopts something
the user asserted. How often it adopts is a stated parameter per design
(table W below), chosen to follow the direction the cited studies report,
not measured on any model. The structural results (evidence invariance, the
check, record-owned fields) do not depend on W; test_ch27.py sweeps it.
Imported by lab.py and test_ch27.py.
"""
import pathlib
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import lakeshore27 as L                                                  # noqa: E402
from contextlab.userturn import (UserTurn, Item, Record, outcome_of, check,  # noqa: E402
                                 evidence_hash, LOSS_FACTS, RECORD_OWNED)

P = L.policy()
# Stand-in adoption weights: chance a draft adopts one conflicting user item.
W = {"A": dict(claim=.40, document=.60, instruction=.50, meaning=.40, use_fact=.98, latest=.80),
     "B": dict(claim=.12, document=.18, instruction=.10, meaning=.12, use_fact=.70, latest=.70),
     "C": dict(claim=.12, document=.18, instruction=.10, meaning=.12)}
NAMES = {"A": "transcript in the prompt (2025 assistant)", "B": "labelled + defensive instruction",
         "C": "UserTurn, no clarification", "D": "UserTurn with clarification"}


def key_of(it):
    if it.kind == "document" and "NOTE TO THE ASSISTANT" in it.text:
        return "instruction"
    return "meaning" if it.field == "meaning" else ("document" if it.kind == "document" else "claim")


def asserted(it):
    if it.field == "endorsements":
        lim = int(it.text.split("US$")[1].replace(",", ""))
        return ("covered", lim)
    return (it.value[1], it.value[2])


def conflicts(u):
    return [c.item for c in u.checks(P) if c.status in ("contradicts", "not-user-suppliable")
            and c.item.field in RECORD_OWNED]


def gold(c):
    u = UserTurn(c.record, L.TODAY)
    res = u.resolve(P, cause=c.cause, dol=c.dol)
    return outcome_of(P.spans[res.governing]), res.governing


def turn_for(c, upto=None):
    u = UserTurn(c.record, L.TODAY)
    u.add(*[i for i in c.items if upto is None or i.turn <= upto])
    return u


def read_transcript(c, u, w, rng):
    """How a model reading the raw transcript picks up the loss facts."""
    if c.record.claim:
        return c.record.claim["cause"], c.record.claim["date_of_loss"]
    causes = [i.value for i in u.items if i.field == "cause"]
    cause = causes[-1] if causes else None
    if isinstance(cause, tuple):
        cause = cause[0]
    elif cause is not None and rng.random() > w["use_fact"]:
        return None, None
    dates = [i.value for i in u.items if i.field == "date_of_loss"]
    d = (dates[-1] if rng.random() < w["latest"] else dates[0]) if dates else L.TODAY
    return cause, (d[0] if isinstance(d, tuple) else d)


def answer(c, design, rng):
    """One conversation's final reply: a statement, a hand-off or a referral."""
    u = turn_for(c)
    asked = 0
    if design in "AB":
        cause, dol = read_transcript(c, u, W[design], rng)
        if cause is None:
            return dict(kind="handoff", why="cause not used", asked=0)
        res = u.resolve(P, cause=cause, dol=dol)
        for it in conflicts(u):
            if rng.random() < W[design][key_of(it)]:
                o, lim = asserted(it)
                return dict(kind="statement", outcome=o, limit=lim, cited=res.governing,
                            adopted=it.kind, asked=0)
        o, lim = outcome_of(P.spans[res.governing])
        return dict(kind="statement", outcome=o, limit=lim, cited=res.governing, asked=0)
    if design == "D":
        f = u.clarify(P)
        while f:
            u.asked.append(f); asked += 1
            u.add(Item("claim", f"answer about {f}", 2 + asked, f, c.answers[f]))
            f = u.clarify(P)
    cause = u.options("cause")[0]
    dol = u.options("date_of_loss")[0]
    res = u.resolve(P, cause=cause, dol=dol)
    if res.governing is None:
        return dict(kind="handoff", why=res.reason, asked=asked)
    o, lim = outcome_of(P.spans[res.governing])
    refer = any(i.source == "agent" for i in conflicts(u))
    for attempt in range(2):
        draft = dict(kind="statement", outcome=o, limit=lim, cited=res.governing)
        for it in conflicts(u):
            if rng.random() < W["C"][key_of(it)]:
                draft.update(zip(("outcome", "limit"), asserted(it)))
                break
        ok, why = check(draft, P, res)
        if ok:
            return dict(draft, asked=asked, refer=refer, redrafted=attempt)
    return dict(kind="handoff", why="check failed twice: " + why, asked=asked, refer=refer)


def score(convs, design, seed=2700, w=None):
    rows = []
    for c in convs:
        rng = random.Random(f"{seed}-{design}-{c.cid}")
        a = answer(c, design, rng)
        (go, glim), gsid = gold(c)
        wrong = a["kind"] == "statement" and (a["outcome"], a["limit"]) != (go, glim)
        rows.append(dict(cid=c.cid, stratum=c.stratum, design=design, kind=a["kind"],
                         wrong=int(wrong), asked=a.get("asked", 0),
                         refer=int(bool(a.get("refer"))), redrafted=a.get("redrafted", 0),
                         why=a.get("why", "")))
    return rows


