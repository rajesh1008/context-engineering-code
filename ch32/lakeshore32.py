"""Chapter 32 fixtures: candidate pools for Lakeshore's full 1,000-determination mix,
a stand-in reader, and the "this week" set for the newest-bulletin failure.

Constructed, seeded and deterministic.  Sizes are stand-ins close to the earlier
chapters' measures; nothing here is a real policy, bulletin or claim.
"""
from __future__ import annotations
import hashlib
import random
from dataclasses import dataclass
from datetime import date
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from contextlab.budgeter import Candidate

TODAY = date(2026, 11, 1)
RESIDUAL = {"DET-0137", "DET-0402", "DET-0655", "DET-0918"}   # wrong for reasons outside selection
SIZES = {"tools": 1_661, "instructions": 640, "claim": 310, "memory": 51,
         "tool_step": 142, "out_step": 60, "out_draft": 420}


def h(*parts) -> float:
    d = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(d[:8], 16) / 0xFFFFFFFF


@dataclass
class Det:
    det_id: str
    cands: list
    governing: list      # span_ids (or group names prefixed "g:") that must be present
    supporting: list     # span_ids behind non-governing claim sentences
    dependent: dict      # endorsement span_id -> definition span_id
    lookalikes: set
    bulletin_new: bool = False


def _mk(rng, det, n, doc, kind, tok, rel, tier, eff, **kw):
    return Candidate(f"{det}:{n}", doc, rng.randint(*tok), round(rng.uniform(*rel), 3),
                     tier, eff, kind, **kw)


def make_det(i: int, this_week: bool = False) -> Det:
    det = f"DET-{i:04d}" if not this_week else f"WK-{i:03d}"
    rng = random.Random(("ch32", det).__repr__())
    c, gov, sup, dep, look = [], [], [], {}, set()
    form_eff = date(2024, 6, 1)
    b = _mk(rng, det, "B", "FORM", "clause", (180, 440), (0.70, 0.92), 2, form_eff, group=f"{det}:gB")
    c.append(b); gov.append(f"g:{det}:gB")
    if rng.random() < 0.5:
        c.append(Candidate(f"{det}:B2", "FORM-ADMIN", b.tokens + rng.randint(-10, 25),
                           round(b.relevance - rng.uniform(0.005, 0.03), 3), 2, form_eff, "clause",
                           group=f"{det}:gB"))
    if rng.random() < 0.45 or this_week or i == 1:      # DET-0001 is Emily Carter's claim (HO 04 95)
        d = _mk(rng, det, "D", "FORM", "definition", (90, 220), (0.58, 0.72), 2, form_eff)
        e = _mk(rng, det, "E", "HO-04-95" if (i == 1 or this_week) else f"END-{rng.randint(1, 9)}", "endorsement", (120, 280), (0.66, 0.88), 2,
                date(2025, rng.randint(1, 12), 1), requires=(d.span_id,))
        c += [d, e]; gov.append(e.span_id); dep[e.span_id] = d.span_id
    if this_week:
        bl = Candidate(f"{det}:L", "BUL-WI-WB-202610", rng.randint(190, 330), round(rng.uniform(0.52, 0.66), 3),
                       1, date(2026, 10, 28), "bulletin", cited_90d=0)
        c.append(bl); gov.append(bl.span_id)
    elif rng.random() < 0.10:
        bl = _mk(rng, det, "L", f"BUL-{rng.randint(1, 30):02d}", "bulletin", (150, 360), (0.60, 0.80), 1,
                 date(2026, rng.randint(1, 9), 1), cited_90d=rng.randint(25, 70))
        c.append(bl); gov.append(bl.span_id)
    for j in range(rng.randint(2, 3)):
        s = _mk(rng, det, f"S{j}", "DEC", "declarations", (40, 110), (0.50, 0.76), 2, date(2026, 1, 1),
                section="DEC")
        c.append(s); sup.append(s.span_id)
    for j in range(rng.randint(14, 22)):
        c.append(_mk(rng, det, f"X{j}", "FORM", "clause", (60, 380), (0.22, 0.60), 2, form_eff,
                     cited_90d=rng.randint(0, 30)))
    for j in range(rng.randint(1, 4)):
        x = _mk(rng, det, f"K{j}", f"END-{rng.randint(10, 19)}", "endorsement", (120, 300), (0.55, 0.78), 2,
                date(2025, 3, 1), cited_90d=rng.randint(0, 40))
        c.append(x); look.add(x.span_id)
    for j in range(rng.randint(1, 3)):
        c.append(_mk(rng, det, f"G{j}", "GUIDE", "guidance", (250, 600), (0.58, 0.80), 4, date(2023, 9, 1),
                     cited_90d=rng.randint(20, 80)))
    for j in range(rng.randint(1, 4)):
        c.append(_mk(rng, det, f"N{j}", "NOTES", "note", (40, 160), (0.40, 0.70), 5, date(2026, 10, 1),
                     cited_90d=rng.randint(0, 10)))
    for x in c:   # the governing and supporting spans carry ordinary usage histories
        if x.cited_90d == 0 and x.kind in ("clause", "definition", "endorsement", "declarations") and x.span_id not in look:
            object.__setattr__(x, "cited_90d", rng.randint(25, 60))
    return Det(det, c, gov, sup, dep, look, this_week)


def full_mix():
    return [make_det(i) for i in range(1, 1001)]


def this_week():
    return [make_det(i, this_week=True) for i in range(1, 41)]


def read(d: Det, packet) -> dict:
    """The stand-in reader: right iff the governing text is present and readable."""
    ids = {s.span_id for s in packet}
    groups = {s.group for s in packet if s.group}
    present = lambda g: (g[2:] in groups) if g.startswith("g:") else (g in ids)
    wrong = d.det_id in RESIDUAL
    missing = [g for g in d.governing if not present(g)]
    if missing and any(not m.endswith(":L") or h(d.det_id, "bul") < 0.35 for m in missing):
        wrong = True
    for e, dfn in d.dependent.items():
        if e in ids and dfn not in ids and h(d.det_id, "dep") < 0.6:
            wrong = True
    sentences = len(d.governing) + len(d.supporting)
    supported = sum(present(g) for g in d.governing) + sum(s in ids for s in d.supporting)
    for k in d.lookalikes & ids:
        if h(d.det_id, k, "cite") < 0.04:
            supported -= 1
    for e, dfn in d.dependent.items():
        if e in ids and dfn not in ids:
            supported -= 0.5
    return {"wrong": wrong, "faithful": max(0, supported) / sentences, "missing": missing}


def needed(d: Det):
    """Labels for calibration: the spans a correct, fully cited determination uses."""
    ids = set(d.supporting) | set(d.dependent.values()) | {g for g in d.governing if not g.startswith("g:")}
    gs = {g[2:] for g in d.governing if g.startswith("g:")}
    return lambda c: c.span_id in ids or (c.group in gs and c.span_id.endswith(":B"))
