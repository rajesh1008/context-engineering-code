"""Chapter 49 fixtures: the packet-review exercise and a stand-in candidate pool.

Deterministic stand-in world (random.Random(49)); no model is called. The packet is
a Wisconsin wind-and-water claim built from the same span objects the context layer
uses, with eight defects seeded on purpose. Candidates are synthetic: each has a
hidden skill, a separate fluency with tool and vendor names, and a score on a
held-out failure-diagnosis drill that stands in for later performance.
"""
import math
import random

from contextlab.hiring import Defect, Review, Span

F = "HO 00 03"
PACKET = [
    Span("S1", "instructions", "Cite the exact form, edition and page for every coverage statement.", "lk-instr", "v14", "1"),
    Span("S2", "facts", "FNOL: wind lifted shingles 12 Aug; rain entered, ceiling stained.", "claim-WI-20931", "2026-08-13", "1"),
    Span("S3", "governing", "Windstorm or hail: covered peril.", F, "05 11", "9"),
    Span("S4", "governing", "Rain to interior: covered only if wind first makes an opening [...]", F, "2026 ed.", "10"),
    Span("S5", "modifier", "Wind/hail deductible 2% of dwelling limit.", "HO 03 12", "2026 ed.", None),
    Span("S6", "governing", "Claimant: 'my agent said water is always covered'.", "call-note-7781", "2026-08-14", "1"),
    Span("S7", "context", "Illinois bulletin: roof-surface schedule.", "IL-BUL-2026-04", "2026-04", "2"),
    Span("S8", "context", "Prior note: 'tree fell on garage, paid 2024'.", "memory/claim-WI-18802", "2024-06", "1"),
    Span("S9", "context", "Windstorm or hail: covered peril.", F, "2026 ed.", "9"),
    Span("S10", "context", "Windstorm or hail: covered peril.", F, "2026 ed.", "9"),
    Span("S11", "question", "Is the wind damage to the roof covered, and what applies?", "planner", "q1", "1"),
]
DEFECTS = [
    Defect("D1", "S3", "superseded edition cited as governing", "retrieve the edition in force on the date of loss"),
    Defect("D2", "S5", "span without a page cannot be cited", "reject at assembly until provenance is complete"),
    Defect("D3", "S6", "a claimant statement placed as governing text", "move to facts; it cannot change what the policy means"),
    Defect("D4", "S7", "another state's bulletin in a Wisconsin packet", "filter bulletins by state and effective date"),
    Defect("D5", "S4", "governing clause trimmed mid-condition", "never trim governing spans; trim context first"),
    Defect("D6", "S10", "duplicate span spends budget twice", "deduplicate by document, edition and page"),
    Defect("D7", "S8", "memory from a different claim", "scope memory reads to this claim and insured"),
    Defect("D8", "S11", "question asks about one peril of two", "one coverage question per peril (Chapter 41)"),
]
DIFFICULTY = {"D1": .30, "D2": .35, "D3": .25, "D4": .45, "D5": .55, "D6": .20, "D7": .50, "D8": .70}


def _sig(x):
    return 1 / (1 + math.exp(-x))


def pool(n=40, seed=49):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        skill, fluency = rng.random(), rng.random()
        r = Review(f"C{i + 1:02d}")
        day = min(1, max(0, skill + rng.gauss(0, .15)))   # the exercise sees one day
        for d in DEFECTS:
            found = rng.random() < _sig(9 * (day - DIFFICULTY[d.did]))
            ev = found and rng.random() < .25 + .7 * day
            fix = ev and rng.random() < .2 + .7 * day
            r.findings[d.did] = (found, ev, fix)
        r.false_findings = int((1 - skill) * 3 * rng.random() + .5)
        r.tool_names = int(fluency * 9 * (.6 + .8 * rng.random()) + .5)
        nfound = sum(f[0] for f in r.findings.values())
        r.claimed = nfound + r.false_findings + int(fluency * 3 + .5)
        drill = 10 * skill + rng.gauss(0, 1.0)
        out.append((r, round(skill, 3), round(fluency, 3), round(drill, 2)))
    return out
