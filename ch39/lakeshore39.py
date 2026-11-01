"""Chapter 39 stand-in world: Lakeshore's full mix, three stages and two judges.

Deterministic. No language model runs. Every probability below is a stand-in
parameter, chosen so the hand-checked baseline matches the scoreboard.
"""
import random
from dataclasses import dataclass

CASES = [("plain", .40), ("limit", .14), ("conditions", .14), ("deductible", .10),
         ("withhold", .08), ("edition", .07), ("conflict", .07)]
STATES = ["WI", "IL", "MI", "MN", "IA"]
FORMS = ["HO-3", "HO 04 95", "DP-3", "HO-5", "HO 04 90"]
UNSUP = {"plain": .015, "limit": .05, "conditions": .045, "deductible": .045,
         "withhold": 0.0, "edition": .03, "conflict": .035}   # per rationale sentence
PACKET = 6
LOOK_CITE = 0.024
SCALE = 0.93
SECONDS_PER_LABEL = 20
HARD = 0.985


@dataclass(frozen=True)
class Claim:
    id: str
    case: str
    state: str
    form: str
    spans: tuple       # (span id, grade, kind)
    n_sent: int

    @property
    def governing(self):
        return next((s for s, g, k in self.spans if k == "governing"), None)

    @property
    def grades(self):
        return {s: g for s, g, _ in self.spans}

    def kind(self, sid):
        return next(k for s, _, k in self.spans if s == sid)


def pick(rng, table):
    x, acc = rng.random(), 0.0
    for name, p in table:
        acc += p
        if x < acc:
            return name
    return table[-1][0]


def make_claims(prefix="F", n=1000, seed=39, new_edition_share=0.0):
    rng, out = random.Random(seed), []
    for i in range(n):
        case = pick(rng, CASES)
        cid = f"{prefix}{i:04d}"
        spans = []
        if case != "withhold":
            spans.append((f"{cid}-g", 3, "governing"))
        spans += [(f"{cid}-s1", 2, "supporting"), (f"{cid}-sch", 2, "schedule")]
        spans += [(f"{cid}-r{j}", 1, "related") for j in range(3)]
        if case in ("edition", "conflict"):
            spans.append((f"{cid}-look", 0, "lookalike"))
        spans += [(f"{cid}-d{j:02d}", 0, "distractor") for j in range(14)]
        out.append(Claim(cid, case, rng.choice(STATES), rng.choice(FORMS), tuple(spans),
                         rng.randint(3, 6)))
    return out


# ---------- retrievers: (base score by grade, schedule bonus, lookalike bonus, noise) ----------
RETRIEVERS = {
    "bm25":        ({3: 2.0, 2: 1.2, 1: 1.0, 0: 0.0}, -1.6, 2.0, 1.0),
    "dense":       ({3: 2.2, 2: 1.3, 1: 1.1, 0: 0.0}, -1.3, 2.1, 1.0),
    "rerank":      ({3: 2.6, 2: 1.5, 1: 1.1, 0: 0.0}, -1.2, 2.4, 0.8),
    "v9":          ({3: 2.7, 2: 1.5, 1: 1.1, 0: 0.0}, 0.1, 2.4, 0.8),
    "edition-aware": ({3: 2.6, 2: 1.5, 1: 1.1, 0: 0.0}, -1.2, -1.5, 0.8),
    "sched-boost": ({3: 2.6, 2: 1.5, 1: 1.1, 0: 0.0}, 1.0, 2.4, 0.8),
    "title-only":  ({3: 1.2, 2: 0.8, 1: 0.8, 0: 0.0}, -1.6, 1.5, 1.0),
}


def score(name, claim, rng):
    base, sch, look, sd = RETRIEVERS[name]
    out = {}
    for s, g, k in claim.spans:
        out[s] = base[g] + (sch if k == "schedule" else 0) + (look if k == "lookalike" else 0) \
            + rng.gauss(0, sd)
    return out


def ranked(name, claim, seed=0, depth=20):
    if name == "hybrid":
        a, b = ranked("bm25", claim, seed), ranked("dense", claim, seed)
        rrf = {}
        for lst in (a, b):
            for i, s in enumerate(lst, 1):
                rrf[s] = rrf.get(s, 0) + 1 / (60 + i)
        return sorted(rrf, key=lambda s: (-rrf[s], s))[:depth]
    rng = random.Random(f"{name}|{claim.id}|{seed}")
    sc = score(name, claim, rng)
    return sorted(sc, key=lambda s: (-sc[s], s))[:depth]


# ---------- assembly ----------
def assemble(claim, run, dedupe_by_form=False):
    packet, forms = [], set()
    for s in run:
        k = claim.kind(s)
        fam = "form" if k in ("governing", "lookalike") else s
        if dedupe_by_form and fam in forms:
            continue
        forms.add(fam)
        packet.append(s)
        if len(packet) == PACKET:
            break
    return packet


# ---------- generation ----------
def determine(claim, packet, seed=0):
    """Returns (cited span or None, per-sentence supported flags)."""
    rng = random.Random(f"gen|{claim.id}|{seed}")
    g = claim.governing
    if g is None or g not in packet:
        if rng.random() < 0.97:
            return None, ()
        return packet[0], tuple(rng.random() > 0.5 for _ in range(claim.n_sent))
    look = f"{claim.id}-look"
    if look in packet and packet.index(look) < packet.index(g) and rng.random() < LOOK_CITE:
        return look, tuple(True for _ in range(claim.n_sent))
    p = UNSUP[claim.case] * SCALE
    if claim.case in ("limit", "deductible"):
        p *= 0.4 if f"{claim.id}-sch" in packet else 1.3
    return g, tuple(rng.random() >= p for _ in range(claim.n_sent))


def record(claim, run, packet, cited, sents):
    g = claim.governing
    return {"governing": g, "rank": (run.index(g) + 1) if g in run else None,
            "in_packet": g in packet if g else False, "cited": cited,
            "supported": all(sents), "sents": sents, "case": claim.case,
            "state": claim.state}


def pipeline(claims, retriever="hybrid", dedupe_by_form=False, seed=0):
    runs, recs = {}, {}
    for c in claims:
        run = ranked(retriever, c, seed)
        pk = assemble(c, run, dedupe_by_form)
        cited, sents = determine(c, pk, seed)
        runs[c.id] = run
        recs[c.id] = record(c, run, pk, cited, sents)
    return runs, recs


# ---------- human labellers: two adjusters and an adjudicator ----------
def adjuster(name, claim, sid, grade):
    rng = random.Random(f"adj|{name}|{sid}")
    k = claim.kind(sid)
    p = .25 if k == "lookalike" else .12 if k == "schedule" else \
        .10 if claim.case in ("edition", "conflict") else .05
    if rng.random() >= p:
        return grade
    if k == "lookalike":
        return 3
    return max(0, min(3, grade + rng.choice((-1, 1))))


# ---------- judges ----------
JUDGES = {   # sentence-level error on supported / unsupported, relevance overrating
    "strict":  {"fp": .08, "fn": .015, "over": .10, "look": .15},
    "lenient": {"fp": .62, "fn": .004, "over": .35, "look": .70},
}


def judge_sentence(jname, sid, truth, difficulty, ask=0):
    j = JUDGES[jname]
    rng = random.Random(f"js|{jname}|{sid}|{ask}")
    err = (j["fn"] if truth else j["fp"]) * (0.3 + 1.4 * difficulty)
    if difficulty > HARD:          # genuinely ambiguous sentences: the judge is close to a coin
        err = max(err, 0.42)
    return truth if rng.random() >= err else (not truth)


def judge_relevance(jname, claim, sid, grade):
    j = JUDGES[jname]
    rng = random.Random(f"jr|{jname}|{sid}")
    if claim.kind(sid) == "lookalike":
        return 3 if rng.random() < j["look"] else 0
    if grade < 3 and rng.random() < j["over"]:
        return grade + 1
    return grade


def difficulty(sid):
    return random.Random(f"diff|{sid}").random()
