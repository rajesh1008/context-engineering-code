"""Chapter 50 stand-ins: four illustrative case layers and Lakeshore's.

Every organisation here is a composite carried from earlier cameos. Every
number the lab prints is produced by these generators; none is a measurement
from a real system. Seeds are fixed, so output.txt is reproducible.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from contextlab.casestudies import LayerCard, Q

TODAY = date(2026, 11, 1)
D0 = date(2020, 1, 1)


def day(rng, a=D0, b=TODAY):
    return a + timedelta(days=rng.randrange((b - a).days))


# ---------------------------------------------------------------- Dutch bank
def dutch_corpus(seed=50):
    """48 topics: 20 EU regulations, 14 directives (via Dutch law), 14 EBA guidelines."""
    rng, topics = random.Random(seed), []
    for i in range(48):
        kind = "reg" if i < 20 else "dir" if i < 34 else "gl"
        t = {"id": f"T{i:02d}", "kind": kind}
        if kind == "reg":
            starts = sorted({day(rng, D0, date(2026, 6, 1)) for _ in range(rng.randint(2, 4))})
            if rng.random() < 0.35:  # amendment published, not yet applying
                starts.append(TODAY + timedelta(days=rng.randint(30, 400)))
            t["versions"] = [{"sid": f"{t['id']}-c{k}", "act": f"{t['id']}-a{k}", "from": s,
                              "pub": s - timedelta(days=rng.randint(60, 540))}
                             for k, s in enumerate([D0 - timedelta(days=900)] + starts)]
        elif kind == "dir":
            t["eu"], t["nl"] = f"{t['id']}-eu", f"{t['id']}-nl"
        else:
            t["gl"], t["nl"] = f"{t['id']}-gl", f"{t['id']}-nl"
            seq, d = [], D0 - timedelta(days=400)
            for _ in range(rng.randint(1, 3)):
                seq.append((d, rng.choice(["comply"] * 5 + ["not", "intends"])))
                d = day(rng, max(d, D0), date(2026, 9, 1))
            t["status"] = sorted(seq)
            t["lag"] = rng.choice([0, 0, 90, 180, 270])  # compliance table updated late
        topics.append(t)
    return topics_fix(topics)


def topics_fix(topics):
    for t in topics:
        if t["kind"] == "gl":
            t["status"] = [s for i, s in enumerate(t["status"])
                           if i == 0 or s[0] > t["status"][i - 1][0]]
    return topics


def in_force(t, when):
    return [v for v in t["versions"] if v["from"] <= when][-1]


def status_at(t, when, lag=0):
    known = [s for d, s in t["status"] if d + timedelta(days=lag) <= when]
    return known[-1] if known else t["status"][0][1]


def dutch_questions(topics, seed=51, n=600):
    rng, qs = random.Random(seed), []
    for i in range(n):
        t = rng.choice(topics)
        intent = rng.choices(["now", "on-date", "what-changed"], [5, 4, 2])[0]
        if t["kind"] != "reg" and intent == "what-changed":
            intent = "now"
        when = TODAY if intent == "now" else day(rng, date(2021, 1, 1))
        if t["kind"] == "reg":
            v = in_force(t, when)
            truth = v["act"] if intent == "what-changed" else v["sid"]
            kind = "pending" if intent == "now" and t["versions"][-1]["from"] > TODAY else intent
        elif t["kind"] == "dir":
            truth, kind = t["nl"], "transposition"
        else:
            truth = t["gl"] if status_at(t, when) == "comply" else t["nl"]
            kind = "comply-or-explain"
        qs.append(Q(f"NL{i:03d}", intent, truth, kind, (t["id"], when, rng.random())))
    return qs


DUTCH_V1 = {"as_of": "publication", "routing": "single", "layers": "flat", "rank": "fixed"}
DUTCH_V2 = {"as_of": "applies", "routing": "intent", "layers": "flat", "rank": "fixed"}
LAKESHORE = {"as_of": "applies", "routing": "single", "layers": "flat", "rank": "fixed"}
DUTCH_V3 = {"as_of": "applies", "routing": "intent", "layers": "transposition", "rank": "dated"}


def dutch_answer(topics, cfg):
    by = {t["id"]: t for t in topics}

    def answer(q):
        t, when = by[q.ctx[0]], q.ctx[1]
        if t["kind"] == "reg":
            if cfg["as_of"] == "publication":
                v = [x for x in t["versions"] if x["pub"] <= when][-1]
            else:
                v = in_force(t, when)
            wants_act = q.intent == "what-changed" and cfg["routing"] == "intent"
            return v["act"] if wants_act else v["sid"]
        if t["kind"] == "dir":
            flat_miss = cfg["layers"] == "flat" and q.ctx[2] < 0.3  # EU text outscores the Dutch act
            return t["eu"] if flat_miss else t["nl"]
        if cfg["rank"] == "fixed":
            return t["gl"]
        return t["gl"] if status_at(t, when, t["lag"]) == "comply" else t["nl"]
    return answer


PAST_CHANGES = 26  # rule changes in the replay archive, 2020 to 2026 (illustrative)


# ---------------------------------------------------------- Boston hospital
def boston_questions(seed=52, n=1000):
    """60 guidelines x 6 sections; revisions replace one to three sections each."""
    rng, gl = random.Random(seed), []
    for g in range(60):
        revs = [(D0 - timedelta(days=700), set(range(6)))]
        for _ in range(rng.randint(1, 5)):
            revs.append((day(rng, D0, date(2026, 10, 1)), set(rng.sample(range(6), rng.randint(1, 3)))))
        gl.append(sorted(revs, key=lambda r: r[0]))
    qs = []
    for i in range(n):
        g, s = rng.randrange(60), rng.randrange(6)
        intent = rng.choices(["dose", "interaction", "retrospective", "judgement"], [59, 20, 17, 4])[0]
        when = day(rng, date(2024, 11, 1)) if intent == "retrospective" else TODAY
        truth = "HANDBACK" if intent == "judgement" else f"G{g}s{s}r{sec_rev(gl[g], s, when)}"
        qs.append(Q(f"BOS{i:04d}", intent, truth, intent, (g, s, when, rng.random(), rng.random())))
    return gl, qs


def sec_rev(revs, s, when):
    return max(k for k, (d, secs) in enumerate(revs) if s in secs and d <= when)


BOSTON_V1 = {"supersession": "document", "as_of": "today", "judgement": "answer", "budget": "shared"}
BOSTON_V2 = {"supersession": "section", "as_of": "purpose", "judgement": "row", "budget": "per-task"}
DROP = {"shared": 0.06, "per-task": 0.01}  # chance an interaction chain loses its second span


def boston_answer(gl, cfg):
    def answer(q):
        g, s, when, u1, u2 = q.ctx
        if q.intent == "judgement":
            return "HANDBACK" if cfg["judgement"] == "row" else f"G{g}s{s}r0"
        at = when if cfg["as_of"] == "purpose" else TODAY
        revs = gl[g]
        if cfg["supersession"] == "document":
            latest = max(k for k, (d, _) in enumerate(revs) if d <= at)
            k = latest if s in revs[latest][1] or u1 < 0.8 else sec_rev(revs, s, at) - 1
            k = sec_rev(revs, s, at) if k == latest else max(k, 0)
        else:
            k = sec_rev(revs, s, at)
        if q.intent == "interaction" and u2 < DROP[cfg["budget"]]:
            return f"G{g}s{s}r-dropped"
        return f"G{g}s{s}r{k}"
    return answer


# ------------------------------------------------------------ Dublin firm
CLAUSES = ("change-of-control", "indemnity-cap", "assignment", "exclusivity")


def dublin_room(seed=53, n_contracts=1400):
    rng, room = random.Random(seed), []
    for c in range(n_contracts):
        inst = [("base", day(rng, date(2015, 1, 1), date(2022, 1, 1)), set(CLAUSES), True)]
        for a in range(rng.choice([0, 0, 1, 1, 2, 3])):
            inst.append((f"amd{a}", day(rng, date(2022, 1, 1), date(2026, 6, 1)),
                         set(rng.sample(CLAUSES, rng.randint(1, 2))), True))
        if rng.random() < 0.18:
            inst.append(("side", day(rng, date(2022, 1, 1), date(2026, 9, 1)),
                         {rng.choice(["change-of-control", "exclusivity"])}, rng.random() < 0.55))
        room.append({"cid": c, "cp": c % 310, "inst": sorted(inst, key=lambda x: x[1]),
                     "registered": rng.random() > 0.01, "u": rng.random()})
    return room


def dublin_questions(room, seed=54, n=800):
    rng, qs = random.Random(seed), []
    for i in range(n):
        c, cl = rng.choice(room), rng.choice(CLAUSES)
        gov = [x for x in c["inst"] if cl in x[2]][-1]
        down = rng.random() < 0.03  # register unavailable for this request
        qs.append(Q(f"DUB{i:03d}", cl, f"C{c['cid']}-{gov[0]}", gov[0][:3], (c["cid"], down, rng.random())))
    return qs


DUBLIN_V1 = {"admission": "similarity", "register": "ignore"}
DUBLIN_V2 = {"admission": "counterparty", "register": "required"}


def dublin_answer(room, cfg):
    def answer(q):
        c, down, u = room[q.ctx[0]], q.ctx[1], q.ctx[2]
        if cfg["register"] == "required" and down and q.intent in ("change-of-control", "exclusivity"):
            return "REFUSE"
        seen = [x for x in c["inst"] if q.intent in x[2]
                and (x[0] != "side" or x[3] or (cfg["register"] == "required" and c["registered"]))]
        gov = seen[-1]
        if cfg["admission"] == "similarity" and gov[0] != "base" and u < 0.09:
            return f"C{(c['cid'] + 310) % 1400}-{gov[0]}"  # same-template twin, other counterparty
        return f"C{c['cid']}-{gov[0]}"
    return answer


def dublin_scans(seed=55, n=1400):
    """Quality score and extraction error chance for each document before extraction."""
    rng, docs = random.Random(seed), []
    for _ in range(n):
        q = 1.0 if rng.random() > 0.22 else rng.betavariate(5, 2)
        docs.append((round(q, 3), 0.004 + 0.6 * (1 - q) ** 2))
    return docs


# ---------------------------------------------------------- Austin company
RELEASES = ("4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "5.0", "5.1")


def austin_questions(seed=56, n=900):
    rng = random.Random(seed)
    custs = [{"rel": rng.choice(RELEASES), "router_knows": rng.random() > 1 / 3,
              "record": rng.random() > 0.03} for _ in range(600)]
    differs = {t for t in range(80) if rng.random() < 0.25}
    qs = []
    for i in range(n):
        c = rng.randrange(600)
        intent = rng.choices(["config", "deploy", "code"], [70, 15, 15])[0]
        if intent == "config":
            t = rng.randrange(80)
            line = custs[c]["rel"][0] if t in differs else "any"
            truth, ctx = f"doc{t}-{line}", (c, t, line)
        elif intent == "deploy":
            same_day = rng.random() < 0.3
            truth, ctx = "deploys-hour-before", (c, same_day)
        else:
            twin = rng.random() < 30 / 120
            truth, ctx = "fn-live", (c, twin, rng.random())
        qs.append(Q(f"AUS{i:03d}", intent, truth, intent, ctx))
    return custs, qs


AUSTIN_V1 = {"unknown_release": "latest", "deploy_log": "nightly", "code": "similarity"}
AUSTIN_V2 = {"unknown_release": "record-or-ask", "deploy_log": "live-view", "code": "reachability"}
TWIN_FIRST = 0.73  # Chapter 12's measured share for the billing service's vendored twins


def austin_answer(custs, cfg):
    def answer(q):
        if q.intent == "config":
            c, t, line = q.ctx
            cu = custs[c]
            if line == "any":
                return q.truth
            if cu["router_knows"]:
                return f"doc{t}-{cu['rel'][0]}"
            if cfg["unknown_release"] == "latest":
                return f"doc{t}-5"
            return f"doc{t}-{cu['rel'][0]}" if cu["record"] else "ASK"
        if q.intent == "deploy":
            stale = cfg["deploy_log"] == "nightly" and q.ctx[1]
            return "deploys-yesterday" if stale else "deploys-hour-before"
        _, twin, u = q.ctx
        if twin and cfg["code"] == "similarity" and u < TWIN_FIRST:
            return "fn-vendored-twin"
        return "fn-live"
    return answer


# ------------------------------------------------------- Lakeshore borrow
def lakeshore_judgement(seed=57, n=1000):
    """Drafts that state a conclusion on a condition the adjuster must judge."""
    rng = random.Random(seed)
    claims = [rng.random() < 0.058 for _ in range(n)]
    return {"claims": n, "judgement_conditions": sum(claims),
            "decided_before": sum(claims), "decided_after": 0,
            "extra_tokens_each": 64}


# ----------------------------------------------------------- layer cards
def cards():
    nl = LayerCard("Dutch bank", {
        "unit": "one compliance question on one obligation", "sources": "journal, consolidations, guidelines, compliance tables",
        "required": "official journal metadata", "as_of": "applies-from by intent", "precedence": "per authority, dated",
        "admission": "in force for the asked date", "budget": "per intent", "gate": "replay every past change",
        "eval": "600 questions by intent", "exit": "rule-change events", "handback": "", "trace": "act and consolidation ids"})
    bos = LayerCard("Boston hospital", {
        "unit": "one clinical question at one as-of", "sources": "guidelines, formulary service, protocols",
        "required": "formulary service", "as_of": "by purpose", "precedence": "section supersession",
        "admission": "minimum necessary", "budget": "per task", "gate": "section-matched ingestion",
        "eval": "1,000 questions by task", "exit": "", "handback": "clinician judgement rows", "trace": "section, revision and purpose"})
    dub = LayerCard("Dublin firm", {
        "unit": "one clause in one contract", "sources": "data room, side-letter register, entity list",
        "required": "side-letter register", "as_of": "signed date", "precedence": "latest operative instrument",
        "admission": "counterparty key, matter wall", "budget": "per draft", "gate": "document quality bands",
        "eval": "800 clause questions", "exit": "", "handback": "solicitor review", "trace": "content hash and pinpoint"})
    aus = LayerCard("Austin company", {
        "unit": "one answer for one customer release", "sources": "release docs, runbooks, deploy view, code",
        "required": "deployment record", "as_of": "customer release", "precedence": "call-graph reachability over similarity",
        "admission": "identity provider per request", "budget": "per task", "gate": "",
        "eval": "900 questions by intent", "exit": "", "handback": "ask when release unknown", "trace": ""})
    lk = LayerCard("Lakeshore", {
        "unit": "one determination", "sources": "six sources", "required": "declarations page",
        "as_of": "date of loss", "precedence": "endorsement and bulletin table", "admission": "version, state, policy",
        "budget": "budgeter per call", "gate": "release gate", "eval": "gold-v1.2",
        "exit": "shadow exit rule by cell", "handback": "", "trace": "answer event and evidence pack"})
    return [nl, bos, dub, aus, lk]
