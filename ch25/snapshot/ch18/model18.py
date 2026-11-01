"""Chapter 18 stand-ins: the tools, the three ways of answering, and a simulated model.

The simulated model is deterministic. Where a real model would sometimes miss
something, a hash of the claim and the item decides, at a rate stated here:
missing the second peril when it decomposes a claim, missing a reference that
is written by name rather than identifier, and erring on a complete packet.
"""
from __future__ import annotations

import hashlib
import re

import corpus18 as C
from contextlab.agentic import Budget, Slot, State, fixed_tokens, run
from contextlab.retrieval.index import InvertedIndex, top

P_MISS_PERIL, P_MISS_NAMEREF, P_ANSWER_ERR, P_MATERIAL = 0.12, 0.20, 0.02, 0.30
QUARANTINED = {("HS", "SP"), ("RS", "OL")}           # Chapter 17 quarantined this edge for a person
CORPUS = C.build()
IX = InvertedIndex()
for _s in CORPUS.values():
    IX.add(_s.sid, _s.text)
NAME = {v[0]: k for k, v in C.FAM.items()}


def chance(*key) -> float:
    return int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def tok(sid):
    return CORPUS[sid].tokens if sid in CORPUS else 0


def admissible(s, claim):
    """Chapter 15's authority rules, applied before anything enters a packet."""
    pol = claim.policy
    if s.kind == "bulletin":
        return s.state == pol.state and C.in_force(s, claim.loss)
    if s.kind == "endorsement":
        return pol.schedule.get(s.family) == s.edition
    return True


def rewrite(q):
    return " ".join(C.SHORTHAND.get(w.lower().strip(",.?;:"), w) for w in q.split())


def search(query, claim, k=8, kind=None, state=None, authority=True):
    allowed = {sid for sid, s in CORPUS.items() if (kind is None or s.kind == kind)
               and (state is None or s.state == state)}
    ranked = top(IX.bm25(query, allowed=allowed), 40)
    if authority:
        ranked = [r for r in ranked if admissible(CORPUS[r], claim)]
    return ranked[:k]


def refs_in(sid):
    """What a careful reader finds in an endorsement: identifiers and names."""
    txt = CORPUS[sid].text
    out = [(m, True) for m in re.findall(r"endorsement LK-([A-Z]{2})", txt)]
    out += [(NAME[n], False) for n in NAME if f"the {n} endorsement" in txt]
    return [(f, by_id) for f, by_id in out if f != CORPUS[sid].family]


# ---- way 1 and way 2 -------------------------------------------------------
def fixed(claim, k=8):
    """Chapters 12 to 16: hybrid retrieval (the BM25 lane here), authority rules, and
    Chapter 15's jurisdiction step, which adds the base clause and the in-force state
    bulletin for every endorsement that made the packet."""
    pol, packet = claim.policy, search(rewrite(claim.question), claim, k)
    for sid in [s for s in packet if CORPUS[s].kind == "endorsement"]:
        f = CORPUS[sid].family
        add = [C.FAM[f][2]] + [b.sid for b in CORPUS.values() if b.kind == "bulletin"
                               and b.family == f and b.state == pol.state and C.in_force(b, claim.loss)]
        packet += [a for a in add if a not in packet]
    return packet


def graph_walk(claim, k=8):
    """Chapter 17: start from the first endorsement retrieved, walk the policy graph."""
    pol, packet = claim.policy, fixed(claim, k)
    starts = [CORPUS[s].family for s in packet if CORPUS[s].kind == "endorsement"]
    if not starts:
        return packet
    todo, fams = list(starts), []
    while todo:
        f = todo.pop(0)
        if f in fams or f not in pol.schedule:
            continue
        fams.append(f)
        todo += [t for t, _ in C.REFS[f] if (f, t) not in QUARANTINED]
    add = [f"LK-{f}@{pol.schedule[f]}" for f in fams] + [C.FAM[f][2] for f in starts]
    add += [s.sid for s in CORPUS.values() if s.kind == "bulletin" and s.family in fams
            and s.state == pol.state and s.effective <= C.GRAPH_BUILT and C.in_force(s, claim.loss)]
    return packet + [a for a in add if a not in packet]


# ---- way 3: the loop -------------------------------------------------------
class StandInModel:
    """Plans sub-queries, reads results and follows what it notices."""

    def __init__(self, claim, slots_driven=True, first="fixed"):
        self.c, self.slots_driven, self.pending, self.done = claim, slots_driven, [], set()
        self.first = first

    def perils(self):
        q, found = self.c.question.lower(), []
        for f in self.c.entries:
            if found and chance(self.c.cid, "peril", f) < P_MISS_PERIL:
                continue                      # the second peril slips past the decomposition
            found.append(f)
        return found

    def plan(self, st):
        if st.rounds == 1 and self.first == "fixed":
            return [("fixed", self.c.question)]
        if st.rounds == 1:
            acts = [("search", (rewrite(C.PHRASE[f]) + " endorsement", "endorsement", None))
                    for f in self.perils()]
            self.done |= {repr(a) for a in acts}
            return acts
        acts, self.pending = self.pending, []
        if self.slots_driven:
            acts += [a for a in (self.act_for(s.name) for s in st.slots) if a]
        acts = [a for a in acts if repr(a) not in self.done]
        self.done |= {repr(a) for a in acts}
        return acts

    def act_for(self, slot):
        kind, _, rest = slot.partition(":")
        pol = self.c.policy
        if kind in ("ref", "peril"):
            f = rest.split("->")[-1]
            if kind == "peril":
                return ("search", (f"{C.FAM[f][0]} endorsement {C.FAM[f][3]}", "endorsement", None))
            return ("lookup", f"LK-{f}@{pol.schedule[f]}") if f in pol.schedule else None
        if kind == "bulletin":
            stt, f = rest.split(":")
            q = f"{C.STATES[stt]} bulletin {C.FAM[f][0]} LK-{f}"
            n = sum(1 for d in self.done if q in d)
            if n:                     # asked before and still open: a model rephrases
                q += " " + ["notice", "circular", "revised", "amended", "current"][n % 5] + f" {n}"
            return ("search", (q, "bulletin", stt))
        if kind == "clause":
            return ("lookup", rest)
        return None

    def admit(self, st, sid):
        if sid not in st.packet:
            st.packet.append(sid)
            if CORPUS[sid].kind == "endorsement" and not self.slots_driven:
                self.notice(st, sid)

    def notice(self, st, sid):
        """Without a slot list, the model follows only what it happens to notice."""
        pol, f0 = self.c.policy, CORPUS[sid].family
        for f, by_id in refs_in(sid):
            if f in pol.schedule and (by_id or chance(self.c.cid, "nameref", f) >= P_MISS_NAMEREF):
                self.pending.append(("lookup", f"LK-{f}@{pol.schedule[f]}"))
        if f0 in self.c.entries:
            self.pending.append(("search", (f"{C.STATES[pol.state]} bulletin {C.FAM[f0][0]} LK-{f0}",
                                            "bulletin", pol.state)))
            if chance(self.c.cid, "clause", f0) < 0.5:
                self.pending.append(("lookup", C.FAM[f0][2]))

    def read(self, st, tool, arg, hits):
        pol = self.c.policy
        if tool == "fixed":
            for h in hits:
                if CORPUS[h].kind != "manual":
                    self.admit(st, h)
            return
        if tool == "lookup":
            for h in hits:
                self.admit(st, h)
        elif arg[1] == "endorsement" and hits:
            words = set(arg[0].lower().replace(",", " ").split())
            fams = sorted(dict.fromkeys(CORPUS[h].family for h in hits[:3]),
                          key=lambda x: -len(words & set(C.FAM[x][3].split())))
            f = next((x for x in fams if x in pol.schedule), None)
            if f:
                self.admit(st, f"LK-{f}@{pol.schedule[f]}")
            else:
                st.absences[f"peril:{fams[0]}"] = "not attached"
        elif arg[1] == "bulletin":
            fam = next((v for n, v in NAME.items() if n in arg[0]), "")
            good = [h for h in hits if CORPUS[h].family == fam and admissible(CORPUS[h], self.c)]
            for h in good[:1]:
                self.admit(st, h)
            if not good:
                st.absences[f"bulletin:{pol.state}:{fam}"] = "none in force"


def tools_for(claim):
    def _search(arg, st):
        q, kind, state = arg
        return search(q, claim, 5, kind=kind, state=state, authority=False)
    def _lookup(sid, st):
        return [sid] if sid in CORPUS else []
    def _fixed(q, st):
        return fixed(claim)
    return {"search": _search, "lookup": _lookup, "fixed": _fixed}


def families_in(packet):
    return {CORPUS[s].family for s in packet if CORPUS[s].kind == "endorsement"}


def endorsements_in(packet):
    return [s for s in packet if CORPUS[s].kind == "endorsement"]


def bulletin_for(f0, packet):
    return any(CORPUS[s].kind == "bulletin" and CORPUS[s].family == f0 for s in packet)


def slot_check(claim, demand_bulletin=False):
    """Lakeshore's sufficiency check: every slot filled by a span or a recorded absence.

    demand_bulletin=True is the failure injection for the sidebar: absences stop counting.
    """
    pol, FAM, absent_ok = claim.policy, C.FAM, not demand_bulletin

    def check(st):
        slots, have = [], families_in(st.packet)
        for f in claim.cause_codes:
            slots.append(Slot(f"peril:{f}", f in have or f not in pol.schedule))
            if f in pol.schedule:
                slots.append(Slot(f"clause:{FAM[f][2]}", FAM[f][2] in st.packet))
        for sid in endorsements_in(st.packet):
            f0 = CORPUS[sid].family
            for f, _ in refs_in(sid):
                slots.append(Slot(f"ref:{f0}->{f}", f in have or f not in pol.schedule))
            key = f"bulletin:{pol.state}:{f0}"
            slots.append(Slot(key, bulletin_for(f0, st.packet) or (absent_ok and key in st.absences)))
        open_ = [s for s in slots if not s.filled]
        return (not open_), open_
    return check


def always_satisfied(st):
    return True, []


def self_judge(st):
    return False, []        # the model stops when it has nothing more it noticed


def agentic(claim, judge="slots", budget=Budget(), isolate=True, first="fixed"):
    model = StandInModel(claim, slots_driven=judge in ("slots", "never", "always"), first=first)
    check = {"slots": slot_check(claim), "never": slot_check(claim, demand_bulletin=True),
             "always": always_satisfied, "self": self_judge}[judge]
    st = State(claim.question, {"policy": claim.policy, "codes": claim.cause_codes})
    return run(st, model, tools_for(claim), check, budget, isolate,
               model_judge=judge in ("self", "always"), span_tokens=tok)


# ---- scoring ---------------------------------------------------------------
def score(claim, packet):
    """Sentences: one per governing span. Wrong if a material span is missing."""
    missing = claim.required - set(packet)
    material = [m for m in missing if chance(claim.cid, "material", m) < P_MATERIAL]
    long_ctx = sum(tok(s) for s in packet) > 3000 and chance(claim.cid, "long") < 0.25
    wrong = bool(material) or chance(claim.cid, "answer") < P_ANSWER_ERR or long_ctx
    return {"wrong": wrong, "sentences": len(claim.required),
            "supported": len(claim.required) - len(missing), "missing": missing}


def evaluate(claims, way, **kw):
    rows = []
    for c in claims:
        if way == "fixed":
            tr = fixed_tokens(fixed(c), tok)
        elif way == "graph":
            tr = fixed_tokens(graph_walk(c), tok)
        else:
            tr = agentic(c, **kw)
        rows.append((c, tr, score(c, tr.packet)))
    return rows
