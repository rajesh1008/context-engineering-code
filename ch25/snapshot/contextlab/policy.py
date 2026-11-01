"""contextlab.policy: authority rules applied at query time (Chapter 15, v3.5).

Similarity says which spans resemble the question. The Policy says which
of them the asker may see, which jurisdiction's rules apply, which were in
force on the date that matters, and which one governs. Every decision has
a reason, and anything the Policy cannot decide is withheld.
Chapter 37 adds the defensive layer (attacks on these rules); this module
is the everyday enforcement the pipeline depends on.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import date

GOVERNING_KINDS = {"form": 10, "endorsement": 20, "bulletin": 30}
CONTEXT_KINDS = {"guideline", "advisory", "note", "summary"}


@dataclass(frozen=True)
class Span:
    sid: str                 # doc_id@version#clause
    doc_id: str
    version: str | None
    kind: str                # form, endorsement, bulletin, guideline, advisory, note, summary
    topic: str               # clause key the span speaks to
    text: str
    jurisdiction: str = "ALL"
    tenant: str = "lakeshore"
    roles: frozenset | None = frozenset({"adjuster", "supervisor"})
    effective_from: date | None = None
    effective_to: date | None = None     # exclusive; None means "not recorded"
    supersedes: str | None = None        # doc_id this one replaces
    superseded_by: str | None = None
    scope: str | None = None             # for endorsements: the clause they modify
    edition_family: str | None = None    # e.g. HO-3, HO 04 95
    page: int | None = None
    derived_from: tuple = ()             # sids a summary was built from


@dataclass(frozen=True)
class Request:
    """Who is asking, for whom, about what, as of when."""
    role: str
    tenant: str
    purpose: str                         # "coverage-determination", ...
    line: str                            # HO or PAP
    policy_state: str | None
    risk_state: str | None               # where the property is, or the loss state for auto
    garaging_state: str | None
    editions: dict                       # family -> edition on the policy
    endorsements: frozenset              # endorsement doc_ids on the policy
    date_of_loss: date | None
    cause: str | None                    # cause-of-loss code mapped to a clause key


@dataclass
class Verdict:
    ok: bool
    reason: str


# Which state's rules apply depends on the kind of rule and the line.
# A Lakeshore compliance decision, written down so the code can apply it.
JURISDICTION_RULES = {
    ("HO", "bulletin"): "risk_state",       # claims-handling rules follow the property
    ("HO", "endorsement"): "policy_state",  # amendatory endorsements follow the contract
    ("PAP", "bulletin"): "garaging_state",
    ("PAP", "endorsement"): "policy_state",
}


class Policy:
    version = "policy-v1.0"

    def __init__(self, spans, rules=JURISDICTION_RULES, scoped_precedence=True,
                 unknown_blocks=True):
        self.spans = {s.sid: s for s in spans}
        self.rules = rules
        self.scoped = scoped_precedence
        self.unknown_blocks = unknown_blocks   # an unknown jurisdiction stops the answer
        self.next_of = {}                    # supersession graph, built both ways
        for s in spans:
            if s.supersedes:
                self.next_of[s.supersedes] = s
        self.fingerprint = hashlib.sha256(
            json.dumps([self.version, sorted(map(str, rules.items())), scoped_precedence]
                       ).encode()).hexdigest()[:12]

    def roles_of(self, s: Span):
        """A derived span is as restricted as the most restricted thing it came from."""
        roles = s.roles
        for parent in s.derived_from:
            p = self.spans.get(parent)
            pr = p.roles if p else None
            roles = None if (roles is None or pr is None) else roles & pr
        return roles

    def entitled(self, s: Span, r: Request) -> Verdict:
        if s.tenant not in ("lakeshore", r.tenant) and s.kind != "bulletin":
            return Verdict(False, f"tenant {s.tenant} not {r.tenant}")
        roles = self.roles_of(s)
        if roles is None:
            return Verdict(False, "no entitlement recorded")  # fail closed
        if r.role not in roles:
            return Verdict(False, f"role {r.role} not in {sorted(roles)}")
        return Verdict(True, "entitled")

    def governing_state(self, s: Span, r: Request):
        key = self.rules.get((r.line, s.kind))
        return getattr(r, key) if key else None

    def in_jurisdiction(self, s: Span, r: Request) -> Verdict:
        if s.jurisdiction == "ALL":
            return Verdict(True, "all states")
        want = self.governing_state(s, r)
        if want is None:
            return Verdict(False, "jurisdiction unknown")
        if s.jurisdiction != want:
            return Verdict(False, f"{s.jurisdiction} rule; {want} governs")
        return Verdict(True, f"{want} governs")

    def end_of(self, s: Span):
        """Effective-to, from the span itself or from the successor that names it."""
        nxt = self.next_of.get(s.doc_id)
        ends = [d for d in (s.effective_to, nxt.effective_from if nxt else None) if d]
        return (min(ends) if ends else None), nxt

    def in_force(self, s: Span, on: date | None) -> Verdict:
        if on is None:
            return Verdict(False, "no date of loss")
        if s.version is None:
            return Verdict(False, "no version")
        if s.effective_from and on < s.effective_from:
            return Verdict(False, f"not yet in force on {on}")
        end, nxt = self.end_of(s)
        if end and on >= end:
            by = nxt.doc_id if nxt else s.superseded_by
            return Verdict(False, f"superseded by {by} from {end}")
        return Verdict(True, f"in force on {on}")

    def on_policy(self, s: Span, r: Request) -> Verdict:
        """Forms and endorsements count only in the edition the policy carries."""
        if s.kind == "form":
            want = r.editions.get(s.edition_family)
            ok = want == s.version
            return Verdict(ok, f"{s.edition_family} ({s.version})" + ("" if ok else f"; policy has {want}"))
        if s.kind == "endorsement" and s.jurisdiction == "ALL":
            ok = f"{s.edition_family}@{s.version}" in r.endorsements
            return Verdict(ok, "on the declarations" if ok else "not on this policy")
        return Verdict(True, "not policy-specific")

    def admit(self, s: Span, r: Request):
        """Run the gates in order; the first failure is the reason."""
        for gate in (self.entitled(s, r), self.on_policy(s, r),
                     self.in_jurisdiction(s, r), self.in_force(s, r.date_of_loss)):
            if not gate.ok:
                return gate
        return Verdict(True, "admitted")

    def precedence(self, s: Span, cause: str) -> int | None:
        """Rank for this cause of loss, or None if the span cannot govern it."""
        if s.kind not in GOVERNING_KINDS:
            return None                              # advisory, notes: context only
        speaks = s.topic == cause or (not self.scoped and s.kind == "endorsement")
        if s.kind == "endorsement" and self.scoped:
            speaks = (s.scope or s.topic) == cause
        return GOVERNING_KINDS[s.kind] if speaks else None

    def resolve(self, candidates, r: Request) -> "Resolution":
        if r.cause is None or r.date_of_loss is None:
            return Resolution(None, [], {}, "withhold",
                              "no cause of loss" if r.cause is None else "no date of loss")
        admitted, excluded = [], {}
        for sid in candidates:
            v = self.admit(self.spans[sid], r)
            (admitted.append(sid) if v.ok else excluded.__setitem__(sid, v.reason))
        states = [getattr(r, f) for (line, _), f in self.rules.items() if line == r.line]
        if self.unknown_blocks and None in states:
            return Resolution(None, admitted, excluded, "withhold", "jurisdiction unknown")
        ranked = sorted(((self.precedence(self.spans[x], r.cause), x) for x in admitted
                         if self.precedence(self.spans[x], r.cause) is not None),
                        key=lambda t: -t[0])
        if not ranked:
            return Resolution(None, admitted, excluded, "withhold", "nothing in force governs")
        top = [x for p, x in ranked if p == ranked[0][0]]
        if len({self.spans[x].doc_id for x in top}) > 1:
            return Resolution(None, admitted, excluded, "withhold",
                              "tie: " + " vs ".join(sorted(top)))
        beaten = [x for p, x in ranked[1:] if x not in top]
        return Resolution(top[0], admitted, excluded, "ok", "governs", beaten)

    def check(self, cited: str | None, res: "Resolution") -> Verdict:
        """The citation must be the span the rules chose, not one that resembles it."""
        if res.status != "ok":
            return Verdict(False, res.reason)
        if cited != res.governing:
            return Verdict(False, f"cited {cited}; rules choose {res.governing}")
        return Verdict(True, "citation follows the rules")

    def audit(self, det_id, r: Request, res: "Resolution", cited, corpus_version):
        g = self.spans.get(res.governing) if res.governing else None
        rec = {"determination": det_id, "as_of": str(r.date_of_loss), "role": r.role,
               "tenant": r.tenant, "cause": r.cause, "status": res.status,
               "reason": res.reason, "cited": cited,
               "governing": None if not g else {
                   "doc": g.doc_id, "edition": g.version, "page": g.page,
                   "jurisdiction": g.jurisdiction, "effective_from": str(g.effective_from),
                   "effective_to": str(self.end_of(g)[0])},
               "beat": res.beaten, "excluded": res.excluded,
               "policy": f"{self.version}#{self.fingerprint}", "corpus": corpus_version}
        rec["record"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:12]
        return rec


@dataclass
class Resolution:
    governing: str | None
    admitted: list
    excluded: dict
    status: str                                    # "ok" or "withhold"
    reason: str
    beaten: list = field(default_factory=list)


def shingles(text: str, n: int = 8) -> set:
    w = [t.strip(".,;:'\"()").lower() for t in text.split()]
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def lineage(span: Span, restricted, n: int = 8) -> tuple:
    """sids of restricted spans that share an n-word run with this one: copied text."""
    mine = shingles(span.text, n)
    return tuple(r.sid for r in restricted if mine & shingles(r.text, n))
