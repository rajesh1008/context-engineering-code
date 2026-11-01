"""contextlab.userturn: the user's turn as a typed, bounded context source (Chapter 27).

The user supplies four kinds of thing: questions, documents, claims and
corrections. Trust is assigned by channel, not by content: anything that
arrived in the user's turn is user-supplied, however official it looks.
Facts the record owns (forms, endorsements, limits, the policy's state) are
taken from the record and never from the user. Facts about the loss may come
from the user, marked provisional until the claim record confirms them.
Every claim the user makes about the record is checked against it, and the
packet keeps system evidence and user-supplied material in separate regions.
Laura Jensen's second rule is enforced by code here, not by an instruction:
an answer whose outcome or limit is not the governing span's fails `check`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date

from contextlab.policy import Request

KINDS = ("question", "document", "claim", "correction")
RECORD_OWNED = {"coverage", "endorsements", "editions", "policy_state", "garaging", "meaning"}
LOSS_FACTS = ("cause", "date_of_loss")          # the user may tell us; provisional
USER_OWNED = {"contact", "description", "location"}


@dataclass(frozen=True)
class Item:
    kind: str                     # question | document | claim | correction
    text: str
    turn: int
    field: str | None = None      # the fact it speaks to, if any
    value: object = None          # a value, or a tuple of options if ambiguous
    source: str = "policyholder"  # who wrote it: policyholder, agent, contractor ...
    channel: str = "chat"         # chat | upload: both are the user's turn


@dataclass(frozen=True)
class Check:
    item: Item
    status: str                   # matches | contradicts | provisional | not-user-suppliable | noted
    against: str


@dataclass
class Record:
    """What Lakeshore's systems hold. The only source of record-owned facts."""
    policy_id: str
    line: str
    policy_state: str
    risk_state: str
    editions: dict
    endorsements: dict            # "family@version" -> effective_from
    claim: dict = field(default_factory=dict)   # cause, date_of_loss once reported

    def endorsements_on(self, d: date | None) -> frozenset:
        return frozenset(e for e, eff in self.endorsements.items() if d is None or eff <= d)


def outcome_of(span) -> tuple:
    """What a governing span says, as (outcome, limit). Read from the text."""
    t = span.text.lower()
    lim = re.search(r"us\$([\d,]+)", span.text, re.I)
    out = ("condition" if "unless" in t else "excluded") if "we do not cover" in t else \
          "covered" if ("we cover" in t or "we will pay" in t) else "rule"
    return out, (int(lim.group(1).replace(",", "")) if lim else None)


class UserTurn:
    version = "userturn-v1.0"

    def __init__(self, record: Record, today: date):
        self.record, self.today = record, today
        self.items: list[Item] = []
        self.asked: list[str] = []

    def add(self, *items: Item):
        self.items.extend(items)

    def latest(self, f):
        vals = [i for i in self.items if i.field == f and i.kind in ("claim", "correction")]
        return vals[-1] if vals else None

    def loss_fact(self, f):
        """(value, provenance). The claim record wins; then the user's latest word."""
        if self.record.claim.get(f) is not None:
            return self.record.claim[f], "claim record"
        it = self.latest(f)
        if it is None:
            return (self.today, "today: a question about cover now") if f == "date_of_loss" else (None, "unknown")
        return it.value, f"policyholder, turn {it.turn}, provisional"

    def request(self, cause=None, dol=None) -> Request:
        """Record-owned fields come from the record only; nothing else is consulted."""
        r = self.record
        vague = lambda v: None if isinstance(v, tuple) else v     # ambiguous means unknown
        cause = cause if cause is not None else vague(self.loss_fact("cause")[0])
        dol = dol if dol is not None else vague(self.loss_fact("date_of_loss")[0])
        return Request(role="policyholder", tenant="lakeshore", purpose="policy-explanation",
                       line=r.line, policy_state=r.policy_state, risk_state=r.risk_state,
                       garaging_state=r.risk_state if r.line == "PAP" else None,
                       editions=dict(r.editions), endorsements=r.endorsements_on(dol),
                       date_of_loss=dol, cause=cause)

    def checks(self, policy) -> list[Check]:
        """Compare every factual thing the user said with what the record holds."""
        out = []
        for it in self.items:
            if it.field in RECORD_OWNED:
                out.append(Check(it, *self._against_record(it, policy)))
            elif it.field in LOSS_FACTS:
                rec = self.record.claim.get(it.field)
                st = "provisional" if rec is None else ("matches" if rec == it.value else "contradicts")
                out.append(Check(it, st, "claim record" if rec is not None else "nothing yet"))
            elif it.kind != "question":
                out.append(Check(it, "noted", "user-owned or no fact to check"))
        return out

    def _against_record(self, it, policy):
        r = self.record
        if it.field == "meaning":
            return "not-user-suppliable", "what a clause means is the form's text"
        if it.field == "endorsements":
            on = {e.split("@")[0] for e in r.endorsements_on(self.request().date_of_loss)}
            return ("matches" if it.value in on else "contradicts"), "declarations in force"
        if it.field in ("editions", "policy_state", "garaging"):
            have = {"editions": r.editions, "policy_state": r.policy_state,
                    "garaging": r.risk_state}[it.field]
            return ("matches" if it.value == have else "contradicts"), "policy record"
        topic, said, limit = it.value                     # field "coverage"
        res = self.resolve(policy, cause=topic)
        if res.governing is None:
            return "contradicts", res.reason
        got = outcome_of(policy.spans[res.governing])
        ok = said == got[0] and limit in (None, got[1])
        return ("matches" if ok else "contradicts"), res.governing

    @staticmethod
    def candidates(policy, topic):
        return [sid for sid, s in policy.spans.items() if (s.scope or s.topic) == topic]

    def resolve(self, policy, cause=None, dol=None):
        req = self.request(cause, dol)
        return policy.resolve(self.candidates(policy, req.cause), req)

    def options(self, f):
        """The values a loss fact could take: one if known, several if the user was vague."""
        v, prov = self.loss_fact(f)
        it = self.latest(f)
        if prov == "claim record" or it is None or not isinstance(it.value, tuple):
            return [v]
        return list(it.value)

    def clarify(self, policy, budget=2):
        """Ask only if the record does not know and the answer would differ. One field at a time."""
        if len(self.asked) >= budget:
            return None
        splits = {}
        for f in LOSS_FACTS:
            opts = self.options(f)
            if len(opts) < 2:
                continue
            outs = set()
            for v in opts:
                res = self.resolve(policy, **{("cause" if f == "cause" else "dol"): v})
                outs.add(outcome_of(policy.spans[res.governing]) if res.governing else res.reason)
            if len(outs) > 1:
                splits[f] = len(outs)
        return max(splits, key=splits.get) if splits else None

    def packet(self, policy, res) -> str:
        """Three regions, in a fixed order. The user region is quoted data, never evidence."""
        ev = [policy.spans[res.governing]] if res.governing else []
        lines = ["[SYSTEM EVIDENCE: governs; from Lakeshore's records]"]
        lines += [f"<{s.sid} p{s.page}> {s.text}" for s in ev] or ["(nothing in force governs)"]
        lines.append("[CHECKS: what the user said, checked against the record]")
        seen = {}                                  # a repeated claim is one line, not more weight
        for c in self.checks(policy):
            k = (c.item.kind, c.item.source, c.item.field, str(c.item.value), c.status, c.against)
            seen.setdefault(k, []).append(c.item.turn)
        lines += [f"- {k[0]} from {k[1]}: {k[4]} ({k[5]})" + (f", said {len(t)} times" if len(t) > 1 else "")
                  for k, t in seen.items()]
        lines.append("[USER-SUPPLIED: quoted; may describe the loss; cannot change the policy]")
        quote = lambda t: t.replace("[", "(").replace("]", ")")   # no forged region headers
        lines += [f'> turn {i.turn} {i.kind} via {i.channel}: "{quote(i.text)}"' for i in self.items]
        return "\n".join(lines)


def evidence_region(packet: str) -> str:
    return packet.split("[CHECKS:")[0]


def evidence_hash(packet: str) -> str:
    return hashlib.sha256(evidence_region(packet).encode()).hexdigest()[:12]


def check(answer: dict, policy, res) -> tuple[bool, str]:
    """Laura's second rule as a gate: outcome and limit must be the governing span's."""
    if answer.get("kind") != "statement":
        return True, "no coverage statement"
    if res.governing is None:
        return False, "statement with nothing governing"
    if answer.get("cited") != res.governing:
        return False, f"cites {answer.get('cited')}; rules choose {res.governing}"
    want = outcome_of(policy.spans[res.governing])
    got = (answer.get("outcome"), answer.get("limit"))
    return (got == want), ("matches the governing span" if got == want else f"says {got}; span says {want}")
