"""The Rulebook: experts' tacit rules as decision tables (Chapter 8).

A Rule is a decision table with a default, owned by an expert. It renders as an
instruction span (the expert's statement) for the claims on its topic, its table
is checked against the documents, and its rows generate adherence probes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import date

from .instructions import Span, count_tokens


@dataclass(frozen=True)
class Row:
    """One decision-table row: every condition given must hold."""
    governs: str                       # doc_id of the governing document
    endorsement: str | None = None     # the policy carries this endorsement
    state: str | None = None
    on_or_after: date | None = None    # date of loss
    before: date | None = None

    def matches(self, det) -> bool:
        if self.endorsement is not None and not det.carries(self.endorsement):
            return False
        if self.state is not None and det.state != self.state:
            return False
        if self.on_or_after is not None and det.loss < self.on_or_after:
            return False
        if self.before is not None and det.loss >= self.before:
            return False
        return True

    def as_dict(self) -> dict:
        return {k: (v.isoformat() if isinstance(v, date) else v)
                for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    topic: str
    statement: str                     # the rule in the expert's words
    rows: tuple[Row, ...]
    default: str                       # doc_id when no row matches
    expert: str = "Tom Brennan"

    def resolve(self, det) -> str:
        return next((r.governs for r in self.rows if r.matches(det)), self.default)

    def exception(self, det) -> bool:
        """True when a row fires: the rule selects something other than its default."""
        return any(r.matches(det) for r in self.rows)

    def as_dict(self) -> dict:
        return {"rule_id": self.rule_id, "topic": self.topic, "statement": self.statement,
                "rows": [r.as_dict() for r in self.rows], "default": self.default,
                "expert": self.expert}


@dataclass(frozen=True)
class Rulebook:
    version: str
    rules: tuple[Rule, ...]
    changelog: str = ""
    header: str = "Lakeshore claims rulebook"

    def topics(self) -> list[str]:
        return sorted({r.topic for r in self.rules})

    def rule_for(self, topic: str) -> Rule | None:
        return next((r for r in self.rules if r.topic == topic), None)

    def fingerprint(self) -> str:
        """sha256 of the rules' content; the version and changelog are not content."""
        body = json.dumps([r.as_dict() for r in self.rules], sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:12]

    def line(self, rule: Rule) -> str:
        return f"{rule.rule_id}. {rule.statement}"

    def render(self, topic: str | None = None) -> str:
        rules = [r for r in self.rules if topic is None or r.topic == topic]
        return "\n".join([f"{self.header} {self.version}:"] + [self.line(r) for r in rules])

    def span(self, det) -> Span | None:
        """The progressive span: only the rules for the claim's topic, or nothing."""
        rule = self.rule_for(det.topic)
        if rule is None:
            return None
        return Span(f"RULEBOOK-{self.version}-{det.topic}", self.render(det.topic),
                    {"kind": "instruction", "tier": 0, "rulebook": self.version,
                     "fingerprint": self.fingerprint(), "rule": rule.as_dict()})

    def tokens(self) -> int:
        return count_tokens(self.render())

    def with_rule(self, rule: Rule, version: str, changelog: str) -> "Rulebook":
        return replace(self, version=version, changelog=changelog,
                       rules=tuple(rule if r.rule_id == rule.rule_id else r for r in self.rules))


@dataclass
class Disagreement:
    det: object
    rule_says: str
    documents_say: object              # the governing Provision


@dataclass
class RuleCheck:
    rule: Rule
    checked: int
    conflicts: list[Disagreement] = field(default_factory=list)

    @property
    def agree(self) -> int:
        return self.checked - len(self.conflicts)


def check(book: Rulebook, corpus) -> list[RuleCheck]:
    """Expert review against the documents: every reviewed determination on a rule's
    topic is resolved by the table and by the documents, and any difference is listed."""
    out = []
    for rule in book.rules:
        dets = [d for d in corpus.determinations if d.topic == rule.topic]
        rc = RuleCheck(rule, len(dets))
        for d in dets:
            gov = corpus.governing(d)
            said = rule.resolve(d)
            if said != gov.doc_id:
                rc.conflicts.append(Disagreement(d, said, gov))
        out.append(rc)
    return out


def probes(book: Rulebook, corpus) -> list:
    """Adherence probes: the reviewed determinations where a rule selects an exception."""
    return [d for d in corpus.determinations
            if (r := book.rule_for(d.topic)) is not None and r.exception(d)]


def load_yaml_rule(path) -> Rule:
    """Read a rule-capture record (Listing 8.1) into a Rule."""
    import yaml
    rec = yaml.safe_load(open(path))
    rows = tuple(Row(governs=t["governs"], endorsement=t.get("endorsement"), state=t.get("state"),
                     on_or_after=t.get("on_or_after"), before=t.get("before"))
                 for t in rec.get("table") or [])
    return Rule(rec["rule_id"], rec["topic"], " ".join(rec["statement"].split()), rows,
                rec["default"], rec.get("owner", "Tom Brennan"))
