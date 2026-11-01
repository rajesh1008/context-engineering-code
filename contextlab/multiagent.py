"""contextlab.multiagent (v10.5): context across steps and agents.

Chapter 47. Standard library only. The shapes meant to be copied are:
PacketSpec (what each step may see), HandoffRecord and its contract
validator (typed facts that carry span handles, never free prose alone),
the summary probe (does a summary keep the constraints it must keep?),
Brief (what a sub-agent is told), Delegation (whose entitlement
a sub-agent's packet is built under), SharedState (one writer per field),
derive() (cross-agent provenance at the minimum trust tier), the token
ledger per agent, and LoopSpec with its preflight check. Model behaviour
in the chapter's lab is a seeded stand-in; these structures are not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

VERSION = "10.5"
TIERS = ("operator", "governed", "record", "party", "public")   # Ch37, highest first


# ---------------------------------------------------------------- packets
@dataclass(frozen=True)
class PacketSpec:
    """What one step of a multi-step flow is allowed to see."""
    step: str
    sources: tuple            # source kinds, e.g. ("record", "spans", "rules")
    budget: int               # tokens for the whole packet
    handles_only: tuple = ()  # sources passed as handles, never as text
    excludes: tuple = ()      # sources this step must not see


def check_spec(spec: PacketSpec, packet: dict) -> list[str]:
    """Return violations of a spec by an assembled packet {source: (tokens, kind)}."""
    issues = []
    used = sum(t for t, _ in packet.values())
    if used > spec.budget:
        issues.append(f"{spec.step}: {used} tokens over budget {spec.budget}")
    for src, (_, kind) in packet.items():
        if src in spec.excludes:
            issues.append(f"{spec.step}: excluded source {src} present")
        elif src not in spec.sources:
            issues.append(f"{spec.step}: undeclared source {src}")
        elif src in spec.handles_only and kind != "handle":
            issues.append(f"{spec.step}: {src} passed as text, spec says handle")
    return issues


# ---------------------------------------------------------------- handoffs
@dataclass(frozen=True)
class Handle:
    span_id: str
    version: str
    tier: str = "governed"


@dataclass(frozen=True)
class Fact:
    name: str
    value: object
    handles: tuple = ()       # tuple[Handle, ...]
    by: str = "parser"        # "parser", "agent:<name>" or "human:<name>"


@dataclass
class HandoffRecord:
    claim_id: str
    sender: str
    receiver: str
    facts: dict = field(default_factory=dict)       # name -> Fact
    open_questions: tuple = ()
    notes: str = ""                                  # free text, never authority

    def tokens(self) -> int:
        n = 40 + sum(12 + 6 * len(f.handles) + len(str(f.value)) // 4
                     for f in self.facts.values())
        return n + len(self.notes) // 4 + 15 * len(self.open_questions)


@dataclass(frozen=True)
class HandoffContract:
    receiver: str
    required: tuple                   # fact names the receiver cannot act without
    must_cite: tuple = ()             # facts that must carry at least one handle
    parser_only: tuple = ()           # facts a parser extracts; an agent may not


def validate(rec: HandoffRecord, c: HandoffContract) -> list[str]:
    """Contract check at the boundary. Empty list means the handoff may proceed."""
    out = [f"missing {n}" for n in c.required if n not in rec.facts]
    for n in c.must_cite:
        if n in rec.facts and not rec.facts[n].handles:
            out.append(f"{n} has no handle")
    for n in c.parser_only:
        if n in rec.facts and rec.facts[n].by != "parser":
            out.append(f"{n} written by {rec.facts[n].by}, contract says parser")
    return out


def summary_probe(summary: str, pinned: list[str]) -> list[str]:
    """Return pinned constraints (form numbers, dates, limits) a summary dropped."""
    low = summary.lower()
    return [p for p in pinned if p.lower() not in low]


# ---------------------------------------------------------------- sub-agents
@dataclass(frozen=True)
class Brief:
    """What a delegating agent tells a sub-agent. Four parts, all required."""
    objective: str
    facts: dict               # the claim facts the subtask depends on
    output_format: str        # e.g. "digest<=400 tokens, every finding with a handle"
    boundaries: str           # what not to do, where to stop


def brief_gaps(b: Brief, needs: tuple) -> list[str]:
    gaps = [k for k in ("objective", "output_format", "boundaries") if not getattr(b, k)]
    return gaps + [f"fact:{n}" for n in needs if n not in b.facts]


# ---------------------------------------------------------------- delegation
@dataclass(frozen=True)
class Delegation:
    """Whose entitlement a sub-agent's packet is built under: the end user's, narrowed."""
    user: str                 # the audience every packet is minted for (Ch46)
    subject: str              # the claim
    actors: tuple             # chain of agents acting for the user
    corpora: frozenset
    budget: int


def delegate(parent: Delegation, actor: str, corpora=None, budget=None) -> Delegation:
    """Narrow, never widen. Raises ValueError on any widening."""
    corpora = frozenset(corpora) if corpora is not None else parent.corpora
    budget = parent.budget if budget is None else budget
    if not corpora <= parent.corpora:
        raise ValueError(f"widening: {sorted(corpora - parent.corpora)}")
    if budget > parent.budget:
        raise ValueError("widening: budget")
    return Delegation(parent.user, parent.subject, parent.actors + (actor,), corpora, budget)


# ---------------------------------------------------------------- shared state
class SharedState:
    """Blackboard with one owner per field. Writes by anyone else are refused."""

    def __init__(self, owners: dict):
        self.owners = dict(owners)
        self.values: dict = {}
        self.log: list = []

    def write(self, agent: str, key: str, value) -> bool:
        ok = self.owners.get(key) == agent
        self.log.append((agent, key, value, ok))
        if ok:
            self.values[key] = value
        return ok

    def refused(self) -> list:
        return [e for e in self.log if not e[3]]


# ---------------------------------------------------------------- provenance
@dataclass(frozen=True)
class Derived:
    text: str
    by: str
    inputs: tuple             # tuple[Handle, ...] the statement rests on
    tier: str


def derive(text: str, by: str, inputs: tuple) -> Derived:
    """Agent output has no tier of its own: it takes the lowest tier of its inputs."""
    worst = max((TIERS.index(h.tier) for h in inputs), default=TIERS.index("public"))
    return Derived(text, by, tuple(inputs), TIERS[worst])


def provenance_coverage(statements: list) -> float:
    """Share of determination statements that trace to at least one span handle."""
    if not statements:
        return 1.0
    return sum(1 for s in statements if s.inputs) / len(statements)


# ---------------------------------------------------------------- tokens
class TokenLedger:
    def __init__(self):
        self.rows: list = []

    def add(self, claim: str, agent: str, tokens: int):
        self.rows.append((claim, agent, tokens))

    def per_claim_by_agent(self) -> dict:
        claims = {c for c, _, _ in self.rows} or {None}
        out: dict = {}
        for _, a, t in self.rows:
            out[a] = out.get(a, 0) + t
        return {a: round(t / len(claims)) for a, t in sorted(out.items())}


# ---------------------------------------------------------------- loops
@dataclass(frozen=True)
class LoopSpec:
    trigger: str              # "schedule:..." or "event:..."
    assembler: str            # component that builds packets
    verifier: str             # component that checks results
    verifier_reads: str       # "store@current" or "assembler-cache"
    stop: str                 # machine-checkable condition
    state_file: str           # where progress persists between runs
    max_tokens: int
    escalate_to: str          # the human on the loop


def preflight(spec: LoopSpec) -> list[str]:
    issues = []
    if spec.verifier == spec.assembler or spec.verifier_reads == "assembler-cache":
        issues.append("verifier shares the assembler's view")
    if not spec.stop:
        issues.append("no machine-checkable stop")
    if not spec.state_file:
        issues.append("no persisted state")
    if spec.max_tokens <= 0:
        issues.append("no token bound")
    if not spec.escalate_to:
        issues.append("no human on the loop")
    return issues
