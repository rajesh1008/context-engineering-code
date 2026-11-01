"""contextlab.platform (v10.3): the context layer as a service for many teams.

Chapter 46. Standard library only. The platform is a deterministic stand-in:
retrieval is a topic lookup and the clock is a number, so the chapter's
failure modes reproduce exactly. The parts meant to be copied are the shapes:
Principal (identity comes from the authenticated token, never the body),
Packet (signed, scoped to one audience, one subject and one entitlement
epoch), the three calls assemble / explain / feedback, TeamBudget and the
chargeback Ledger, the contract renderers with consumer contract tests, and
the SLO evaluator.
"""
from __future__ import annotations

import hashlib
import hmac
import heapq
import json
from dataclasses import dataclass, field

VERSION = "10.3"
TIERS = ("governing", "guidance", "team-note", "public")   # authority order


def _h(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


# ---------------------------------------------------------------- content
@dataclass(frozen=True)
class Span:
    sid: str
    corpus: str
    topic: str
    tier: str                      # one of TIERS
    tokens: int
    roles: frozenset               # roles that may read it
    policy_id: str | None = None   # subject-scoped: one policy's own documents
    agency: str | None = None      # book of business the policy belongs to
    current: bool = True           # False once superseded


@dataclass(frozen=True)
class Corpus:
    name: str
    owner: str                     # owning team; "platform" for shared corpora
    shared: bool                   # readable by every onboarded team (subject to roles)
    readers: frozenset = frozenset()   # extra teams a private corpus is shared with


# --------------------------------------------------------------- identity
@dataclass(frozen=True)
class Principal:
    """Built by the platform from a verified token. Callers cannot set it."""
    team: str
    service: str
    user: str
    role: str
    agency: str | None = None
    policy_ids: frozenset = frozenset()


class Directory:
    """Entitlement source of truth: grants plus an epoch that moves on change."""

    def __init__(self):
        self.users: dict[str, Principal] = {}
        self.epoch = 1
        self.agency_of: dict[str, str] = {}      # policy_id -> agency

    def grant(self, p: Principal):
        self.users[p.user] = p

    def revoke(self, user: str):
        self.users.pop(user, None)
        self.epoch += 1

    def move_policy(self, policy_id: str, agency: str):
        self.agency_of[policy_id] = agency
        self.epoch += 1

    def resolve(self, token: dict) -> Principal | None:
        """A token is {'user', 'service', 'sig'}; the stand-in checks the sig."""
        if token.get("sig") != _h("tok", token.get("user"), token.get("service"))[:16]:
            return None
        return self.users.get(token["user"])


def issue_token(user: str, service: str) -> dict:
    return {"user": user, "service": service,
            "sig": _h("tok", user, service)[:16]}


# ------------------------------------------------------------ entitlement
def entitled(span: Span, p: Principal, corpora: dict, agency_of: dict) -> tuple[bool, str]:
    c = corpora[span.corpus]
    if not (c.shared or c.owner == p.team or p.team in c.readers):
        return False, f"corpus {c.name} is private to {c.owner}"
    if p.role not in span.roles:
        return False, f"role {p.role} not permitted"
    if span.policy_id is not None:
        if p.role == "policyholder" and span.policy_id not in p.policy_ids:
            return False, "another policyholder's document"
        if p.role == "agent" and agency_of.get(span.policy_id, span.agency) != p.agency:
            return False, "policy outside the agent's book"
    return True, "entitled"


# ----------------------------------------------------------------- packets
@dataclass
class Packet:
    packet_id: str
    audience: str                  # the end user the packet was built for
    team: str
    subject: str                   # claim or policy the packet is about
    topic: str
    epoch: int
    issued_at: float
    expires_at: float
    spans: list
    withheld: list                 # (sid, reason)
    corpus_versions: dict
    sig: str = ""

    def canonical(self) -> str:
        return json.dumps([self.packet_id, self.audience, self.team, self.subject,
                           self.epoch, self.issued_at, self.expires_at,
                           [s.sid for s in self.spans]], sort_keys=True)


def sign(packet: Packet, key: bytes) -> str:
    return hmac.new(key, packet.canonical().encode(), hashlib.sha256).hexdigest()[:24]


def verify_packet(packet: Packet, p: Principal, subject: str, now: float,
                  epoch: int, key: bytes) -> tuple[bool, str]:
    """Run where the packet is used (the model gateway), not where it was cached."""
    if not hmac.compare_digest(packet.sig, sign(packet, key)):
        return False, "signature"
    if packet.audience != p.user:
        return False, "audience"
    if packet.subject != subject:
        return False, "subject"
    if packet.epoch != epoch:
        return False, "entitlement epoch"
    if now >= packet.expires_at:
        return False, "expired"
    return True, "ok"


# ----------------------------------------------------------------- budgets
@dataclass
class TeamBudget:
    team: str
    monthly_tokens: int            # packet tokens the team may draw per month
    per_request_cap: int           # largest packet one call may ask for
    concurrency: int               # share of assembly workers
    hard: bool = True              # hard cap refuses; soft cap only alerts


@dataclass
class Charge:
    team: str
    packet_tokens: int
    retrieval_units: int
    t: float


class Ledger:
    def __init__(self):
        self.charges: list[Charge] = []

    def add(self, c: Charge):
        self.charges.append(c)

    def used(self, team: str) -> int:
        return sum(c.packet_tokens for c in self.charges if c.team == team)

    def bill(self, prices: dict, shared_monthly: float, method="usage") -> dict:
        """Direct cost per team plus shared cost allocated by usage share or evenly."""
        direct = {}
        for c in self.charges:
            d = direct.setdefault(c.team, 0.0)
            direct[c.team] = d + c.packet_tokens * prices["per_token"] \
                + c.retrieval_units * prices["per_unit"]
        total = sum(direct.values()) or 1.0
        teams = sorted(direct)
        out = {}
        for t in teams:
            share = direct[t] / total if method == "usage" else 1 / len(teams)
            out[t] = {"direct": direct[t], "shared": shared_monthly * share,
                      "total": direct[t] + shared_monthly * share}
        return out


# ---------------------------------------------------------------- platform
class Platform:
    """assemble / explain / feedback over shared and private corpora."""

    def __init__(self, spans, corpora, directory: Directory, budgets: dict,
                 key: bytes = b"lakeshore-46", ttl: float = 0.25,
                 corpus_versions: dict | None = None):
        self.by_topic: dict[str, list[Span]] = {}
        self.by_subject: dict[str, list[Span]] = {}
        for s in spans:
            if s.policy_id is not None:
                self.by_subject.setdefault(s.policy_id, []).append(s)
            else:
                self.by_topic.setdefault(s.topic, []).append(s)
        self.corpora = {c.name: c for c in corpora}
        self.dir = directory
        self.budgets = budgets
        self.key = key
        self.ttl = ttl
        self.versions = corpus_versions or {c.name: "v1" for c in corpora}
        self.ledger = Ledger()
        self.packets: dict[str, Packet] = {}
        self.feedback_log: list[dict] = []

    def assemble(self, token: dict, subject: str, topic: str, budget: int,
                 now: float) -> tuple[Packet | None, str]:
        p = self.dir.resolve(token)
        if p is None:
            return None, "401 unauthenticated"
        b = self.budgets.get(p.team)
        if b is None:
            return None, "403 team not onboarded"
        if budget > b.per_request_cap:
            return None, f"400 budget {budget} over cap {b.per_request_cap}"
        if b.hard and self.ledger.used(p.team) + budget > b.monthly_tokens:
            return None, "429 monthly quota"
        kept, withheld, used = [], [], 0
        cands = sorted(self.by_topic.get(topic, []) + self.by_subject.get(subject, []),
                       key=lambda s: (TIERS.index(s.tier), not s.current, s.sid))
        for s in cands:
            ok, why = entitled(s, p, self.corpora, self.dir.agency_of)
            if ok and not s.current:
                ok, why = False, "superseded"
            if ok and used + s.tokens > budget:
                ok, why = False, "over budget"
            if ok:
                kept.append(s)
                used += s.tokens
            else:
                withheld.append((s.sid, why))
        pid = _h("pkt", p.user, subject, topic, now)[:12]
        pk = Packet(pid, p.user, p.team, subject, topic, self.dir.epoch, now,
                    now + self.ttl, kept, withheld,
                    {c: self.versions[c] for c in sorted({s.corpus for s in kept})})
        pk.sig = sign(pk, self.key)
        self.packets[pid] = pk
        self.ledger.add(Charge(p.team, used, 1, now))
        return pk, "200"

    def explain(self, packet_id: str) -> dict:
        pk = self.packets[packet_id]
        return {"packet_id": pk.packet_id, "audience": pk.audience,
                "epoch": pk.epoch, "corpus_versions": pk.corpus_versions,
                "included": [(s.sid, s.tier, s.corpus) for s in pk.spans],
                "withheld": list(pk.withheld)}

    def feedback(self, packet_id: str, signal: str, detail: str = "") -> dict:
        if packet_id not in self.packets:
            return {"status": "404"}
        rec = {"packet_id": packet_id, "signal": signal, "detail": detail,
               "team": self.packets[packet_id].team}
        self.feedback_log.append(rec)
        return {"status": "202"}


# ---------------------------------------------------------------- contract
def render(packet: Packet, version: str) -> dict:
    """v1: flat spans with text-ish fields. v2: adds tier, withheld count, epoch;
    renames 'source' to 'corpus' and orders spans by authority, not by score."""
    if version == "v1":
        return {"id": packet.packet_id,
                "spans": [{"sid": s.sid, "source": s.corpus, "tokens": s.tokens}
                          for s in sorted(packet.spans, key=lambda s: s.sid)]}
    if version == "v2":
        return {"contract": "v2", "id": packet.packet_id, "epoch": packet.epoch,
                "expires_at": packet.expires_at,
                "spans": [{"sid": s.sid, "corpus": s.corpus, "tier": s.tier,
                           "tokens": s.tokens} for s in packet.spans],
                "withheld": len(packet.withheld)}
    raise ValueError(version)


@dataclass
class ContractTest:
    consumer: str
    name: str
    check: object                  # callable(rendered) -> bool


def run_contract_tests(tests, rendered: dict) -> list[tuple[str, str, bool]]:
    out = []
    for t in tests:
        try:
            ok = bool(t.check(rendered))
        except (KeyError, TypeError, IndexError):
            ok = False
        out.append((t.consumer, t.name, ok))
    return out


# ------------------------------------------------------------------ load
def simulate(requests, workers: int, mode: str, shares: dict | None = None):
    """Discrete-event queue. requests: (t_arrive, team, klass, service_s).
    Returns (team, klass, latency_s, t_arrive) per request.
    mode 'fifo' serves in arrival order; 'fair' caps each team's concurrent
    workers at its share and serves interactive before batch."""
    reqs = sorted(requests)
    waiting: list = []
    running: list = []                      # heap of (t_end, team)
    busy: dict = {}
    done = []
    i, now = 0, 0.0
    seq = 0
    while i < len(reqs) or waiting or running:
        nxt_arr = reqs[i][0] if i < len(reqs) else float("inf")
        nxt_end = running[0][0] if running else float("inf")
        if nxt_end <= nxt_arr:
            now, team = heapq.heappop(running)
            busy[team] -= 1
        else:
            now = nxt_arr
            t, team, klass, svc = reqs[i]
            pr = 0 if (mode == "fair" and klass == "interactive") else 1
            heapq.heappush(waiting, (pr if mode == "fair" else 0, t, seq, team, klass, svc))
            seq += 1
            i += 1
        # dispatch
        free = workers - len(running)
        skipped = []
        while free > 0 and waiting:
            item = heapq.heappop(waiting)
            _, t, _, team, klass, svc = item
            if mode == "fair" and busy.get(team, 0) >= shares[team]:
                skipped.append(item)
                continue
            busy[team] = busy.get(team, 0) + 1
            heapq.heappush(running, (now + svc, team))
            done.append((team, klass, now + svc - t, t))
            free -= 1
        for it in skipped:
            heapq.heappush(waiting, it)
    return done


def pct(xs, q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[k]


# ------------------------------------------------------------------- SLOs
@dataclass
class SLO:
    name: str
    target: float                  # fraction of good events required
    budgeted: bool = True          # False: any bad event is an incident

    def evaluate(self, good: int, total: int) -> dict:
        frac = good / total if total else 1.0
        allowed = (1 - self.target) * total if self.budgeted else 0
        bad = total - good
        return {"slo": self.name, "target": self.target, "good": good,
                "total": total, "attained": frac, "bad": bad,
                "budget_left": (allowed - bad) / allowed if allowed else
                (1.0 if bad == 0 else -1.0),
                "met": bad <= allowed}
