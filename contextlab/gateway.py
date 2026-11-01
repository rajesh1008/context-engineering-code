"""contextlab.gateway (v8.4): the context layer behind an enterprise model gateway.

Chapter 38. Standard library only. Everything here is a stand-in: the
gateway, its deployments, keys and caches are simulated so the chapter's
failure modes can be reproduced deterministically. The adapter half
(CredentialProvider, ResidencyPolicy, verify_record, cache_namespace,
probe_check, join_traces) is the part meant to be copied.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

VERSION = "8.4"
HOUR = 1.0
WEEK = 168.0


def _h(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


# ---------------------------------------------------------------- catalogue
@dataclass
class Deployment:
    name: str                 # deployment, e.g. "sim-large@2026-04/us-1"
    model: str                # dated model identifier, e.g. "sim-large@2026-04"
    kind: str                 # "chat", "embed" or "verify"
    region: str               # "us", "eu" or "global"
    retire_at: float | None = None   # hour on the simulation clock
    successor: str | None = None     # model the gateway may alias to


@dataclass
class UsageRecord:
    request_id: str
    caller: str
    requested: str
    served: str | None        # model the gateway says it served
    deployment: str | None
    region: str | None
    key_id: str | None
    status: int
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    fallback: bool = False
    attempts: int = 1
    transforms: tuple = ()
    body_hash: str = ""
    t: float = 0.0


# --------------------------------------------------------------------- keys
class KeyIssuer:
    """Issues one application key per rotation period with an overlap window."""

    def __init__(self, period_h: float = WEEK, overlap_h: float = 24.0, salt="k"):
        self.period, self.overlap, self.salt = period_h, overlap_h, salt

    def generation(self, t: float) -> int:
        return int(t // self.period)

    def current(self, t: float) -> tuple[str, str]:
        g = self.generation(t)
        return f"key-{g:03d}", _h(self.salt, g)[:24]

    def valid(self, secret: str, t: float) -> str | None:
        g = self.generation(t)
        if secret == _h(self.salt, g)[:24]:
            return f"key-{g:03d}"
        if g > 0 and secret == _h(self.salt, g - 1)[:24] \
                and t < g * self.period + self.overlap:
            return f"key-{g - 1:03d}"
        return None


class CredentialProvider:
    """Adapter side. mode='at_start' reproduces the defect; 'per_batch' is the fix.

    per_batch fetches from the secret store for every batch and refreshes once
    on a 401 before giving up; a second 401 is an error, never a retry loop.
    """

    def __init__(self, issuer: KeyIssuer, mode: str = "per_batch", started_at=0.0):
        self.issuer, self.mode = issuer, mode
        self._held = issuer.current(started_at)[1]
        self.fetches = 0

    def secret(self, t: float) -> str:
        if self.mode == "at_start":
            return self._held
        self.fetches += 1
        return self.issuer.current(t)[1]

    def refresh(self, t: float) -> str | None:
        if self.mode == "at_start":
            return None
        self.fetches += 1
        return self.issuer.current(t)[1]


# ------------------------------------------------------------------ gateway
class SimGateway:
    """A stand-in enterprise gateway.

    alias_mode: 'refuse' answers 404 for a retired pinned model; 'forward'
    serves the successor and says so; 'forward_masked' serves the successor
    and reports the pinned name (the dishonest case only a probe catches).
    cache_mode: 'pooled' (one upstream credential for every tenant),
    'client_key' (namespace from a caller-supplied key) or 'identity'
    (namespace derived from the authenticated caller).
    fallback: 'none', 'in_region' (another deployment of the same model in
    an allowed region) or 'any' (whatever is up, anywhere).
    """

    def __init__(self, deployments, issuer, alias_mode="refuse",
                 cache_mode="identity", fallback="in_region", transforms=()):
        self.deps = list(deployments)
        self.issuer = issuer
        self.alias_mode, self.cache_mode, self.fallback = alias_mode, cache_mode, fallback
        self.transforms = tuple(transforms)
        self.cache: dict[str, set] = {}
        self.down: set[str] = set()
        self.log: list[UsageRecord] = []
        self._n = 0

    def _rid(self) -> str:
        self._n += 1
        return f"gw-{self._n:07d}"

    def _of(self, model):
        return [d for d in self.deps if d.model == model]

    def resolve(self, model: str, t: float):
        """Return (model actually served, name to report) or (None, None)."""
        ds = self._of(model)
        if not ds:
            return None, None
        d = ds[0]
        if d.retire_at is not None and t >= d.retire_at:
            if self.alias_mode == "refuse" or not d.successor:
                return None, None
            rep = model if self.alias_mode == "forward_masked" else d.successor
            return d.successor, rep
        return model, model

    def _route(self, model, allowed):
        ups = [d for d in self._of(model) if d.name not in self.down]
        first = self._of(model)[0]
        inreg = [d for d in ups if not allowed or d.region in allowed]
        pref = [d for d in self._of(model) if not allowed or d.region in allowed]
        primary = pref[0] if pref else first
        if primary.name not in self.down:
            return primary, False, 1
        if inreg:
            return inreg[0], True, 2
        if self.fallback == "any":
            pool = ups or [d for d in self.deps if d.kind == first.kind
                           and d.name not in self.down]
            if pool:
                return pool[0], True, 2
        return None, False, 2

    def _namespace(self, caller: str, client_key: str | None) -> str:
        if self.cache_mode == "pooled":
            return "org-upstream"
        if self.cache_mode == "client_key":
            return client_key or caller
        return cache_namespace(caller)

    def call(self, caller, model, secret, t, prefix="", suffix="", out_tokens=0,
             client_key=None, allowed_regions=None) -> UsageRecord:
        rid = self._rid()
        key_id = self.issuer.valid(secret, t)
        if key_id is None:
            return self._log(UsageRecord(rid, caller, model, None, None, None, None, 401, t=t))
        served, reported = self.resolve(model, t)
        if served is None:
            return self._log(UsageRecord(rid, caller, model, None, None, None, key_id, 404, t=t))
        dep, fb, attempts = self._route(served, allowed_regions)
        if self.fallback == "none" and fb:
            dep = None
        if dep is None:
            return self._log(UsageRecord(rid, caller, model, None, None, None, key_id, 503,
                                         attempts=attempts, t=t))
        if dep.model != served:
            reported = dep.model
        body = prefix + suffix
        if "redact" in self.transforms:
            body = body.replace("POLICYHOLDER", "[name]")
        if "truncate" in self.transforms and len(body) > 4000:
            body = body[-4000:]
        ptoks = len(prefix) // 4
        seen = self.cache.setdefault(self._namespace(caller, client_key), set())
        key = _h(dep.model, prefix)
        cached = ptoks if (prefix and key in seen) else 0
        if prefix:
            seen.add(key)
        return self._log(UsageRecord(rid, caller, model, reported, dep.name, dep.region,
                                     key_id, 200, ptoks + len(suffix) // 4, cached,
                                     out_tokens, fb, attempts, self.transforms,
                                     _h(body)[:16], t))

    def _log(self, rec):
        self.log.append(rec)
        return rec


# ------------------------------------------------------------ adapter side
def cache_namespace(authenticated_caller: str) -> str:
    """Cache namespace from the identity the gateway authenticated, never from the body."""
    return "ns-" + _h("cache-ns", authenticated_caller)[:16]


@dataclass
class ResidencyPolicy:
    rules: dict = field(default_factory=lambda: {"claim_file": {"us"},
                                                 "public": {"us", "eu", "global"}})

    def allowed(self, data_class: str) -> set:
        return self.rules.get(data_class, set())

    def permits(self, data_class: str, region: str | None) -> bool:
        return region in self.allowed(data_class)


def verify_record(rec: UsageRecord, pinned: str, data_class: str,
                  policy: ResidencyPolicy, sent_hash: str | None = None,
                  declared=("redact",)) -> list[str]:
    """Findings for one gateway usage record; an empty list means it matches the trace."""
    out = []
    if rec.status != 200:
        return [f"status_{rec.status}"]
    if rec.served != pinned:
        out.append("served_differs")
    if not policy.permits(data_class, rec.region):
        out.append("region")
    if rec.fallback:
        out.append("fallback")
    undeclared = [x for x in rec.transforms if x not in declared]
    if undeclared:
        out.append("undeclared_transform")
    if sent_hash is not None and rec.body_hash != sent_hash and not rec.transforms:
        out.append("body_changed")
    return out


# ------------------------------------------------------- embedding probes
def sim_embed(model: str, text: str, dim: int = 256) -> list[float]:
    """Stand-in embedder: model-specific token directions, mean-pooled, unit length."""
    v = [0.0] * dim
    for w in text.lower().split():
        hx = _h(model, w)
        rnd = random.Random(hx)
        for i in range(dim):
            v[i] += rnd.gauss(0.0, 1.0)
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def probe_check(embed, probes, reference, tol: float = 0.98) -> tuple[bool, float]:
    """Chapter 14's check, run through the gateway: minimum cosine over probe texts."""
    m = min(cosine(embed(p), r) for p, r in zip(probes, reference))
    return m >= tol, m


# ------------------------------------------------------------- trace join
def join_traces(traces: list[dict], records: list[UsageRecord]) -> dict:
    """Join determination traces to gateway records on the gateway request id."""
    by_id = {r.request_id: r for r in records}
    matched = [t for t in traces if t.get("request_id") in by_id]
    return {"traces": len(traces), "matched": len(matched),
            "unmatched": len(traces) - len(matched),
            "answerable": sum(1 for t in matched
                              if by_id[t["request_id"]].served
                              and by_id[t["request_id"]].region)}
