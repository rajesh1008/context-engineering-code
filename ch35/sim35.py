"""Chapter 35 stand-in: what each tier does with a determination.

Note: deterministic stand-in. No model runs. A tier reads the cause right
when the claim's hardness h is within its capability for the reasoning budget
(CAP); a draft carries a flagged sentence when the claim's fixed uniform draw
falls under FLAG_P. The same draws are used under every configuration, so
two configurations differ only where their choices differ. Prices are the
illustrative ones in contextlab.costs. Token profiles are Lakeshore's packet
after Chapters 32 to 34.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from contextlab.costs import BUDGETS, Call, CacheLayer, Ledger, Route, Router
import lakeshore35 as L

CAP = {("small", "none"): .30, ("small", "low"): .42, ("small", "medium"): .50, ("small", "high"): .55,
       ("mid", "none"): .52, ("mid", "low"): .64, ("mid", "medium"): .76, ("mid", "high"): .84,
       ("large", "none"): .74, ("large", "low"): .86, ("large", "medium"): 1.0, ("large", "high"): 1.0}
FLAG_BASE = {"small": .46, "mid": .40, "large": .36}
FLAG_MULT = {"none": 1.25, "low": 1.10, "medium": 1.0, "high": .62}
RERANK_MISS = {"small": .060, "mid": .028, "large": .020}
EXTRACT_ERR = {"small": .060, "mid": .030, "large": .010}
PREFIX_HIT = 0.92           # prefix cache hit share after Chapter 34's layout
DAY_SIZE = 3200

PROFILE = {   # stage: (stable prefix, dynamic input, output)
    "rewrite": (800, 300, 60),
    "rerank": (400, 2200, 80),
    "extract": (1900, 700, 220),
    "draft": (1500, 1700, 480),
    "verify": (700, 1100, 160),
}


def call(stage, tier, extra_in=0, thinking=0, drop_dynamic=0):
    pre, dyn, out = PROFILE[stage]
    dyn = max(dyn + extra_in - drop_dynamic, 0)
    cached = int(pre * PREFIX_HIT)
    return Call(stage, tier, pre - cached + dyn, cached, out, thinking)


def think(budget, h):
    return min(BUDGETS[budget], int(600 + 7000 * h)) if BUDGETS[budget] else 0


def flagged(tier, budget, h, u):
    return u < min(FLAG_BASE[tier] * FLAG_MULT[budget] * (0.4 + h), 0.95)


@dataclass
class Config:
    name: str
    router: Router | None = None
    fixed: Route = Route("large", "medium", False)
    retrieval: tuple = ("large", "large")      # rewrite tier, rerank tier
    extract: str = "large"
    retrieval_cache: str = "off"               # off | versioned | unversioned
    tool_cache: bool = False
    mechanical_exit: bool = False
    hard_redraft: bool = False
    escalate: Route = Route("large", "medium", False)
    shadow: float = 0.0                        # share of small exits re-run large
    rename: dict = field(default_factory=dict)  # failure injection: a feature renamed upstream


@dataclass
class Outcome:
    d: object
    calls: list
    route: str
    wrong: bool
    flag: bool
    exited: bool = False
    escalated: bool = False
    stale: bool = False
    shadow_disagree: bool = False


def run(cfg: Config, dets, bulletin_change_at=None):
    bulletin_change_at = L.BULLETIN_CHANGE_AT if bulletin_change_at is None else bulletin_change_at
    rcache = CacheLayer("retrieval", lambda x: x[0])
    tcache = CacheLayer("tool results", lambda d: (d.policy, "watermark"))
    outs = []
    for d in dets:
        calls, stale, support_miss = [], False, False
        bv = 2 if d.i >= bulletin_change_at else 1
        # retrieval side: rewrite + rerank, or a retrieval-cache hit
        def retrieve():
            calls.append(call("rewrite", cfg.retrieval[0]))
            calls.append(call("rerank", cfg.retrieval[1]))
            return {"miss": d.u_rerank < RERANK_MISS[cfg.retrieval[1]], "bv": bv}
        if cfg.retrieval_cache == "off":
            r = retrieve()
        else:
            key = d.retrieval_key(cfg.retrieval_cache == "versioned", bv)
            r, hit = rcache.get_or((key, d), retrieve)
            stale = (r["bv"] != bv) and d.state == "WI" and d.intent == "water"
        support_miss = r["miss"]
        # extraction from the record and notes, checked against the record
        def extract_record():
            return 700
        tool_in = 700
        if cfg.tool_cache:
            _, hit = tcache.get_or(d, extract_record)
            tool_in = 0 if hit else 700
        calls.append(call("extract", cfg.extract, extra_in=d.notes_tokens, drop_dynamic=700 - tool_in))
        if cfg.extract != "large" and d.u_extract < EXTRACT_ERR[cfg.extract]:
            calls.append(call("extract", "large", extra_in=d.notes_tokens))   # cross-check failed: retry
        # exit before the model: the outcome is fixed by rule
        if cfg.mechanical_exit and d.mechanical:
            calls.append(call("verify", "large"))
            outs.append(Outcome(d, calls, "rule", d.record_error, False, exited=True))
            continue
        feats = {cfg.rename.get(k, k): v for k, v in d.feats().items()}
        route = cfg.router.route(feats) if cfg.router else cfg.fixed
        band = cfg.router.band(feats) if cfg.router else 1

        def draft(rt, u):
            calls.append(call("draft", rt.tier, extra_in=30 * min(d.photos, 10), thinking=think(rt.budget, d.h)))
            calls.append(call("verify", "large", thinking=think("low", d.h)))
            misread = d.h > CAP[(rt.tier, rt.budget)]
            return {"misread": misread, "flag": flagged(rt.tier, rt.budget, d.h, u) or support_miss}

        dr = draft(route, d.u_flag)
        exited, escalated, shadow_dis = False, False, False
        if route.exit_check:
            if dr["flag"]:
                escalated = True
                dr = draft(cfg.escalate, d.u_flag2)
            else:
                exited = True
                if cfg.shadow and (d.i * 7919) % 1000 < cfg.shadow * 1000:
                    sh = draft(cfg.escalate, d.u_flag2)
                    shadow_dis = dr["misread"] != sh["misread"]
        elif cfg.hard_redraft and band == 2 and dr["flag"]:
            dr = draft(route, d.u_flag2)
        wrong = d.record_error or dr["misread"] or stale
        outs.append(Outcome(d, calls, f"{route.tier}-{route.budget}", wrong, dr["flag"] and not wrong,
                            exited, escalated, stale, shadow_dis))
    return outs, rcache.stats(), tcache.stats()


def summary(outs):
    led = Ledger()
    for o in outs:
        led.add(o.calls)
    rev = [o for o in outs if o.d.reviewed]
    wrong = sum(o.wrong for o in rev)
    flag = sum(o.flag for o in rev)
    return {"cost": led.per_determination(), "tokens": led.tokens_per_determination(),
            "wrong": wrong, "flagged": flag, "clean": len(rev) - wrong - flag,
            "day": led.per_determination() * DAY_SIZE, "ledger": led}
