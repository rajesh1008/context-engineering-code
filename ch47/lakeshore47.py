"""Chapter 47 stand-in world: Lakeshore's three-agent claims flow in shadow.

Everything here is a seeded simulation. Model behaviour (what a summariser
keeps, how often a long window loses a detail, what a reviewer catches) is
a parameterised stand-in chosen to reproduce the mechanisms the chapter
describes; the structures it exercises live in contextlab.multiagent.
"""
from __future__ import annotations

import math
import random

from contextlab.multiagent import (Brief, Fact, Handle, HandoffContract, HandoffRecord,
                                   LoopSpec, SharedState, TokenLedger, brief_gaps, derive,
                                   preflight, provenance_coverage, summary_probe, validate)

SEED = 47
SINGLE_TOKENS = 16_003          # Ch35: tokens per determination, single pipeline
RATE = 0.0348 / 16_003          # Ch35 blended US$ per token
CASES = {"plain": 395, "conditions": 135, "edition": 72, "deductible": 86,
         "limit": 162, "conflict": 64, "withhold": 86}
P_GOV_END = {"plain": .25, "conditions": .35, "edition": .20, "deductible": .90,
             "limit": .85, "conflict": 1.0, "withhold": 0.0}
FOUND_ANYWAY = 0.6              # endorsement lands in packet without the list
REVIEW_CATCH = {"single": 0.0, "record": 0.85, "share": 0.6, "summary": 0.45,
                "summary-injected": 0.45}   # summary flows give the reviewer no handles
FACTS_TOKENS = 600              # claim facts the single pipeline gets from the system


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def rot(tokens: int) -> float:
    """Stand-in context-rot curve: chance a long window loses one pinned detail."""
    return 0.01 + 0.12 * sigmoid((tokens - 30_000) / 6_000)


def world(seed=SEED):
    rng = random.Random(seed)
    claims = []
    for case, n in CASES.items():
        for _ in range(n):
            gov_end = rng.random() < P_GOV_END[case]
            claims.append(dict(
                id=f"C{len(claims):04d}", case=case, gov_end=gov_end,
                discussed=gov_end and rng.random() < 0.45,
                n_end=rng.randint(2, 6),
                transcript=max(3000, min(25000, int(rng.lognormvariate(math.log(8500), .35)))),
                guess=rng.random() < 0.09, misread=rng.random() < 0.004,
                bulletin=case in ("edition", "conflict") or rng.random() < 0.06,
                superseded=rng.random() < 0.35, lookalike=rng.random() < 0.40,
                searches=rng.randint(4, 7), u=[rng.random() for _ in range(8)]))
    order = list(range(len(claims)))
    rng.shuffle(order)
    claims = [claims[i] for i in order]
    pool = [c for c in claims if c["case"] in ("limit", "conflict") and not c["misread"]]
    for c in pool[:3]:
        c["baseline_wrong"] = True     # the context layer's own residual errors (Ch39: 3)
    return claims


# ---------------------------------------------------------------- handoffs
def summary_facts(c, loss_only=False):
    """(name, tokens, salience) facts a summariser chooses from."""
    end_sal = 0.15 if loss_only else 0.5
    facts = [("peril", 15, .95), ("date_of_loss", 8, .9), ("state", 4, .85),
             ("form_edition", 10, .7 if not loss_only else .35)]
    facts += [(f"narrative{i}", 30, .3 + .6 * c["u"][i % 8]) for i in range(10)]
    facts += [(f"letter{i}", 35, .2 + .4 * c["u"][(i + 3) % 8]) for i in range(4)]
    facts.append(("endorsements", 12 * c["n_end"], end_sal))
    if c["discussed"]:
        facts.append(("gov_endorsement_in_narrative", 30, .8))
    return facts


def summarise(c, budget=400, loss_only=False):
    kept, used = set(), 0
    for name, tok, _ in sorted(summary_facts(c, loss_only), key=lambda f: -f[2]):
        if used + tok <= budget:
            kept.add(name)
            used += tok
    return kept, used


def record_for(c) -> HandoffRecord:
    h = lambda s, t="record": (Handle(f"{c['id']}:{s}", "v1", t),)
    facts = {n: Fact(n, n, h(n)) for n in ("endorsements", "form_edition", "state")}
    facts["date_of_loss"] = Fact("date_of_loss", "dol", h("fnol", "party"), by="agent:intake")
    facts["peril"] = Fact("peril", "peril", h("fnol", "party"), by="agent:intake")
    return HandoffRecord(c["id"], "intake", "coverage", facts)


CONTRACT = HandoffContract(
    receiver="coverage",
    required=("peril", "date_of_loss", "state", "form_edition", "endorsements"),
    must_cite=("peril", "date_of_loss", "form_edition", "endorsements"),
    parser_only=("endorsements", "form_edition"))


def determine(c, strategy, rng, budget=400):
    """Return (wrong_clause, coverage_tokens, handoff_tokens, endorsement_survived)."""
    wrong = c.get("baseline_wrong", False)
    survived = True
    if strategy == "single":
        return wrong, SINGLE_TOKENS, 0, True
    if c["misread"] and c["case"] in ("plain", "conditions"):
        wrong = True                                   # intake reads the peril wrong
    if strategy == "record":
        rec = record_for(c)
        assert not validate(rec, CONTRACT)
        hand = rec.tokens()
    elif strategy == "share":
        hand = c["transcript"]
        lost = c["gov_end"] and rng.random() < rot(SINGLE_TOKENS + hand)
        anchored = c["guess"] and rng.random() < 0.08
        survived = not lost
        wrong = wrong or (lost and rng.random() > FOUND_ANYWAY) or anchored
    else:
        kept, hand = summarise(c, budget, loss_only=(strategy == "summary-injected"))
        has_end = "endorsements" in kept or "gov_endorsement_in_narrative" in kept
        survived = has_end or not c["gov_end"]
        if c["gov_end"] and not has_end and rng.random() > FOUND_ANYWAY:
            wrong = True
        if c["case"] == "edition" and "form_edition" not in kept and rng.random() < .5:
            wrong = True
    return wrong, SINGLE_TOKENS - FACTS_TOKENS + hand, hand, survived


def run_strategies(claims):
    out = {}
    for strat in ("single", "share", "summary", "summary-injected", "record"):
        rng = random.Random(f"{SEED}-{strat}")
        wrong = surv = tok = hand = 0
        probe_flags = after = 0
        rv = random.Random(f"{SEED}-{strat}-review")
        for c in claims:
            w, t, h, s = determine(c, strat, rng)
            wrong += w; tok += t; hand += h
            if w:   # clean-context review sees errors the flow added, not the layer's own
                after += c.get("baseline_wrong", False) or rv.random() > REVIEW_CATCH[strat]
            surv += s if c["gov_end"] else 0
            if strat.startswith("summary"):
                kept, _ = summarise(c, 400, strat == "summary-injected")
                text = " ".join(sorted(kept))
                probe_flags += bool(summary_probe(text, ["endorsements", "form_edition"]))
        n_gov = sum(c["gov_end"] for c in claims)
        out[strat] = dict(wrong=wrong, per_1000=round(1000 * wrong / len(claims), 1),
                          after_review=after,
                          coverage_tokens=round(tok / len(claims)),
                          handoff_tokens=round(hand / len(claims)),
                          endorsement_survival=round(surv / n_gov, 3),
                          probe_flagged=probe_flags)
    return out


def budget_sweep(claims, budgets=(150, 200, 250, 300, 400, 500, 650, 800)):
    gov = [c for c in claims if c["gov_end"]]
    rows = []
    for b in budgets:
        for inj in (False, True):
            ok = 0
            for c in gov:
                kept, _ = summarise(c, b, inj)
                ok += "endorsements" in kept or "gov_endorsement_in_narrative" in kept
            rows.append(dict(budget=b, injected=inj, survival=round(ok / len(gov), 3)))
    return rows


def rot_curve():
    return [dict(tokens=t, p=round(rot(t), 4)) for t in range(10_000, 52_001, 2_000)]


# ---------------------------------------------------------------- sub-agents
FULL_BRIEF = Brief("Find the state bulletins that govern this claim's coverage question",
                   {"state": "WI", "date_of_loss": "dol", "form_edition": "HO 3 ed.",
                    "peril": "water back-up"},
                   "digest <= 400 tokens; every finding with a span handle and version",
                   "read only; no determination; stop after 8 searches")
THIN_BRIEF = Brief("Research water back-up bulletins in Wisconsin", {"state": "WI"},
                   "", "")


def research(claims):
    need = ("state", "date_of_loss", "form_edition", "peril")
    modes = {}
    for mode in ("inline", "fork", "isolated-full", "isolated-thin", "isolated-small"):
        rng = random.Random(f"{SEED}-{mode}")
        wrong = peak = tok = cost = 0
        rows = [c for c in claims if c["bulletin"]]
        for c in rows:
            debris = c["searches"] * 1_400
            if mode == "inline":
                window = SINGLE_TOKENS + debris
                p = 0.03 + rot(window) * 1.6
                sub, price = 0, 1.0
            elif mode == "fork":
                window = SINGLE_TOKENS + 350
                p = 0.03 + rot(SINGLE_TOKENS + debris) * 1.3
                sub, price = SINGLE_TOKENS + debris + 350, 1.0
            else:
                window = SINGLE_TOKENS + 350
                brief = THIN_BRIEF if mode == "isolated-thin" else FULL_BRIEF
                p = 0.03
                if brief_gaps(brief, need) and c["superseded"]:
                    p = 0.8                    # picks the newest bulletin, not the one in force
                if mode == "isolated-small" and c["lookalike"]:
                    p += 0.18                  # look-alike bulletins need judgement
                sub, price = 1_500 + debris + 350, (0.25 if mode == "isolated-small" else 1.0)
            wrong += rng.random() < p
            peak = max(peak, window)
            tok += sub + (debris if mode == "inline" else 0)
            cost += (sub * price + (debris if mode == "inline" else 0)) * RATE
        modes[mode] = dict(n=len(rows), bulletin_wrong=wrong,
                           wrong_rate=round(wrong / len(rows), 3),
                           coverage_window_mean=SINGLE_TOKENS + (
                               round(sum(c["searches"] * 1400 for c in rows) / len(rows))
                               if mode == "inline" else 350),
                           research_tokens_per_claim=round(tok / len(rows)),
                           research_cost_per_claim=round(cost / len(rows), 4))
    return modes


# ---------------------------------------------------------------- review
def review(claims, seed=SEED):
    rng = random.Random(f"{seed}-review")
    seeded = []
    clean = [c for c in claims if not c.get("baseline_wrong")]
    for i, c in enumerate(clean[:150]):
        seeded.append(dict(id=c["id"], kind=("endorsement", "amount", "edition")[i % 3],
                           anchor=rng.random()))
    # detection: needs the evidence for that error kind, and not to share the anchoring
    reviewers = {
        "same-session": dict(evidence={"endorsement": .65, "amount": .85, "edition": .7},
                             shares=0.6, false_alarm=.008),
        "briefed-subagent": dict(evidence={"endorsement": .6, "amount": .85, "edition": .65},
                                 shares=0.4, false_alarm=.012),
        "clean-context": dict(evidence={"endorsement": .9, "amount": .8, "edition": .75},
                              shares=0.0, false_alarm=.025)}
    out = {}
    for name, r in reviewers.items():
        rr = random.Random(f"{seed}-{name}")
        caught = {"endorsement": 0, "amount": 0, "edition": 0}
        for s in seeded:
            sees = rr.random() < r["evidence"][s["kind"]]
            fooled = rr.random() < r["shares"] * s["anchor"]
            caught[s["kind"]] += sees and not fooled
        fa = sum(rr.random() < r["false_alarm"] for _ in range(1000))
        out[name] = dict(caught=sum(caught.values()), of=len(seeded),
                         by_kind=caught, false_alarms_per_1000=fa,
                         recall=round(sum(caught.values()) / len(seeded), 3))
    return out


# ---------------------------------------------------------------- shared state
def blackboard(claims):
    owners = {"facts": "intake", "determination": "coverage", "findings": "review",
              "approval": "adjuster"}
    open_board = SharedState({k: None for k in owners})
    owned = SharedState(owners)
    rng = random.Random(f"{SEED}-board")
    clashes = refused = 0
    for c in claims:
        writes = [("intake", "facts"), ("coverage", "determination"), ("review", "findings")]
        if rng.random() < 0.05:
            writes.append(("review", "determination"))   # reviewer "fixes" the draft
        if rng.random() < 0.02:
            writes.append(("coverage", "facts"))         # coverage edits a fact it doubts
        seen = {}
        for agent, key in writes:
            if key in seen and seen[key] != agent:
                clashes += 1
            seen[key] = agent
            owned.write(agent, key, c["id"])
        refused = len(owned.refused())
    return dict(open_board_overwrites=clashes, single_writer_refused=refused)


# ---------------------------------------------------------------- provenance
def provenance(claims, strategy):
    stmts = []
    for c in claims[:200]:
        if strategy == "record":
            ins = (Handle(f"{c['id']}:decl", "v1", "record"), Handle(f"{c['id']}:form", "v3"))
        elif strategy == "share":
            ins = (Handle(f"{c['id']}:form", "v3"),) if c["u"][0] > .3 else ()
        else:
            ins = (Handle(f"{c['id']}:form", "v3"),) if c["u"][0] > .45 else ()
        stmts.append(derive("covered under ...", "agent:coverage", ins))
        stmts.append(derive("facts as handed off", "agent:intake",
                            (Handle(f"{c['id']}:fnol", "v1", "party"),) if strategy == "record" else ()))
    tiers = {}
    for s in stmts:
        tiers[s.tier] = tiers.get(s.tier, 0) + 1
    return dict(coverage=round(provenance_coverage(stmts), 3), tiers=tiers)


# ---------------------------------------------------------------- tokens per agent
def tokens_per_agent(claims):
    led = TokenLedger()
    for c in claims:
        led.add(c["id"], "1 intake", 1_400 + c["transcript"] + 900 + record_for(c).tokens())
        led.add(c["id"], "2 coverage", SINGLE_TOKENS - FACTS_TOKENS + record_for(c).tokens()
                + (350 if c["bulletin"] else 0))
        led.add(c["id"], "3 research sub-agent",
                (1_500 + c["searches"] * 1_400 + 350) if c["bulletin"] else 0)
        led.add(c["id"], "4 review", 900 + 550 + 1_800 + 400 + 600 + 700 + 250)
    per = led.per_claim_by_agent()
    total = sum(per.values())
    return dict(per_agent=per, total=total, ratio=round(total / SINGLE_TOKENS, 2),
                cost=round(total * RATE, 4))


# ---------------------------------------------------------------- multi-peril
def fan_out(seed=SEED, n=104, coupled=38):
    rng = random.Random(f"{seed}-peril")
    items = [dict(coupled=i < coupled, perils=2 if rng.random() < .8 else 3) for i in range(n)]
    modes = {"one agent, one question per peril": (.04, .08, lambda k: SINGLE_TOKENS * (1 + .35 * (k - 1))),
             "fan-out, isolated briefs": (.04, .45, lambda k: 12_000 * k + 4_000),
             "fan-out, coupling in briefs": (.04, .10, lambda k: 12_300 * k + 4_000)}
    for x in items:
        x["u"] = rng.random()            # common random numbers across modes
    out = {}
    for name, (pi, pc, tok) in modes.items():
        wi = sum(x["u"] < pi for x in items if not x["coupled"])
        wc = sum(x["u"] < pc for x in items if x["coupled"])
        out[name] = dict(wrong_independent=wi, of_independent=n - coupled,
                         wrong_coupled=wc, of_coupled=coupled,
                         tokens=round(sum(tok(x["perils"]) for x in items) / n))
    return out


# ---------------------------------------------------------------- unattended loop
def bulletin_loop(open_claims=212, affected=37):
    good = LoopSpec("event:bulletin.published", "assembler", "verifier", "store@current",
                    "all affected claims re-checked or token cap reached",
                    "state/bulletin-recheck.json", 900_000, "Tom Brennan")
    bad = LoopSpec("event:bulletin.published", "assembler", "assembler", "assembler-cache",
                   "", "", 0, "")
    flagged_good = affected          # verifier reads the store at the current epoch
    flagged_bad = 0                  # verifier reads the same stale cache as the assembler
    screen = open_claims * 120       # citation-index lookup per open claim
    redo = affected * (SINGLE_TOKENS + 5_900)
    return dict(open_claims=open_claims, affected=affected,
                preflight_good=preflight(good), preflight_bad=preflight(bad),
                flagged_independent=flagged_good, flagged_shared_view=flagged_bad,
                tokens=screen + redo, cost=round((screen + redo) * RATE, 2),
                within_cap=screen + redo <= good.max_tokens)
