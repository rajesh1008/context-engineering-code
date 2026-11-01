"""Deterministic stand-ins for Chapter 51. No model runs; numbers are illustrative."""
import random
from contextlab.finish import EdgeCase, Check, Runbook, Diagnostic

SEED = 51

# Margaret Hale's three targets, and the scoreboard rows the book has carried.
GOALS = {"lookup_minutes": ("<", 2.0),
         "wrong_clause_pct": ("<", 1.0),
         "traceable_pct": (">=", 100.0)}
BASELINE = {"lookup_minutes": 11.0, "wrong_clause_pct": 7.0, "traceable_pct": 0.0}

# The edge-case backlog: per 10,000 determinations, severity 1-5,
# share caught by today's diagnostics, engineering days to fix and test.
BACKLOG = [
    EdgeCase("E01", "Two endorsements amend one clause; order by edition", 4.0, 5, 0.50, 6),
    EdgeCase("E02", "Bulletin effective mid-term; loss before, report after", 3.0, 5, 0.40, 4),
    EdgeCase("E03", "Stamp over the limit on a scanned endorsement", 2.0, 4, 0.75, 8),
    EdgeCase("E04", "Third peril on a multi-peril claim", 6.0, 4, 0.60, 5),
    EdgeCase("E05", "Claimant quotes what an agent promised", 9.0, 3, 0.90, 3),
    EdgeCase("E06", "Boat on Great Lakes waters vs inland endorsement", 1.5, 4, 0.20, 3),
    EdgeCase("E07", "Continuous leak straddling a renewal", 5.0, 5, 0.30, 7),
    EdgeCase("E08", "Transferred policy with no term start reaches the retirement gate", 2.5, 2, 1.00, 2),
    EdgeCase("E09", "Mortgagee clause questions from partner banks", 12.0, 2, 0.70, 4),
    EdgeCase("E10", "Claim notes written in Spanish", 30.0, 2, 0.80, 10),
    EdgeCase("E11", "Businessowners file over the packet budget", 8.0, 3, 0.85, 6),
    EdgeCase("E12", "Bulletin published as a scanned letter", 0.8, 5, 0.25, 5),
    EdgeCase("E13", "Storm week: four times the volume", 40.0, 1, 0.95, 4),
    EdgeCase("E14", "Policyholder asks about an open determination", 15.0, 3, 0.95, 2),
]

# The hardening checklist: 24 checks in six areas. Each test reads the config.
def _c(key, area, text, f):
    return Check(key, area, text, f)

CHECKS = [
    _c("H01", "inputs", "Every source in the inventory has an owner", lambda c: c["sources_without_owner"] == 0),
    _c("H02", "inputs", "Document-quality gate fails closed", lambda c: c["docgate_fail_closed"]),
    _c("H03", "inputs", "Claimant text never enters the instruction slot", lambda c: c["claimant_text_in_instructions"] == 0),
    _c("H04", "inputs", "Ingest lag alarm under 24 hours", lambda c: c["ingest_lag_alarm_h"] <= 24),
    _c("H05", "retrieval", "Precedence and effective-date rules have tests", lambda c: c["policy_tests"] >= 1),
    _c("H06", "retrieval", "Term-aware retirement gate on", lambda c: c["term_aware_gate"]),
    _c("H07", "retrieval", "Entitlement enforced at the platform", lambda c: c["entitlement_at_platform"]),
    _c("H08", "retrieval", "Weekly reconciliation ran in the last 8 days", lambda c: c["days_since_reconcile"] <= 8),
    _c("H09", "model", "Every call goes through the gateway", lambda c: c["calls_outside_gateway"] == 0),
    _c("H10", "model", "Model pinned by version with a probe fingerprint", lambda c: c["model_pinned"]),
    _c("H11", "model", "Retire-by date at least 90 days away or a plan", lambda c: c["retire_by_days"] >= 90 or c["swap_plan"]),
    _c("H12", "model", "Degraded mode drilled this quarter", lambda c: c["degraded_mode_drilled"]),
    _c("H13", "outputs", "Citation parse rate alarm has an on-call owner", lambda c: c["parse_alarm_owner"] is not None),
    _c("H14", "outputs", "Nothing reaches a policyholder without a person", lambda c: c["auto_send"] == 0),
    _c("H15", "outputs", "Evidence pack sealed for every determination", lambda c: c["unsealed_packs"] == 0),
    _c("H16", "outputs", "Manual determinations carry a reason code", lambda c: c["manual_without_reason"] == 0),
    _c("H17", "operations", "Each runbook drilled in the last 6 months", lambda c: c["runbooks_undrilled"] == 0),
    _c("H18", "operations", "Write-once store restore tested", lambda c: c["restore_tested"]),
    _c("H19", "operations", "Platform SLO with an error budget", lambda c: c["slo_defined"]),
    _c("H20", "operations", "Per-team budgets and alerts", lambda c: c["team_budgets"]),
    _c("H21", "governance", "Asset register has no gaps", lambda c: c["register_gaps"] == 0),
    _c("H22", "governance", "Erasure drill passed this quarter", lambda c: c["erasure_drill_passed"]),
    _c("H23", "governance", "Gold set re-adjudication date set", lambda c: c["gold_readjudication_set"]),
    _c("H24", "governance", "Behaviour probes run before any model change", lambda c: c["behaviour_probes"] > 0),
]

CONFIG_BEFORE = dict(
    sources_without_owner=0, docgate_fail_closed=True, claimant_text_in_instructions=0,
    ingest_lag_alarm_h=24, policy_tests=212, term_aware_gate=True, entitlement_at_platform=True,
    days_since_reconcile=4, calls_outside_gateway=0, model_pinned=True, retire_by_days=160,
    swap_plan=True, degraded_mode_drilled=False, parse_alarm_owner=None, auto_send=0,
    unsealed_packs=0, manual_without_reason=1, runbooks_undrilled=2, restore_tested=False,
    slo_defined=True, team_budgets=True, register_gaps=0, erasure_drill_passed=True,
    gold_readjudication_set=True, behaviour_probes=0)

CONFIG_AFTER = dict(CONFIG_BEFORE, degraded_mode_drilled=True, parse_alarm_owner="context on-call",
                    manual_without_reason=0, runbooks_undrilled=0, restore_tested=True,
                    behaviour_probes=40)

# Six runbooks: (step, minutes). Discovery = minutes to work out what is happening.
RUNBOOKS = [
    Runbook("R1", "A governing document cited after it stopped governing", "context team",
            (("mark rows retired in every serving store", 10), ("drop cached results citing it", 5),
             ("replay since the change date", 25), ("hand affected files to Compliance", 10)),
            "restore prior lifecycle state from the catalogue"),
    Runbook("R2", "Gateway or model unavailable for 10 minutes", "platform on-call",
            (("switch drafting off; search-only mode on", 3), ("banner to adjusters", 2),
             ("check queue and cache health", 10), ("restore and replay the held queue", 20)),
            "stay in search-only mode"),
    Runbook("R3", "Model deprecation notice or forced swap", "platform team",
            (("open change record; pin current model", 15), ("run gold set and behaviour probes", 60),
             ("shadow per cell under the written rule", 30), ("release by cell", 20)),
            "alias back to the pinned model"),
    Runbook("R4", "Suspected injected or poisoned span", "security with context team",
            (("quarantine the source and its spans", 10), ("search packets that admitted it", 20),
             ("re-draft affected determinations", 30), ("add the pattern to the red-team set", 15)),
            "keep the source quarantined"),
    Runbook("R5", "Erasure request or partner exit", "privacy office",
            (("look up the register", 5), ("fan out to every asset", 30),
             ("compact and check absence", 20), ("send the response with evidence", 15)),
            "none: erasure is not rolled back"),
    Runbook("R6", "Volume above three times the daily norm", "claims operations with platform",
            (("raise the batch window; pause renewals job", 5), ("shed policyholder chat to FAQ", 5),
             ("watch p95 and cost per determination", 15), ("stand down and review", 20)),
            "restore normal quotas"),
]
DISCOVERY = {"R1": 35, "R2": 12, "R3": 40, "R4": 30, "R5": 20, "R6": 15}

# Examination drill: 30 determinations drawn from 1 October to 27 November 2026.
def exam_sample():
    rnd = random.Random(SEED)
    sample = []
    for i in range(30):
        manual = (i == 17)            # drafted by hand during a gateway outage
        sample.append({"id": f"D{rnd.randint(100000, 999999)}", "manual": manual,
                       "reason_code": not manual, "verifies": True,
                       "minutes": round(rnd.uniform(1.5, 3.5), 1)})
    return sample

# Two generations newer: the gold set and two behaviour changes.
GOLD_FULL_MIX = 1000
GOLD_EXTRA = 144
PROBES = 40          # older state bulletin narrows a newer endorsement

def gold_items():
    rnd = random.Random(SEED + 1)
    items = []
    for i in range(GOLD_FULL_MIX + GOLD_EXTRA):
        items.append({"i": i, "full_mix": i < GOLD_FULL_MIX,
                      "two_docs": rnd.random() < 0.09,
                      "cell": rnd.randrange(24), "precedence_probe": False})
    return items

def probe_items():
    return [{"i": 10000 + k, "full_mix": False, "two_docs": True, "cell": k % 24,
             "precedence_probe": True} for k in range(PROBES)]

PROD_WRONG = {41, 388, 902, 1077, 1130}         # 3 on the full mix, 2 on the extra 144
CAND_WRONG = {388, 1130}                         # the candidate fixes three of five

def run_model(items, model):
    rnd = random.Random(SEED + (7 if model == "candidate" else 3))
    out = []
    for it in items:
        if it["precedence_probe"]:
            wrong = model == "candidate" and (it["i"] - 10000) % 40 in {1, 4, 7, 9, 13, 18, 22, 26, 29, 33, 38}
        else:
            wrong = it["i"] in (CAND_WRONG if model == "candidate" else PROD_WRONG)
        new_format = model == "candidate" and it["two_docs"]
        tokens = 16003 * (1.04 if model == "candidate" else 1.0)
        latency = (6.1 if model == "candidate" else 6.4) * rnd.uniform(0.8, 1.3)
        faithful = (it["i"] * 37) % 1000 >= (76 if model == "candidate" else 92)
        out.append({**it, "wrong": wrong, "new_format": new_format, "tokens": tokens,
                    "latency": latency, "faithful": faithful})
    return out

def parse_ok(r, adapter=False):
    if r["i"] == 1100:                # one malformed citation in production too
        return False
    return adapter or not r["new_format"]

def _rate(rs, f):
    return sum(1 for r in rs if f(r)) / len(rs)

def _p95(xs):
    xs = sorted(xs)
    return xs[int(0.95 * (len(xs) - 1))]

def diagnostics(adapter=False):
    fm = lambda rs: [r for r in rs if r["full_mix"] and not r["precedence_probe"]]
    gold = lambda rs: [r for r in rs if not r["precedence_probe"]]
    def worst_cell(rs):
        per = {}
        for r in gold(rs):
            per.setdefault(r["cell"], 0)
            per[r["cell"]] += r["wrong"] or not parse_ok(r, adapter)
        return max(per.values())
    return [
        Diagnostic("citation parse rate, %", lambda rs: 100 * _rate(gold(rs), lambda r: parse_ok(r, adapter)), 99.5),
        Diagnostic("wrong-clause per 1,000, full mix", lambda rs: 1000 * _rate(fm(rs), lambda r: r["wrong"] or not parse_ok(r, adapter)), 4.0, False),
        Diagnostic("faithfulness-v2, %", lambda rs: 100 * _rate(gold(rs), lambda r: r["faithful"]), 88.0),
        Diagnostic("tokens per determination", lambda rs: sum(r["tokens"] for r in gold(rs)) / len(gold(rs)), 16003 * 1.10, False),
        Diagnostic("p95 latency, s", lambda rs: _p95([r["latency"] for r in gold(rs)]), 6.4 * 1.3 * 1.25, False),
        Diagnostic("worst cell, wrong of its items", worst_cell, 3, False),
    ]
