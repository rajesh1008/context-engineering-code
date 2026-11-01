"""Chapter 48 stand-in world: Lakeshore's three configurations, the world they point
at, and a behaviour model that turns a configuration into outcomes.

Deterministic. No language model runs. Each anti-pattern has a stated exposure (the
share of determinations it touches) and a stated chance of harm when it touches one.
The numbers echo the chapters where each failure was first measured; they are
stand-in parameters, not copies of those chapters' results. Draws come from a hash
of (label, item, month), so every rerun gives the same counts.
"""
import copy
import hashlib

from contextlab.configlint import World

N_DET = 1_000            # sampled determinations per month
N_EPISODES = 200         # subrogation-agent episodes per month
BASE_WRONG = 3           # the scoreboard's 0.3% (3/1,000), carried from Chapter 39
P_HOLD_BROKEN = 0.25     # unpinned hold broken per episode under threshold compaction


def u(*key):
    h = hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()
    return int(h[:12], 16) / 16 ** 12


EDITIONS = ["HO3-2022-01", "HO3-2024-06", "HO0495-2019", "HO0495-2024", "LMHO2311-2022",
            "LMHO2311-2025"]
FEEDS = {"bulletins-WI": "wm_wi", "bulletins-MN": "wm_mn", "bulletins-IL": "wm_il"}
RULES = {"R1-PRECEDENCE", "R2-CITE-PAGE", "R3-NO-USER-OVERRIDE", "SUBRO-HOLD-01",
         "DIARY-RESTRICT", "LEGAL-INSTR"}

DETERMINATION = {
    "name": "determination",
    "instructions": {"id": "claims-core", "version": "4.2", "sha": "9f1c2e"},
    "index": {"version": "spans-v9", "filter": "pre", "fill_missing_version": False},
    "ranking": {"order": ["authority", "similarity"], "precedence": list(EDITIONS)},
    "budget": {"max_tokens": 3_000, "floors": {"endorsements": 600}},
    "layout": {"prefix": ["instructions", "tool_defs"], "volatile": ["now", "claim"]},
    "memory": {"summaries": True, "provenance": True},
    "compaction": {"mode": "off"},
    "cache": {"ttl_s": 900, "key": ["query", "policy", "form_edition", "index_version",
                                    "wm_wi", "wm_mn", "wm_il"]},
    "feeds": [{"name": k, "watermark": v} for k, v in FEEDS.items()],
    "evals": {"judge_layer": 3},
    "tools": [{"name": "policy_record", "description": "Return the policy's form edition "
               "and endorsement list from the policy administration system."},
              {"name": "claim_facts", "description": "Return loss date, peril, location "
               "and reserve for one claim from the claims system."}],
}

ASSISTANT = copy.deepcopy(DETERMINATION)
ASSISTANT.update(name="policyholder-assistant")
ASSISTANT["instructions"] = {"id": "ph-assistant", "version": "1.6", "sha": "41d0aa"}
ASSISTANT["layout"] = {"prefix": ["instructions", "greeting", "today"],
                       "volatile": ["today", "turn"]}

SUBRO = copy.deepcopy(DETERMINATION)
SUBRO.update(name="subrogation-agent")
SUBRO["instructions"] = {"id": "subro-agent", "version": "2.1", "sha": "c7e913"}
SUBRO["compaction"] = {"mode": "threshold", "pins": ["SUBRO-HOLD-01", "DIARY-RESTRICT",
                                                     "LEGAL-INSTR"]}
SUBRO["tools"] = DETERMINATION["tools"] + [
    {"name": "claim_notes", "description": "Return the adjuster notes recorded on the "
     "claim file, newest first, with author and date."},
    {"name": "claim_diary", "description": "Return the diary entries recorded on the "
     "claim file, newest first, with author and date."},
    {"name": "send_demand", "description": "Send a subrogation demand to the third "
     "party's insurer; blocked while a hold is in force."}]

CONFIGS = [DETERMINATION, ASSISTANT, SUBRO]
ALL_TOOLS = {"policy_record", "claim_facts", "claim_notes", "claim_diary", "send_demand"}


def world(month):
    """The world as of a stand-in month (0 = the audit). Three changes land later."""
    rules, editions, feeds = set(RULES), set(EDITIONS), dict(FEEDS)
    if month >= 2:      # instructions v4.3: the hold rule is split in two
        rules -= {"SUBRO-HOLD-01"}
        rules |= {"SUBRO-HOLD-01A", "SUBRO-HOLD-01B"}
    if month >= 3:      # Illinois bulletins move to a new feed with its own watermark
        feeds["bulletins-IL-2"] = "wm_il2"
    if month >= 4:      # a new HO 04 95 edition is filed for renewals
        editions.add("HO0495-2027")
    return World(rule_ids=rules, editions=editions, feeds=feeds, tools=set(ALL_TOOLS))


# ---- behaviour: what each defect does to outcomes (stand-in parameters) ----
NEW_EDITION_SHARE = {4: 0.06, 5: 0.12}    # renewals reaching the new edition
P_WRONG_NO_PRECEDENCE = 0.30              # similarity picks the older edition
IL_SHARE, IL_STALE, P_WRONG_STALE = 0.15, 0.20, 0.50


def baseline_wrong(month):
    ranked = sorted(range(N_DET), key=lambda i: u("base", i, month))
    return set(ranked[:BASE_WRONG])


def month_outcomes(cfg_det, cfg_sub, month):
    """Wrong determinations per 1,000 and broken holds per 1,000 episodes."""
    w = world(month)
    wrong = baseline_wrong(month)
    missing_eds = w.editions - set(cfg_det["ranking"]["precedence"])
    unkeyed = [m for m in w.feeds.values() if m not in cfg_det["cache"]["key"]]
    for i in range(N_DET):
        if missing_eds and u("ed", i, month) < NEW_EDITION_SHARE.get(month, 0):
            if u("edp", i, month) < P_WRONG_NO_PRECEDENCE:
                wrong.add(i)
        if "wm_il2" in unkeyed and u("il", i, month) < IL_SHARE * IL_STALE:
            if u("ilp", i, month) < P_WRONG_STALE:
                wrong.add(i)
    pinned = [p for p in cfg_sub["compaction"]["pins"] if p in w.rule_ids]
    hold_pinned = any(p.startswith("SUBRO-HOLD") for p in pinned)
    broken = 0 if hold_pinned else sum(u("ep", e, month) < P_HOLD_BROKEN
                                       for e in range(N_EPISODES))
    return len(wrong), round(broken * 1000 / N_EPISODES)


def fixed(cfg, month):
    """What the owner changes when resolve() reports against the month's world."""
    w, c = world(month), copy.deepcopy(cfg)
    if c["compaction"].get("pins"):
        c["compaction"]["pins"] = [p for p in c["compaction"]["pins"] if p in w.rule_ids]
        c["compaction"]["pins"] += sorted(r for r in w.rule_ids if r.startswith("SUBRO-HOLD"))
    c["ranking"]["precedence"] = sorted(set(c["ranking"]["precedence"]) | w.editions)
    c["cache"]["key"] = list(dict.fromkeys(c["cache"]["key"] + list(w.feeds.values())))
    return c


# ---- the catalogue experiment: each anti-pattern applied to the clean configuration ----
# (metric, exposure share, chance of harm when exposed, cost index with clean = 100)
# metrics: wrong determinations, broken holds per 1,000 episodes, red runs per 1,000
# commits on a clean system (flaky), or cost alone
EFFECTS = {
    "AP01": ("wrong", 1.00, 0.020, 310),
    "AP02": ("wrong", 0.08, 0.60, 100),
    "AP03": ("wrong", 0.10, 0.15, 100),
    "AP04": ("wrong", 0.05, 0.30, 100),
    "AP05": ("broken", 1.00, P_HOLD_BROKEN, 100),
    "AP06": ("wrong", 0.12, 0.50, 100),
    "AP07": ("wrong", 0.04, 0.45, 100),
    "AP08": ("wrong", 0.04, 0.50, 100),
    "AP09": ("cost", 0.00, 0.00, 260),
    "AP10": ("wrong", 0.03, 0.60, 100),
    "AP11": ("flaky", 1.00, 0.30, 100),
    "AP12": ("wrong", 0.06, 0.20, 135),
}


def catalogue_run(rule):
    metric, exposure, p, cost = EFFECTS[rule]
    n = N_EPISODES if metric == "broken" else N_DET
    bad = sum(u(rule, "x", i) < exposure and u(rule, "p", i) < p for i in range(n))
    if metric in ("broken", "flaky"):
        return {"rule": rule, "metric": metric, "per_1000": round(bad * 1000 / n),
                "cost_index": cost}
    return {"rule": rule, "metric": metric, "per_1000": BASE_WRONG + bad, "cost_index": cost}


def with_antipattern(cfg, rule):
    """Return a copy of a clean configuration showing one anti-pattern on its face."""
    c = copy.deepcopy(cfg)
    if rule == "AP01": c["budget"] = {}
    if rule == "AP02": c["ranking"] = {"order": ["similarity"]}
    if rule == "AP03": c["instructions"] = {"id": c["instructions"]["id"], "version": "latest"}
    if rule == "AP04": c["memory"] = {"summaries": True, "provenance": False}
    if rule == "AP05": c["compaction"] = {"mode": "threshold", "pins": []}
    if rule == "AP06": c["index"]["version"] = "latest"
    if rule == "AP07": c["index"]["fill_missing_version"] = "file_mtime"
    if rule == "AP08": c["index"]["filter"] = "post"
    if rule == "AP09": c["layout"] = {"prefix": ["instructions", "now"], "volatile": ["now"]}
    if rule == "AP10": c["cache"]["key"] = ["query", "policy"]
    if rule == "AP11": c["evals"] = {"judge_layer": 1}
    if rule == "AP12": c["tools"] = c["tools"] + [dict(c["tools"][1], name="claim_facts_v2")]
    return c
