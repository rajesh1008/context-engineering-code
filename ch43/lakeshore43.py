"""Lakeshore stand-in for Chapter 43: mirrored claims, two drafting arms, the
adjuster's final determination, and a Wisconsin determination pool for the
examination. Everything is synthetic and seeded. Behaviour rates are
parameters of the stand-in, not measurements, and the chapter says so."""
from __future__ import annotations

import random
from datetime import date, timedelta

from contextlab.shadow import DocStore, Item, Pack, Pair, sha

LINES = {"PA": 0.40, "HO": 0.38, "BOP": 0.15, "BOAT": 0.07}
STATES = {"WI": 0.38, "MN": 0.20, "MI": 0.14, "IL": 0.12, "IA": 0.10, "IN": 0.06}
FORM = {"PA": "PA (2023-01)", "HO": "HO-3 (2024-06)", "BOP": "BOP (2025-01)", "BOAT": "BOAT (2021-05)"}
AMEND = "LM-HO 0122 MN"          # Minnesota amendatory endorsement (Chapter 13)
BUDGET = 575                     # packet token budget for this assistant
BASE_TOKENS = 360

P = dict(ineligible=0.06, timeout=0.005, parse_fail=0.0015, still_open=0.03,
         facts_gap=0.019, judgement=0.012, judgement_same=0.8, wrong=0.0035, wrong_shared=0.7,
         adjuster_error=0.004, accept_wrong_draft=0.3, late_share=0.17)


def pick(rng, table):
    r, acc = rng.random(), 0.0
    for k, w in table.items():
        acc += w
        if r < acc:
            return k
    return k


def business_days(start: date, n: int):
    d, out = start, []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_claim(rng, i, day):
    line, state = pick(rng, LINES), pick(rng, STATES)
    late = line == "HO" and state == "MN" and rng.random() < P["late_share"]
    notes = max(40, int(rng.gauss(178 if late else 120, 34)))
    if late:
        truth = {"verdict": "covered", "clause": f"{AMEND}#duties"}
    else:
        peril = rng.choice(["wb", "fz", "id", "rf", "col", "th", "liab"])
        truth = {"verdict": rng.choice(["covered", "covered", "denied", "partial"]),
                 "clause": f"{FORM[line]}#{peril}"}
    return {"id": f"S-{i:05d}", "day": day.isoformat(), "line": line, "state": state,
            "late": late, "notes": notes, "truth": truth, "u": [rng.random() for _ in range(8)]}


def alt(t):
    """A plausible wrong answer: another clause, opposite verdict."""
    v = "denied" if t["verdict"] != "denied" else "covered"
    return {"verdict": v, "clause": t["clause"].split("#")[0] + "#other"}


def draft(c, arm, factor=1.0, pin=False):
    """One arm's draft. Shared failure modes use the same random draws for both arms."""
    t, u = c["truth"], c["u"]
    if c["late"]:
        tokens = (BASE_TOKENS + c["notes"]) * factor
        if tokens > BUDGET and not pin:       # amendment ranked last, dropped for budget
            return {"verdict": "denied", "clause": f"{FORM['HO']}#duties"}
        return dict(t)
    if u[0] < P["facts_gap"]:                 # file lacked what the adjuster later had
        return {"verdict": "covered" if t["verdict"] == "denied" else "denied", "clause": t["clause"]}
    if u[1] < P["judgement"]:                 # a judgement call: partial vs full
        pflip = u[2] < 0.5
        flip = pflip if (arm == "prod" or u[3] < P["judgement_same"]) else not pflip
        return {"verdict": "partial" if flip else t["verdict"], "clause": t["clause"]}
    if u[4] < P["wrong"]:                     # wrong clause: mostly shared by both arms
        if u[5] < P["wrong_shared"] or (arm == "prod") == (u[3] < 0.5):
            return alt(t)
    return dict(t)


def final(c, shown):
    """The adjuster's determination; `shown` is the draft on screen (or None if blind)."""
    t, u = c["truth"], c["u"]
    if u[6] < P["adjuster_error"]:
        return alt(t)
    if shown is not None and shown != t and u[0] >= P["facts_gap"] and u[7] < P["accept_wrong_draft"]:
        return dict(shown)                    # anchoring: a wrong visible draft accepted
    return dict(t)


def shadow_run(n=2000, seed=43, start=date(2026, 9, 21), days=10, factor=1.062, pin=False,
               cell=None):
    rng = random.Random(seed)
    ds = business_days(start, days)
    arrived = sum(int(rng.gauss(3200, 140)) for _ in ds)
    eligible = int(arrived * (1 - P["ineligible"]))
    pairs = []
    i = 0
    while len(pairs) < n:
        c = make_claim(rng, i, ds[i % days]); i += 1
        if cell and (c["line"], c["state"]) != cell:
            continue
        pairs.append(pair_for(c, rng, factor, pin))
    return arrived, eligible, pairs


def pair_for(c, rng, factor, pin):
    prod = draft(c, "prod")
    cand = None if rng.random() < P["timeout"] else draft(c, "cand", factor, pin)
    parsed = cand is not None and rng.random() >= P["parse_fail"]
    fin = None if rng.random() < P["still_open"] else final(c, prod)
    truth = dict(c["truth"]) if (fin is not None and cand is not None and cand != fin) else None
    ptok = BASE_TOKENS + c["notes"]
    pms = rng.lognormvariate(7.75, 0.25)
    return Pair(c["id"], (c["line"], c["state"]), prod, cand, fin, truth, parsed,
                prod_ms=pms, cand_ms=pms * rng.uniform(1.0, 1.14),
                prod_cost=ptok * 3e-6, cand_cost=ptok * factor * 3e-6)


def holdback(days=10, per_day=3000, share=0.02, seed=4302):
    """After release: agreement of the adjuster with a visible draft versus a hidden one."""
    rng = random.Random(seed)
    vis = [0, 0]; blind = [0, 0]; accepted = 0
    for _ in range(days * per_day):
        c = make_claim(rng, 0, date(2026, 10, 7))
        d = draft(c, "cand", 1.062, pin=True)
        if rng.random() < share:
            f = final(c, None); blind[0] += d == f; blind[1] += 1
        else:
            f = final(c, d); vis[0] += d == f; vis[1] += 1
            accepted += d == f and d != c["truth"]
    return vis, blind, accepted


# ---- the Wisconsin pool for the examination ---------------------------------------

WI_AMEND = "LM-HO 0121 WI"
OLD = ("If you fail to give prompt notice, we may deny the claim only as set out in "
       "Condition 2.b of Section I.")
NEW = ("If you fail to give prompt notice, we may deny the claim only as set out in "
       "Condition 2.c of Section I.")
NAMES = ["A. Lindqvist", "R. Okafor", "M. Brennan", "J. Nowak", "S. Patel", "T. Hughes", "K. Moreau"]


def documents(mode="mutable"):
    ds = DocStore(mode)
    ds.put(WI_AMEND, "2025-03", "Wisconsin amendatory endorsement. Section I conditions. " + OLD
           + " Appraisal is replaced by the Wisconsin appraisal provision.", "2025-03-01T00:00")
    ds.put("HO-3", "2024-06", "Homeowners 3. Section I exclusions: flood, earth movement, "
           "water below the surface. Section I conditions: duties after loss.", "2024-06-01T00:00")
    ds.put("PA", "2023-01", "Personal auto. Part D: collision, other than collision. "
           "Exclusions: racing, business use without endorsement.", "2023-01-01T00:00")
    ds.put("BOP", "2025-01", "Businessowners. Section I: building, business personal property. "
           "Exclusions: wear and tear, mould.", "2025-01-01T00:00")
    ds.put("WI-OCI-2025-03", "2025-03", "Bulletin on prompt payment of property claims.", "2025-03-15T00:00")
    return ds


def wi_pool(ds: DocStore, key: bytes, start=date(2026, 10, 7), days=13, per_day=360, seed=4303,
            edit_day=date(2026, 10, 16)):
    """Wisconsin denials and closed-without-payment determinations. Each is sealed
    when issued. On edit_day the endorsement is replaced in place (if the store allows)."""
    rng = random.Random(seed)
    pool, edited = [], None
    for di, d in enumerate(business_days(start, days)):
        if d == edit_day:
            try:
                ds.put(WI_AMEND, "2025-03", ds.current[(WI_AMEND, "2025-03")].replace(OLD, NEW),
                       f"{d.isoformat()}T11:20")
                edited = d.isoformat()
            except Exception:
                ds.put(WI_AMEND, "2025-03r2", ds.current[(WI_AMEND, "2025-03")].replace(OLD, NEW),
                       f"{d.isoformat()}T11:20")
                edited = "new edition " + d.isoformat()
        ds.snapshot(d.isoformat())
        for k in range(per_day):
            line = pick(rng, {"PA": 0.42, "HO": 0.40, "BOP": 0.14, "BOAT": 0.04})
            doc = (WI_AMEND, "2025-03") if line == "HO" and rng.random() < 0.25 else \
                {"PA": ("PA", "2023-01"), "HO": ("HO-3", "2024-06"), "BOP": ("BOP", "2025-01"),
                 "BOAT": ("WI-OCI-2025-03", "2025-03")}[line]
            if doc[0] == WI_AMEND and (WI_AMEND, "2025-03r2") in ds.current:
                doc = (WI_AMEND, "2025-03r2")
            pool.append(build(rng, ds, key, f"D-W{di:02d}{k:03d}", d, line, doc))
    return pool, edited


def build(rng, ds, key, det_id, d, line, doc):
    text = ds.current[doc]
    name, tp = rng.choice(NAMES), rng.choice(NAMES)
    phone = f"608-555-{rng.randint(1000, 9999)}"
    quoted = rng.random() < 0.10
    seg = lambda t, cls="", q=False: {"text": t, "cls": cls, "quoted": q}  # noqa: E731
    items = [
        Item("request", "claim", [seg(f"Claim {det_id[2:]}, {line}, policyholder "), seg(name, "policyholder"),
                                  seg(", phone "), seg(phone, "policyholder"), seg(f", reported {d}")]),
        Item("output", "determination", [seg("denied; governing " + f"{doc[0]}@{doc[1]}")]),
        Item("version", "stack", [seg("contextlab 9.4; prompt claims-v31; Rulebook r12; index snap "
                                      f"{d}; reranker v9; verifier us-0.55")]),
        Item("version", "model", [seg("sim-large@2026-10 via gateway, region us-central [record]")]),
        Item("span", f"{doc[0]}@{doc[1]}#p2", [seg(text[:140])]),
        Item("span", f"note:{det_id}", [seg("Neighbour "), seg(tp, "third_party", quoted),
                                        seg(" reported water at the property at 06:00.")]),
        Item("document", f"{doc[0]}@{doc[1]}", [seg(sha(text)[:16] + " " + text)]),
        Item("citation", "c1", [seg(f"{doc[0]}@{doc[1]}#p2 checked: supported (0.91)")]),
        Item("tool", "policy_lookup", [seg(f"read {d}T10:02 forms on policy at loss date")]),
        Item("tool", "payment_lookup", [seg("account ****"), seg(f"{rng.randint(1000, 9999)}", "payment")]),
        Item("review", "adjuster", [seg(f"adjuster ADJ-{rng.randint(100, 180)} accepted with edits "
                                        f"{rng.randint(0, 3)} at {d}T{rng.randint(11, 16)}:{rng.randint(10, 59)}")]),
    ]
    p = Pack(det_id, items)
    p.seal(key)
    return {"det_id": det_id, "day": d.isoformat(), "line": line, "doc": doc, "doc_hash": sha(text),
            "pack": p}
