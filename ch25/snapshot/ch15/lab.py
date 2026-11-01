"""Chapter 15 lab: similarity is not authority.

Run: python3 ch15/lab.py   (from 04-code/). Writes ch15/output.txt and ch15/out/*.json.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import replace
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import corpus15 as C
import standin as M
from contextlab.policy import Policy, lineage
from contextlab.retrieval.index import InvertedIndex, top

K = 8
RATE = 0.00573 / 1609          # Chapter 9's blended US$ per token, held fixed


def build(dets, extra=()):
    spans = C.governing_spans() + C.context_spans() + list(extra)
    restricted = [s for s in spans if s.kind == "note"] + \
                 [n for d in dets for n in d.notes if n.roles == C.SIU]
    notes = []
    for d in dets:
        for n in d.notes:
            notes.append(replace(n, derived_from=lineage(n, restricted)) if n.roles == C.ADJ else n)
    allspans = spans + notes
    ix = InvertedIndex()
    for s in allspans:
        ix.add(s.sid, s.text)
    return {s.sid: s for s in allspans}, ix, [s.sid for s in spans]


def source_roles(s):
    """Chapter 9 checks entitlement per source, not per span."""
    if s.doc_id.startswith("UW-NOTE"):
        return C.UW
    if s.doc_id.startswith("NOTE-"):
        return C.ADJ
    return s.roles


def ch14_allowed(s, r):
    """The Chapter 14 configuration's filters: tenant, source role, edition, the claim's states, end date."""
    if s.tenant not in ("lakeshore", r.tenant):
        return False
    if r.role not in (source_roles(s) or ()):
        return False
    if s.kind == "form" and r.editions.get(s.edition_family) != s.version:
        return False
    if s.kind == "endorsement" and s.jurisdiction == "ALL" and \
            f"{s.edition_family}@{s.version}" not in r.endorsements:
        return False
    if s.jurisdiction not in ("ALL", r.policy_state, r.risk_state):
        return False
    if s.version is None or (s.effective_from and r.date_of_loss < s.effective_from):
        return False
    return not (s.effective_to and r.date_of_loss >= s.effective_to)


def own(d, sid):
    return not sid.startswith("NOTE-") or sid.startswith(f"NOTE-{d.did}-")


def run(dets, spans, ix, global_ids, config="ch14", model="sim-large", policy=None,
        prefilter=True, k=K, as_of=None):
    """Returns one result per determination: cited sid, status, packet, reason."""
    out = []
    for d in dets:
        r = d.req if as_of is None else replace(d.req, date_of_loss=as_of(d))
        pool = global_ids + [n.sid for n in d.notes]
        if d.upstream_withheld:
            out.append({"did": d.did, "status": "withheld", "reason": "document gate (Ch11)",
                        "cited": None, "packet": []}); continue
        if config == "ch14":
            allowed = {x for x in pool if ch14_allowed(spans[x], r)}
            sims = ix.bm25(d.question, allowed=allowed)
            packet = top(sims, k)
            cited = M.cite(model, d.did, packet, spans, sims, topic=r.cause, state=r.policy_state)
            out.append({"did": d.did, "status": "answered", "cited": cited, "packet": packet,
                        "reason": ""}); continue
        if prefilter:
            allowed = {x for x in pool if policy.admit(spans[x], r).ok}
            sims = ix.bm25(d.question, allowed=allowed)
            packet = top(sims, k)
        else:                               # retrieve first, filter after
            sims = ix.bm25(d.question, allowed=set(pool))
            packet = [x for x in top(sims, k) if policy.admit(spans[x], r).ok]
        res = policy.resolve(packet, r)
        if res.status != "ok":
            out.append({"did": d.did, "status": "withheld", "reason": res.reason,
                        "cited": None, "packet": packet}); continue
        cited = M.cite(model, d.did, packet, spans, sims, governs=res.governing, topic=r.cause)
        chk = policy.check(cited, res)
        out.append({"did": d.did, "status": "answered" if chk.ok else "withheld",
                    "reason": chk.reason, "cited": cited if chk.ok else None,
                    "packet": packet, "governing": res.governing, "beat": res.beaten})
    return out


def clause_key(sid, spans):
    s = spans.get(sid)
    if s is None:
        return None
    if s.kind == "bulletin":
        return ("bulletin", s.jurisdiction, s.topic)
    return (s.doc_id, s.topic)


def score(dets, res, spans, scorer="audit"):
    wrong, withheld, leaks = [], 0, 0
    for d, x in zip(dets, res):
        if any(spans[p].roles != C.ADJ and spans[p].kind == "note" or
               (spans[p].doc_id.startswith("NOTE-") and is_restricted(spans[p], spans))
               for p in x["packet"]):
            leaks += 1
        if x["status"] == "withheld":
            withheld += 1; continue
        gold = d.gold
        if isinstance(gold, tuple):
            wrong.append(d.did); continue
        ok = x["cited"] == gold if scorer == "audit" else \
            clause_key(x["cited"], spans) == clause_key(gold, spans)
        if not ok:
            wrong.append(d.did)
    return {"wrong": len(wrong), "withheld": withheld, "leaks": leaks, "wrong_ids": wrong}


def is_restricted(s, spans):
    if s.roles != C.ADJ:
        return True
    return bool(s.derived_from) or "pasted" in s.text   # ground truth: any pasted underwriting text


class NoGraph(Policy):
    """Supersession read only from the span's own end date (as before)."""
    def end_of(self, s):
        return s.effective_to, None


LABEL = 6                      # tokens for a GOVERNS / CONTEXT label and its reason


def tokens(res, spans):
    t = [sum(len(spans[p].text.split()) * 1.33 + (LABEL if "governing" in x else 0)
             for p in x["packet"]) for x in res]
    return sum(t) / len(t)


def pct(n, d=1000):
    return f"{100 * n / d:.1f}% ({n}/{d})"


def main():
    lines = []
    P = lambda *a: lines.append(" ".join(str(x) for x in a))
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    dets = C.full_mix()
    spans, ix, g = build(dets)
    pol = Policy(list(spans.values()))
    by = {d.did: d for d in dets}
    P("== 0. Corpus", C.corpus_version(list(spans.values())))
    P("governing spans", Counter(s.kind for s in C.governing_spans()))
    P("context spans", Counter(s.kind for s in C.context_spans()), "claim notes", sum(len(d.notes) for d in dets))
    P("full mix", dict(Counter(d.stratum for d in dets)))
    P("tenants", dict(Counter(d.req.tenant for d in dets)), "policy", pol.version, pol.fingerprint)

    base = run(dets, spans, ix, g, "ch14")
    full = run(dets, spans, ix, g, "policy", policy=pol)
    b_cl, b_au = score(dets, base, spans, "clause"), score(dets, base, spans, "audit")
    f_cl, f_au = score(dets, full, spans, "clause"), score(dets, full, spans, "audit")

    P("\n== 1. Similarity is not authority (Chapter 14 configuration, sim-large)")
    lead_not_gov = 0
    for d, x in zip(dets, base):
        if x["packet"] and not isinstance(d.gold, tuple):
            on = [p for p in x["packet"] if spans[p].topic == d.req.cause] or x["packet"]
            if on[0] != d.gold:
                lead_not_gov += 1
    P("determinations whose most similar on-topic span is not the governing one:", lead_not_gov)
    ex = next(d for d, x in zip(dets, base) if d.did in b_au["wrong_ids"] and d.did not in b_cl["wrong_ids"])
    sims = ix.bm25(ex.question, allowed={p for p in g if ch14_allowed(spans[p], ex.req)})
    P(f"example {ex.did}: {ex.question!r} ({ex.req.risk_state}, loss {ex.req.date_of_loss})")
    for sid in top(sims, 4):
        P(f"   {sims[sid]:6.2f}  {sid:40s} {pol.admit(spans[sid], ex.req).reason}")
    P("   governing:", ex.gold, "| cited by Chapter 14 configuration:", base[dets.index(ex)]["cited"])
    sc = []
    for d, x in zip(dets, base):
        if not x["packet"] or isinstance(d.gold, tuple):
            continue
        s2 = ix.bm25(d.question, allowed=set(x["packet"]))
        mx = max(s2.values()) or 1
        for p in x["packet"]:
            st = "governs" if p == d.gold else ("context" if spans[p].kind not in ("form", "endorsement", "bulletin") else
                                                ("superseded" if not pol.in_force(spans[p], d.req.date_of_loss).ok else "other"))
            sc.append((round(s2.get(p, 0) / mx, 3), st, spans[p].topic == d.req.cause))
    json.dump(sc, open(os.path.join(HERE, "out", "sim_vs_authority.json"), "w"))
    P("candidate spans scored:", len(sc), Counter(s for _, s, _ in sc))
    P("superseded spans that top their packet:", sum(1 for v, st, _ in sc if st == "superseded" and v == 1.0))

    P("\n== 2. Failure injection: the 2023 Wisconsin roof bulletin re-scanned without its end date")
    inj = C.injected_copy()
    spans2, ix2, g2 = build(dets, extra=[inj])
    pol2 = Policy(list(spans2.values()))
    wi = [d for d in dets if d.req.cause == "roof" and d.req.risk_state == "WI" and d.req.policy_state == "WI"
          and d.req.date_of_loss >= date(2025, 3, 15) and not d.upstream_withheld]
    b2 = run(wi, spans2, ix2, g2, "ch14"); f2 = run(wi, spans2, ix2, g2, "policy", policy=pol2)
    P("Wisconsin roof losses on or after 15 March 2025:", len(wi))
    P("Chapter 14 configuration cites the re-scan:", sum(x["cited"] == inj.sid for x in b2),
      "| wrong:", score(wi, b2, spans2)["wrong"])
    P("Policy: wrong", score(wi, f2, spans2)["wrong"], "| re-scan excluded as:",
      pol2.in_force(inj, wi[0].req.date_of_loss).reason)
    s_inj = ix2.bm25(wi[0].question, allowed={inj.sid, "WI-OCI-2025-03@2025-03-15#roof"})
    P("similarity, re-scan vs governing:", {k: round(v, 2) for k, v in s_inj.items()})

    P("\n== 3. Entitlement")
    kinds = Counter()
    for d in dets:
        for n0 in d.notes[1:]:
            n = spans[n0.sid]
            kinds["siu" if n.roles == C.SIU else ("copied" if n.derived_from else "paraphrased")] += 1
    P("restricted claim notes:", dict(kinds))
    P("packets carrying restricted text: Chapter 14 config", b_au["leaks"], "| Policy", f_au["leaks"])
    spans_nl = {**spans, **{n.sid: replace(n, derived_from=()) for d in dets for n in d.notes if n.roles == C.ADJ}}
    r_nl = run(dets, spans_nl, ix, g, "policy", policy=Policy(list(spans_nl.values())))
    P("Policy without lineage (span roles only):", score(dets, r_nl, spans_nl)["leaks"])
    def leak_kinds(res, sp):
        c = Counter()
        for d, x in zip(dets, res):
            for p in x["packet"]:
                n = spans[p]
                if p.startswith("NOTE-") and is_restricted(sp[p], sp):
                    c["siu" if n.roles == C.SIU else ("copied" if n.derived_from else "paraphrased")] += 1
        return dict(c)
    lk = {"Chapter 14 configuration": leak_kinds(base, spans), "span roles only": leak_kinds(r_nl, spans_nl),
          "Policy (roles and lineage)": leak_kinds(full, spans)}
    P("restricted notes reaching packets, by kind:", lk)
    json.dump(lk, open(os.path.join(HERE, "out", "leaks.json"), "w"))
    P("pre-filter versus post-filter: governing span in the packet")
    pf = {}
    for k in (3, 5, 8, 12):
        row = []
        for mode in (True, False):
            r = run(dets, spans, ix, g, "policy", policy=pol, prefilter=mode, k=k)
            hit = sum(1 for d, x in zip(dets, r) if not isinstance(d.gold, tuple)
                      and not d.upstream_withheld and d.gold in x["packet"])
            n_ = sum(1 for d in dets if not isinstance(d.gold, tuple) and not d.upstream_withheld)
            wd = score(dets, r, spans)
            row.append((round(hit / n_, 3), wd["wrong"], wd["withheld"]))
        pf[k] = row
        P(f"  k={k:2d}  pre {row[0][0]:.3f} (wrong {row[0][1]}, withheld {row[0][2]})   post {row[1][0]:.3f} (wrong {row[1][1]}, withheld {row[1][2]})")
    json.dump(pf, open(os.path.join(HERE, "out", "prepost.json"), "w"))

    P("\n== 4. Jurisdiction (multi-state stratum, 60)")
    ms = [d for d in dets if d.stratum == "multi-state"]
    variants = {"policy state for every rule": {k: "policy_state" for k in pol.rules},
                "property state for every rule": {k: ("garaging_state" if k[0] == "PAP" else "risk_state") for k in pol.rules},
                "rule by kind (Policy)": pol.rules}
    jur = {}
    for name, rules in variants.items():
        pv = Policy(list(spans.values()), rules=rules)
        r = run(ms, spans, ix, g, "policy", policy=pv)
        s_ = score(ms, r, spans); jur[name] = (s_["wrong"], s_["withheld"])
        P(f"  {name:32s} wrong {s_['wrong']:2d}  withheld {s_['withheld']}")
    s_ = score(ms, [x for d, x in zip(dets, base) if d.stratum == "multi-state"], spans)
    jur["Chapter 14 configuration"] = (s_["wrong"], s_["withheld"])
    P(f"  {'Chapter 14 configuration':32s} wrong {s_['wrong']:2d}  withheld {s_['withheld']}")
    P("  multi-state wrong under Policy, by reason:",
      dict(Counter(by[w].carried or "other" for w in score(ms, run(ms, spans, ix, g, "policy", policy=pol), spans)["wrong_ids"])))
    fo = Policy(list(spans.values()), unknown_blocks=False)
    s_ = score(ms, run(ms, spans, ix, g, "policy", policy=fo), spans)
    jur["unknown state falls through"] = (s_["wrong"], s_["withheld"])
    P(f"  {'unknown state falls through':32s} wrong {s_['wrong']:2d}  withheld {s_['withheld']}")
    json.dump(jur, open(os.path.join(HERE, "out", "jurisdiction.json"), "w"))
    reg = [w for w in f_au["wrong_ids"] if by[w].stratum == "docstore-new"]
    P("  regressions (record's property state wrong):", reg, [by[w].req.risk_state + " recorded, " + by[w].req.policy_state + " actual" for w in reg])

    P("\n== 5. Precedence")
    glob = Policy(list(spans.values()), scoped_precedence=False)
    r_g = run(dets, spans, ix, g, "policy", policy=glob)
    s_g = score(dets, r_g, spans)
    P("global 'endorsement beats form':", "wrong", s_g["wrong"], "withheld", s_g["withheld"])
    P("scoped to the clause (Policy):    wrong", f_au["wrong"], "withheld", f_au["withheld"])
    newly = [w for w in s_g["wrong_ids"] if w not in f_au["wrong_ids"]]
    ex2 = by[newly[0]] if newly else None
    if ex2:
        x = r_g[dets.index(ex2)]
        P(f"  e.g. {ex2.did} cause={ex2.req.cause} endorsements={sorted(ex2.req.endorsements)} cited {x['cited']} governing {ex2.gold}")
    P("  newly wrong by cause:", dict(Counter(by[w].req.cause for w in newly)))
    json.dump({"global": [s_g["wrong"], s_g["withheld"]], "scoped": [f_au["wrong"], f_au["withheld"]],
               "by_cause": dict(Counter(by[w].req.cause for w in newly))},
              open(os.path.join(HERE, "out", "precedence.json"), "w"))
    adv = sum(1 for d, x in zip(dets, base) if x["cited"] and spans[x["cited"]].kind in ("guideline", "advisory", "note"))
    P("Chapter 14 configuration citing context as governing:", adv, "| Policy:",
      sum(1 for x in full if x["cited"] and spans[x["cited"]].kind in ("guideline", "advisory", "note")))
    prec_err = [w for w in b_au["wrong_ids"] if by[w].carried is None and spans[base[dets.index(by[w])]["cited"]].kind == "form"
                and not isinstance(by[w].gold, tuple) and by[w].gold in base[dets.index(by[w])]["packet"]]
    P("Chapter 14 configuration: governing span in the packet, form cited instead:", len(prec_err))

    P("\n== 6. Freshness: the date that matters")
    today = date(2026, 11, 1)
    r_t = run(dets, spans, ix, g, "policy", policy=pol, as_of=lambda d: today)
    s_t = score(dets, r_t, spans)
    nw = [w for w in s_t["wrong_ids"] if w not in f_au["wrong_ids"]]
    P("in force today (latest wins):", "wrong", s_t["wrong"], "| on the date of loss:", f_au["wrong"])
    P("  newly wrong:", len(nw), dict(Counter((by[w].req.risk_state, by[w].req.cause) for w in nw)))
    ng = NoGraph(list(spans.values()))
    s_n = score(dets, run(dets, spans, ix, g, "policy", policy=ng), spans)
    P("end dates read only from the old document (no supersession graph): wrong", s_n["wrong"])
    mn = [d for d in dets if d.req.risk_state == "MN" and d.req.cause == "roof"]
    P("Minnesota roof losses:", len(mn), "after 2 June 2025:", sum(d.req.date_of_loss >= date(2025, 6, 2) for d in mn))
    fresh = {"date of loss": [f_au["wrong"], f_au["withheld"]], "today": [s_t["wrong"], s_t["withheld"]],
             "no graph": [s_n["wrong"], s_n["withheld"]], "ch14": [b_au["wrong"], b_au["withheld"]]}
    P("  withheld: date of loss", f_au["withheld"], "today", s_t["withheld"], "no graph", s_n["withheld"])
    tl = [(d.req.risk_state, str(d.req.date_of_loss), d.gold.split("@")[0]) for d in dets
          if d.req.cause == "roof" and d.req.risk_state in ("WI", "MN") and isinstance(d.gold, str)
          and d.req.risk_state == d.req.policy_state]
    json.dump({"counts": fresh, "timeline": tl}, open(os.path.join(HERE, "out", "freshness.json"), "w"))

    P("\n== 7. Citation an auditor can follow")
    P("Chapter 14 configuration: clause scorer", pct(b_cl["wrong"]), "| audit scorer", pct(b_au["wrong"]))
    P("Policy:                   clause scorer", pct(f_cl["wrong"]), "| audit scorer", pct(f_au["wrong"]))
    diff = [w for w in b_au["wrong_ids"] if w not in b_cl["wrong_ids"]]
    P("right clause, wrong authority under Chapter 14:", len(diff), [base[dets.index(by[w])]["cited"].split("@")[0] for w in diff])
    i = dets.index(ex)
    rp = Policy(list(spans.values()))
    res = rp.resolve(full[i]["packet"], ex.req)
    rec = rp.audit(ex.did, ex.req, res, full[i]["cited"], C.corpus_version(list(spans.values())))
    rec2 = Policy(list(spans.values())).audit(ex.did, ex.req, Policy(list(spans.values())).resolve(full[i]["packet"], ex.req),
                                              full[i]["cited"], C.corpus_version(list(spans.values())))
    json.dump(rec, open(os.path.join(HERE, "out", "audit_record.json"), "w"), indent=1)
    P(json.dumps(rec, indent=1))
    P("replay gives the same record:", rec["record"] == rec2["record"])

    P("\n== 8. Fail closed")
    P("withheld under Policy:", dict(Counter(x["reason"] for x in full if x["status"] == "withheld")))
    mods = {}
    for m in M.PROFILES:
        b_ = score(dets, run(dets, spans, ix, g, "ch14", model=m), spans)
        rm = run(dets, spans, ix, g, "policy", policy=pol, model=m)
        f_ = score(dets, rm, spans)
        chk = sum(1 for x in rm if x["status"] == "withheld" and x["reason"].startswith("cited"))
        mods[m] = (b_["wrong"], f_["wrong"], f_["withheld"], chk)
        P(f"  {m:9s} Chapter 14 config wrong {b_['wrong']:2d} | Policy wrong {f_['wrong']:2d} withheld {f_['withheld']:2d} (citation check {chk})")
    json.dump(mods, open(os.path.join(HERE, "out", "models.json"), "w"))
    P("fall-through on unknown jurisdiction, full mix: wrong",
      score(dets, run(dets, spans, ix, g, "policy", policy=fo), spans)["wrong"])

    P("\n== 9. Scoreboard ladder (full mix, sim-large, audit scorer)")
    ladder = [("Chapter 14 configuration", base)]
    ps = {k: "policy_state" for k in pol.rules}
    steps = [("+ entitlement, precedence in code, check", NoGraph(list(spans.values()), rules=ps)),
             ("+ supersession graph", Policy(list(spans.values()), rules=ps)),
             ("+ jurisdiction by kind of rule (Policy v1.0)", pol)]
    for name, pv in steps:
        ladder.append((name, run(dets, spans, ix, g, "policy", policy=pv)))
    out = []
    for name, r in ladder:
        s_ = score(dets, r, spans)
        out.append((name, s_["wrong"], s_["withheld"], s_["leaks"], round(tokens(r, spans)), tokens(r, spans) * RATE))
        P(f"  {name:44s} wrong {pct(s_['wrong']):14s} withheld {s_['withheld']:3d} leaks {s_['leaks']:2d} "
          f"evidence tokens {tokens(r, spans):6.1f} US${tokens(r, spans) * RATE:.5f}")
    json.dump(out, open(os.path.join(HERE, "out", "ladder.json"), "w"))
    P("remaining wrong under Policy:", dict(Counter(by[w].carried or by[w].stratum for w in f_au["wrong_ids"])))
    open(os.path.join(HERE, "output.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
