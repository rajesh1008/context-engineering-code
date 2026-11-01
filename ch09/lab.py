"""Chapter 9 lab: build the Lakeshore knowledge inventory and use it.

Note: deterministic stand-in. No language model runs; the SQL writer and
the determination outcomes are rules plus seeded noise (see fullmix.py).
Run: python3 lab.py > output.txt
"""
import random
from datetime import date, datetime, timedelta

import fullmix as F
import inventory_lakeshore as L
import sql_context as Q
from contextlab.knowledge import Record

NOW = datetime(2026, 10, 30, 9, 0)


def section(title):
    print(f"\n== {title}")


def show_inventory(inv):
    section(f"1. Inventory {inv.version} #{inv.fingerprint()}: {len(inv.sources)} sources")
    print(f"{'source':22}{'class':12}{'access':10}{'volatility':11}"
          f"{'version':10}owner")
    for s in inv.sources:
        print(f"{s.name:22}{s.evidence_class.value:12}{s.access.value:10}"
              f"{s.volatility.value:11}{s.version_rule.value:10}{s.owner}")


def show_gaps():
    section("2. Gaps")
    for inv in (L.draft(), L.signed()):
        gaps = inv.gaps()
        print(f"{inv.version}: {len(gaps)} gap(s)")
        for g in gaps:
            print("  -", g)


def ch02_admit(rec):
    """Chapter 2's rule: reject a record only when it has no version."""
    return rec.version is not None


def failure_injection(inv):
    section("3. Failure injection: a scan with no printed edition")
    scan = Record("LM 112#scan-0457", "scanned-documents",
                  "Sump overflow: we cover direct physical loss ... up to $5,000",
                  version=None, version_origin=None, page=2)
    # the imaging connector 'helpfully' fills the version from file metadata
    filled = Record(scan.record_id, scan.source, scan.text,
                    version="2026-03-14", version_origin="file-mtime", page=2)
    loss = date(2026, 2, 7)
    print("connector output :", filled.record_id, filled.version,
          f"(origin {filled.version_origin})")
    print("Chapter 2 admit  :", "admitted" if ch02_admit(filled) else "rejected")
    print("  packet header  : [S2] LM 112 (2026-03) | page 2 | authority endorsement")
    print(f"  date of loss {loss}; cited edition starts 2026-03: cited before it existed")
    ev = inv.offer(filled, on=loss, now=NOW)
    print("Chapter 9 offer  :", "admitted" if ev else "withheld")
    rid, owner, why = inv.quarantine[-1]
    print(f"  quarantine     : {rid} -> owner {owner}: {why}")
    print("  packet note    : [W1] governing evidence withheld: LM 112 scan, "
          "edition not printed; see imaging archive page 2")


def other_admissions(inv):
    section("4. Admission across the inventory (date of loss 2026-02-07)")
    loss = date(2026, 2, 7)
    cases = [
        (Record("HO 04 95@2024-06#water_backup", "endorsements", "...",
                "2024-06", page=1), "adjuster", "claims_determination"),
        (Record("WI-OCI-2025-03#matching", "state-bulletins", "...", "2025-03",
                effective_from=date(2025, 3, 15),
                observed_at=NOW - timedelta(hours=20)), "adjuster",
         "claims_determination"),
        (Record("IL-DOI-2026-09#frozen_pipe", "state-bulletins", "...",
                "2026-09", effective_from=date(2026, 9, 1),
                observed_at=NOW - timedelta(hours=3)), "adjuster",
         "claims_determination"),
        (Record("IA-ID-2024-08#debris", "state-bulletins", "...", "2024-08",
                effective_from=date(2024, 8, 15),
                observed_at=NOW - timedelta(days=6)), "adjuster",
         "claims_determination"),
        (Record("UW-NOTE-0004", "underwriting-notes", "...", None), "adjuster",
         "claims_determination"),
        (Record("parcel-55-0192#roof_age", "property-data-feed", "...",
                "2026-10-29T06:00", version_origin="system",
                observed_at=NOW - timedelta(days=1)), "adjuster",
         "claims_determination"),
        (Record("DET-0074#summary", "generated-summaries", "...", "2026-09-30",
                version_origin="system"), "adjuster", "claims_determination"),
    ]
    for rec, role, purpose in cases:
        src = inv.by_name(rec.source)
        adm = src.admit(rec, on=loss, now=NOW, role=role, purpose=purpose)
        print(f"  {rec.record_id:32}{'admit' if adm.ok else 'refuse':8}{adm.reason}")


def build_policy_db(dets):
    rng = random.Random(19)
    rows, cases = [], []
    for i, d in enumerate(dets):
        pol = f"WI-HO-{40100 + i}"
        change = date(2025, 4, 1) + timedelta(days=rng.randrange(0, 300))
        old, new = ("2022-01", "2024-06") if i % 2 else ("2019-03", "2024-06")
        form = "HO-3" if i % 2 else "HO 04 95"
        rows += [(pol, form, old, "2023-04-01", change.isoformat(), 3),
                 (pol, form, new, change.isoformat(), None, 4)]
        if d.pointer_wrong:
            loss = change - timedelta(days=rng.randrange(5, 120))
        else:
            loss = change + timedelta(days=rng.randrange(5, 120))
        truth = old if loss < change else new
        cases.append((d, pol, form, loss, truth, new))
    return Q.connect(rows), cases


def passage_pick(form, old, new):
    """Similarity ranking over declarations pages: both pages share the
    same words; the tie goes to the page indexed last, the renewal."""
    return new


def database_answers(dets):
    section("5. When the database already has the answer (130 determinations)")
    pa = [d for d in dets if d.stratum == "policy-admin"]
    db, cases = build_policy_db(pa)
    hits = {"pointer": 0, "passages": 0, "query": 0}
    rejected = 0
    for d, pol, form, loss, truth, newest in cases:
        hits["pointer"] += newest == truth
        hits["passages"] += passage_pick(form, None, newest) == truth
        ok, why, sql, rows = Q.ask(db, pol, loss, fault=d.sql_fault)
        if not ok:
            rejected += 1
            continue
        hits["query"] += rows[0][1] == truth
    n = len(cases)
    for k, v in hits.items():
        print(f"  {k:9} edition in force on the date of loss: {v}/{n}")
    print(f"  query rejected by the verifier (no date predicate): {rejected}")
    d, pol, form, loss, truth, newest = next(c for c in cases if c[0].pointer_wrong)
    ok, why, sql, rows = Q.ask(db, pol, loss)
    print("  span:", Q.as_span(pol, loss, sql, rows, NOW))
    ok, why, sql, rows = Q.ask(db, pol, loss, fault=True)
    print(f"  injected fault: {len(rows)} rows returned -> {why}")


def scoreboard(dets):
    section("6. Scoreboard: Chapter 8 configuration against Chapter 9")
    sets = {"eval 200": [d for d in dets if d.split == "eval"],
            "holdout 300": [d for d in dets if d.split == "holdout"],
            "full mix 1,000": dets}
    print(f"  {'set':16}{'config':7}{'wrong':>8}{'rate':>7}{'withheld':>10}"
          f"{'median':>8}{'mean':>6}{'tokens':>8}{'US$':>9}")
    for name, sub in sets.items():
        for cfg in ("ch08", "ch09"):
            s = F.score(sub, cfg)
            print(f"  {name:16}{cfg:7}{s['wrong']:>8}{s['rate']:>7.1%}"
                  f"{s['withheld']:>10}{s['median_min']:>8.1f}{s['mean_min']:>6.1f}"
                  f"{s['tokens']:>8,.0f}{s['usd']:>9.5f}")
    from collections import Counter
    c = Counter(d.stratum if d.split == "new" else d.split
                for d in dets if F.outcome(d, "ch09") == "wrong")
    print("  remaining wrong (ch09, full mix):", dict(sorted(c.items())))


def main():
    inv = L.signed()
    show_inventory(inv)
    show_gaps()
    failure_injection(inv)
    other_admissions(inv)
    dets = F.build()
    database_answers(dets)
    scoreboard(dets)
    section("7. Quarantine, visible to owners")
    for rid, owner, why in inv.quarantine:
        print(f"  {owner}: {rid}: {why}")


if __name__ == "__main__":
    main()
