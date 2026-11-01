"""Chapter 16 lab: claims-history numbers as structured spans.

Note: deterministic stand-in. No language model runs. The 'model' reads
numbers by fixed rules: from passages it copies what the summary says,
with schema-only SQL it writes the obvious query, and from a table it
splits each line on commas. Latency and price are stated constants.
Run: python3 lab.py > output.txt
"""
import random
from datetime import date, datetime, timedelta
from decimal import Decimal

import lakeshore16 as L
from contextlab import structured as S

NOW = datetime(2026, 10, 30, 9, 0)
PRICE_PER_MTOK = Decimal("3.00")          # US$ per million input tokens (stand-in)
MS = {"retrieve": 120, "sql_model": 900, "pick_measure": 300, "query": 30, "verify": 2}
LOOKUPS = ["prior_claims_5y", "deductible", "backup_remaining", "term_premium"]
CARTER = {"id": "CLM-2026-01187", "pol": L.EMILY, "loss": "2026-03-14", "asks": LOOKUPS}


def section(t):
    print(f"\n== {t}")


def term_of(db, pol, loss):
    return db.execute("SELECT term_start FROM v_terms WHERE policy_number=? AND "
                      "term_start<=? AND term_end>?", (pol, loss, loss)).fetchone()[0]


def determinations(policies, seed=16, n=1000):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if rng.random() >= 0.3:
            continue
        pol, _ = rng.choice(policies)
        loss = date(2025, 11, 1) + timedelta(days=rng.randrange(0, 360))
        asks = LOOKUPS + (["paid_all_perils_term"] if rng.random() < 0.08 else [])
        out.append({"id": f"D{i:04d}", "pol": pol, "loss": loss.isoformat(), "asks": asks})
    return out


def run_measure(db, sl, name, pol, loss, term):
    m = sl.measures[name]
    sql = S.compile_measure(m, ("policy_number",))
    lookback = date.fromisoformat(loss).replace(year=int(loss[:4]) - 5).isoformat()
    params = {"policy_number": pol, "loss_date": loss, "lookback": lookback,
              "term_start": term}
    value, n = db.execute(sql, params).fetchone()
    grain_sql = sql.replace(f"{m.expr} AS value, COUNT(*) AS n",
                            "1, COUNT(DISTINCT {})".format(
                                {"claim": "claim_id", "payment": "payment_id",
                                 "term": "term_start"}[m.grain]))
    n_grain = db.execute(grain_sql, params).fetchone()[1]
    ok, why = S.verify(sql, m, L.PUBLISHED, n, n_grain)
    value = value or 0
    if m.unit == "USD":
        value = Decimal(value) / 100
    return ok, why, value, sql


def truth(db, sl, d):
    term = term_of(db, d["pol"], d["loss"])
    t = {}
    for name in ("prior_claims_5y", "deductible", "term_premium", "backup_sublimit",
                 "backup_paid_term"):
        t[name] = run_measure(db, sl, name, d["pol"], d["loss"], term)[2]
    t["backup_remaining"] = max(Decimal(0), t["backup_sublimit"] - t["backup_paid_term"])
    paid_all = db.execute("SELECT SUM(amount_cents) FROM v_payments WHERE policy_number=? "
                          "AND term_start=? AND kind IN ('indemnity','recovery')",
                          (d["pol"], term)).fetchone()[0] or 0
    t["paid_all_perils_term"] = Decimal(paid_all) / 100
    return term, t


# ---------- configuration P: passages (Chapter 15's path) ----------
def passages(db, d, term):
    pol, loss = d["pol"], d["loss"]
    latest = db.execute("SELECT term_start, annual_premium_cents, deductible_cents, "
                        "backup_sublimit_cents FROM v_terms WHERE policy_number=? AND "
                        "term_start<=? ORDER BY term_start DESC LIMIT 1",
                        (pol, NOW.date().isoformat())).fetchone()
    summaries = db.execute("SELECT claim_id, loss_date, peril FROM v_claims WHERE "
                           "policy_number=? AND loss_date<? ORDER BY loss_date DESC "
                           "LIMIT 3", (pol, loss)).fetchall()
    paid_written = 0
    for cid, ld, peril in summaries:     # written 45 days after loss; no term on it
        if peril == "sewer_backup":
            cut = (date.fromisoformat(ld) + timedelta(days=45)).isoformat()
            paid_written += db.execute("SELECT COALESCE(SUM(amount_cents),0) FROM payments "
                                       "WHERE claim_id=? AND paid_date<=? AND amount_cents>0",
                                       (cid, cut)).fetchone()[0]
    text = "\n".join(f"Claim {c} ({p}, {l}): summary ..." for c, l, p in summaries)
    tokens = 160 + 90 * len(summaries)
    ans = {"prior_claims_5y": len(summaries),
           "deductible": Decimal(latest[2]) / 100,
           "term_premium": Decimal(latest[1]) / 100,
           "backup_remaining": max(Decimal(0), Decimal(latest[3] - paid_written) / 100),
           "paid_all_perils_term": None}
    return ans, tokens, MS["retrieve"], text


# ---------- configuration Q: schema-only text-to-SQL ----------
NAIVE = {   # what a model writes from the schema alone
    "prior_claims_5y": "SELECT COUNT(*) AS value, COUNT(*) AS n FROM v_claims "
                       "WHERE policy_number = :policy_number AND loss_date < :loss_date",
    "deductible": "SELECT SUM(deductible_cents) AS value, COUNT(*) AS n FROM v_terms "
                  "WHERE policy_number = :policy_number AND term_start = :term_start",
    "term_premium": "SELECT SUM(annual_premium_cents) AS value, COUNT(*) AS n FROM "
                    "v_term_endorsements WHERE policy_number = :policy_number "
                    "AND term_start = :term_start",
    "backup_sublimit": "SELECT SUM(backup_sublimit_cents) AS value, COUNT(*) AS n FROM "
                       "v_terms WHERE policy_number = :policy_number "
                       "AND term_start = :term_start",
    "backup_paid_term": "SELECT SUM(amount_cents) AS value, COUNT(*) AS n FROM v_payments "
                        "WHERE policy_number = :policy_number AND term_start = :term_start "
                        "AND peril = 'sewer_backup'",
    "paid_all_perils_term": "SELECT SUM(amount_cents) AS value, COUNT(*) AS n FROM "
                            "v_payments WHERE policy_number = :policy_number "
                            "AND term_start = :term_start",
}


def schema_sql(db, d, term, verifier, sl=None):
    params = {"policy_number": d["pol"], "loss_date": d["loss"], "term_start": term}
    q = {}
    for k, sql in NAIVE.items():
        value, n = db.execute(sql, params).fetchone()
        value = value or 0
        ok = True
        if verifier and k in sl.measures:
            m = sl.measures[k]
            ok, _ = S.verify(sql, m, L.PUBLISHED, n, n)
        q[k] = value if ok else None
    usd = lambda c: None if c is None else Decimal(c) / 100
    rem = (None if q["backup_sublimit"] is None or q["backup_paid_term"] is None
           else max(Decimal(0), usd(q["backup_sublimit"]) - usd(q["backup_paid_term"])))
    ans = {"prior_claims_5y": q["prior_claims_5y"], "deductible": usd(q["deductible"]),
           "term_premium": usd(q["term_premium"]), "backup_remaining": rem,
           "paid_all_perils_term": usd(q["paid_all_perils_term"])}
    calls = len(d["asks"])
    ms = calls * (MS["sql_model"] + MS["query"]) + (calls * MS["verify"] if verifier else 0)
    return ans, 420 + 60 * calls, ms


# ---------- configuration R: rows pasted into the prompt ----------
def rows_in_prompt(db, d, term, rng):
    ans, _, _ = schema_sql(db, d, term, verifier=False)
    n_rows = db.execute("SELECT COUNT(*) FROM v_payments WHERE policy_number=?",
                        (d["pol"],)).fetchone()[0] + 8
    for k in ("backup_remaining", "paid_all_perils_term"):
        if n_rows > 12 and rng.random() < 0.3:          # long sums slip (stand-in)
            ans[k] = ans[k] + Decimal(100)
    return ans, 300 + 38 * n_rows, MS["retrieve"] + 40


# ---------- configuration S: semantic layer + verifier + structured span ----------
def structured(db, sl, d, term):
    ans, spans = {}, []
    for k in d["asks"]:
        parts = ["backup_sublimit", "backup_paid_term"] if k == "backup_remaining" else [k]
        if any(p not in sl.measures for p in parts):
            ans[k] = None
            continue
        vals = []
        for p in parts:
            ok, why, v, sql = run_measure(db, sl, p, d["pol"], d["loss"], term)
            vals.append(v if ok else None)
            spans.append(S.StructuredSpan(f"S{len(spans)+1}", sl.measures[p],
                                          {"policy": d["pol"], "term": term}, v, sql,
                                          NOW, sl.version))
        ans[k] = (None if None in vals else
                  max(Decimal(0), vals[0] - vals[1]) if len(vals) == 2 else vals[0])
    tokens = sum(len(s.text()) for s in spans) // 4 + len(sl.card()) // 4
    ms = MS["pick_measure"] + len(spans) * (MS["query"] + MS["verify"])
    return ans, tokens, ms, spans


def score(results):
    stated = right = withheld = 0
    for ans, t, asks in results:
        for k in asks:
            if ans.get(k) is None:
                withheld += 1
            else:
                stated += 1
                right += ans[k] == t[k]
    return stated, right, withheld


def table_reading(db, dets, grouping, check):
    """The prior-claims table quoted in the letter, read by comma split."""
    read = right = refused = 0
    for d in dets:
        rows = db.execute("SELECT claim_id, loss_date, peril, SUM(amount_cents) FROM "
                          "v_payments WHERE policy_number=? AND loss_date<? GROUP BY "
                          "claim_id ORDER BY loss_date", (d["pol"], d["loss"])).fetchall()
        rows = [[c, l, p, Decimal(a) / 100] for c, l, p, a in rows]
        if not rows:
            continue
        units = ["id", "date", "text", "USD"]
        try:
            text = S.render(["claim", "loss_date", "peril", "paid"], rows, units,
                            grouping=grouping, check=check)
        except ValueError:
            refused += 1
            text = S.render(["claim", "loss_date", "peril", "paid"], rows, units)
        for src, line in zip(rows, text.splitlines()[1:]):
            read += 1
            right += Decimal(line.split(",")[3]) == src[3]
    return read, right, refused


def main():
    db, policies = L.build()
    sl = L.layer()
    dets = determinations(policies)
    rng = random.Random(1616)
    section(f"1. Data: {len(policies)} policies, "
            f"{db.execute('SELECT COUNT(*) FROM claims').fetchone()[0]} claims, "
            f"{db.execute('SELECT COUNT(*) FROM payments').fetchone()[0]} payments")
    n_lookups = sum(len(d['asks']) for d in dets)
    print(f"history-bearing determinations in the full mix: {len(dets)} of 1000; "
          f"lookups: {n_lookups}")
    section(f"2. Semantic layer {sl.version}: {len(sl.measures)} measures")
    print(sl.card())

    cfgs = {"passages": [], "rows in prompt": [], "schema SQL": [],
            "schema SQL + verifier": [], "structured span": []}
    cost = {k: [0, 0] for k in cfgs}
    errs_by = {k: {} for k in cfgs}
    for d in dets:
        term, t = truth(db, sl, d)
        for name in cfgs:
            if name == "passages":
                ans, tok, ms, _ = passages(db, d, term)
            elif name == "rows in prompt":
                ans, tok, ms = rows_in_prompt(db, d, term, rng)
            elif name == "schema SQL":
                ans, tok, ms = schema_sql(db, d, term, False)
            elif name == "schema SQL + verifier":
                ans, tok, ms = schema_sql(db, d, term, True, sl)
            else:
                ans, tok, ms, spans = structured(db, sl, d, term)
            cfgs[name].append((ans, t, d["asks"]))
            cost[name][0] += tok
            cost[name][1] += ms
            for k in d["asks"]:
                if ans.get(k) is not None and ans[k] != t[k]:
                    errs_by[name][k] = errs_by[name].get(k, 0) + 1

    section("3. Numeric faithfulness on claims-history lookups")
    print(f"{'configuration':24}{'stated':>7}{'right':>7}{'faithful':>10}{'withheld':>9}"
          f"{'tokens/det':>11}{'ms/det':>8}{'US$/1k det':>11}")
    for name, res in cfgs.items():
        st, rt, wh = score(res)
        tok = cost[name][0] / len(dets)
        usd = Decimal(tok) * 1000 * PRICE_PER_MTOK / 1_000_000
        print(f"{name:24}{st:7}{rt:7}{rt/st:10.1%}{wh:9}{tok:11.0f}"
              f"{cost[name][1]/len(dets):8.0f}{usd:11.2f}")

    section("4. Wrong numbers by lookup")
    print(f"{'configuration':24}" + "".join(f"{k[:16]:>18}" for k in LOOKUPS + ['paid_all_perils_term']))
    for name in cfgs:
        print(f"{name:24}" + "".join(f"{errs_by[name].get(k, 0):18}"
                                     for k in LOOKUPS + ['paid_all_perils_term']))

    section("5. Failure injection: thousands separators in the rendered table")
    for label, g, c in (("plain", False, True), ("separators, no check", True, False),
                        ("separators, round-trip check", True, True)):
        read, right, refused = table_reading(db, dets, g, c)
        print(f"{label:30} numbers read {read:5}  right {right:5} ({right/read:.1%})  "
              f"renderings refused {refused}")

    section("6. Sidebar: the premium that was summed twice")
    fan = [d for d in dets if db.execute(
        "SELECT COUNT(*) FROM term_endorsements WHERE policy_number=? AND term_start=?",
        (d["pol"], term_of(db, d["pol"], d["loss"]))).fetchone()[0] > 1]
    print(f"terms with more than one endorsement: {len(fan)} of {len(dets)}")
    d = CARTER
    term = term_of(db, d["pol"], d["loss"])
    m = sl.measures["term_premium"]
    naive = NAIVE["term_premium"]
    params = {"policy_number": d["pol"], "term_start": term}
    wrong = db.execute(naive, params).fetchone()[0]
    rows = db.execute("SELECT COUNT(*), COUNT(DISTINCT term_start) FROM v_term_endorsements "
                      "WHERE policy_number=:policy_number AND term_start=:term_start",
                      params).fetchone()
    ok, why = S.verify(naive, m, L.PUBLISHED | {"v_term_endorsements"}, rows[0], rows[1])
    right = run_measure(db, sl, "term_premium", d["pol"], d["loss"], term)[2]
    print(f"{d['id']} {d['pol']} term {term}: naive SUM {Decimal(wrong)/100:.2f} USD, "
          f"measure {right:.2f} USD; verifier: {why}")

    section("7. A structured span and a derived number")
    d = CARTER
    term = term_of(db, d["pol"], d["loss"])
    _, _, _, spans = structured(db, sl, d, term)
    p_ans = passages(db, d, term)[0]
    print(f"claim {d['id']}, policy {d['pol']}, loss {d['loss']}, term {term}")
    print("passages said: " + ", ".join(f"{k}={p_ans[k]}" for k in LOOKUPS))
    for s in spans[:4]:
        print(s.text())
    sub, paid = spans[2], spans[3]
    print(f"[S5] derived | backup_remaining = {sub.label} - {paid.label} | "
          f"layer {sl.version}\nbackup_remaining = "
          f"{max(Decimal(0), sub.value - paid.value):.2f} USD")

    section("8. Determinations whose amount line changes")
    wrong_amt = {n: sum(1 for ans, t, _ in r if ans.get("backup_remaining") is not None
                        and ans["backup_remaining"] != t["backup_remaining"])
                 for n, r in cfgs.items()}
    for n, v in wrong_amt.items():
        print(f"{n:24} wrong remaining back-up limit in {v} of {len(dets)} determinations")


if __name__ == "__main__":
    main()
