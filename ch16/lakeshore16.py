"""Lakeshore's policy and claims history as a small SQLite database.

Synthetic and seeded. Amounts are stored in cents so that every sum is exact.
Published views are the only things a query may read.
"""
import pathlib
import random
import sqlite3
import sys
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from contextlab.structured import Measure, SemanticLayer

SCHEMA = """
CREATE TABLE terms (policy_number TEXT, term_start TEXT, term_end TEXT,
  annual_premium_cents INTEGER, deductible_cents INTEGER, backup_sublimit_cents INTEGER);
CREATE TABLE term_endorsements (policy_number TEXT, term_start TEXT, form_id TEXT);
CREATE TABLE claims (claim_id TEXT, policy_number TEXT, loss_date TEXT,
  peril TEXT, status TEXT);
CREATE TABLE payments (payment_id TEXT, claim_id TEXT, paid_date TEXT,
  kind TEXT, amount_cents INTEGER);
CREATE VIEW v_terms AS SELECT * FROM terms;
CREATE VIEW v_claims AS SELECT * FROM claims;
CREATE VIEW v_payments AS
  SELECT p.*, c.policy_number, c.peril, c.loss_date, t.term_start
  FROM payments p JOIN claims c USING (claim_id)
  JOIN terms t ON t.policy_number = c.policy_number
   AND c.loss_date >= t.term_start AND c.loss_date < t.term_end;
CREATE VIEW v_term_endorsements AS
  SELECT t.*, e.form_id FROM terms t JOIN term_endorsements e
  USING (policy_number, term_start);
"""
PUBLISHED = {"v_terms", "v_claims", "v_payments"}
PERILS = ["sewer_backup", "frozen_pipe", "wind_hail", "theft", "fire"]
FORMS = ["HO 04 95", "HO 04 90", "HO 04 61", "WI-OCI-2025-03"]


def layer() -> SemanticLayer:
    sl = SemanticLayer("sl-1.2")
    sl.add(Measure("prior_claims_5y", "COUNT(DISTINCT claim_id)", "v_claims", "claim",
                   "count", ("loss_date < :loss_date", "loss_date >= :lookback",
                             "status != 'void'"), "Claims platform",
                   ("prior claims", "claims in the last five years"),
                   "claims before this loss, five-year lookback, voids excluded"))
    sl.add(Measure("backup_paid_term", "SUM(amount_cents)", "v_payments", "payment",
                   "USD", ("peril = 'sewer_backup'", "term_start = :term_start",
                           "kind IN ('indemnity', 'recovery')"), "Claims finance",
                   ("back-up paid this term",),
                   "indemnity net of recoveries; expenses excluded"))
    sl.add(Measure("term_premium", "SUM(annual_premium_cents)", "v_terms", "term",
                   "USD", ("term_start = :term_start",), "Policy administration",
                   ("premium",), "annual premium of the term in force"))
    sl.add(Measure("deductible", "SUM(deductible_cents)", "v_terms", "term", "USD",
                   ("term_start = :term_start",), "Policy administration",
                   ("deductible",), "all-peril deductible of the term in force"))
    sl.add(Measure("backup_sublimit", "SUM(backup_sublimit_cents)", "v_terms", "term",
                   "USD", ("term_start = :term_start",), "Policy administration",
                   ("back-up limit",), "HO 04 95 limit for the term in force"))
    return sl


def build(seed: int = 16, n_policies: int = 400):
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    policies, pay_id = [], 0
    for i in range(n_policies):
        pol = f"LMH-WI-{48000 + i:07d}"
        start = date(2019, 1, 1) + timedelta(days=rng.randrange(0, 365))
        prem = rng.randrange(900, 4200) * 100 + rng.choice([0, 25, 50, 75])
        for y in range(8):
            ts = start.replace(year=start.year + y)
            te = ts.replace(year=ts.year + 1)
            prem = int(prem * rng.choice([1.0, 1.04, 1.07, 1.12]))
            db.execute("INSERT INTO terms VALUES (?,?,?,?,?,?)",
                       (pol, ts.isoformat(), te.isoformat(), prem,
                        rng.choice([50000, 100000, 250000]),
                        rng.choice([500000, 1000000, 2500000])))
            for f in rng.sample(FORMS, rng.choice([1, 2, 2, 3])):
                db.execute("INSERT INTO term_endorsements VALUES (?,?,?)",
                           (pol, ts.isoformat(), f))
        for c in range(rng.choice([0, 1, 1, 2, 3, 4, 5])):
            loss = start + timedelta(days=rng.randrange(30, 7 * 365))
            cid = f"CL-{pol[-5:]}-{c}"
            db.execute("INSERT INTO claims VALUES (?,?,?,?,?)",
                       (cid, pol, loss.isoformat(), rng.choice(PERILS),
                        "void" if rng.random() < 0.06 else "closed"))
            for k in range(rng.choice([1, 2, 3])):
                pay_id += 1
                kind = rng.choice(["indemnity", "indemnity", "expense", "recovery"])
                amt = rng.randrange(150, 9000) * 100 + rng.randrange(0, 100)
                db.execute("INSERT INTO payments VALUES (?,?,?,?,?)",
                           (f"P{pay_id}", cid, (loss + timedelta(days=20 + 30 * k)).isoformat(),
                            kind, -amt // 4 if kind == "recovery" else amt))
        policies.append((pol, start))
    carter(db)
    db.commit()
    return db, policies


EMILY = "LMH-WI-0048213"


def carter(db):
    """Emily Carter's policy, as Chapter 1 describes it (synthetic figures)."""
    for t in ("terms", "term_endorsements", "claims"):
        db.execute(f"DELETE FROM {t} WHERE policy_number=?", (EMILY,))
    db.execute("DELETE FROM payments WHERE claim_id LIKE 'CL-48213-%'")
    prem = 158000
    for y in range(2019, 2027):
        ts, te = f"{y}-06-01", f"{y + 1}-06-01"
        prem = int(prem * 1.04)
        db.execute("INSERT INTO terms VALUES (?,?,?,?,?,?)",
                   (EMILY, ts, te, prem, 100000, 1000000))
        for f in ("HO 04 95", "HO 04 90"):
            db.execute("INSERT INTO term_endorsements VALUES (?,?,?)", (EMILY, ts, f))
    db.execute("INSERT INTO claims VALUES (?,?,?,?,?)",
               ("CL-48213-0", EMILY, "2024-04-20", "sewer_backup", "closed"))
    db.executemany("INSERT INTO payments VALUES (?,?,?,?,?)",
                   [("P-E1", "CL-48213-0", "2024-05-10", "indemnity", 385000),
                    ("P-E2", "CL-48213-0", "2024-06-21", "expense", 41250)])
