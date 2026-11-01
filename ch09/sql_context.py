"""When the database already has the answer: a verified query as context.

The SQL writer is a deterministic stand-in for a model writing SQL. It
fills a template; with fault=True it drops the date predicate, which is
the commonest temporal mistake text-to-SQL systems make.
"""
import hashlib
import re
import sqlite3
from datetime import date, datetime

SCHEMA = """
CREATE TABLE policy_form_history (
  policy_number TEXT, form_id TEXT, edition TEXT,
  effective_from TEXT, effective_to TEXT, row_version INTEGER);
CREATE VIEW v_forms_on_policy AS
  SELECT policy_number, form_id, edition, effective_from, effective_to,
         row_version FROM policy_form_history;
"""
ALLOWED_VIEWS = {"v_forms_on_policy"}


def connect(rows):
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO policy_form_history VALUES (?,?,?,?,?,?)", rows)
    return db


def write_sql(question: dict, fault: bool = False) -> str:
    sql = ("SELECT form_id, edition, row_version FROM v_forms_on_policy "
           "WHERE policy_number = :policy")
    if not fault:
        sql += (" AND effective_from <= :loss_date"
                " AND (effective_to IS NULL OR effective_to > :loss_date)")
    return sql


def verify(sql: str, rows: list) -> tuple[bool, str]:
    if not sql.lstrip().upper().startswith("SELECT"):
        return False, "not a read"
    views = set(re.findall(r"FROM\s+(\w+)", sql, re.I))
    if not views <= ALLOWED_VIEWS:
        return False, f"view not allowed: {sorted(views - ALLOWED_VIEWS)}"
    if ":loss_date" not in sql:
        return False, "no date-of-loss predicate"
    forms = [r[0] for r in rows]
    if len(forms) != len(set(forms)):
        return False, "more than one edition per form"
    return True, "verified"


def as_span(policy: str, loss_date: date, sql: str, rows: list,
            read_at: datetime) -> str:
    qhash = hashlib.sha256(sql.encode()).hexdigest()[:8]
    body = "; ".join(f"{f} ({e}) row v{v}" for f, e, v in rows)
    return (f"[Q1] policy admin | v_forms_on_policy | query {qhash} | "
            f"as of {read_at:%Y-%m-%d %H:%M} | date of loss {loss_date} | "
            f"policy {policy}: {body}")


def ask(db, policy: str, loss_date: date, fault: bool = False):
    sql = write_sql({"policy": policy}, fault)
    rows = db.execute(sql, {"policy": policy,
                            "loss_date": loss_date.isoformat()}).fetchall()
    ok, why = verify(sql, rows)
    return ok, why, sql, rows
