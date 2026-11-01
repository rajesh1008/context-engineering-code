"""contextlab.structured: the structured span (Chapter 16, v3.7).

A number that enters a context packet should be computed by the system
that owns it, not read by the model out of rows. This module holds the
pieces that make that safe:

* Measure and SemanticLayer: the business meaning of a number, written once.
* compile_measure: turns a measure into one parametrised, read-only query.
* verify: refuses queries that touch other views, miss a required filter
  or fan out across a join.
* render: writes a small table for a model and refuses any rendering
  that does not parse back to the same values.
* StructuredSpan: the value with its unit, grain, query hash and as-of time.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class Measure:
    name: str
    expr: str                      # aggregate over the base view
    view: str                      # the published view it reads
    grain: str                     # one row of the base view is one ...
    unit: str                      # "USD", "count"
    filters: tuple[str, ...]       # predicates that are part of the meaning
    owner: str
    synonyms: tuple[str, ...] = ()
    note: str = ""


@dataclass
class SemanticLayer:
    version: str
    measures: dict[str, Measure] = field(default_factory=dict)

    def add(self, m: Measure) -> None:
        self.measures[m.name] = m

    def resolve(self, phrase: str) -> Measure | None:
        p = phrase.lower()
        for m in self.measures.values():
            if p == m.name or p in m.synonyms:
                return m
        return None

    def card(self) -> str:
        """The few lines a model reads: names, meanings, units, owners."""
        return "\n".join(f"- {m.name} ({m.unit}, per {m.grain}): {m.note} "
                         f"[owner: {m.owner}]" for m in self.measures.values())


def compile_measure(m: Measure, keys: tuple[str, ...]) -> str:
    where = " AND ".join([f"{k} = :{k}" for k in keys] + list(m.filters))
    return f"SELECT {m.expr} AS value, COUNT(*) AS n FROM {m.view} WHERE {where}"


def qhash(sql: str) -> str:
    return hashlib.sha256(sql.encode()).hexdigest()[:8]


def verify(sql: str, m: Measure, allowed: set[str], n_rows: int,
           n_grain: int) -> tuple[bool, str]:
    if not sql.lstrip().upper().startswith("SELECT"):
        return False, "not a read"
    views = set(re.findall(r"(?:FROM|JOIN)\s+(\w+)", sql, re.I))
    if not views <= allowed:
        return False, f"view not allowed: {sorted(views - allowed)}"
    missing = [f for f in m.filters if f not in sql]
    if missing:
        return False, f"missing filter: {missing[0]}"
    if n_rows != n_grain:
        return False, f"fan-out: {n_rows} rows for {n_grain} {m.grain}(s)"
    return True, "verified"


def fmt(value: Decimal | int, unit: str, grouping: bool = False) -> str:
    if unit == "count":
        return str(int(value))
    return f"{value:,.2f}" if grouping else f"{value:.2f}"


def render(header: list[str], rows: list[list], units: list[str],
           grouping: bool = False, check: bool = True) -> str:
    """Comma-separated rows for a model; refuse what does not round-trip."""
    out = io.StringIO()
    out.write(",".join(f"{h} ({u})" for h, u in zip(header, units)) + "\n")
    for r in rows:
        out.write(",".join(fmt(v, u, grouping) if isinstance(v, (int, Decimal))
                           else str(v) for v, u in zip(r, units)) + "\n")
    text = out.getvalue()
    if check:
        back = list(csv.reader(io.StringIO(text)))[1:]
        for src, got in zip(rows, back):
            if len(got) != len(src) or any(
                    isinstance(v, (int, Decimal)) and Decimal(g) != Decimal(v)
                    for v, g in zip(src, got)):
                raise ValueError(f"rendering does not round-trip: {got}")
    return text


@dataclass(frozen=True)
class StructuredSpan:
    label: str
    measure: Measure
    keys: dict
    value: Decimal | int
    sql: str
    read_at: datetime
    layer_version: str

    def header(self) -> str:
        keys = ", ".join(f"{k}={v}" for k, v in self.keys.items())
        return (f"[{self.label}] {self.measure.view} | {self.measure.name} | "
                f"layer {self.layer_version} | query {qhash(self.sql)} | "
                f"as of {self.read_at:%Y-%m-%d %H:%M} | {keys}")

    def text(self) -> str:
        return (f"{self.header()}\n{self.measure.name} = "
                f"{fmt(self.value, self.measure.unit)} {self.measure.unit}")
