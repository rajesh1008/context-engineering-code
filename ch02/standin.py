"""Chapter 2 stand-in: Lakeshore's synthetic corpus, its policies and determinations, the
applicability policy, a keyword retriever and SimulatedModel.

STAND-IN NOTICE. Nothing here is a language model or a real search engine.
* The corpus is 378 synthetic spans (22,326 tokens): four base forms in nine editions, 22
  endorsements, ten state bulletins, three pieces of NAIC guidance and 14 underwriting notes,
  with 500 policies, 500 determinations and 157 claim-history records (world.py).
* The retriever ("bm25" in the configuration) is a simulated keyword ranker. Each question has
  a fixture of keyword scores for the spans that share its words (calibration.json); the query's
  claim facts then add a real, computed bonus to every span whose document or edition label the
  query names, and to bulletins of the state it names. Change the query and the ranking changes.
* SimulatedModel reads the rendered packet: the claim header (parsed from the text) and the
  evidence blocks in the order they were placed. It scores each span on the question's topic
  (the stand-in is told the topic, as a stand-in for comprehension) by authority (bulletins and
  endorsements a little above forms, notes and guidance a little below), a pull towards spans
  whose label the claim header names, whether a bulletin's state and dates fit the claim, and a
  small preference for earlier positions. A fixture adds, per determination, the misreadings a
  reader makes (a few spans it over- or under-weights) and the answers it embellishes with a
  sentence its cited span does not support. With no span on the topic it cites the first span.
* calibrate.py chose the keyword scores, the span lengths, the misreadings and the
  embellishments so that this mechanism reproduces the chapter's illustrative numbers exactly.
  The mechanism is real and inspectable; the profile is illustrative, not measured.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from contextlab.chunker import count_tokens                      # noqa: E402
from contextlab.models import (Applicability, Determination, Kind,  # noqa: E402
                               Provenance, Source, Span)
from contextlab.pipeline import Answer, Rendered                 # noqa: E402
import world                                                     # noqa: E402

DATA = HERE / "data"
CAL = DATA / "calibration.json"
ROLE = "claims"

FILLER = ("The insurer applies this provision together with the declarations and every other provision of the "
          "policy as amended by any endorsement that is attached to it on the date of the loss and the adjuster "
          "records the facts that support the payment or the denial in the claim file before any letter is sent to "
          "the insured or to the mortgagee named in the declarations and keeps a copy of each document relied on "
          "for the period that the record retention schedule requires for claims of this type").split()


def filler(n: int, key: str) -> str:
    """Exactly n tokens of boilerplate sentences (words plus a full stop each)."""
    if n <= 0:
        return ""
    sizes, left = [], n
    while left > 0:
        s = min(left, 13)
        if left - s in (1, 2):
            s = left - 3 if left - 3 >= 3 else left
        sizes.append(s)
        left -= s
    off = int(hashlib.sha256(key.encode()).hexdigest()[:6], 16) % len(FILLER)
    out, i = [], off
    for s in sizes:
        words = [FILLER[(i + j) % len(FILLER)] for j in range(s - 1)]
        i += s - 1
        out.append(" ".join(words).capitalize() + ".")
    return " ".join(out)


def span_text(spec: dict, n: int | None) -> str:
    if spec.get("fixed_text"):
        if spec["kind"] == "uw_note" and n:          # notes run on past their first sentence
            body = filler(n - count_tokens(spec["fixed_text"]), spec["span_id"])
            return " ".join(x for x in (spec["fixed_text"], body) if x)
        return spec["fixed_text"]
    label = f"{spec['doc_id']} ({spec['version']})"
    page = f", page {spec['page']}" if spec.get("page") is not None else ""
    head = f"{spec['title']}. {spec['holding']}."
    tail = f"[{label}{page}]"
    fixed = count_tokens(head) + count_tokens(tail)
    body = filler((n or fixed) - fixed, spec["span_id"])
    return " ".join(x for x in (head, body, tail) if x)


def operative(s: Span) -> str:
    """The sentence of a span that states what it decides (the one an answer quotes)."""
    if s.kind == Kind.UW_NOTE:
        return s.text.split(". ")[0].rstrip(".")
    body = s.text.split(". ", 1)[1] if ". " in s.text else s.text
    return body.split(". ")[0].split(" [")[0].rstrip(".")


# ------------------------------------------------------------------ corpus
def to_span(spec: dict, n: int | None) -> Span:
    ent = frozenset(spec.get("entitlement", ["claims", "underwriting"]))
    prov = Provenance(doc_id=spec["doc_id"], version=spec["version"], effective_from=spec.get("effective_from"),
                      effective_to=spec.get("effective_to"), superseded_by=spec.get("superseded_by"),
                      jurisdiction=spec.get("jurisdiction", "ALL"), entitlement=ent,
                      precedence=spec.get("precedence", 0), page=spec.get("page"),
                      source_system=spec.get("source_system", "docstore"), trust=spec.get("trust", 1.0))
    meta = {"verdict": spec["verdict"]} if spec.get("verdict") else {}
    return Span(span_id=spec["span_id"], kind=Kind(spec["kind"]), text=span_text(spec, n), provenance=prov,
                topic=spec.get("topic"), section=spec.get("section"), meta=meta)


SOURCES = [
    Source(name="policy forms", kind=Kind.FORM, owner="product", system="docstore"),
    Source(name="endorsements", kind=Kind.ENDORSEMENT, owner="product", system="docstore"),
    Source(name="state bulletins", kind=Kind.BULLETIN, owner="compliance", system="regulatory-feed",
           refresh="daily"),
    Source(name="NAIC guidance", kind=Kind.GUIDANCE, owner="compliance", system="naic"),
    Source(name="underwriting notes", kind=Kind.UW_NOTE, owner="underwriting", system="sharepoint",
           entitlement=frozenset({"underwriting"})),
]


class Policy:
    """The applicability policy: does this span govern this claim? Reasons are recorded."""

    def __init__(self, role: str = ROLE):
        self.role = role

    def applies(self, s: Span, det: Determination) -> Applicability:
        p = s.provenance
        if self.role not in p.entitlement:
            return Applicability(ok=False, reason=f"not entitled ({self.role})")
        if s.kind == Kind.FORM:
            doc, ver = det.form.split(" (")[0], det.form.split(" (")[1].rstrip(")")
            if p.doc_id != doc:
                return Applicability(ok=False, reason="not on policy")
            if p.version == ver:
                return Applicability(ok=True)
            return Applicability(ok=False, reason="superseded" if (p.version or "") < ver else "not on policy")
        if s.kind == Kind.ENDORSEMENT:
            if p.label in det.endorsements:
                return Applicability(ok=True)
            mine = {e.split(" (")[0]: e.split(" (")[1].rstrip(")") for e in det.endorsements}
            if p.doc_id in mine and (p.version or "") < mine[p.doc_id]:
                return Applicability(ok=False, reason="superseded")
            return Applicability(ok=False, reason="endorsement not on policy")
        if s.kind == Kind.BULLETIN:
            if p.jurisdiction != det.state:
                return Applicability(ok=False, reason="wrong state")
            if not p.in_force(det.date_of_loss):
                return Applicability(ok=False, reason="bulletin not in force")
        return Applicability(ok=True)


def bonus(s: Span, facts: str, edition: float = 3.55, doc: float = 2.5, state: float = 1.2) -> float:
    p = s.provenance
    b = 0.0
    if p.version and p.version in facts:
        b += edition
    if re.search(rf"(^|\s){re.escape(p.doc_id)} \(", facts):
        b += doc
    if s.kind == Kind.BULLETIN and re.search(rf"\b{p.jurisdiction}\b", facts):
        b += state
    return b


class Retriever:
    """Simulated keyword ranker: fixture keyword scores per question plus a computed facts bonus."""

    EDITION, DOC, STATE = 3.55, 2.5, 1.2

    def __init__(self, corpus: "Corpus", tables: dict[str, dict[str, dict[str, float]]]):
        self.corpus = corpus
        self.tables = tables                     # mode -> question -> span_id -> keyword score
        self._keys = {m: sorted(t, key=len, reverse=True) for m, t in tables.items()}

    def add_mode(self, mode: str, table: dict[str, dict[str, float]]) -> None:
        """Register another ranker's keyword scores (question -> span_id -> score) under `mode`."""
        self.tables[mode] = table
        self._keys[mode] = sorted(table, key=len, reverse=True)

    def bonus(self, s: Span, facts: str) -> float:
        """What the claim facts in the query add: the edition date, the form number, the state."""
        return bonus(s, facts, self.EDITION, self.DOC, self.STATE)

    def score_table(self, query: str, mode: str = "bm25") -> dict[str, float]:
        if mode not in self.tables:
            raise ValueError(f"retrieval mode {mode!r} has no fixture in this lab")
        q = next((k for k in self._keys[mode] if query.startswith(k)), None)
        if q is None:
            return {}
        facts = query[len(q):]
        return {sid: round(x + self.bonus(self.corpus.span(sid), facts), 2) for sid, x in self.tables[mode][q].items()}

    def search(self, query: str, k: int = 30, mode: str = "bm25") -> list[tuple[Span, float]]:
        table = self.score_table(query, mode)
        ranked = sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.corpus.span(sid), x) for sid, x in ranked]


AUTH = {Kind.FORM: 1.0, Kind.ENDORSEMENT: 1.2, Kind.BULLETIN: 1.2, Kind.GUIDANCE: 0.7, Kind.UW_NOTE: 0.9}
PULL, JUR, POS = 1.0, 0.8, 0.03
FACT_RE = re.compile(r"Policy form: (?P<form>[^.]+\)|[^.]+)\. Endorsements: (?P<ends>.*?)\. State: (?P<state>[A-Z]{2})\. "
                     r"Date of loss: (?P<dol>\d{4}-\d{2}-\d{2})\.")
EMBELLISH = "The same terms apply under every edition of this form"


def read_header(facts: str) -> dict:
    m = FACT_RE.search(facts)
    if not m:
        return {"labels": set(), "state": None, "dol": None}
    ends = [] if m["ends"] == "none" else [e.strip() for e in m["ends"].split(",")]
    return {"labels": {m["form"], *ends}, "state": m["state"], "dol": date.fromisoformat(m["dol"])}


class SimulatedModel:
    """A deterministic stand-in reader. It does not understand language; see the module notice."""

    def __init__(self, name: str, misreads: dict[str, dict[str, float]], embellish: set[str]):
        self.name = name
        self.misreads = misreads                 # determination id -> span id (or "span|other") -> weight change
        self.embellish = embellish               # determination ids answered with one unsupported sentence

    def score(self, e, hdr: dict, topic: str, n: int, mis: dict[str, float], present: frozenset = frozenset()) -> float:
        s = e.span
        if s.topic != topic:
            return -10.0 - POS * n
        x = AUTH[s.kind] - POS * n + mis.get(s.span_id, 0.0)
        # interference: "A|B" changes the weight of A only when B is also in the packet
        x += sum(v for k, v in mis.items() if "|" in k and k.split("|")[0] == s.span_id
                 and k.split("|")[1] in present)
        if e.labelled:
            p = s.provenance
            if p.label in hdr["labels"]:
                x += PULL
            if s.kind == Kind.BULLETIN:
                fits = p.jurisdiction == hdr["state"] and hdr["dol"] is not None and p.in_force(hdr["dol"])
                x += JUR if fits else -JUR
        return x

    def choose(self, rendered: Rendered, key: str, topic: str):
        hdr = read_header(rendered.facts)
        mis = self.misreads.get(key, {})
        best = None
        present = frozenset(e.span.span_id for e in rendered.entries)
        for n, e in enumerate(rendered.entries):
            x = self.score(e, hdr, topic, n, mis, present)
            if best is None or x > best[0]:
                best = (x, e)
        return None if best is None else best[1]

    def generate(self, rendered: Rendered, key: str, topic: str) -> Answer:
        e = self.choose(rendered, key, topic)
        if e is None:
            text = "The documents provided contain no provision on this question, so I cannot draft a determination."
            return Answer(text=text, cited_span_id=None, cited_label=None, citation=None,
                          tokens_out=count_tokens(text) + 1, abstained=True)
        claims = [operative(e.span)]
        if key in self.embellish:
            claims.append(EMBELLISH)
        if e.labelled:
            citation = e.span.citation()
            gov = f"Governing provision: {citation} [{e.tag}]."
        else:
            citation = None
            gov = "Governing provision: the clause quoted above, as given."
        text = " ".join(c + "." for c in claims) + " " + gov
        return Answer(text=text, cited_span_id=e.span.span_id, cited_label=e.tag or None, citation=citation,
                      tokens_out=count_tokens(text) + 1, claims=claims)


# ------------------------------------------------------------------ world loading
class Corpus:
    def __init__(self, spans: list[Span], dets: list[Determination], policies: dict[str, dict],
                 history: list[Span], cal: dict):
        self.spans = spans
        self._by_id = {s.span_id: s for s in spans}
        self.dets = dets
        self.policies = policies
        self.history = history
        self.cal = cal

    def span(self, sid: str) -> Span:
        return self._by_id[sid]

    def eval_set(self, n: int = 200) -> list[Determination]:
        return self.dets[:n]

    def holdout(self) -> list[Determination]:
        return self.dets[200:]

    def history_for(self, det: Determination) -> list[Span]:
        return [h for h in self.history if h.meta.get("policy_number") == det.policy_number]

    def retriever(self) -> Retriever:
        return Retriever(self, self.cal["tables"])

    def policy(self) -> Policy:
        return Policy(ROLE)

    def model(self, name: str = "sim-large") -> SimulatedModel:
        m = self.cal["models"][name]
        return SimulatedModel(name, m["misreads"], set(m["embellish"]))


def gold_for(row: dict, by_id: dict[str, Span]) -> Span:
    """The governing span by Lakeshore's precedence rule: bulletin in force, then endorsement, then form."""
    dol = row["date_of_loss"]
    for s in by_id.values():
        if (s.kind == Kind.BULLETIN and s.topic == row["topic"] and s.provenance.jurisdiction == row["state"]
                and s.provenance.in_force(dol)):
            return s
    for e in row["endorsements"]:
        doc, ver = e.split(" (")[0], e.split(" (")[1].rstrip(")")
        sid = f"{doc}@{ver}#{row['topic']}"
        if sid in by_id:
            return by_id[sid]
    doc, ver = row["form"].split(" (")[0], row["form"].split(" (")[1].rstrip(")")
    return by_id[f"{doc}@{ver}#{row['topic']}"]


def build_rows(cal: dict | None = None) -> list[dict]:
    rows = world.build_working() + world.build_holdout()
    for r in rows:
        q = (cal or {}).get("questions", {}).get(r["determination_id"])
        if q:
            r["question"] = q
    return rows


def history_records(dets: list[Determination]) -> list[Span]:
    out = []
    r = world.rng("history")
    for i, d in enumerate(r.sample(dets, 157), 1):
        text = (f"Prior claim on policy {d.policy_number}: {r.choice(['hail', 'wind', 'water', 'theft', 'glass'])} "
                f"loss closed {r.randint(2019, 2024)}, paid without dispute.")
        out.append(Span(span_id=f"HIST-{i:04d}", kind=Kind.GUIDANCE, text=text,
                        provenance=Provenance(doc_id="claim-history", version=f"{i:04d}", source_system="claims"),
                        meta={"policy_number": d.policy_number}))
    return out


@lru_cache(maxsize=1)
def load_corpus() -> Corpus:
    cal = json.loads(CAL.read_text()) if CAL.exists() else {"lengths": {}, "tables": {"bm25": {}},
                                                             "models": {m: {"misreads": {}, "embellish": []}
                                                                        for m in ("sim-large", "sim-mid", "sim-small")}}
    return build_corpus(cal)


def build_corpus(cal: dict) -> Corpus:
    specs = world.all_span_specs()
    spans = [to_span(s, cal["lengths"].get(s["span_id"])) for s in specs]
    by_id = {s.span_id: s for s in spans}
    dets, policies = [], {}
    for r in build_rows(cal):
        g = gold_for(r, by_id)
        kind = g.kind.value
        assert kind == r["kind"], (r["determination_id"], kind, r["kind"])
        det = Determination(**{k: v for k, v in r.items() if k != "kind"}, gold_span_id=g.span_id,
                            gold_label=g.provenance.label, gold_kind=kind, verdict=g.meta["verdict"],
                            difficulty="simple" if kind == "form" else "complex")
        dets.append(det)
        policies[det.policy_number] = {"form": det.form, "endorsements": list(det.endorsements),
                                       "state": det.state, "line": det.line}
    return Corpus(spans, dets, policies, history_records(dets), cal)


EVAL_COLUMNS = ["determination_id", "state", "form", "topic", "gold_label", "gold_span_id", "gold_kind", "verdict",
                "difficulty"]


def eval_csv(dets: list[Determination]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(EVAL_COLUMNS)
    for d in dets:
        w.writerow([getattr(d, c) for c in EVAL_COLUMNS])
    return buf.getvalue()
