"""contextlab.assembler: the context schema and the packet assembler (Chapter 31).

contextlab v6.7. Retrieval and selection (Chapters 20 and 32) decide which spans enter the packet;
the assembler decides how the reader meets them: in which sections, in which
order, next to what, under which identifiers and in which syntax. A context
schema declares the sections an agent's packet must have, which roles may be
shortened, and how spans are named. The assembler renders the packet, assigns
identifiers after the order is final, freezes an id -> span map for the record,
and refuses to assemble when two different spans would share an identifier.

Token counts use the o200k_base tokeniser when tiktoken is installed and a
regex proxy otherwise; `TOKENIZER` says which one produced a number.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from contextlab.citations import Entry, HANDLE, Manifest

VERSION = "6.7"
ROLES = ("instructions", "facts", "governing", "modifier", "context", "question")
SHORT = re.compile(r"\[(S\d+)\]")

try:                                                    # real tokeniser if available
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
    TOKENIZER = "o200k_base"

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:                                       # pragma: no cover
    TOKENIZER = "regex-proxy"

    def count_tokens(text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text))


@dataclass
class Span:
    """A packet span with the role the pipeline gave it."""
    entry: Entry
    role: str = "context"            # facts | governing | modifier | context
    modifies: Entry | None = None    # for a modifier: the clause it changes
    topic: str = ""


@dataclass
class Section:
    role: str
    tag: str
    required: bool = True
    elastic: bool = False            # may be shortened under budget


@dataclass
class ContextSchema:
    """What one agent's packet must contain, in order."""
    name: str
    version: str
    sections: list
    repeat_question: bool = True
    ids: str = "short"

    def validate(self, spans: list) -> list:
        problems, roles = [], {s.role for s in spans}
        for sec in self.sections:
            if sec.required and sec.role not in roles and sec.role not in ("instructions", "question"):
                problems.append(f"missing required section {sec.tag}")
        inpacket = {s.entry for s in spans}
        for s in spans:
            if s.role not in ROLES:
                problems.append(f"unknown role {s.role} for {s.entry.handle}")
            if s.role == "modifier" and s.modifies not in inpacket:
                problems.append(f"modifier {s.entry.handle} without the clause it modifies")
        return problems


CLAIMS_SCHEMA = ContextSchema("claims-determination", "1.0", [
    Section("instructions", "instructions"),
    Section("facts", "claim_facts"),
    Section("governing", "governing"),
    Section("context", "context", required=False, elastic=True),
    Section("question", "question")])


class CollisionError(ValueError):
    """Two different spans would carry the same identifier."""


def positional_id(e: Entry) -> str:
    return f"{e.doc}@{e.edition}/p{e.page}/s{e.n}"


def make_ids(entries: list, scheme: str) -> list:
    if scheme == "short":
        return [f"S{i + 1}" for i in range(len(entries))]
    if scheme == "handle":
        return [e.handle for e in entries]
    if scheme == "positional":
        return [positional_id(e) for e in entries]
    return [None] * len(entries)


def trim(text: str, words: int) -> str:
    w = text.split()
    return text if len(w) <= words else " ".join(w[:words]) + " [...]"


def order_spans(spans: list, how: str) -> list:
    if how == "retrieval":
        return list(spans)
    if how == "document":
        return sorted(spans, key=lambda s: (s.entry.doc, s.entry.edition, s.entry.page, s.entry.n))
    if how == "edges":                       # strongest first and last, weakest in the middle
        front, back = list(spans[0::2]), list(spans[1::2])
        return front + back[::-1]
    if how == "schema":                      # facts, governing (modifier after its clause), context
        rank = {"facts": 0, "governing": 1, "modifier": 1, "context": 2}
        base = [s for s in spans if s.role != "modifier"]
        base.sort(key=lambda s: (rank[s.role], s.topic, s.entry.doc, s.entry.page, s.entry.n))
        out = []
        for s in base:
            out.append(s)
            out += [m for m in spans if m.role == "modifier" and m.modifies == s.entry]
        return out + [m for m in spans if m.role == "modifier" and m not in out]
    raise ValueError(how)


@dataclass
class Assembly:
    text: str
    order: list                      # Span objects in rendered order
    ids: list                        # identifier shown for each span (or None)
    shown: dict                      # source Entry -> Entry as shown (text may be trimmed)
    idmap: dict = field(default_factory=dict)      # identifier -> shown Entry (frozen)
    collisions: list = field(default_factory=list)

    def manifest(self) -> Manifest:
        return Manifest([self.shown[s.entry] for s in self.order])

    def position(self, entry: Entry) -> float:
        ev = [s.entry for s in self.order if s.role != "facts"]
        if entry not in ev or len(ev) < 2:
            return 0.0
        return ev.index(entry) / (len(ev) - 1)

    def to_handles(self, sentence: str) -> str:
        """Rewrite the identifiers the reader saw into stored handles."""
        sub = lambda i: f"[[{self.idmap[i].handle}]]" if i in self.idmap else f"[[unknown:{i}]]"
        if self.ids and self.ids[0] == "S1":
            return SHORT.sub(lambda m: sub(m.group(1)), sentence)
        return HANDLE.sub(lambda m: sub(m.group(1)), sentence)


class Assembler:
    def __init__(self, fmt="schema", order="schema", ids="short", schema=CLAIMS_SCHEMA,
                 trim_context=None, check_ids=True, trim_roles=("context",)):
        self.trim_roles = trim_roles
        self.fmt, self.how, self.scheme, self.schema = fmt, order, ids, schema
        self.trim_context, self.check_ids = trim_context, check_ids

    def _shown(self, s: Span) -> Entry:
        """The entry as the reader will see it: context spans may be trimmed."""
        t = s.entry.text
        if self.trim_context and s.role in self.trim_roles:
            t = trim(t, self.trim_context)
        return Entry(s.entry.doc, s.entry.edition, s.entry.page, s.entry.n, t)

    def assemble(self, instructions: str, question: str, spans: list) -> Assembly:
        order = order_spans(spans, self.how)
        order = _by_section(order, self.schema) if self.fmt == "schema" else order
        ids = make_ids([s.entry for s in order], self.scheme)   # assigned last
        shown = {s.entry: self._shown(s) for s in order}        # elastic trim, context only
        idmap, seen, coll = {}, {}, []
        for i, s in ((i, s) for i, s in zip(ids, order) if i is not None):
            if i in seen and seen[i] != s.entry:
                coll.append(i)                         # one identifier, two spans
            seen[i], idmap[i] = s.entry, shown[s.entry]   # a dict keeps the last writer
        if coll and self.check_ids:
            raise CollisionError(f"{len(coll)} identifier(s) name two spans: {sorted(set(coll))}")
        text = RENDER[self.fmt](self, instructions, question, order, ids, shown)
        return Assembly(text, order, ids, shown, idmap, coll)


def _in(sec, s):
    return s.role == sec.role or (sec.role == "governing" and s.role == "modifier")


def _by_section(order: list, schema: ContextSchema) -> list:
    out = [s for sec in schema.sections for s in order if _in(sec, s)]
    return out + [s for s in order if s not in out]


def _attrs(e: Entry):
    return f'doc="{e.doc}" edition="{e.edition}" page="{e.page}"'


def _plain(a, ins, q, order, ids, shown):
    out, last = [ins, ""], None
    for s in order:
        out.append(shown[s.entry].text)
        nxt = order[order.index(s) + 1].entry if s is not order[-1] else None
        if nxt is None or (nxt.doc, nxt.page) != (s.entry.doc, s.entry.page):
            out.append(f"(Source: {s.entry.doc} ({s.entry.edition}), page {s.entry.page})")
    return "\n".join(out + ["", q])


def _json(a, ins, q, order, ids, shown):
    return json.dumps({"instructions": ins, "spans": [
        {"id": i, "doc": s.entry.doc, "edition": s.entry.edition, "page": s.entry.page,
         "text": shown[s.entry].text} for i, s in zip(ids, order)], "question": q}, indent=2)


def _markdown(a, ins, q, order, ids, shown):
    out = ["## Instructions", ins, "", "## Documents"]
    for i, s in zip(ids, order):
        out += [f"### [{i}] {s.entry.doc} ({s.entry.edition}) | page {s.entry.page}", shown[s.entry].text, ""]
    return "\n".join(out + ["## Question", q])


def _xml(a, ins, q, order, ids, shown):
    out = [f"<instructions>{ins}</instructions>", "<documents>"]
    out += [f'<span id="{i}" {_attrs(s.entry)}>{shown[s.entry].text}</span>' for i, s in zip(ids, order)]
    return "\n".join(out + ["</documents>", f"<question>{q}</question>"])


def _schema(a, ins, q, order, ids, shown):
    idof = {s.entry: i for i, s in zip(ids, order)}
    out = ["# Instructions", ins, "", f"<question>{q}</question>"]
    for sec in a.schema.sections:
        if sec.role in ("instructions", "question"):
            continue
        rows = [(i, s) for i, s in zip(ids, order) if _in(sec, s)]
        if not rows:
            continue
        out.append(f"<{sec.tag}>")
        for i, s in rows:
            rel = f' modifies="{idof[s.modifies]}"' if s.role == "modifier" and s.modifies in idof else ""
            out.append(f'<span id="{i}" {_attrs(s.entry)}{rel}>{shown[s.entry].text}</span>')
        out.append(f"</{sec.tag}>")
    if a.schema.repeat_question:
        out.append(f"<question>{q}</question>")
    return "\n".join(out)


RENDER = {"plain": _plain, "json": _json, "markdown": _markdown, "xml": _xml, "schema": _schema}


# Tables: the same rows in four encodings, for the token comparison.
def _cell(v, delim=","):
    s = str(v)
    return f'"{s}"' if any(c in s for c in (delim, ":", '"')) or s != s.strip() else s


def encode_table(name: str, rows: list, how: str) -> str:
    keys = list(rows[0])
    if how == "json":
        return json.dumps({name: rows}, indent=2)
    if how == "json-compact":
        return json.dumps({name: rows}, separators=(",", ":"))
    if how == "toon":
        head = f"{name}[{len(rows)}]{{{','.join(keys)}}}:"
        return "\n".join([head] + ["  " + ",".join(_cell(r[k]) for k in keys) for r in rows])
    if how == "csv":
        return "\n".join([",".join(keys)] + [",".join(_cell(r[k]) for k in keys) for r in rows])
    if how == "markdown":
        return "\n".join(["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] +
                         ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows])
    raise ValueError(how)
