"""contextlab.tools: descriptions, loadouts, result shaping and source trust (Chapter 26, v5.8).

Chapter 26. Everything here is deterministic. The 'selector' is a stand-in
for a model choosing a tool: it scores each loaded description against the
step by a concept-vector cosine and adds noise whose scale grows with the
tokens of tool text in the window. That growth is an assumption of the
stand-in, stated in the chapter's Note, not a measurement of any model.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from .retrieval.fusion import rrf
from .retrieval.index import InvertedIndex

DIM = 256
WORD = re.compile(r"[a-z0-9]+")
STOP = set("a an and are as at be by for from in into is it its of on or "
           "the to was with this that which use when".split())
GENERIC = {"get", "return", "returns", "details", "detail", "information",
           "info", "data", "record", "records", "lookup", "tool", "fetch"}


def tokens(text: str) -> int:
    """Token estimate: four characters per token, the usual rule of thumb."""
    return max(1, math.ceil(len(text) / 4))


def _direction(name: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    v = np.random.default_rng(seed).standard_normal(DIM)
    return v / np.linalg.norm(v)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: tuple[str, ...] = ()
    server: str = "lakeshore-claims"
    trust: str = "internal"            # internal | partner | public
    examples: tuple[str, ...] = ()

    returns: tuple[str, ...] = ()

    def definition(self) -> str:
        """The text that enters the window: name, description, schemas, examples."""
        props = {p: {"type": "string", "description": PARAM_DOC.get(p, p.replace(
            "_", " ") + ", exactly as recorded in the source system")} for p in self.params}
        schema = {"name": self.name, "description": self.description,
                  "input_schema": {"type": "object", "properties": props,
                                   "required": list(self.params)},
                  "output_schema": {"type": "object", "properties": {
                      f: {"type": "string"} for f in self.returns}}}
        if self.examples:
            schema["input_examples"] = list(self.examples)
        return json.dumps(schema)

    def text(self) -> str:
        """What a reader of the definition sees, minus JSON punctuation."""
        docs = [PARAM_DOC[p] for p in self.params if p in PARAM_DOC]
        return " ".join([self.name.replace("_", " "), self.description, *docs,
                         *self.examples])

    def tokens(self) -> int:
        return tokens(self.definition())

    def pin(self) -> str:
        return hashlib.sha256(self.definition().encode()).hexdigest()[:16]


PARAM_DOC = {
    "as_of_date": "the date the answer must be true on; for a claim, the date of loss (YYYY-MM-DD)",
    "policy_number": "Lakeshore policy number, for example WI-HO-104233",
    "claim_id": "claim identifier, for example CLM-2026-01187",
}
IDS = [(re.compile(r"^clm-"), "claim"), (re.compile(r"^[a-z]{2}-ho-"), "polnum"),
       (re.compile(r"^ho3-\d"), "span"), (re.compile(r"^r-\d"), "rule")]


class Encoder:
    """Concept vectors: synonyms share a direction; generic words weigh little."""

    def __init__(self, lexicon: dict[str, str]):
        self.lexicon = lexicon
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        """Sentences that begin 'Not for' or 'Does not' count against a concept."""
        if text in self._cache:
            return self._cache[text]
        acc = np.zeros(DIM)
        for sentence in re.split(r"(?<=[.;])\s+", text.lower()):
            sign = -0.4 if sentence.startswith(("not for", "does not")) else 1.0
            for c, w in self._concepts(sentence).items():
                acc += sign * w * _direction(c)
        n = np.linalg.norm(acc)
        self._cache[text] = acc / n if n else acc
        return self._cache[text]

    def _concepts(self, text: str) -> dict[str, float]:
        words = set(WORD.findall(text))
        for ident in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", text):
            for pat, concept in IDS:
                if pat.match(ident):
                    words.add("__" + concept)
        out: dict[str, float] = {}
        for w in words - STOP:
            c = (w[2:] if w.startswith("__") else None) or self.lexicon.get(w) \
                or self.lexicon.get(w.rstrip("s")) or w
            out[c] = max(out.get(c, 0.0), 0.25 if w in GENERIC else 1.0)
        return out


def noise_scale(window_tool_tokens: int, base: float = 0.01,
                per_doubling: float = 0.01) -> float:
    """Stand-in assumption: confusion grows with log tool-text tokens."""
    return base + per_doubling * max(0.0, math.log2(window_tool_tokens / 1000))


def select(step: str, loadout: list[ToolSpec], enc: Encoder, key: str,
           extra_tokens: int = 0) -> str:
    """The stand-in model's choice of tool for one step."""
    sigma = noise_scale(sum(t.tokens() for t in loadout) + extra_tokens)
    q = enc.encode(step)
    best, best_s = "", -9.0
    for t in loadout:
        seed = int.from_bytes(hashlib.sha256(f"{key}|{t.name}".encode())
                              .digest()[:8], "big")
        eps = np.random.default_rng(seed).standard_normal() * sigma
        s = float(enc.encode(t.text()) @ q) + eps
        if s > best_s:
            best, best_s = t.name, s
    return best


class ToolIndex:
    """Hybrid retrieval over tool descriptions (and their examples)."""

    def __init__(self, tools: list[ToolSpec], enc: Encoder, expand: bool = True):
        self.tools = {t.name: t for t in tools}
        self.enc = enc
        self.lex = InvertedIndex()
        self.rows = {}
        for t in tools:
            text = t.text() if expand else t.name.replace("_", " ") + " " + t.description
            self.lex.add(t.name, text)
            self.rows[t.name] = enc.encode(text)

    def search(self, step: str, k: int) -> list[str]:
        q = self.enc.encode(step)
        dense = {n: float(v @ q) for n, v in self.rows.items()}
        fused = rrf(self.lex.bm25(step), dense)
        return [n for n, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]]


def loadout(step: str, index: ToolIndex, core: list[str], k: int) -> list[ToolSpec]:
    """Core tools always; the top k retrieved for this step; no duplicates."""
    names = list(core) + [n for n in index.search(step, k + len(core))
                          if n not in core][:k]
    return [index.tools[n] for n in names]


# Tool results -------------------------------------------------------------

@dataclass
class Shape:
    keep: tuple[str, ...]
    units: dict[str, str] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)


def shape_result(raw: dict, shape: Shape, tool: str, read_at: str) -> str:
    """A tool result as a span: named fields, units, notes, source and read time."""
    lines = [f"[tool {tool} | read {read_at} | {len(shape.keep)} of {len(raw)} fields]"]
    for k in shape.keep:
        unit = f" {shape.units[k]}" if k in shape.units else ""
        note = f"  ({shape.notes[k]})" if k in shape.notes else ""
        lines.append(f"{k}: {raw[k]}{unit}{note}")
    return "\n".join(lines)


# Source trust ---------------------------------------------------------------

TRUST_ORDER = {"internal": 0, "partner": 1, "public": 2}


def admit_tool(t: ToolSpec, pins: dict[str, str], max_trust: str) -> tuple[bool, str]:
    """A description enters the window only if approved, unchanged and trusted."""
    if t.name not in pins:
        return False, "not approved"
    if pins[t.name] != t.pin():
        return False, "description changed since approval"
    if TRUST_ORDER[t.trust] > TRUST_ORDER[max_trust]:
        return False, f"server trust {t.trust} above {max_trust}"
    return True, "ok"
