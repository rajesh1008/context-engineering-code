"""contextlab v6.4: the Citation Checker (Chapter 28).

A determination is a list of sentences. Each sentence carries zero or more
pointers to spans in the context packet, or an explicit tag saying it needs
none. The checker resolves every pointer against the packet manifest, asks a
verifier whether the span supports the sentence, tries one re-attribution
inside the packet when it does not, and flags what it cannot fix. It never
deletes a sentence or a pointer: it records what it found.

The verifier here is a deterministic stand-in (token coverage with number and
polarity checks). Swap in any entailment model with the same signature:
verify(sentence, span_text) -> score in [0, 1].
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

VERSION = "6.4"
STOP = set("""a an the of to in on for by or and is are be was were it its this that
with as at from any which who we you your our us do does not no but if all
will may shall under than then there their they has have had been being""".split())
NEG = {"not", "no", "none", "excluded", "exclude", "excludes", "never", "denied", "deny"}
COND = {"unless", "except", "only"}             # a negative clause with a way out
NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
HANDLE = re.compile(r"\[\[([^\]]+)\]\]")      # [[HO 04 95@2024-06/p1/s2#a1b2c3d4]]
INDEX = re.compile(r"\[(\d+)\]")               # [3]
DOCPAGE = re.compile(r"\[([A-Z][A-Z0-9 \-]+?) \(([\w\-]+)\)(?:, page (\d+))?\]")
NOSOURCE = "[no-source]"
CODE = re.compile(r"\b[A-Z]{2,}[- ]?\d+(?: \d+)*\b")   # form names such as HO-3, HO 04 95


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class Entry:
    """One span as it sits in the packet manifest."""
    doc: str
    edition: str
    page: int
    n: int                    # span number on the page
    text: str

    @property
    def handle(self) -> str:
        return f"{self.doc}@{self.edition}/p{self.page}/s{self.n}#{text_hash(self.text)}"


@dataclass
class Manifest:
    """The packet as assembled: order matters only for index pointers."""
    entries: list
    by_handle: dict = field(init=False)

    def __post_init__(self):
        self.by_handle = {e.handle: e for e in self.entries}

    def resolve(self, kind: str, ref) -> list:
        """Candidate spans a pointer can mean. One for a handle or index."""
        if kind == "handle":
            e = self.by_handle.get(ref)
            return [e] if e else []
        if kind == "index":
            return [self.entries[ref - 1]] if 1 <= ref <= len(self.entries) else []
        doc, ed, page = ref                      # document- or page-level
        return [e for e in self.entries if e.doc == doc and e.edition == ed
                and (page is None or e.page == page)]


def parse(sentence: str):
    """Pointers in a sentence: list of (kind, ref)."""
    out = [("handle", h) for h in HANDLE.findall(sentence)]
    plain = HANDLE.sub("", sentence)
    out += [("index", int(i)) for i in INDEX.findall(plain)]
    for doc, ed, page in DOCPAGE.findall(plain):
        out.append(("page" if page else "doc", (doc, ed, int(page) if page else None)))
    return out


def strip(sentence: str) -> str:
    s = HANDLE.sub("", sentence).replace(NOSOURCE, "")
    return DOCPAGE.sub("", INDEX.sub("", s)).strip()


def _norm(w):
    w = w.lower().strip(".,;:()'\"")
    for suf in ("ing", "ed", "es", "e", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content(text: str) -> list:
    return [w for w in map(_norm, NUM.sub(" ", text).split()) if w and w not in STOP]


def numbers(text: str) -> set:
    return {n.replace("$", "").replace(",", "").rstrip(".") for n in NUM.findall(text)}


def verify(sentence: str, span: str) -> float:
    """Stand-in verifier: how much of the sentence the span accounts for."""
    sentence, span = CODE.sub(" ", strip(sentence)), CODE.sub(" ", span)
    s, t = content(sentence), set(content(span))
    if not s:
        return 0.0
    score = sum(w in t for w in s) / len(s)
    if not numbers(strip(sentence)) <= numbers(span):
        score *= 0.3                                  # a number the span does not contain
    raw = lambda x: {w.lower().strip(".,;:()'\"") for w in x.split()}
    ws, wt = raw(sentence), raw(span)
    neg_s, neg_t = bool(NEG & ws), bool(NEG & wt)
    if neg_s != neg_t and not (neg_t and COND & wt and not neg_s):
        score *= 0.5            # polarity differs, unless the span's negative has a way out
    return round(score, 4)


def bm25_rank(query: str, entries: list, k1=1.2, b=0.75) -> list:
    docs = [content(e.text) for e in entries]
    avg = sum(map(len, docs)) / max(1, len(docs))
    df = Counter(w for d in docs for w in set(d))
    q, scores = content(strip(query)), []
    for e, d in zip(entries, docs):
        tf, s = Counter(d), 0.0
        for w in q:
            if tf[w]:
                idf = math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        scores.append((s, e))
    return [e for s, e in sorted(scores, key=lambda x: -x[0]) if s > 0]


@dataclass
class Finding:
    sentence: str
    pointers: list            # [(kind, ref)]
    resolved: list            # spans each pointer resolved to (flattened)
    score: float              # best verifier score over resolved spans
    status: str               # supported | reattributed | unverified | missing | exempt | edition
    moved_to: str | None = None
    ambiguous: bool = False   # pointer resolved to more than one span


class CitationChecker:
    """Verify, re-attribute once, flag. Edition check against what governs."""

    def __init__(self, threshold=0.6, verifier=verify, reattribute=True):
        self.threshold, self.verifier, self.reattribute = threshold, verifier, reattribute

    def check_sentence(self, sentence, manifest, in_force=None):
        if NOSOURCE in sentence:
            return Finding(sentence, [], [], 0.0, "exempt")
        ptrs = parse(sentence)
        if not ptrs:
            return Finding(sentence, [], [], 0.0, "missing")
        spans = [e for k, r in ptrs for e in manifest.resolve(k, r)]
        amb = any(len(manifest.resolve(k, r)) > 1 for k, r in ptrs)
        best = max((self.verifier(sentence, e.text) for e in spans), default=0.0)
        if len(spans) > 1:                      # a sentence may rest on two spans jointly
            best = max(best, self.verifier(sentence, " ".join(e.text for e in spans)))
        if in_force and any(in_force.get(e.doc) not in (None, e.edition) for e in spans):
            return Finding(sentence, ptrs, spans, best, "edition", ambiguous=amb)
        if best >= self.threshold:
            return Finding(sentence, ptrs, spans, best, "supported", ambiguous=amb)
        if self.reattribute:
            for e in bm25_rank(sentence, manifest.entries)[:1]:
                s = self.verifier(sentence, e.text)
                if s >= self.threshold and e not in spans:
                    return Finding(sentence, ptrs, spans, s, "reattributed", e.handle, amb)
        return Finding(sentence, ptrs, spans, best, "unverified", ambiguous=amb)

    def check(self, sentences, manifest, in_force=None):
        return [self.check_sentence(s, manifest, in_force) for s in sentences]


def mark(f: Finding) -> str:
    """How the sentence is shown to the adjuster: never deleted."""
    if f.status in ("unverified", "missing", "edition"):
        return f"{f.sentence} [UNVERIFIED: {f.status}]"
    if f.status == "reattributed":
        return f"{f.sentence} [moved to [[{f.moved_to}]]]"
    return f.sentence


def precision_recall(labels):
    """labels: per sentence dict(needs, cites, supported_cites, total_cites).

    precision = supporting pointers / all pointers
    recall    = claim sentences with at least one supporting pointer / claim sentences
    """
    tot = sum(l["total_cites"] for l in labels)
    sup = sum(l["supported_cites"] for l in labels)
    need = [l for l in labels if l["needs"]]
    rec = sum(1 for l in need if l["supported_cites"] > 0)
    return (sup / tot if tot else float("nan"), rec / len(need) if need else float("nan"))
