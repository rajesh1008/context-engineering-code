"""contextlab.compress: compressing a packet without making it wrong (Chapter 33).

contextlab v6.9. Three compressor families and the checks that decide whether
a compressed packet can be trusted:

* `extract`      sentence selection with span handles kept (extractive);
* `prune_tokens` token deletion by an importance score (the shape of learned
                 token-classification compressors), with a force-keep set;
* abstractive compressors are model calls; the lab supplies a stand-in.

Checks: `negation_profile` and `polarity_diff` (cheap, catch inversions),
`probe` (later questions answered on the original and the compressed packet),
and `survival` (which decision facts survived verbatim, altered or not at all).
Compaction helpers for long-running agents: `Pinned`, `compact_threshold`,
`trim_results`. Token counts use four characters per token.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

NEG = ("not", "no", "never", "unless", "except", "excluded", "exclusion", "without", "void")
_WORD = re.compile(r"[A-Za-z]+|US\$[\d,]+|\d[\d,/:.]*")


def tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class Sentence:
    handle: str            # span_id#sN, the thing a citation points to
    text: str
    kind: str = ""         # source kind: policy, endorsement, notes, letters ...

    @property
    def tokens(self) -> int:
        return tokens(self.text)


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def negation_profile(text: str) -> int:
    """Count of negation and exception cues: a cheap fingerprint of polarity."""
    return sum(w.lower() in NEG for w in words(text))


def polarity_diff(original: str, compressed: str) -> int:
    """Cues lost (positive) or gained (negative) between two versions of a fact."""
    return negation_profile(original) - negation_profile(compressed)


# ------------------------------------------------------------- extractive
def extract(sents: list[Sentence], score: Callable[[Sentence], float], budget: int,
            protect: Callable[[Sentence], bool] = lambda s: False,
            link: Callable[[Sentence, Sentence], bool] | None = None) -> list[Sentence]:
    """Keep the best-scoring sentences within `budget` tokens, in original order.

    `protect` marks sentences that are always kept (exceptions, conditions,
    findings). `link(a, b)` pulls in b when a is kept and b qualifies a, so a
    clause never travels without its exception: the clause-group rule."""
    keep = [s for s in sents if protect(s)]
    used = sum(s.tokens for s in keep)
    for s in sorted(sents, key=score, reverse=True):
        if s in keep or used + s.tokens > budget:
            continue
        keep.append(s); used += s.tokens
    if link:
        for a in list(keep):
            keep += [b for b in sents if b not in keep and link(a, b)]
    order = {s.handle: i for i, s in enumerate(sents)}
    return sorted(keep, key=lambda s: order[s.handle])


# ------------------------------------------------------------- token pruning
def prune_tokens(text: str, ratio: float, score: Callable[[str], float],
                 force: Iterable[str] = ()) -> str:
    """Delete the lowest-scoring tokens until about `ratio` of them remain.

    Tokens in `force` (lower-cased) are kept whatever their score: negations,
    exception words and anything that looks like a number or a form code."""
    toks = text.split()
    forced = {f.lower() for f in force}
    must = [i for i, t in enumerate(toks) if t.strip(".,;:").lower() in forced
            or any(c.isdigit() for c in t)] if forced else []
    rest = sorted((i for i in range(len(toks)) if i not in set(must)),
                  key=lambda i: score(toks[i]), reverse=True)
    n = max(0, round(ratio * len(toks)) - len(must))
    kept = sorted(set(must) | set(rest[:n]))
    return " ".join(toks[i] for i in kept)


def prune_packet(texts: list[str], ratio: float, score: Callable[[str, str], float],
                 force: Iterable[str] = ()) -> list[str]:
    """`prune_tokens` with one threshold across the whole packet. `score(tok,
    sentence)` sees the token's sentence, as a bidirectional scorer would."""
    forced = {f.lower() for f in force}
    toks = [(i, j, t) for i, x in enumerate(texts) for j, t in enumerate(x.split())]
    must = {(i, j) for i, j, t in toks if forced and (t.strip(".,;:").lower() in forced
                                                   or any(c.isdigit() for c in t))}
    rest = sorted(((i, j) for i, j, t in toks if (i, j) not in must),
                  key=lambda ij: score(texts[ij[0]].split()[ij[1]], texts[ij[0]]),
                  reverse=True)
    keep = must | set(rest[:max(0, round(ratio * len(toks)) - len(must))])
    out = [" ".join(t for j, t in enumerate(x.split()) if (i, j) in keep) for i, x in enumerate(texts)]
    return [x for x in out if x]


# ------------------------------------------------------------- checks
@dataclass
class Probe:
    name: str
    ask: Callable[[str], object]       # a question answered from packet text


def probe(original, compressed, probes: list[Probe]) -> list[str]:
    """Names of the probes whose answer changed: a summary passes when none do."""
    return [p.name for p in probes if p.ask(original) != p.ask(compressed)]


@dataclass(frozen=True)
class Fact:
    kind: str           # exclusion, exception, limit, deductible, condition ...
    key: str            # what must appear for the fact to be present
    value: str          # the value that must come with it
    polarity: int = 0   # negation cues in the source sentence


def survival(facts: list[Fact], read: Callable[[Fact], tuple[bool, str, int]]) -> dict:
    """Classify each decision fact as kept, altered or dropped.

    `read(fact)` returns (present, value_found, negation_cues) from the
    compressed packet. Altered means present with a different value or polarity."""
    out = {"kept": [], "altered": [], "dropped": []}
    for f in facts:
        present, value, pol = read(f)
        if not present:
            out["dropped"].append(f.kind)
        elif value != f.value or (f.polarity > 0) != (pol > 0):
            out["altered"].append(f.kind)
        else:
            out["kept"].append(f.kind)
    return out


# ------------------------------------------------------------- long-running agents
@dataclass
class Pinned:
    """Constraints kept outside the region a compactor may summarise."""
    items: dict[str, str] = field(default_factory=dict)

    def pin(self, cid: str, text: str) -> None:
        self.items[cid] = text

    def render(self) -> str:
        return "\n".join(f"[{k}] {v}" for k, v in sorted(self.items.items()))


def compact_threshold(window_tokens: int, threshold: int) -> bool:
    """Threshold compaction: run when the rendered window crosses the line."""
    return window_tokens >= threshold


def trim_results(items: list[dict], keep_last: int) -> list[dict]:
    """Trimming: replace all but the last `keep_last` tool results with stubs
    that keep the call id and handle, so the result can be fetched again."""
    res = [i for i, it in enumerate(items) if it["role"] == "tool"]
    old = set(res[:-keep_last]) if keep_last else set(res)
    return [dict(it, text=f"[cleared {it['id']}; refetch by handle]")
            if i in old else it for i, it in enumerate(items)]
