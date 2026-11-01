"""Listing 3.13: cache identity, cost and latency of preloading the whole corpus, and what a
stale snapshot behind a name-keyed cache does to the answers."""
from __future__ import annotations

import copy
import hashlib
import json

from common import tee
import standin as S
from standin import CUT, Pipeline, PipelineConfig, PromptCache, cost_usd, latency_s
from contextlab.instructions import V22


def main():
    dets, corpus = S.load_dets(), S.CORPUS
    stale, rows = copy.copy(corpus), []

    def cache_key(**parts: str) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]

    stale.spans = [s for s in corpus.spans
                   if s.provenance.effective_from is None or s.provenance.effective_from <= CUT]
    for label, c in (("current corpus", corpus), ("stale snapshot (name-keyed cache)", stale)):
        p = Pipeline(PipelineConfig(stuff="all", order="cache"), corpus=c, cache=PromptCache())
        ts = p.run_many(dets)
        rows.append((label, ts))

    print("cache identity (hash of tenant, corpus, model, tokenizer, instructions, policy):")
    for label, c in (("current corpus", corpus), (f"{CUT} snapshot", stale)):
        key = cache_key(tenant=c.tenant, corpus=c.version(), model="sim-large", tokenizer="contextlab-0.8",
                        instructions=V22.version, policy="applicability-v2")
        print(f"  {label} {c.version()} -> key {key}")
    missing = len(corpus.spans) - len(stale.spans)
    print(f"  spans effective after {CUT} missing from the snapshot: {missing} of {len(corpus.spans)}")
    print()
    print(f"preloaded whole corpus, cache order, {len(dets)} determinations (sim-large)")
    for label, ts in rows:
        n, tok = len(ts), sum(t.tokens_in for t in ts)
        w = sum(t.wrong for t in ts)
        print(f"  {label:<35}wrong {w:>3}/{n} ({w / n:.1%})  gold not in packet "
              f"{sum(not t.gold_in_packet for t in ts):>3}{round(tok / n):>9,} tokens"
              f"  cached {sum(t.cached for t in ts) / tok:>6.1%}  ${sum(t.cost for t in ts) / n:.5f}"
              f"  {sum(t.latency for t in ts) / n:.2f} s")
    rt = Pipeline(PipelineConfig(policy_filter=True)).run_many(dets)
    print(f"  {'retrieve (filter, k=10), for comparison':<39}{round(sum(t.tokens_in for t in rt) / len(rt)):>8,} tokens"
          f"  {sum(t.latency for t in rt) / len(rt):.2f} s")
    print()
    ts = rows[0][1]
    tok = round(sum(t.tokens_in for t in ts) / len(ts))
    warm = max(t.cached for t in ts)
    print(f"one cold call on the preloaded corpus ${cost_usd('sim-large', tok, S.OUT_TOKENS):.5f}, "
          f"one warm call ${cost_usd('sim-large', tok, S.OUT_TOKENS, warm):.5f}; latency cold "
          f"{latency_s('sim-large', tok, S.OUT_TOKENS) + S.CLIENT_S:.2f} s, warm "
          f"{latency_s('sim-large', tok, S.OUT_TOKENS, warm) + S.CLIENT_S:.2f} s")


if __name__ == "__main__":
    tee("3.13_snapshot", main)
