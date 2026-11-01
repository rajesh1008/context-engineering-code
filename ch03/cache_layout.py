"""Listings 3.10 and 3.12: prompt-cache share and cost by layout, and three routes to the window
(retrieve, stuff the form family, stuff the whole corpus), 200 determinations in a row."""
from __future__ import annotations

from common import tee
import standin as S
from standin import Assembler, Pipeline, PipelineConfig, PromptCache, claim_facts, cost_usd

LAYOUTS = [("rank order: instructions, facts, evidence", "rank", False),
           ("cache order: instructions, documents, facts", "cache", False),
           ("cache order, timestamp in the instructions", "cache", True)]


def layout_test(dets, pipe, order, stamp):
    cache, tokens, cached = PromptCache(), [], []
    for i, d in enumerate(dets):
        pk = pipe.build_packet(d)[0]
        if stamp:
            ins = pk.instructions[0]
            pk = pk.model_copy(update={"instructions": [ins.model_copy(
                update={"text": f"Current time: 2026-10-31 09:{i // 60:02d}:{i % 60:02d}. " + ins.text})]})
        r = Assembler("markdown", order).render(pk, claim_facts(d))
        c = cache.lookup([(s.text, s.end - s.start) for s in r.segments])
        tokens.append(r.tokens)
        cached.append(c)
    cost = sum(cost_usd("sim-large", t, S.OUT_TOKENS, c) for t, c in zip(tokens, cached)) / len(dets)
    return round(sum(tokens) / len(dets)), sum(cached) / sum(tokens), cost


def layouts():
    dets = S.load_dets()
    for title, cfg in (("retrieved packets (filter on, k=10)", dict(policy_filter=True)),
                       ("stuffed packets (the policy's whole form family)", dict(stuff="family"))):
        print(f"Layout test, {title}, {len(dets)} calls in a row")
        pipe = Pipeline(PipelineConfig(**cfg))
        for label, order, stamp in LAYOUTS:
            tok, share, cost = layout_test(dets, pipe, order, stamp)
            print(f"  {label:<47}{tok:>6,} tokens  cached {share:>6.1%}  ${cost:.5f}/det")


def strategies():
    dets, corpus = S.load_dets(), S.CORPUS
    print("Strategy test: wrong-clause, tokens, cache share and cost per determination (sim-large)")
    for label, cfg in (("retrieve (filter, k=10)", dict(policy_filter=True)),
                       ("retrieve, cache order", dict(policy_filter=True, order="cache")),
                       ("stuff the policy's form family", dict(stuff="family")),
                       ("stuff the family, cache order", dict(stuff="family", order="cache")),
                       ("stuff the whole corpus, cache order", dict(stuff="all", order="cache"))):
        p = Pipeline(PipelineConfig(**cfg), corpus=corpus, cache=PromptCache())
        ts = p.run_many(dets)
        n = len(ts)
        print(f"  {label:<39}wrong {sum(t.wrong for t in ts) / n:>6.1%}"
              f"{round(sum(t.tokens_in for t in ts) / n):>9,} tokens"
              f"  cached {sum(t.cached for t in ts) / sum(t.tokens_in for t in ts):>6.1%}"
              f"  ${sum(t.cost for t in ts) / n:.5f}")


if __name__ == "__main__":
    tee("3.10_layout", layouts)
    tee("3.12_strategies", strategies)
