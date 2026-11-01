"""Listings 3.1-3.3: token counts three ways, one packet in four formats, cost by packet size.

tiktoken is optional: without it the cl100k_base and o200k_base columns are skipped."""
from __future__ import annotations

from common import tee
import standin as S
from standin import count_tokens
from contextlab.tokens import WINDOWS, cost_usd

try:
    import tiktoken
    ENCS = {n: tiktoken.get_encoding(n) for n in ("cl100k_base", "o200k_base")}
except ImportError:  # pragma: no cover
    ENCS = {}

def listing_3_1():
    print(f"{'text':<20}{'contextlab':>11}" + "".join(f"{n:>13}" for n in ENCS))
    SAMPLES = {
        "endorsement title": "Water Back-Up and Sump Discharge or Overflow",
        "policy number": "LMH-WI-0048213",
        "form label": "HO-3 (2024-06)",
        "dollar limit": "$10,000 limit, $1,000 deductible",
        "date": "2026-03-14",
    }
    for name, text in SAMPLES.items():
        print(f"{name:<20}{count_tokens(text):>11}" + "".join(f"{len(e.encode(text)):>13}" for e in ENCS.values()))
    texts = [s.text for s in S.CORPUS.spans]
    print(f"{'whole corpus':<20}{sum(map(count_tokens, texts)):>11,}"
          + "".join(f"{sum(len(e.encode(t)) for t in texts):>13,}" for e in ENCS.values()))


def listing_3_2():
    d = S.load_dets()[0]
    pk = S.Pipeline(S.PipelineConfig(policy_filter=True)).build_packet(d)[0]
    print(f"{d.determination_id} ({d.holder}) fixed packet, {len(pk.evidence)} evidence spans, by format:")
    for fmt in ("plain", "markdown", "xml", "json"):
        r = S.Assembler(fmt).render(pk, S.claim_facts(d))
        real = "".join(f"  {n} {len(e.encode(r.text)):,}" for n, e in ENCS.items())
        print(f"  {fmt:<9}{r.tokens:>6,} tokens (contextlab){real}")


def listing_3_3():
    ts = S.Pipeline(S.PipelineConfig(policy_filter=True)).run_many(S.load_dets())
    mean_in = round(sum(t.tokens_in for t in ts) / len(ts))
    out = round(sum(t.tokens_out for t in ts) / len(ts))
    print(f"fixed pipeline on {len(ts)}: mean {mean_in:,} tokens in, {out} out")
    print(f"Cost of one determination ({out} output tokens), US dollars, illustrative prices:")
    models = ("sim-large", "sim-mid", "sim-small")
    print(f"{'packet tokens':>14}" + "".join(f"{m:>11}" for m in models))
    for n in (mean_in, 15_000, 60_000, 180_000):
        cells = [f"{cost_usd(m, n, out):>11.5f}" if n <= WINDOWS[m] else f"{'n/a':>11}" for m in models]
        print(f"{n:>14,}" + "".join(cells))
    day = lambda n: 3_200 * cost_usd("sim-large", n, out)  # noqa: E731
    print(f"per day at 3,200 determinations, sim-large: {mean_in:,} tokens ${day(mean_in):,.2f}; "
          f"60,000 tokens ${day(60_000):,.2f}")


if __name__ == "__main__":
    tee("3.1_tokens", listing_3_1)
    tee("3.2_formats", listing_3_2)
    tee("3.3_cost", listing_3_3)
