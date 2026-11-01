"""Chapter 2 smoke tests (Listing 2.9). Writes nothing. `--stale` applies the stale form pointer."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from contextlab import scoreboard                              # noqa: E402
from contextlab.models import Kind, Provenance, Source, Span      # noqa: E402
from contextlab.pipeline import Pipeline                          # noqa: E402
from standin import eval_csv, load_corpus                        # noqa: E402

SOURCES = [
    Source(name="policy forms", kind=Kind.FORM, owner="product", system="docstore"),
    Source(name="endorsements", kind=Kind.ENDORSEMENT, owner="product", system="docstore"),
    Source(name="state bulletins", kind=Kind.BULLETIN, owner="compliance", system="regulatory-feed",
           refresh="daily"),
    Source(name="NAIC guidance", kind=Kind.GUIDANCE, owner="compliance", system="naic"),
    Source(name="underwriting notes", kind=Kind.UW_NOTE, owner="underwriting", system="sharepoint",
           entitlement=frozenset({"underwriting"})),
]
FROZEN = HERE / "eval200.csv"
WINDOW = 200_000


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<37} {detail}")
    return ok


def main(stale: bool = False) -> int:
    corpus = load_corpus()
    pipe = Pipeline(corpus=corpus)
    by_kind = {s.kind: s for s in SOURCES}
    dets = corpus.eval_set(200)
    if stale:
        from lab import stale_view
        dets = [stale_view(d) for d in dets]
    results: list[bool] = []

    rejected = [s.span_id for s in corpus.spans if not by_kind[s.kind].admit(s)[0]]
    results.append(check("every span admitted by a source", not rejected,
                         f"{len(corpus.spans) - len(rejected)}/{len(corpus.spans)} admitted"))

    bad = Span(span_id="HO-3#scan-0001", kind=Kind.FORM, text="Section I Conditions ...",
               provenance=Provenance(doc_id="HO-3", version=None))
    ok, why = by_kind[Kind.FORM].admit(bad)
    results.append(check("unversioned span fails closed", not ok, why))

    drift = [d.determination_id for d in dets
             if corpus.policies[d.policy_number]["form"] != d.form]
    results.append(check("claim form agrees with policy record", not drift,
                         f"{len(dets) - len(drift)}/{len(dets)} agree"
                         + (f"; first mismatch {drift[0]}" if drift else "")))

    frozen = FROZEN.read_text()
    now = eval_csv(dets)
    results.append(check("eval set matches frozen file", sha(now) == sha(frozen),
                         f"sha256 {sha(frozen)}, {len(dets)} rows"))

    a, b = pipe.run(dets[0]), pipe.run(dets[0])
    results.append(check("same input, same answer", a.answer.text == b.answer.text
                         and a.rendered.tokens == b.rendered.tokens,
                         f"{dets[0].determination_id}: {a.rendered.tokens:,} tokens twice"))

    closed = pipe.with_config(k=0).run(dets[0])
    results.append(check("no evidence, no answer", closed.answer.abstained,
                         f"k=0 packet of {closed.rendered.tokens} tokens abstains"))

    traces = [pipe.run(d) for d in dets]
    largest = max(t.rendered.tokens for t in traces)
    results.append(check("every packet far below the window", largest < WINDOW // 10,
                         f"largest {largest:,} tokens; window {WINDOW:,}"))

    ids = {s.span_id for s in corpus.spans}
    resolve = sum(t.answer.cited_span_id in ids for t in traces)
    results.append(check("every citation resolves to a span", resolve == len(traces),
                         f"{resolve}/{len(traces)} resolve"))

    exact = sum(scoreboard.exact(t) for t in traces)
    results.append(check("every citation names version, page", exact == len(traces),
                         f"{exact}/{len(traces)} exact"))

    full = sum(bool(t.retrieved and t.packet.spans and t.rendered.text and t.answer.text) for t in traces)
    results.append(check("every run leaves a trace", full == len(traces),
                         f"retrieved, packet, rendered, answer on all {full}"))

    print()
    print(f"{sum(results)}/{len(results)} smoke tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale", action="store_true", help="apply the stale form pointer to the claim facts")
    raise SystemExit(main(ap.parse_args().stale))
