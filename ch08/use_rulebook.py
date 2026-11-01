"""Chapter 8, Listing 8.13: using the Rulebook in the pipeline.

    python3 use_rulebook.py      # writes out/use_rulebook.txt
"""
from __future__ import annotations

import contextlib
import io

from lab import HERE, R10
from standin import CORPUS as corpus, Pipeline, PipelineConfig
from contextlab.instructions import V22
from contextlab.rulebook import check

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    book = R10                                                 # r1.0, twelve rules
    print(book.version, book.fingerprint(), len(book.rules), "rules,", len(book.topics()), "topics")
    conflicts = [c for c in check(book, corpus) if c.conflicts]
    print("rules in conflict with the documents:", len(conflicts))
    pipe = Pipeline(PipelineConfig(policy_filter=True, model="sim-large", instructions=V22,
                                   rulebook=book), corpus=corpus)

    det = next(d for d in corpus.determinations if d.determination_id == "DET-0030")
    t = pipe.run(det)
    print(det.determination_id, det.topic, "->", t.answer.citation)
    print("rulebook span in packet:",
          [s.span.span_id for s in t.rendered.segments if s.span and s.span.span_id.startswith("RULEBOOK")])
    print("trace fingerprints:", t.config.instructions.fingerprint(), t.config.rulebook.fingerprint())

(HERE / "out" / "use_rulebook.txt").write_text(buf.getvalue())
print(buf.getvalue(), end="")
