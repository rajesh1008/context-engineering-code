# Chapter 7: How Instructions Break (code)

Code for Chapter 7 of *The Context Engineering Handbook for LLMs and AI Agents*.

    python3 lab.py            # writes output.txt sections 1-4 (diff, clash check, before/after, incident)
    python3 explore.py        # appends parts A-L to output.txt (seeds, leak, history, hot edit, gates, resolve)
    python3 -m pytest -q      # the chapter's numbers, fingerprints, gates; seed 2026 reproduces Chapter 6
    python3 calibrate.py      # optional: rebuilds calibration.json from ../ch06/calibration.json

Needs Python 3.11+. Run from this folder; the scripts put `04-code` on the path for `contextlab`.
`calibrate.py` and one test also read `../ch06/calibration.json` and `../ch06/standin.py`
(both are in this zip).

**Stand-in notice.** No language model runs here. The three models (`sim-large`, `sim-mid`,
`sim-small`) are deterministic stand-ins (`standin.py`). The mechanism is real and
inspectable: the governing document is computed from the documents, packets are assembled
from the instruction spans and counted, fingerprints are the sha256 of the rendered
instructions (first 12 hex characters, version line excluded), the clash check reads the
skills' `negates` annotations, and lint resolves each example against the documents.

How a model settles a clash is a modelled behaviour, set by two parameters chosen to resemble
the published conflict studies, not measured: `ModelProfile.recency` (0.75, the later
instruction wins without a hierarchy rule) and `ModelProfile.hierarchy` (0.96, the higher tier
wins with one).

**Fixtures (calibration.json, built by calibrate.py), disclosed.** Which claims each model gets
wrong when the precedence rule is dropped or a form-leaning skill tips it (`noprec`,
`skill_pull`); the per-seed clash draws; unsupported additions per seed; the gold span's rank in
the retriever's list (hit@5); passage lengths (hold-out fitted to a mean of 1,522 prompt tokens
under v2.2; DET-0150's passages 8 tokens longer than in Chapter 6 so v3.2's evaluation mean
rounds to 1,550); and a handful of world adjustments to hold-out claims listed at the top of
`calibrate.py`. These are illustrative, chosen so the lab reproduces the chapter; they are not
measurements of any production model.

Files: `standin.py` (world, models, pipeline), `common.py` (helpers `MODELS`, `corpus`, `pipe`,
`adh`), `lab.py`, `explore.py` (Listings 7.11, 7.14 and 7.15 verbatim), `calibrate.py`,
`test_ch07.py`, `output.txt`. Shared: `../contextlab/instructions.py` (versions v1.0-v3.2,
`clashes`, `lint`, `adherence_probes`).
