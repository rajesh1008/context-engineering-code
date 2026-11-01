# Chapter 6 lab: instructions as a versioned artefact

    python3 lab.py            # a few seconds; writes output.txt (sections 1-4), versions.csv, skills.csv,
                              # injection.csv, gate.csv, instructions_v2.2.md, CHANGELOG-instructions.md
    python3 explore.py        # the chapter's exercises; writes out_AB.txt ... out_N.txt
    python3 -m pytest -q      # the chapter's numbers, the fingerprints and the release gate
    python3 calibrate.py      # optional, about ten seconds: rebuilds calibration.json (reads ../ch08/calibration.json)

Needs Python 3.11+. Run from this folder; the scripts put `04-code` on the path for `contextlab`.

- `../contextlab/instructions.py`: `Instructions`, `Skill`, `Example`, render, spans, fingerprint and
  diff, and the versions v1.0 to v2.3b (Listings 6.20 to 6.22). Shared with Chapters 7 and 8.
- `standin.py`: the claims corpus (200 evaluation and 300 hold-out determinations), the documents that
  govern them (bulletin in force, then endorsement on the topic, then form edition), `lint`, the
  packet, `SimulatedModel` and the metrics (wrong-clause rate, faithfulness, tokens, cost, lookup proxy).
- `lab.py`: sections 1-4 of the chapter's output and the release gate (block if the evaluation rate
  rises by more than 1 point or the hard set by more than 2).
- `explore.py`: parts A to N of the exercises.
- `calibrate.py`: builds `calibration.json`, the per-claim fixtures.
- `AGENTS.md`: the repository instructions file of Section 6.3.

**Stand-in notice.** No language model runs here. The mechanism is real and inspectable: the
documents decide the gold answer, the instruction set is rendered and fingerprinted, skills load by
claim topic, `lint` checks each example against the documents, and the simulated reader follows the
precedence rule when it is present. What each simulated model gets wrong on its own (retrieval misses,
which claims an example pulls, which answers carry an unsupported addition) is a calibrated fixture,
illustrative rather than measured, and says nothing about how any hosted model would score. The
imitation strength (0.45) and quote-only retention (0.25) are stand-in parameters. Token prices
($3/$15, $1/$5 and $0.25/$1.25 per million input/output tokens) and the lookup-time proxy (1.3 minutes,
plus 2.2 when an answer does not state the version it checked) are illustrative constants. Prompt
tokens are counted with the labs' simple tokenizer (words and punctuation); the retrieved passages'
length is a per-claim fixture, and the smaller models' chat template adds one token.

**How the fixtures were built.** `calibrate.py` starts from the shared determinations of Chapter 8,
re-assigns a handful of claims so the chapter's named cases exist (for example DET-0042 is an auto
claim on PAP 2018-09, DET-0236 and DET-0288 carry the facts of worked Examples A and B, and five
evaluation claims are moved to the dog-bite, wear-and-tear and snow-load topics), and builds the
example pool for Section 6.4 by cloning evaluation facts, with shifted loss dates, onto unnamed
hold-out sources (seeded search, `random.Random(606)`). The roof, deer and hard slices are drawn
from the hold-out set.
