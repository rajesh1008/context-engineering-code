# Chapter 8 lab: the Rulebook

    python3 lab.py            # under a second; writes output.txt and review.csv, injection.csv, adherence.csv, probes.csv
    python3 explore.py        # parts A, B, C, D, G, H; writes out/explore.txt
    python3 gate.py           # PASS for r0.9 -> r1.0, BLOCK for r1.0 -> r1.1; writes out/gate.txt
    python3 use_rulebook.py   # Listing 8.13; writes out/use_rulebook.txt
    python3 -m pytest -q      # the chapter's numbers and the gate
    python3 calibrate.py      # optional, about a second: rebuilds calibration.json

Needs Python 3.11+ and PyYAML. Run from this folder; the scripts put `04-code` on the path for `contextlab`.

- `../contextlab/rulebook.py`: `Row`, `Rule` (Listings 8.2 and 8.3), `Rulebook` (render, progressive
  span, fingerprint), `check` (review against the documents) and `probes`.
- `../contextlab/instructions.py`: the v2.2 instructions the pipeline carries (Chapter 6).
- `R11.yaml`: the rule-capture record of Listing 8.1, loaded as `R11_FIXED`.
- `lab.py`: `RULES` (r0.9), `R11_FIXED`, r1.0 and r1.1, and sections 1-5 of the chapter's output.
- `standin.py`: the 500 reviewed determinations, the documents that govern them (bulletin in force,
  then endorsement on the policy, then form edition), the packet and `SimulatedModel`.
- `calibrate.py`: builds the determinations (the claims the chapter names are pinned) and the
  model fixtures.

**Stand-in notice.** No language model runs here. The mechanism is real and inspectable: the
documents decide the gold answer, rule tables are resolved and checked against them, the rule span
enters the packet only for its topic, and the reader follows a rule whose row fires, follows a
rule with no rows (it states its default for every claim), and otherwise reads the documents. What
each simulated model gets wrong on its own (which claims it misses, which ones an example pulls,
which other clauses it cites) is a calibrated fixture, illustrative rather than measured, and says
nothing about how any hosted model would score. Prompt tokens are counted with the labs' simple
tokenizer (words and punctuation); the retrieved passages' length is a per-claim fixture, and the
smaller models' chat template adds one token.

The instruction fingerprint in `use_rulebook.py` is that of `contextlab/instructions.py` V22,
shared with Chapters 6 and 7.
