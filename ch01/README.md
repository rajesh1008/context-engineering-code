# Chapter 1 lab: the same question, four assemblies

    python3 lab.py            # about three seconds; writes output.txt and assemblies.csv
    python3 -m pytest -q      # the chapter's numbers, the Carter block and the fail-closed policy
    python3 calibrate.py      # optional, about 80 s: rebuilds calibration.json

Needs Python 3.11+ and pydantic 2 (numpy only for calibrate.py).

- `standin.py`: Lakeshore's synthetic corpus (378 spans, 22,326 tokens), the applicability
  policy (Listing 1.2), the assembler and `SimulatedModel`, a deterministic stand-in reader.
- `lab.py`: assemblies A-E, the failure injection (Listing 1.3), the required-endorsement check
  (Listing 1.4) and the tables in Section 1.8.
- `calibrate.py`: builds the 500 determinations and their fixtures (per-determination match
  strengths, keyword and retrieval adjustments, and the five misread questions) and fits them so
  the lab reproduces the chapter's profile.

The stand-in's mechanism is real and inspectable: every answer is computed from the packet the
assembly built, so changing the packet (an empty header, no rules, a smaller budget) changes the
result. Its behaviour profile is a calibrated fixture, illustrative rather than measured, and says
nothing about how any hosted model would score.
