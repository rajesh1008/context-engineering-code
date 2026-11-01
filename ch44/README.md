# Chapter 44 code: Governing Knowledge, Embeddings and Memory

Deterministic stand-ins; no language model runs. Rerun and you get the same numbers.

- `lakeshore44.py`: the Michigan forms (the 2011 national edition, the Michigan amendatory endorsement retired for terms from 1 September 2026, the new Michigan edition), a year of Michigan loss-payment determinations, a stand-in catalogue with five derived stores and eight injected defects, the asset register in two versions (v0.9 with gaps, v1.0), the holdings for erasure request ER-2026-0131 and the Northwoods Savings exit, the quarter's index changes and the controls.
- `lab.py`: nine numbered sections; writes `output.txt` (via stdout) and `out/results.json`.
- `test_ch44.py`: twelve tests.
- Library: `../contextlab/governance.py` (contextlab v9.6): `Register` (gaps, downstream), `Doc`, `TermAwarePolicy` (Chapter 15's Policy plus term-aware retirement), `reconcile`, `classify` and `impact` for index changes, `fan_out`, `exit_drill`, `coverage`, `calendar`.

Run from this folder: `python3 lab.py > output.txt && python3 -m pytest -q`
