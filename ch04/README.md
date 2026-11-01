# Chapter 4 code: context rot

> **Stand-in note.** Every model here is a deterministic `SimulatedModel` (standin.py); the
> corpus (378 spans, 22,326 tokens) and the 200 determinations are synthetic. Each model
> reduces a determination to two margins (rival clause, look-alike) that length noise, a
> position weight and look-alike pull can overturn. `calibrate.py` fits the per-determination
> margins once so the stand-ins reproduce the chapter's illustrative profile; the result is
> committed as `calibration.json`. Prices (US$ per million tokens) and lookup minutes are
> illustrative lab constants, not any provider's figures.

| File | What it does |
| --- | --- |
| `standin.py` | corpus, determinations, padding (`neutral`, `lookalike`), Budgeter, `SimulatedModel`, `Pipeline` |
| `calibrate.py` | fits `calibration.json` (run once; about 15 s) |
| `lab.py` | Sections 4.2, 4.3, 4.5 and 4.7: writes `output.txt` and the CSVs behind Figures 4.1-4.5, 4.8, 4.9, 4.12 |
| `verify_ch04.py` | A: scoreboard model on 200 determinations, with and without the cap; B: Listing 4.1 |
| `monitor_ch04.py` | Listing 4.8, the two-point check |
| `terminal_states.py` | Section 4.4 stand-in agent and terminal states |
| `instruction_drift.py` | Section 4.6 drift check and probes |
| `../contextlab/diagnostics.py` | `rot_curve`, `knee` (Listings 4.5-4.7) and the clash/confusion/distraction/rot checks |
| `test_ch04.py` | `python -m pytest 04-code/ch04 -q` |
