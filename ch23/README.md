# Chapter 23 code: Memory Lifecycle and State

Deterministic stand-ins; no language model runs. Rerun and you get the same numbers.

- `session23.py`: Tom Brennan's two-hour afternoon on the workbench (Thursday 22 October 2026, from 13:00) and four ways to hold it: full history, truncation, summarisation (compaction), structured state (pinned constraints, per-claim state with sources, folded claims, tool-result clearing with re-fetch by id). Constants at the top are the stand-in's own; their shapes follow the research the chapter cites.
- `lifecycle23.py`: Chapter 22's full-mix replay (`../ch22/lakeshore22.py`, `../ch22/replay22.py`, imported unchanged) with the lifecycle policy applied to the store it starts from: re-validation of earlier-claim records against the policy record, a read budget of six, retention by scope.
- `lab.py`: eleven numbered sections; writes `output.txt` (via stdout) and `out/results.json`.
- `test_ch23.py`: eight tests, including the numbers the chapter quotes.
- Library: `../contextlab/statestore.py` (contextlab v5.3): `StateStore`, checkpoint by reference, `LifecyclePolicy`, `sweep`, `expire`, `revalidate`, `may_read`, `erase` with a tombstone.

Run from this folder: `python3 lab.py > output.txt && python3 -m pytest -q`
