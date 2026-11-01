# Chapter 51 lab: the last twenty percent

Deterministic stand-ins only: no model, no network, standard library only.
Every number here is illustrative and comes from `lakeshore51.py`.

- `lab.py`: runs the seven sections (targets, backlog, hardening checklist, runbooks,
  examination drill, a model two generations newer, targets after) and writes `out/results.json`.
- `lakeshore51.py`: the backlog, checklist, runbooks, drill sample, gold set and probe set.
- `test_ch51.py`: tests, including a failure injection (missing evidence fails a check closed).
- `conftest.py`: puts this folder and its parent on `sys.path`.
- `output.txt`: the output of `python3 lab.py`.
- Module: `../contextlab/finish.py` (contextlab v12.0).

Run: `python3 lab.py > output.txt` and `python3 -m pytest -q`.
