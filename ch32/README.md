# Chapter 32 code: Budgeting and Selection

Deterministic stand-ins; no language model runs. Rerun and you get the same numbers (packing times vary by machine).

- `lakeshore32.py`: the full mix of 1,000 determinations with about 30 candidate spans each (relevance, authority tier, effective date, dependencies, redundancy groups), the labelled set (the first 200), the reader that scores a packet (wrong clause, faithfulness, missing required spans), and this week's bulletin BUL-WI-WB-202610 with claims WK-001 to WK-040.
- `lab.py`: seven numbered sections: four budgeters and the Budgeter on the full mix; a budget sweep; the Budgeter's ablation; tokens per determination for the five-call loop with and without per-step budgets; the prefix-cache simulation of six layouts over one day (3,200 determinations); the failure injection (usage scorer, newest bulletin); packing times. Writes `output.txt` (via stdout) and `out/results.json`.
- `example.py`: Listing 32.4, packing DET-0001 and checking the packed packet.
- `test_ch32.py`: thirteen tests, including brute force against the knapsack on 40 random pools, the newest-bulletin test and the numbers the chapter quotes.
- Library: `../contextlab/budgeter.py` (contextlab v7.2 and v7.3): `Candidate`, `score`, `score_usage`, `Calibrator`, `top_k`, `fill_by_score`, `greedy_density`, `knapsack`, `Budgeter` (pins, document slots with dependency bundles, redundancy groups, floor, target, hard cap, trace), `StepBudget`, `CachePrices`, `PrefixCache`.
- Figures: `../../03-figures/src/ch32/make_figs.py` reads `out/results.json` and the fixtures.

The cache model is a simulator of the rules the providers publish (exact prefix, breakpoints, write 1.25x, read 0.1x, five-minute lifetime, 1,024-token minimum), assuming the builder renders the intended layout byte for byte. Chapter 34 audits the real builder.

Run from this folder: `python3 lab.py > output.txt && python3 -m pytest -q`
