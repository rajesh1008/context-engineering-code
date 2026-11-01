# Chapter 46 lab: the context layer as a platform

Run from `04-code`:

    python3 ch46/lab.py > ch46/output.txt   # writes ch46/out/results.json
    python3 -m pytest -q ch46               # 12 tests

Library: `contextlab/platform.py` (contextlab v10.3). Scenario data: `ch46/lakeshore46.py` (seed 46).

Sections of the output map to the chapter:

- 46.1 fork audit: the agency team's copy of the index against the platform.
- 46.2 assemble / explain / feedback on one agency request.
- 46.3 fair shares versus FIFO under the Monday renewal batch; a month of quotas (soft and hard); chargeback by usage share and even split.
- 46.4 600 entitlement probes against three designs; failure injection with revocations and policy moves (caller cache, gateway verification, platform cache keyed by entitlement class).
- 46.5 private producer-guide notes against governing forms.
- 46.6 consumer contract tests against contract v1 and v2; migration by week.
- 46.7 platform SLOs over 28 simulated days.

Everything is a deterministic stand-in: invented corpus, lookup retrieval, a hash token rather than a signed credential, a simulated queue and illustrative prices. Compare designs, not absolute numbers.
