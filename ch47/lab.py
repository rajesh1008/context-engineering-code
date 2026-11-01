"""Chapter 47 lab: three handoff strategies, sub-agent isolation, review, loops."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import lakeshore47 as L  # noqa: E402


def main():
    claims = L.world()
    res = dict(n=len(claims), gov_end=sum(c["gov_end"] for c in claims),
               bulletin=sum(c["bulletin"] for c in claims),
               strategies=L.run_strategies(claims), sweep=L.budget_sweep(claims),
               rot=L.rot_curve(), research=L.research(claims), review=L.review(claims),
               board=L.blackboard(claims),
               provenance={s: L.provenance(claims, s) for s in ("share", "summary", "record")},
               tokens=L.tokens_per_agent(claims), fan_out=L.fan_out(),
               loop=L.bulletin_loop())
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"claims {res['n']}  governing endorsement {res['gov_end']}  bulletin {res['bulletin']}")
    for k, v in res["strategies"].items():
        print(f"{k:17s} wrong/1000 {v["per_1000"]:5} after review {v["after_review"]:3}  coverage tok {v['coverage_tokens']:6}"
              f"  handoff tok {v['handoff_tokens']:5}  endorsement survival {v['endorsement_survival']}"
              f"  probe flagged {v['probe_flagged']}")
    for k, v in res["research"].items():
        print(f"research {k:15s} wrong {v['bulletin_wrong']:3}/{v['n']} ({v['wrong_rate']})"
              f"  window {v['coverage_window_mean']}  tok {v['research_tokens_per_claim']}"
              f"  US${v['research_cost_per_claim']}")
    for k, v in res["review"].items():
        print(f"review {k:17s} caught {v['caught']}/{v['of']} {v['by_kind']}  FA/1000 {v['false_alarms_per_1000']}")
    print("board", res["board"])
    print("provenance", res["provenance"])
    print("tokens", res["tokens"])
    for k, v in res["fan_out"].items():
        print(f"fan-out {k:35s} {v}")
    print("loop", res["loop"])


if __name__ == "__main__":
    main()
