"""Chapter 48 lab: lint Lakeshore's configurations against the catalogue, then let
the world move under a configuration that passes.

Note: deterministic stand-in; see lakeshore48.py for every stated parameter.
Run from 04-code/: python3 ch48/lab.py > ch48/output.txt
"""
import copy
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import lakeshore48 as L                                              # noqa: E402
from contextlab.configlint import CATALOGUE, lint, report, resolve   # noqa: E402

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
results = {}


def section(t):
    print(f"\n== {t}")


section("1. The catalogue")
for rule, title in CATALOGUE.items():
    print(f"{rule}  {title}")

section("2. The audit: lint, month 0")
audit = {}
for cfg in L.CONFIGS:
    found = lint(cfg)
    audit[cfg["name"]] = [str(f) for f in found]
    print(f"{cfg['name']:<23} {report(found)}")
    for f in found:
        print("   ", f)
results["audit_lint"] = audit

section("3. Each anti-pattern injected into the clean determination configuration")
print(f"{'rule':<5} {'caught by lint':<15} {'metric':<7} {'per 1,000':>9} {'cost index':>10}")
rows = []
for rule in CATALOGUE:
    bad = L.with_antipattern(L.DETERMINATION, rule)
    caught = sorted({f.rule for f in lint(bad)})
    run = L.catalogue_run(rule)
    rows.append(dict(run, caught=caught))
    print(f"{rule:<5} {','.join(caught) or '-':<15} {run['metric']:<7} "
          f"{run['per_1000']:>9} {run['cost_index']:>10}")
results["catalogue"] = rows

section("4. The audit: resolve, month 0")
for cfg in L.CONFIGS:
    print(f"{cfg['name']:<23} {report(resolve(cfg, L.world(0)))}")

section("5. Failure injection: the configuration passes, the world moves")
print("month  lint errors  resolve errors  wrong/1,000  holds broken/1,000")
monthly = []
for m in range(6):
    w = L.world(m)
    lint_err = sum(f.severity == "error" for c in L.CONFIGS for f in lint(c))
    rf = [f for c in L.CONFIGS for f in resolve(c, w)]
    wrong, broken = L.month_outcomes(L.DETERMINATION, L.SUBRO, m)
    print(f"{m:>5}  {lint_err:>11}  {len(rf):>14}  {wrong:>11}  {broken:>18}")
    monthly.append({"month": m, "lint_errors": lint_err, "resolve": len(rf),
                    "wrong": wrong, "broken": broken})
results["rot_lint_only"] = monthly
print("resolve findings at month 5:")
for f in sorted({(f.rule, f.message) for c in L.CONFIGS for f in resolve(c, L.world(5))}):
    print("   ", *f)

section("6. The same months with resolve run on every world change")
cfgs = [copy.deepcopy(c) for c in L.CONFIGS]
gated = []
for m in range(6):
    found = [f for c in cfgs for f in resolve(c, L.world(m))]
    if found:
        cfgs = [L.fixed(c, m) for c in cfgs]
    after = [f for c in cfgs for f in resolve(c, L.world(m))]
    wrong, broken = L.month_outcomes(cfgs[0], cfgs[2], m)
    print(f"month {m}: {len(found)} finding(s) on the change, {len(after)} after the fix; "
          f"wrong {wrong}/1,000, holds broken {broken}/1,000")
    gated.append({"month": m, "found": len(found), "wrong": wrong, "broken": broken})
results["rot_resolved"] = gated

section("7. Re-audit after the owners' changes")
det2, assist2, sub2 = cfgs
assist2["layout"] = {"prefix": ["instructions"], "volatile": ["greeting", "today", "turn"]}
sub2["tools"] = [t for t in sub2["tools"] if t["name"] != "claim_diary"]
sub2["tools"][2]["description"] = ("Return the claim file's adjuster notes and diary "
                                   "entries; diary restrictions come back marked.")
final = {}
for cfg in (det2, assist2, sub2):
    found = lint(cfg) + resolve(cfg, L.world(5))
    final[cfg["name"]] = report(found)
    print(f"{cfg['name']:<23} {report(found)}")
results["reaudit"] = final

(OUT / "results.json").write_text(json.dumps(results, indent=1))
