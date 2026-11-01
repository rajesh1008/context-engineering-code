"""Chapter 8 gate: a Rulebook change ships only if the review against the documents
finds no conflicts and no model gets worse on the evaluation set or the probes.

    python3 gate.py      # PASS for r0.9 -> r1.0, BLOCK for r1.0 -> r1.1; writes out/gate.txt
"""
from __future__ import annotations

from lab import HERE, MODELS, R09, R10, R11_BOOK, evaluate
from standin import CORPUS
from contextlab.rulebook import Rulebook, check

SHOW = 3   # conflicts printed; the count line gives them all


def gate(old: Rulebook, new: Rulebook) -> tuple[bool, list[str]]:
    out = [f"{old.version} ({old.fingerprint()}) -> {new.version} ({new.fingerprint()})"]
    for a, b in zip(old.rules, new.rules):
        if a != b:
            out.append(f"  changed {b.rule_id}: table rows {len(a.rows)} -> {len(b.rows)}")
    conflicts = [c for rc in check(new, CORPUS) for c in rc.conflicts]
    out.append(f"  review: conflicts with the documents {len(conflicts)}")
    worse = False
    for m in MODELS:
        e0, e1 = evaluate(m, old), evaluate(m, new)
        worse |= e1["wrong"] > e0["wrong"] or e1["probes_ok"] < e0["probes_ok"]
        out.append(f"  {m:<10} eval {100 * e0['wrong'] / e0['n']:>4.1f}% -> {100 * e1['wrong'] / e1['n']:>4.1f}%  "
                   f"probes {e1['probes_ok']}/{e1['probes']}")
    ok = not conflicts and not worse
    out.append(f"  gate: {'PASS' if ok else 'BLOCK'}")
    if conflicts:
        out.append("")
        for c in conflicts[:SHOW]:
            d = c.det
            out.append(f"  {d.determination_id} ({d.state}, {d.loss}): rule says {c.rule_says}, "
                       f"documents say {c.documents_say}")
    return ok, out


def main() -> None:
    lines: list[str] = []
    for old, new in ((R09, R10), (R10, R11_BOOK)):
        _, out = gate(old, new)
        lines += out + [""]
    text = "\n".join(lines).rstrip() + "\n"
    (HERE / "out" / "gate.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
