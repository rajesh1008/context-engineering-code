"""Listing 3.11: a clashing instruction with and without a stated hierarchy, on 15 roof-topic
probes where a state bulletin governs (the experiment Chapter 7 runs in full).

Mechanism: the packet holds the governing bulletin and the policy form's roof clause. With no
declared clash the stand-in follows rule 1 (precedence) and cites the bulletin. When a loaded
skill countermands rule 1, the later instruction wins when the probe's draw u < RECENCY (0.75)
without a hierarchy rule, and only when u >= HIERARCHY (0.96) with one. The draws are fixtures."""
from __future__ import annotations

from common import tee
import standin as S
from contextlab.instructions import V22, V30, V31, clashes

ROWS = [("v2.2 rules only, no clash", V22), ("v3.0 clashing skill after the rules", V30),
        ("v3.1 same clash, hierarchy rule stated", V31)]


def packet(d: S.Determination, ins) -> S.Rendered:
    form = S.CORPUS.by_id[f"{d.form}|{d.topic}"]
    pk = S.Packet(determination_id=d.determination_id, instructions=S.instructions_for(d, ins),
                  evidence=[S.ev_of(S.CORPUS.by_id[d.gold_span_id]), S.ev_of(form)], question=d.question)
    return S.Assembler().render(pk, S.claim_facts(d))


def cite(d: S.Determination, ins, u: float) -> str:
    r = packet(d, ins)
    skill_rules = [c for c in clashes(ins) if any(t == d.topic for t in c.split("(")[1].split(")")[0].split(", "))]
    later_wins = bool(skill_rules) and (u >= S.HIERARCHY if "hierarchy" in ins.rules else u < S.RECENCY)
    ids = [s.span_id for s in r.evidence]
    return ids[1] if later_wins else ids[0]


def main():
    probes, us = S.load_dets("probes"), S.CAL["probe_u"]
    print(f"{len(probes)} roof-topic probes where a state bulletin governs (sim-large)")
    for label, ins in ROWS:
        ok = sum(cite(d, ins, u) == d.gold_span_id for d, u in zip(probes, us))
        check = clashes(ins) or "none"
        print(f"  {label:<40}cites the bulletin {ok}/{len(probes)}  static check: {check}")


if __name__ == "__main__":
    tee("3.11_system_turn", main)
