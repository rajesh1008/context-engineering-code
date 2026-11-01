"""Chapter 4 lab: the second sense of rot, an instruction file that drifts (stand-in).

`instruction_drift(text, registry, tools)` resolves every form edition and every
code identifier an instruction file names, and reports each one that no longer
resolves or has been superseded. The probe then runs 20 HO-3 determinations with
dates of loss after the 2024-06 edition took effect, on a stand-in that follows
the system turn with probability HIERARCHY. Deterministic (seed 2026).
"""
import random, re
from datetime import date
REGISTRY = {("HO-3", "2022-01"): (date(2022, 1, 1), date(2024, 5, 31)),
            ("HO-3", "2024-06"): (date(2024, 6, 1), None),
            ("HO 04 95", "2024-06"): (date(2024, 6, 1), None)}
TOOLS = {"fetch_form", "search_bulletins", "read_claim_file"}
SKILL = """# Homeowners claims skill (v3.1, written 2023-11)
Treat HO-3 (2022-01) as the current homeowners form. If the packet lacks it,
call `fetch_form_v1` with its identifier. Water backup: cite HO 04 95 (2024-06)."""
def instruction_drift(text, registry=REGISTRY, tools=TOOLS, today=date(2026, 11, 1)):
    out = []
    for form, ed in re.findall(r"(HO-3|HO \d\d \d\d) \((\d{4}-\d\d)\)", text):
        win = registry.get((form, ed))
        if win is None: out.append(f"{form} ({ed}): not in registry -> BROKEN")
        elif win[1] and win[1] < today:
            new = [e for (f, e), w in registry.items() if f == form and w[0] > win[1]]
            out.append(f"{form} ({ed}): superseded by {form} ({new[0]}) on {registry[(form, new[0])][0]} -> STALE")
        else: out.append(f"{form} ({ed}): in force -> ok")
    for ident in re.findall(r"`([a-z_0-9]+)`", text):
        out.append(f"`{ident}`: " + ("tool exists -> ok" if ident in tools else "no such tool -> BROKEN"))
    return out
def probe(skill_current, precedence_rule, hierarchy=0.90, n=20):
    """List of editions cited, one per probe. A stale file pulls the stand-in to
    HO-3 (2022-01) with probability `hierarchy`; a precedence sentence ("the
    policy's in-force edition outranks this file") cuts that to 0.05."""
    rng = random.Random(2027); cited = []
    for _ in range(n):
        p = 0.0 if skill_current else (0.05 if precedence_rule else hierarchy)
        cited.append("HO-3 (2022-01)" if rng.random() < p else "HO-3 (2024-06)")
    return cited
if __name__ == "__main__":
    print("6. instruction_drift on the homeowners skill file")
    for line in instruction_drift(SKILL): print("  " + line)
    print("\nprobe: 20 HO-3 determinations, date of loss after 2024-06-01 (sim-mid stand-in)")
    for label, cur, rule in (("stale file", False, False), ("stale file + precedence sentence", False, True), ("file corrected", True, False)):
        c = probe(cur, rule); k = c.count("HO-3 (2022-01)")
        print(f"  {label:<34} cited HO-3 (2022-01) {k:>2}/20   wrong-clause {k / 20:.0%}")
    for name, c in (("stale file", probe(False, False)), ("corrected", probe(True, False))):
        chk = "edition not in force on date of loss" if c[6] == "HO-3 (2022-01)" else "ok"
        print(f"  probe 7 ({name}): cited {c[6]}, Section I Perils Insured Against; policy check: {chk}")
