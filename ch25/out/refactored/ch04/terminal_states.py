"""Chapter 4 lab: premature termination in a multi-step lookup (stand-in agent).

Deterministic (seed 2026). A stand-in agent pages through a claim file and the
bulletin index, one tool call per turn, looking for the governing span. The
governing bulletin is always the 12th result (difficulty held fixed); what
varies is how much history the session already carries when the search starts.
Each turn, the stand-in may (a) find the span if it has reached it, (b) stop
and report "insufficient information" (gave-up), (c) answer from a look-alike
with low stated confidence (uncertain-wrong), or (d) answer from a look-alike
with high confidence (confident-wrong). The give-up and wrong hazards grow with
accumulated tokens past an onset, the stand-in's configured analogue of the
premature-termination finding in arXiv 2606.29718. Numbers are the stand-in's.
"""
import random
N, GOLD_AT, TOOL_TOK, BASE = 100, 12, 1_200, 1_100
ONSET = 12_000            # sim-mid's rot onset (Chapter 4 profiles)
def hazards(acc):
    x = max(0, acc - ONSET) / 1_000
    return min(0.12, 0.0016 * x), min(0.03, 0.0005 * x), min(0.03, 0.0004 * x)  # give-up, uncertain-wrong, confident-wrong
def terminal_state(status, cited, gold, confidence):
    """Classify how a lookup ended. Deterministic scorer: reads the final status,
    the cited span and the stated confidence; no model judges."""
    if status in ("insufficient_information", "turn_cap") or cited is None:
        return "gave-up"                      # a turn-cap hand-off counts here too
    if cited == gold:
        return "confident-right"
    return "confident-wrong" if confidence >= 0.5 else "uncertain-wrong"
OUTCOME = {"confident-right": ("answered", "GOLD", 0.9), "gave-up": ("insufficient_information", None, 0.0),
           "uncertain-wrong": ("answered", "LOOKALIKE", 0.3), "confident-wrong": ("answered", "LOOKALIKE", 0.8),
           "turn-cap": ("turn_cap", None, 0.0)}
def run_one(rng, history, cap=None, compact_at=None, compact_to=1_500):
    st, t, a = _run(rng, history, cap, compact_at, compact_to)
    status, cited, conf = OUTCOME[st]
    return terminal_state(status, cited, "GOLD", conf), t, a, st == "turn-cap"
def _run(rng, history, cap, compact_at, compact_to):
    acc, turn, kept = BASE + history, 0, BASE + history
    while True:
        turn += 1
        acc_now = kept + TOOL_TOK
        kept = acc_now
        if turn >= GOLD_AT and rng.random() < 0.97:
            return "confident-right", turn, acc_now
        g, u, c = hazards(acc_now)
        r = rng.random()
        if r < g: return "gave-up", turn, acc_now
        if r < g + u: return "uncertain-wrong", turn, acc_now
        if r < g + u + c: return "confident-wrong", turn, acc_now
        if compact_at and kept > compact_at:
            kept = BASE + compact_to + TOOL_TOK   # compaction checkpoint: history -> a 1,500-token note
        if cap and turn >= cap:
            return "turn-cap", turn, acc_now
STATES = ["confident-right", "confident-wrong", "uncertain-wrong", "gave-up"]
def sweep(history, **kw):
    rng = random.Random(2026 + history)
    res = [run_one(rng, history, **kw) for _ in range(N)]
    counts = {s: sum(r[0] == s for r in res) for s in STATES}
    gave = [r for r in res if r[0] == "gave-up"]
    mean_turn = sum(r[1] for r in gave) / len(gave) if gave else 0
    mean_acc = sum(r[2] for r in res) / N
    return counts, mean_turn, mean_acc, sum(r[3] for r in res)
if __name__ == "__main__":
    print("5. Terminal states by history carried into the search (sim-mid stand-in, 100 lookups, bulletin at call 12)")
    print(f"{'history':>8} {'mean tokens':>11}  " + "  ".join(f"{s:>15}" for s in STATES) + "  gave-up turn")
    for h in (0, 8_000, 16_000, 24_000, 32_000, 48_000, 64_000):
        c, t, a, _ = sweep(h)
        print(f"{h:>8,} {a:>11,.0f}  " + "  ".join(f"{c[s]:>15}" for s in STATES) + f"  {t:>12.1f}")
    print("\nfix at 32,000 history: turn cap 20 + compaction checkpoint at 12,000 tokens (history -> 1,500-token note)")
    hand = []
    for label, kw in (("no fix", {}), ("cap 20 + compaction", {"cap": 20, "compact_at": 12_000})):
        c, t, a, k = sweep(32_000, **kw); hand.append(k)
        print(f"  {label:<19} mean tokens {a:>6,.0f}  right {c['confident-right']}  confident-wrong {c['confident-wrong']:>2}"
              f"  uncertain-wrong {c['uncertain-wrong']}  gave-up {c['gave-up']}")
    print(f"  turn-cap hand-offs: {hand[0]} and {hand[1]}")
    # one trace: the first lookup at 16,000 history that gives up at turn 9
    rng = random.Random(2026 + 16_000)
    for i in range(N):
        s, t, a, _ = run_one(rng, 16_000)
        if s == "gave-up" and t == 9:
            print(f"\ntrace: lookup {i + 1} at 16,000 history -> {s} at turn {t}, {a:,} tokens; governing bulletin at call {GOLD_AT}")
            for k in range(1, t + 1):
                print(f"  turn {k:>2}  tokens {BASE + 16_000 + k * TOOL_TOK:>6,}  " + (f"open_bulletin_result(state='IL', topic='roof', rank={k - 6})" if k > 6 else f"read_claim_file(page={k})"))
            print(f"  final: status=insufficient_information  cited=None  (rank {GOLD_AT - 6} holds the governing bulletin)")
            break
