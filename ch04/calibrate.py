"""Fit the stand-ins' per-determination margins to the chapter's illustrative profile.

The profile (TARGETS below) is the set of counts the chapter reports. For every model
and determination the script computes the two pressures SimulatedModel.scores()
returns under each lab condition, then searches (simulated annealing, fixed seed)
for one rival margin and one look-alike margin per determination such that the
counts come out exactly. Writes calibration.json. Run once; lab.py only reads it.
"""
import json, math, random
import standin as S

T13 = [0] + S.NEUTRAL_TARGETS
NEUTRAL = {  # wrong of 100 at each neutral target (overflow included)
    "sim-large": [4, 4, 4, 4, 4, 4, 4, 6, 13, 27, 63, 86, 96],
    "sim-mid": [6, 6, 6, 6, 8, 9, 13, 20, 57, 78, 96, 100, 100],
    "sim-small": [6, 6, 6, 8, 8, 11, 37, 100, 100, 100, 100, 100, 100]}
LOOK = {"sim-large": [5, 5, 5], "sim-mid": [7, 8, 23], "sim-small": [12, 28, 54]}
AT80 = {"sim-large": (46, 68), "sim-mid": (92, 96), "sim-small": (100, 100)}
CAP = {"sim-large": (4, 4), "sim-mid": (8, 7), "sim-small": (6, 6)}   # (injected + cap, clean + cap)
T8 = [0, 4_000, 8_000, 12_000, 16_000, 24_000, 32_000, 48_000]
TASK = {("T", "form clause"): [5, 5, 5, 5, 5, 8, 17, 85], ("T", "endorsement or bulletin"): [10, 10, 10, 14, 17, 21, 22, 29],
        ("F", "HO-3"): [13, 13, 13, 15, 18, 21, 25, 64], ("F", "PAP"): [0, 0, 0, 2, 2, 4, 6, 28],
        ("F", "BOAT"): [2, 2, 2, 2, 2, 2, 4, 13], ("F", "BOP"): [0, 0, 0, 0, 0, 2, 4, 9]}

CONDS = [("N", t) for t in T13] + [("N", 80_000), ("L", 4_000), ("L", 8_000), ("L", 16_000), ("L", 80_000), ("CC", 0), ("IC", 0)]


def pressures(model: S.SimulatedModel, d):
    out = {}
    for kind, t in CONDS:
        pad = (lambda x, t=t: S.neutral(x, t)) if kind == "N" else (lambda x, t=t: S.lookalike(x, t)) if kind == "L" \
            else (lambda x: S.lookalike(x, 16_000)) if kind == "IC" else None
        p = S.Pipeline(model=model, pad_fn=pad, budget=3_000 if kind in ("CC", "IC") else None)
        r = p.packet(d)
        if r.tokens > model.window:
            out[(kind, t)] = None
        else:
            u, v, _, has = model.scores(d, r)
            out[(kind, t)] = (u, v if has else None)
    return out


def outcome(p, mr, ml):
    if p is None:
        return "over"
    u, v = p
    if v is not None and v >= ml:
        return "la"
    return "rival" if u >= mr else "right"


def constraints(name, dets):
    """(condition, det indices, predicate, target) rows the counts must hit."""
    k100 = list(range(min(100, len(dets))))
    wrong = lambda o: o != "right"
    rows = [(("N", t), k100, wrong, NEUTRAL[name][j]) for j, t in enumerate(T13)]
    rows += [(("L", t), k100, wrong, LOOK[name][j]) for j, t in enumerate((4_000, 8_000, 16_000))]
    rows += [(("N", 80_000), k100, wrong, AT80[name][0]), (("L", 80_000), k100, wrong, AT80[name][1]),
             (("IC", 0), k100, wrong, CAP[name][0]), (("CC", 0), k100, wrong, CAP[name][1])]
    if name == "sim-mid":
        rows.append((("L", 16_000), k100, lambda o: o == "la", 14))
        for (axis, key), tgt in TASK.items():
            idx = [i for i, d in enumerate(dets) if (d.task_type if axis == "T" else d.family) == key]
            rows += [(("N", t), idx, wrong, tgt[j]) for j, t in enumerate(T8)]
        # DET-0004 is the first determination the feed misleads and the cap rescues
        rows.append((("D4",), [3], lambda o: o, 1))
        rows.append((("D4",), [0, 1, 2], lambda o: o, 0))
    if name == "sim-large" and len(dets) == 200:
        rows += [(("N", 0), list(range(200)), wrong, 12), (("CC", 0), list(range(200)), wrong, 12)]
    return rows


def fit(name, dets, seed=7, steps=400_000):
    m = S.SimulatedModel(name)
    P = [pressures(m, d) for d in dets]
    n = len(dets)
    cand_r = [[-math.inf, math.inf] + sorted({p[0] for p in pr.values() if p and p[0] > 0}) for pr in P]
    cand_l = [[math.inf] + sorted({p[1] for p in pr.values() if p and p[1] is not None}) for pr in P]
    rows = constraints(name, dets)
    by_det = [[] for _ in range(n)]
    for r, (c, idx, pred, tgt) in enumerate(rows):
        for i in idx:
            by_det[i].append(r)

    def val(r, i, mr, ml):
        c, _, pred, _ = rows[r]
        if c == ("D4",):
            return int(outcome(P[i][("L", 16_000)], mr, ml) == "la" and outcome(P[i][("IC", 0)], mr, ml) == "right")
        return int(bool(pred(outcome(P[i][c], mr, ml))))

    rng = random.Random(seed)
    mr, ml = [math.inf] * n, [math.inf] * n
    cnt = [sum(val(r, i, mr[i], ml[i]) for i in rows[r][1]) for r in range(len(rows))]
    cost = lambda: sum(abs(cnt[r] - rows[r][3]) * (3 if rows[r][0] == ("D4",) else 1) for r in range(len(rows)))
    cur, temp = cost(), 1.5
    for step in range(steps):
        i = rng.randrange(n)
        nr, nl = mr[i], ml[i]
        if rng.random() < 0.55:
            nr = rng.choice(cand_r[i])
        else:
            nl = rng.choice(cand_l[i])
        if nr == -math.inf:
            nl = math.inf                        # wrong at every length: it cites the rival
        delta = 0
        changes = []
        for r in by_det[i]:
            dv = val(r, i, nr, nl) - val(r, i, mr[i], ml[i])
            if dv:
                w = 3 if rows[r][0] == ("D4",) else 1
                delta += w * (abs(cnt[r] + dv - rows[r][3]) - abs(cnt[r] - rows[r][3]))
                changes.append((r, dv))
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            mr[i], ml[i] = nr, nl
            for r, dv in changes:
                cnt[r] += dv
            cur += delta
        temp = max(0.02, temp * 0.99997)
        if cur == 0:
            break
    out = {}
    for i, d in enumerate(dets):
        def mid(x, cands):
            if x in (math.inf, -math.inf):
                return x
            lower = [c for c in cands if c != -math.inf and c < x]
            return (x + (lower[-1] if lower else 0.0)) / 2
        out[d.determination_id] = {"rot": mid(mr[i], cand_r[i]), "la": mid(ml[i], cand_l[i])}
    return out, cur, step


if __name__ == "__main__":
    res = {}
    for name in S.PROFILES:
        dets = S.DETS_ALL if name != "sim-small" else S.DETS_ALL[:100]
        for seed in range(7, 40):                     # restart until every count is hit exactly
            res[name], L, steps = fit(name, dets, seed=seed)
            if L == 0:
                break
        print(f"{name}: residual {L} after {steps:,} steps (seed {seed})")
        assert L == 0, f"{name} did not calibrate"
    json.dump(res, open(S.HERE / "calibration.json", "w"), indent=0,
              default=lambda x: x)
