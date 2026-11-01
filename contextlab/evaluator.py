"""contextlab.evaluator (v8.6, Chapter 39): measure retrieval, assembly and grounding
separately, validate the judges that score them, and gate releases on the result.

Standard library only. Positive class for judges is "pass" (relevant, supported).
"""
import hashlib
import json
import math
import random
from dataclasses import dataclass, field


# ---------- retrieval metrics (Chapter 12 definitions, graded gains are linear) ----------
def hit_at_k(ranked, gold, k):
    return 1.0 if any(s in gold for s in ranked[:k]) else 0.0


def recall_at_k(ranked, gold, k):
    return sum(1 for s in ranked[:k] if s in gold) / len(gold) if gold else 0.0


def mrr(ranked, gold):
    for i, s in enumerate(ranked, 1):
        if s in gold:
            return 1 / i
    return 0.0


def ndcg_at_k(ranked, grades, k):
    dcg = sum(grades.get(s, 0) / math.log2(i + 1) for i, s in enumerate(ranked[:k], 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def judged_at_k(ranked, judged, k):
    return sum(1 for s in ranked[:k] if s in judged) / k


# ---------- agreement ----------
def cohen_kappa(a, b, weighted=False):
    labels = sorted(set(a) | set(b))
    n, idx = len(a), {l: i for i, l in enumerate(labels)}
    m = len(labels)
    w = [[(abs(i - j) / (m - 1) if weighted and m > 1 else float(i != j)) for j in range(m)]
         for i in range(m)]
    obs = [[0] * m for _ in range(m)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    ra = [sum(r) for r in obs]
    cb = [sum(obs[i][j] for i in range(m)) for j in range(m)]
    do = sum(w[i][j] * obs[i][j] for i in range(m) for j in range(m)) / n
    de = sum(w[i][j] * ra[i] * cb[j] for i in range(m) for j in range(m)) / (n * n)
    return 1 - do / de if de else 1.0


def agreement(judge, human):
    """Judge verdicts against adjudicated human verdicts (True = pass)."""
    tp = sum(1 for j, h in zip(judge, human) if j and h)
    tn = sum(1 for j, h in zip(judge, human) if not j and not h)
    pos, neg = sum(human), len(human) - sum(human)
    return {"n": len(human), "agreement": (tp + tn) / len(human),
            "kappa": cohen_kappa(list(judge), list(human)),
            "tpr": tp / pos if pos else float("nan"), "tnr": tn / neg if neg else float("nan")}


def corrected_rate(p_obs, tpr, tnr):
    """Estimate the true pass rate from a judge's observed pass rate (Rogan-Gladen)."""
    d = tpr + tnr - 1
    return min(1.0, max(0.0, (p_obs + tnr - 1) / d)) if d > 0 else float("nan")


def corrected_ci(judge_cal, human_cal, judge_target, n_boot=2000, seed=39):
    rng, est = random.Random(seed), []
    cal = list(zip(judge_cal, human_cal))
    for _ in range(n_boot):
        c = [rng.choice(cal) for _ in cal]
        t = [rng.choice(judge_target) for _ in judge_target]
        a = agreement([x for x, _ in c], [y for _, y in c])
        est.append(corrected_rate(sum(t) / len(t), a["tpr"], a["tnr"]))
    est = sorted(e for e in est if e == e)
    a = agreement(judge_cal, human_cal)
    point = corrected_rate(sum(judge_target) / len(judge_target), a["tpr"], a["tnr"])
    return point, est[int(0.025 * len(est))], est[int(0.975 * len(est)) - 1]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def paired_bootstrap(before, after, n_boot=2000, seed=39):
    d = [b - a for a, b in zip(before, after)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return sum(d) / len(d), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


# ---------- stage attribution ----------
STAGES = ("ok", "correct referral", "retrieval", "assembly", "grounding: wrong citation",
          "grounding: unsupported")


def attribute(r, k=10):
    """r: dict with governing (id or None), rank (1-based or None), in_packet, cited,
    supported (bool). Returns the first stage that lost the governing span."""
    if r["governing"] is None:
        return "correct referral" if r["cited"] is None else "grounding: wrong citation"
    if r["rank"] is None or r["rank"] > k:
        return "retrieval"
    if not r["in_packet"]:
        return "assembly"
    if r["cited"] != r["governing"]:
        return "grounding: wrong citation"
    return "ok" if r["supported"] else "grounding: unsupported"


# ---------- gold set ----------
def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class GoldSet:
    version: str
    items: tuple          # claim ids
    labels: tuple         # (claim id, span id, grade) adjudicated

    @property
    def key(self):
        return digest([self.version, list(self.items), list(self.labels)])


# ---------- the evaluator ----------
@dataclass
class Evaluator:
    gold: GoldSet
    k: int = 10
    stages: dict = field(default_factory=dict)

    def retrieval(self, runs, grades, governing):
        """runs: claim -> ranked span ids; grades: claim -> {span: grade}."""
        rows = [c for c in runs if governing.get(c)]
        g2 = {c: {s for s, g in grades[c].items() if g >= 2} for c in rows}
        out = {"n": len(rows),
               f"hit@{self.k}": sum(hit_at_k(runs[c], {governing[c]}, self.k) for c in rows) / len(rows),
               f"recall@{self.k}": sum(recall_at_k(runs[c], g2[c], self.k) for c in rows) / len(rows),
               "mrr": sum(mrr(runs[c], {governing[c]}) for c in rows) / len(rows),
               f"ndcg@{self.k}": sum(ndcg_at_k(runs[c], grades[c], self.k) for c in rows) / len(rows)}
        self.stages["retrieval"] = out
        return out

    def run(self, records):
        """records: claim -> dict for `attribute`. Returns counts by stage and headline rates."""
        counts = {s: 0 for s in STAGES}
        for r in records.values():
            counts[attribute(r, self.k)] += 1
        n = len(records)
        wrong = sum(1 for r in records.values() if r["cited"] is not None
                    and r["cited"] != r["governing"])
        out = {"n": n, "by_stage": counts,
               "faithfulness": (counts["ok"] + counts["correct referral"]) / n,
               "wrong_clause": wrong,
               "packet_sufficiency": sum(1 for r in records.values() if r["governing"]
                                         and r["in_packet"]) /
               max(1, sum(1 for r in records.values() if r["governing"]))}
        self.stages["end_to_end"] = out
        return out

    @staticmethod
    def validate_judge(judge, human, min_kappa=0.6, min_tnr=0.8):
        a = agreement(judge, human)
        a["valid"] = a["kappa"] >= min_kappa and a["tnr"] >= min_tnr
        return a


@dataclass
class ReleaseGate:
    """Adopt a release only if every stage holds and the judge that scored it is valid."""
    margin: float = 0.01

    def check(self, base, cand, judge, faith_ci, recall_ci):
        m, why = self.margin, []
        if not judge["valid"]:
            why.append(f"judge not valid (kappa {judge['kappa']:.2f}, TNR {judge['tnr']:.2f})")
        if recall_ci[1] < -m:
            why.append(f"recall@10 interval reaches {100 * recall_ci[1]:+.1f} points")
        b, c = base["packet_sufficiency"], cand["packet_sufficiency"]
        if c < b - m:
            why.append(f"packet sufficiency {100 * b:.1f}% -> {100 * c:.1f}%")
        if faith_ci[1] < -m:
            why.append(f"faithfulness interval reaches {100 * faith_ci[1]:+.1f} points")
        if cand["wrong_clause"] > base["wrong_clause"]:
            why.append(f"wrong-clause {base['wrong_clause']} -> {cand['wrong_clause']}")
        return (not why), why
