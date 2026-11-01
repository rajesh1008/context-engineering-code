"""contextlab.optimise (v7.8, Chapter 36): prompt optimisation you can audit.

Standard library only. The optimisers here are small, deterministic versions of
the three families the chapter describes (bootstrapped few-shot, evolutionary
search over instructions, reflective proposal with a per-item Pareto front).
They are not DSPy's implementations; they share its contract: a program, a
metric that returns a score and feedback, a set to optimise on, and a set the
optimiser never sees. Everything an optimiser writes passes through `Gate`.
"""
import hashlib
import json
import random
from dataclasses import dataclass, field, replace


def digest(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Prompt:
    """Pinned rules are human-owned; rules and demos are optimiser-owned."""
    pinned: tuple
    rules: tuple = ()
    demos: tuple = ()

    @property
    def key(self):
        return digest([self.pinned, sorted(self.rules), sorted(self.demos)])

    @property
    def pinned_hash(self):
        return digest(list(self.pinned))

    def with_rule(self, r):
        return self if r in self.rules else replace(self, rules=self.rules + (r,))

    def without_rule(self, r):
        return replace(self, rules=tuple(x for x in self.rules if x != r))


@dataclass
class Score:
    mean: float
    per_item: dict
    feedback: dict = field(default_factory=dict)


def evaluate(program, prompt, items, metric):
    per, fb = {}, {}
    for it in items:
        s, f = metric(it, program(prompt, it))
        per[it.id] = s
        if s < 1:
            fb[it.id] = f
    return Score(sum(per.values()) / len(items), per, fb)


def transitions(before, after):
    """Item-level change between two runs on the same items."""
    rec = sum(1 for k in before.per_item if before.per_item[k] < 1 <= after.per_item[k])
    reg = sum(1 for k in before.per_item if after.per_item[k] < 1 <= before.per_item[k])
    return {"recoveries": rec, "regressions": reg, "churn": rec + reg, "net": rec - reg}


def paired_ci(before, after, n_boot=2000, seed=36):
    """Bootstrap 95% interval on the paired difference in mean score."""
    keys = sorted(before.per_item)
    d = [after.per_item[k] - before.per_item[k] for k in keys]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return sum(d) / len(d), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def bootstrap_fewshot(program, metric, base, trainset, devset, k=3, trials=12, seed=36,
                      as_demo=lambda it: it.id):
    """Run the base prompt on train, keep passing traces as demos, pick on dev."""
    passing = [as_demo(it) for it in trainset if metric(it, program(base, it))[0] >= 1]
    rng, history = random.Random(seed), []
    best = (evaluate(program, base, devset, metric).mean, base)
    for _ in range(trials):
        cand = replace(base, demos=tuple(rng.sample(passing, k)))
        s = evaluate(program, cand, devset, metric).mean
        history.append((cand, s))
        if s > best[0]:
            best = (s, cand)
    return best[1], history


def evolve(program, metric, base, pool, devset, generations=6, children=6, seed=36,
           may_edit_pinned=False, pinned_variants=()):
    """Mutate the optimisable block (add or drop one rule); keep the best."""
    rng, history = random.Random(seed), []
    best = (evaluate(program, base, devset, metric).mean, base)
    for _ in range(generations):
        parent = best[1]
        for _ in range(children):
            op = rng.random()
            if may_edit_pinned and pinned_variants and op < 0.2:
                old, new = rng.choice(pinned_variants)
                cand = replace(parent, pinned=tuple(new if p == old else p for p in parent.pinned))
            elif parent.rules and op < 0.45:
                cand = parent.without_rule(rng.choice(parent.rules))
            else:
                cand = parent.with_rule(rng.choice(pool))
            s = evaluate(program, cand, devset, metric).mean
            history.append((cand, s))
            if s > best[0]:
                best = (s, cand)
    return best[1], history


def reflect(program, metric, base, devset, propose, rounds=8, seed=36):
    """Read failures with their feedback, propose one edit, keep a per-item
    Pareto front, and sample parents from it (exploration); return the best
    aggregate (selection)."""
    rng, front, history, seen = random.Random(seed), [base], [], {base.key: base}
    scores = {base.key: evaluate(program, base, devset, metric)}
    for r in range(rounds):
        parent = rng.choice(front)
        fails, cand = scores[parent.key].feedback, None
        for attempt in range(len(fails)):
            c = propose(parent, fails, r + attempt)
            if c is not None and c.key not in scores:
                cand = c
                break
        if cand is None:
            continue
        seen[cand.key] = cand
        scores[cand.key] = evaluate(program, cand, devset, metric)
        history.append((cand, scores[cand.key].mean))
        front.append(cand)
        front = [p for p in front if not _dominated(p, front, scores)]
    best = max(scores, key=lambda k: (scores[k].mean, -len(seen[k].rules)))
    return seen[best], history


def _dominated(p, front, scores):
    a = scores[p.key].per_item
    for q in front:
        if q.key == p.key:
            continue
        b = scores[q.key].per_item
        if all(b[k] >= a[k] for k in a) and any(b[k] > a[k] for k in a):
            return True
    return False


@dataclass
class Manifest:
    """What goes into version control next to an optimised prompt."""
    prompt: Prompt
    optimiser: str
    seed: int
    model: str
    metric_version: str
    trainset: str
    devset: str
    heldout: str
    dev_score: float
    heldout_score: float

    def record(self):
        return {"prompt_key": self.prompt.key, "pinned_hash": self.prompt.pinned_hash,
                "rules": list(self.prompt.rules), "demos": list(self.prompt.demos),
                "optimiser": self.optimiser, "seed": self.seed, "model": self.model,
                "metric": self.metric_version, "trainset": self.trainset,
                "devset": self.devset, "heldout": self.heldout,
                "dev": round(self.dev_score, 4), "heldout_score": round(self.heldout_score, 4)}


def diff(a, b):
    return {"pinned_changed": a.pinned != b.pinned,
            "rules_added": [r for r in b.rules if r not in a.rules],
            "rules_dropped": [r for r in a.rules if r not in b.rules],
            "demos_added": [d for d in b.demos if d not in a.demos]}


@dataclass
class Gate:
    """Adopt only if pinned text is untouched, held-out does not regress,
    and the scoreboard's hard number does not move."""
    pinned_hash: str
    max_rules: int = 8
    margin: float = 0.01      # held-out may not plausibly lose more than 1 point

    def check(self, cand, heldout_ci, wrong_before, wrong_after, narrow):
        reasons = []
        if cand.pinned_hash != self.pinned_hash:
            reasons.append("pinned block edited")
        if heldout_ci[0] <= 0:
            reasons.append(f"held-out gain {100 * heldout_ci[0]:+.2f} points, not positive")
        if heldout_ci[1] < -self.margin:
            reasons.append(f"held-out interval reaches {100 * heldout_ci[1]:+.2f} points")
        if wrong_after > wrong_before:
            reasons.append(f"wrong-clause {wrong_before} -> {wrong_after}")
        if narrow:
            reasons.append(f"{len(narrow)} item-specific rule(s)")
        if len(cand.rules) > self.max_rules:
            reasons.append("rule budget exceeded")
        return (not reasons), reasons


@dataclass
class Playbook:
    """An evolving context: bullets with counters, updated by deltas or rewrites."""
    bullets: dict = field(default_factory=dict)   # rule id -> [helpful, harmful]
    queued: list = field(default_factory=list)    # waiting for a human

    def delta(self, proposals, needs_human, keep=lambda r: True):
        for r, helpful in proposals:
            if not keep(r):
                continue
            if needs_human(r):
                if r not in self.queued and r not in self.bullets:
                    self.queued.append(r)
                continue
            c = self.bullets.setdefault(r, [0, 0])
            c[0 if helpful else 1] += 1
        for r in [r for r, (h, x) in self.bullets.items() if x > h + 1]:
            del self.bullets[r]

    def rewrite(self, budget, generic, length):
        """Monolithic rewrite to a length budget. The rewriter keeps the shortest
        lines and folds the rest into one generic line: brevity bias, and over
        repeated rewrites, context collapse."""
        if sum(length(r) for r in self.bullets) <= budget:
            return
        kept, used = {}, length(generic)
        for r in sorted(self.bullets, key=length):
            if used + length(r) <= budget:
                kept[r], used = self.bullets[r], used + length(r)
        kept[generic] = [0, 0]
        self.bullets = kept

    def rules(self):
        return tuple(sorted(self.bullets))
