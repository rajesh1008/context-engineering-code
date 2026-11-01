"""Lakeshore's simulated year, November 2026 to October 2027 (Chapter 45 stand-in).

Nothing here happened. The year is generated from growth rates and seasonal shapes
chosen to resemble what an insurer's context layer meets, and every number the
lab prints after November 2026 is a property of this stand-in. No model is called.

Mechanisms, each deliberately simple:
* Retrieval: the governing span's first-stage rank is 1 + floor(N_eff * q), where
  N_eff counts competing pages (weighted by how much each kind resembles policy
  language) and q is a per-claim difficulty. Adding look-alike pages pushes every
  governing span down the list in proportion. The reranker sees the top K = 40.
* A determination whose governing span is not in the pool is referred by the
  withhold-and-refer instruction 55% of the time and cites a look-alike otherwise.
* Withhold floor: a determination is also referred when its top dense score is
  under the floor. A new embedding model shifts every score down.
* Bulletins: a determination made between a bulletin's effective date and its
  publication is decided under the superseded rule, and one in four of those
  changes the answer. A bulletin touches between 2% and 10% of the claims in its
  state and line.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache
from dataclasses import dataclass, field
from datetime import date, timedelta

START = date(2026, 11, 1)
MONTHS = ["Nov 2026", "Dec 2026", "Jan 2027", "Feb 2027", "Mar 2027", "Apr 2027",
          "May 2027", "Jun 2027", "Jul 2027", "Aug 2027", "Sep 2027", "Oct 2027"]
SHORT = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
FIRST = [date(2026, 11, 1), date(2026, 12, 1)] + [date(2027, m, 1) for m in range(1, 11)]
DAYS = [30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31]
PER_DAY = 3200
VOLUME = [1.00, 0.96, 1.06, 1.24, 1.12, 1.10, 1.36, 1.44, 1.28, 1.12, 1.02, 1.00]

NOTES_INGEST = date(2027, 3, 13)        # underwriting notes migrated from the file shares
NOTICE = date(2027, 5, 12)              # embedding deprecation notice
CUTOVER = date(2027, 6, 9)
RETIRE = date(2027, 6, 30)
REBASELINES = [date(2027, 1, 4), date(2027, 4, 5), date(2027, 7, 5), date(2027, 10, 4)]
MEMORY_LIVE = date(2026, 10, 5)

CAUSES = ["water_backup", "freeze", "wind_hail", "fire", "theft", "marine", "other"]
FAMILY = {"water_backup": "water", "freeze": "water", "wind_hail": "wind", "fire": "property",
          "theft": "property", "marine": "marine", "other": "property"}
#            Nov  Dec  Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct
MIX = {"water_backup": [16, 13, 12, 12, 22, 24, 17, 15, 14, 15, 16, 16],
       "freeze":       [3, 12, 22, 26, 10, 3, 1, 0, 0, 0, 1, 2],
       "wind_hail":    [14, 8, 6, 6, 12, 20, 34, 38, 34, 27, 20, 16],
       "fire":         [17, 19, 18, 17, 14, 12, 10, 10, 11, 13, 15, 17],
       "theft":        [16, 15, 13, 12, 13, 13, 13, 13, 15, 17, 17, 16],
       "marine":       [9, 10, 8, 7, 8, 8, 8, 8, 9, 10, 10, 10],
       "other":        [25, 23, 21, 20, 21, 20, 17, 16, 17, 18, 21, 23]}
DIFF = {"water_backup": 1.25, "freeze": 1.15, "wind_hail": 0.9, "fire": 0.85,
        "theft": 0.7, "marine": 1.1, "other": 1.0}
WITHHOLD_BASE = {"water_backup": 0.060, "freeze": 0.150, "wind_hail": 0.035, "fire": 0.045,
                 "theft": 0.030, "marine": 0.055, "other": 0.045}
STATES = {"WI": 0.40, "MN": 0.15, "IL": 0.15, "MI": 0.12, "IA": 0.10, "IN": 0.08}

MU, SIGMA = -9.80, 1.10                 # per-claim retrieval difficulty, lognormal
K = 40                                  # reranker candidate pool, tuned November 2026
P_MISS_WITHHELD = 0.55
P_BASE_WRONG = 0.0018
P_NOTE_WRONG = 0.0025                   # a note quoting old wording sits in the packet
P_BULLETIN_WRONG = 0.25
REPEAT_SHARE = 0.12
OLD_FLOOR = 0.62
SCORE_SHIFT = 0.07                      # new model's scores sit lower
WEIGHT = {"forms": 1.0, "endorsements": 1.0, "bulletins": 1.0, "marine": 1.0,
          "manuals": 0.6, "notes": 0.9}


def month_of(d: date) -> int:
    for i in range(11, -1, -1):
        if d >= FIRST[i]:
            return i
    return 0


# ------------------------------------------------------------------ corpus
def corpus(d: date) -> dict[str, int]:
    """Pages by kind on day d."""
    c = {"forms": 14000, "endorsements": 9000, "bulletins": 11000, "marine": 2000,
         "manuals": 4000, "notes": 0}
    m = month_of(d)
    frac = (d - FIRST[m]).days / DAYS[m]
    for i in range(m + 1):
        f = 1.0 if i < m else frac
        c["bulletins"] += round(150 * f)
        if SHORT[i] in ("Jan", "Feb", "Mar"):
            c["bulletins"] += round(500 * f)
        if SHORT[i] in ("May", "Jun", "Jul"):
            c["bulletins"] += round(120 * f)
        if SHORT[i] == "Jan" and (i < m or d >= date(2027, 1, 1)):
            c["forms"] += 1200
            c["endorsements"] += 700
    if d >= NOTES_INGEST:
        c["notes"] = 34000 + round(900 * max(0, (d - NOTES_INGEST).days) / 30.4)
    return c


def pages(d: date) -> int:
    return sum(corpus(d).values())


@lru_cache(maxsize=None)
def n_eff(d: date, partitioned: bool) -> float:
    c = corpus(d)
    return sum(v * WEIGHT[k] for k, v in c.items() if not (partitioned and k == "notes"))


# ----------------------------------------------------------------- regimes
@dataclass
class Regime:
    name: str
    partition_from: date | None          # notes moved to a context-only index
    floors: list[tuple[date, float]]     # floor in force under the new model, from date
    memory_from: date | None             # memory index on the new model from this date
    staffing: float = 0.6                # bulletin approvals per working day, Dec to Mar
    priority_lane: bool = False          # emergency bulletins reviewed first, backfill bounded
    pending_flag: bool = False           # affected claims referred until publication
    prune: bool = False
    jitter: bool = False
    k: int = K

    def partitioned(self, d: date) -> bool:
        return self.partition_from is not None and d >= self.partition_from

    def floor(self, d: date) -> float:
        if d < CUTOVER:
            return OLD_FLOOR
        f = OLD_FLOOR
        for since, val in self.floors:
            if d >= since:
                f = val
        return f

    def memory_ok(self, d: date) -> bool:
        return d < CUTOVER or (self.memory_from is not None and d >= self.memory_from)


# ---------------------------------------------------------------- bulletins
@dataclass
class Bulletin:
    id: str
    state: str
    family: str
    issued: date
    effective: date
    emergency: bool
    scope: float                 # share of that state's family claims the bulletin touches


BULLETINS_PER_MONTH = [3, 9, 16, 12, 14, 5, 8, 7, 6, 4, 3, 4]
FAM_W = {"water": 0.35, "wind": 0.25, "property": 0.30, "marine": 0.10}


def bulletins() -> list[Bulletin]:
    rng = random.Random("ch45-bulletins")
    out = []
    for m, n in enumerate(BULLETINS_PER_MONTH):
        for j in range(n):
            issued = FIRST[m] + timedelta(days=rng.randrange(DAYS[m]))
            emergency = rng.random() < (0.5 if SHORT[m] in ("May", "Jun", "Jul") else 0.1)
            if SHORT[m] == "Dec" and not emergency:
                eff = date(2027, 1, 1)           # January editions
                if eff <= issued:
                    eff = issued + timedelta(days=3)
            else:
                lead = 0 if emergency else max(2, int(rng.expovariate(1 / 14)))
                eff = issued + timedelta(days=lead)
            st = rng.choices(list(STATES), weights=list(STATES.values()))[0]
            fam = rng.choices(list(FAM_W), weights=list(FAM_W.values()))[0]
            scope = round(rng.uniform(0.02, 0.10), 3)
            out.append(Bulletin(f"B{len(out)+1:03d}", st, fam, issued, eff, emergency, scope))
    return sorted(out, key=lambda b: (b.issued, b.id))


BASE_CAPACITY = 0.6                     # approvals per working day outside the season
MIN_REVIEW = 3                          # days


def _working(d: date) -> bool:
    return d.weekday() < 5


def publish_dates(items: list[Bulletin], reg: Regime) -> dict[str, date]:
    """Two approvers and compliance review in order of arrival (or, with the priority
    lane, emergencies first and then by effective date);
    the publish pipeline adds one day, or four while a bulk backfill holds it."""
    pending: list[Bulletin] = []
    done: dict[str, date] = {}
    queue = sorted(items, key=lambda b: b.issued)
    credit = 0.0
    d = queue[0].issued
    i = 0
    while len(done) < len(items):
        while i < len(queue) and queue[i].issued <= d:
            pending.append(queue[i])
            i += 1
        if _working(d):
            season = d.month in (12, 1, 2, 3)
            credit += reg.staffing if season else BASE_CAPACITY
            if reg.priority_lane:
                pending.sort(key=lambda b: (not b.emergency, b.effective, b.issued))
            while credit >= 1 and pending:
                b = pending[0]
                if (d - b.issued).days < MIN_REVIEW:   # two approvers and compliance
                    break
                pending.pop(0)
                credit -= 1
                backfill = (not reg.priority_lane) and NOTES_INGEST <= d <= NOTES_INGEST + timedelta(days=3)
                done[b.id] = d + timedelta(days=4 if backfill else 1)
            if not pending:
                credit = min(credit, 1.0)
        d += timedelta(days=1)
    return done


def windows(reg: Regime) -> list[tuple[Bulletin, date]]:
    """Bulletins whose publication came after their effective date, with that date."""
    items = bulletins()
    pub = publish_dates(items, reg)
    return [(b, pub[b.id]) for b in items if pub[b.id] > b.effective]


def family_rate(fam: str, d: date) -> float:
    m = month_of(d)
    return sum(MIX[c][m] for c in CAUSES if FAMILY[c] == fam) / sum(MIX[c][m] for c in CAUSES)


def exposure(reg: Regime) -> dict[int, float]:
    """Production determinations decided under a superseded rule, by month."""
    out = {m: 0.0 for m in range(12)}
    for b, pub in windows(reg):
        d = b.effective
        while d < pub:
            m = month_of(d)
            out[m] += PER_DAY * VOLUME[m] * STATES[b.state] * family_rate(b.family, d) * b.scope
            d += timedelta(days=1)
    return out


# ------------------------------------------------------------------ claims
@dataclass
class Claim:
    id: str
    day: date
    cause: str
    state: str
    q: float
    s_old: float
    s_new: float
    repeat: bool
    u: list[float] = field(default_factory=list)


def claims(m: int, n: int, tag: str = "review") -> list[Claim]:
    rng = random.Random(f"ch45-{tag}-{m}-{n}")
    w = [MIX[c][m] for c in CAUSES]
    out = []
    for i in range(n):
        day = FIRST[m] + timedelta(days=rng.randrange(DAYS[m]))
        cause = rng.choices(CAUSES, weights=w)[0]
        state = rng.choices(list(STATES), weights=list(STATES.values()))[0]
        q = math.exp(rng.gauss(MU, SIGMA)) * DIFF[cause]
        s_old = 0.74 + 0.06 * rng.gauss(0, 1)
        s_new = s_old - SCORE_SHIFT + 0.012 * rng.gauss(0, 1)
        out.append(Claim(f"{tag}-{m:02d}-{i:05d}", day, cause, state, q, s_old, s_new,
                         rng.random() < REPEAT_SHARE, [rng.random() for _ in range(5)]))
    return out


def _touch(cid: str, bid: str) -> float:
    return random.Random(f"{cid}|{bid}").random()


def decide(c: Claim, reg: Regime, wins: list[tuple[Bulletin, date]] | None = None) -> dict:
    part = reg.partitioned(c.day)
    rank = 1 + int(n_eff(c.day, part) * c.q)
    hit = rank <= reg.k
    new_model = c.day >= CUTOVER
    score = c.s_new if new_model else c.s_old
    notes_in_ctx = corpus(c.day)["notes"] > 0 and not part
    outcome = "ok"
    if not hit:
        outcome = "withheld" if c.u[0] < P_MISS_WITHHELD else "wrong"
    else:
        if c.u[1] < P_BASE_WRONG or (notes_in_ctx and c.u[2] < P_NOTE_WRONG):
            outcome = "wrong"
        elif score < reg.floor(c.day) or c.u[3] < WITHHOLD_BASE[c.cause]:
            outcome = "withheld"
    superseded = False
    for b, pub in (wins or []):
        if (b.state == c.state and b.family == FAMILY[c.cause] and b.effective <= c.day < pub
                and _touch(c.id, b.id) < b.scope):
            superseded = True
    if superseded and outcome == "ok":
        if reg.pending_flag:
            outcome = "withheld"
        elif c.u[4] < P_BULLETIN_WRONG:
            outcome = "wrong"
    return {"id": c.id, "rank": rank, "hit": hit, "outcome": outcome, "score": score,
            "memory_missing": c.repeat and not reg.memory_ok(c.day), "superseded": superseded,
            "notes_in_ctx": notes_in_ctx, "cause": c.cause, "repeat": c.repeat}


# --------------------------------------------------------------- cost model
ENTRY_COST = 0.0348          # US$ per determination at entry (Chapter 35's close)
ENTRY_INPUT = 3077           # stand-in input tokens: fixed 2,022 + instructions 640 + packet 415


def cost_per_det(instr_lines: int, notes_tokens: float, k: int = K) -> float:
    inp = 2022 + 640 * instr_lines / 46 + 415 + notes_tokens
    return ENTRY_COST * (0.30 + 0.55 * inp / ENTRY_INPUT + 0.15 * k / K)


# ----------------------------------------------------- instruction ledger
ADDS = [1, 2, 7, 5, 6, 4, 4, 6, 5, 6, 3, 2]      # lines appended after incidents, per month


def instruction_lines(prune_quarterly: bool) -> list[int]:
    """Determination instruction lines at the end of each month.

    Every appended line answers an incident; 45% stop being needed once the root
    cause is fixed. As run, 40% of appended lines carry no rationale and no test,
    so nothing is ever removed. With the ledger, each line has both and the quarterly
    prune removes those whose test passes without them.
    """
    rng = random.Random("ch45-instructions")
    lines = [(True, rng.random() < 0.74) for _ in range(46)]  # (needed, documented)
    out = []
    for m in range(12):
        for _ in range(ADDS[m]):
            needed = rng.random() >= 0.45
            documented = True if prune_quarterly else rng.random() >= 0.40
            lines.append((needed, documented))
        if prune_quarterly and SHORT[m] in ("Dec", "Mar", "Jun", "Sep"):
            lines = [ln for ln in lines if ln[0] or not ln[1]]
        out.append(len(lines))
    return out


# ------------------------------------------------------------ cohort clocks
def revalidation_due(jitter: bool) -> dict[date, int]:
    """Earlier-claim records due for their 180-day re-validation, by day.

    At memory go-live 14,000 records were backfilled on one day; about 60 a day
    have been written since. With jitter, each record's first re-validation is
    drawn between day 150 and day 180."""
    rng = random.Random("ch45-cohort")
    due: dict[date, int] = {}
    made = [(MEMORY_LIVE, 14000)] + [(MEMORY_LIVE + timedelta(days=i), 60) for i in range(1, 400)]
    for d0, n in made:
        for _ in range(n):
            gap = rng.randint(150, 180) if jitter else 180
            d = d0 + timedelta(days=gap)
            due[d] = due.get(d, 0) + 1
    return due


def revalidation_backlog(jitter: bool, capacity: int = 2000) -> tuple[dict[date, int], int]:
    """Records past due and not yet re-validated, by day; and the repeat-claimant
    determinations that met one (and so ran without that earlier-claim record)."""
    due = revalidation_due(jitter)
    backlog, d, out, hits = 0, min(due), {}, 0
    live = 14000 + 60 * 180
    while d <= date(2027, 10, 31):
        backlog = max(0, backlog + due.get(d, 0) - capacity)
        out[d] = backlog
        m = month_of(d) if d >= START else 0
        hits += PER_DAY * VOLUME[m] * REPEAT_SHARE * min(1.0, backlog / live)
        d += timedelta(days=1)
    return out, round(hits)


# ---------------------------------------------------------------- gold ageing
def gold_aged_share() -> list[float]:
    """Share of the 1,040 gold items whose governing edition has been superseded."""
    rng = random.Random("ch45-gold")
    hazard = {"Jan": 0.050, "Feb": 0.008, "Mar": 0.012, "May": 0.006, "Jun": 0.006, "Jul": 0.004}
    aged = [False] * 1040
    out = []
    for m in range(12):
        h = hazard.get(SHORT[m], 0.002)
        aged = [a or rng.random() < h for a in aged]
        out.append(sum(aged) / len(aged))
    return out


def replay_sample(n: int = 2000) -> list[Claim]:
    """Last month's determinations before the cutover (May 2027), replayed."""
    return claims(6, n, tag="replay")


def recalibrated_floor() -> float:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from contextlab.operations import recalibrate_floor
    rs = replay_sample()
    return recalibrate_floor([c.s_old for c in rs], [c.s_new for c in rs], OLD_FLOOR)


NEW_FLOOR = recalibrated_floor()
AS_RUN = Regime("as run", partition_from=date(2027, 8, 4),
                floors=[(CUTOVER, OLD_FLOOR), (date(2027, 6, 18), 0.58), (date(2027, 8, 12), NEW_FLOOR)],
                memory_from=date(2027, 8, 12))
AS_OPERATED = Regime("as operated", partition_from=date(2027, 3, 26),
                     floors=[(CUTOVER, NEW_FLOOR)], memory_from=CUTOVER,
                     staffing=1.5, priority_lane=True, pending_flag=True, prune=True, jitter=True)
