"""Chapter 22 stand-in: Lakeshore's full mix replayed in date order with memory on.

Note: deterministic stand-in. No language model runs. 1,000 determinations
over Chapter 21's two weeks (ten working days, 5 to 16 October 2026); a memory
store shaped like Chapter 21's (subject, adjuster and team scopes, the go-live
back-fill, nightly consolidation); a 'determination model' that answers from
its packet and follows a memory that contradicts the governing span with a
stated probability. Every probability is a constant below. Seeded: rerun and
you get the same numbers. Which determinations accept lesson M02116's cause is
not drawn here: it is carried from Chapter 21's replay (04-code/ch21, section 8),
so the six determinations this chapter traces are the six that chapter counted.
"""
import random
import zlib
from dataclasses import dataclass
from datetime import date, timedelta

from contextlab.memguard import MemoryRecord as M, Origin as O, ReadLog

SEED = 22
DAYS = [date(2026, 10, 5) + timedelta(d) for d in range(12)
        if (date(2026, 10, 5) + timedelta(d)).weekday() < 5]       # 10 working days
POISON_NIGHT = date(2026, 10, 4)             # go-live back-fill: consolidation writes M02116
SECOND_NIGHT = date(2026, 10, 12)            # failure injection: M-T0419
FOLLOWED = {"D0025", "D0355", "D0450", "D0807", "D0848", "D0978"}  # Chapter 21, section 8
MOULD = {"D0807"}                            # recorded as mould at first notice in Chapter 21
MOULD_READER = "D0030"                       # the second mould claim among Chapter 21's 37 readers
STATES = ["WI", "IA", "MN", "IL", "MI", "IN"]
STATE_W = [34, 14, 16, 16, 12, 8]
PERILS = ["water_backup", "burst_pipe", "seepage", "hail", "wind", "fire", "theft", "other"]
P_FOLLOW = {"team": 0.15, "adjuster": 0.20, "policy": 0.50}   # follow a contradicting memory
P_DISTRACT = 0.002            # per memory read beyond six (over-remembering)
K_MAX = 8                     # most records Chapter 21's per-kind caps and 160-token budget admit
LESSON_READERS = 37           # Chapter 21: questions that pass M02116's lesson floor


def u(*parts) -> float:
    return zlib.crc32("|".join(map(str, parts)).encode()) / 2**32


@dataclass
class Det:
    did: str
    day: date
    state: str
    peril: str
    policy: str
    adjuster: str
    repeat: bool
    ho0495: bool        # water back-up endorsement on the policy today
    metal: bool = False
    cde: bool = False   # cosmetic damage exclusion endorsement
    residual: bool = False

    def truth(self) -> str:
        if self.peril == "water_backup":
            return "covered:HO0495" if self.ho0495 else "excluded:HO3-2.c"
        if self.peril == "mould":
            return "excluded:HO3-mould"
        if self.peril == "burst_pipe":
            return "covered:HO3-sudden"
        if self.peril == "seepage":
            return "excluded:HO3-seepage"
        if self.peril == "hail" and self.state == "IL" and self.metal:
            return "excluded:CDE" if self.cde else "covered:HO3-hail"
        return "covered:" + self.peril

    def keys(self):
        return (self.policy, self.adjuster, self.state)


def make_mix():
    rng = random.Random(SEED)
    dets = []
    for i in range(1000):
        st = rng.choices(STATES, STATE_W)[0]
        pw = [12 if st == "WI" else 8, 12, 6, 18, 14, 8, 8, 17]
        d = Det(f"D{i+1:04d}", DAYS[i * len(DAYS) // 1000], st, rng.choices(PERILS, pw)[0],
                f"P-{st}-{rng.randrange(10**5):05d}", f"A{STATES.index(st)*2 + rng.randrange(2) + 1:02d}",
                rng.random() < 0.30, rng.random() < 0.45)
        d.metal = st == "IL" and rng.random() < 0.35
        d.cde = d.metal and rng.random() < 0.40
        dets.append(d)
    for d in dets:                         # Chapter 21's six: WI water claims at first notice
        if d.did in FOLLOWED:
            d.state, d.peril = "WI", "mould" if d.did in MOULD else "water_backup"
            d.policy = "P-WI" + d.policy[4:]
        if d.did == MOULD_READER:
            d.peril = "mould"
    pool = [d for d in dets if d.peril in ("wind", "fire", "theft", "other") and d.did not in FOLLOWED]
    for d in rng.sample(pool, 7):          # Chapter 20's seven residual errors
        d.residual = True
    return dets


def seed_store(dets):
    """The store as the go-live back-fill wrote it on 4 October, before the replay.

    Earlier-claim episodes (scope 'policy', the claimant/policy subject scope) carry the
    outcome of the earlier claim, including the endorsement it was settled under. That is
    a fact about the past; read as a fact about today it goes stale at renewal."""
    rng = random.Random(SEED + 1)
    store = {}
    for d in dets:
        if not d.repeat:
            continue
        changed = rng.random() < 0.08          # endorsement added or dropped at renewal
        snap = (not d.ho0495) if changed else d.ho0495
        store[f"M-P{d.did}"] = M(f"M-P{d.did}", "policy", d.policy,
            f"earlier water claim on {d.policy}: settled " + ("under HO 04 95" if snap else "under the HO-3, no HO 04 95"),
            date(2025, rng.randrange(1, 13), 1),
            (O("record", f"claim:old-{d.did}"),), asserts=("ho0495", snap))
        for j in range(rng.randrange(6, 15)):                       # earlier claim notes
            store[f"M-C{d.did}-{j}"] = M(f"M-C{d.did}-{j}", "claimant", d.policy,
                f"prior claim note {j}", date(2025, rng.randrange(1, 13), 1),
                (O("determination", f"old-{d.did}-{j}"),))
    for a in range(1, 13):
        for j in range(3):
            mid = f"M-A{a:02d}-{j}"
            store[mid] = M(mid, "adjuster", f"A{a:02d}", f"working note {j}",
                           date(2026, 9, 1), (O("note", mid),))
    return store


def poison(dets=None):
    """Chapter 21's lesson: a 41-word correction by tbrennan (17 February) consolidated
    at go-live into 24 words that drop the qualifiers and name a cause."""
    return M("M02116", "team", "WI",
             "Wisconsin basement water: check for a frozen supply line first, and code the "
             "loss as freezing unless back-up residue is shown at the drain.", POISON_NIGHT,
             (O("note", "correction:tbrennan:2026-02-17"),), asserts=("wi_water_cause", "freezing"))


def second_poison():
    return M("M-T0419", "team", "IL",
             "Illinois hail claims: cosmetic roof damage is not covered.", SECOND_NIGHT,
             (O("note", "N-ADJ-1019a"), O("note", "N-ADJ-1019b"), O("note", "N-ADJ-1019c")),
             asserts=("il_hail", "excluded"))


def correct_positions():
    """Six team memories that state a right position from unverified origins."""
    tx = [("IA", "burst_pipe", "Iowa burst pipe: sudden discharge is covered"),
          ("MN", "burst_pipe", "Minnesota burst pipe: sudden discharge is covered"),
          ("MI", "seepage", "Michigan seepage over weeks is excluded"),
          ("IN", "seepage", "Indiana seepage over weeks is excluded"),
          ("WI", "seepage", "Wisconsin seepage over weeks is excluded"),
          ("IL", "burst_pipe", "Illinois burst pipe: sudden discharge is covered")]
    return [M(f"M-TOK{i}", "team", s, t, date(2026, 9, 14), (O("determination", f"old-ok-{i}"),),
              asserts=(f"{s.lower()}_{p}", "ok")) for i, (s, p, t) in enumerate(tx)]


def lesson_passes(dets):
    """The determinations whose question passes the lesson floor: Chapter 21's six plus
    WI water claims ranked by a seeded hash, to Chapter 21's mix of 37 readers
    (back-up 21, frozen pipe 14, mould 2)."""
    want = {"water_backup": 21, "burst_pipe": 14, "mould": 2}
    out = set(FOLLOWED)
    for d in dets:
        if d.did in FOLLOWED:
            want[d.peril] -= 1
    for d in sorted(dets, key=lambda d: u(d.did, "floor")):
        if d.state == "WI" and d.did not in out and want.get(d.peril, 0) > 0:
            out.add(d.did); want[d.peril] -= 1
    return out
