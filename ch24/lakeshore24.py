"""Chapter 24 stand-in: Lakeshore's bulletin stream, policy changes and determinations.

Everything here is synthetic and seeded. Volumes follow the book's Lakeshore
facts (six states, about 3,200 determinations a business day); timings of each
pipeline stage are stated constants, not measurements of any product.
Times are naive datetimes in US Central time.
"""
import random
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta

STATES = ["WI", "IA", "MN", "IL", "IN", "MI"]
PERILS = {"WB": 0.18, "WN": 0.16, "HL": 0.14, "BP": 0.14,
          "FR": 0.08, "TH": 0.10, "MO": 0.08, "EB": 0.12}
START, END = date(2026, 7, 1), date(2026, 9, 30)
GRAPH_FULL_BUILD = datetime(2026, 6, 30, 23, 0)
PER_DAY = 3200
GRAPH_WALK_SHARE = 0.065 * 32 / 65     # graph-routed and not sent to the loop (Ch18 router v2)
CHAIN_TOUCH = 0.25                     # walk's chain reaches the bulletin's peril


def rng(tag):
    return random.Random(zlib.crc32(tag.encode()))


@dataclass
class Bulletin:
    doc: str
    version: int
    state: str
    peril: str
    committed: datetime
    effective_from: date
    supersedes: str | None = None
    fixed: bool = False


def business_days(a=START, b=END):
    d = a
    while d <= b:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def bulletins():
    r, out = rng("bulletins"), []
    for d in business_days():
        if r.random() < 0.30:                       # about two a week across six states
            st, pe = r.choice(STATES), r.choice(list(PERILS))
            t = datetime(d.year, d.month, d.day, r.randint(9, 16), r.randint(0, 59))
            u = r.random()
            lag = r.randint(3, 30) if u < 0.7 else (0 if u < 0.9 else -r.randint(1, 10))
            prev = [b for b in out if (b.state, b.peril) == (st, pe)]
            doc = f"BUL-{st}-{pe}-{d:%Y%m}"
            doc += "B" if any(b.doc == doc for b in out) else ""
            out.append(Bulletin(doc, 1, st, pe, t, d + timedelta(days=lag),
                                prev[-1].doc if prev and r.random() < 0.5 else None))
    out = [b for b in out if not (b.state == "IA" and b.peril == "EB")]
    out.append(Bulletin("BUL-IA-EB-202608", 1, "IA", "EB", datetime(2026, 7, 28, 14, 20),
                        date(2026, 8, 1), None, True))
    return sorted(out, key=lambda b: b.committed)


def cache_day_events():
    """15 September 2026: a corrected reissue in Wisconsin, a new hail bulletin in Minnesota."""
    wi = Bulletin("BUL-WI-WB-202609", 2, "WI", "WB", datetime(2026, 9, 15, 10, 5),
                  date(2026, 9, 1), "BUL-WI-WB-202609@1", True)
    mn = Bulletin("BUL-MN-HL-202609", 1, "MN", "HL", datetime(2026, 9, 15, 13, 30),
                  date(2026, 9, 8), None, True)
    return wi, mn


@dataclass
class Determination:
    did: int
    decided: datetime
    state: str
    peril: str
    loss: date
    graph_walk: bool


def determinations(day, tag="dets"):
    r = rng(f"{tag}-{day}")
    ps, ws = list(PERILS), list(PERILS.values())
    out = []
    for i in range(PER_DAY):
        t = datetime(day.year, day.month, day.day, 8) + timedelta(seconds=r.randint(0, 10 * 3600 - 1))
        st, pe = r.choice(STATES), r.choices(ps, ws)[0]
        walk = r.random() < GRAPH_WALK_SHARE and r.random() < CHAIN_TOUCH
        out.append(Determination(i, t, st, pe, day - timedelta(days=r.randint(1, 20)), walk))
    return sorted(out, key=lambda x: x.decided)
