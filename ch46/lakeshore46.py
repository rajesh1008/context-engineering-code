"""Lakeshore stand-in data for Chapter 46 (seed 46). Everything is invented.

Three consuming teams: claims (the determination assistant, role adjuster),
policyholder (the policyholder assistant, role policyholder) and agency (the
assistant for the independent agents who sell Lakeshore policies, role agent).
"""
import random

from contextlab.platform import Span, Corpus, Principal, Directory, TeamBudget

SEED = 46
N_TOPICS = 30
N_POLICIES = 2000
N_AGENCIES = 40
ALL = frozenset({"adjuster", "supervisor", "agent", "policyholder"})
STAFF = frozenset({"adjuster", "supervisor"})

CORPORA = [
    Corpus("forms", "platform", True),
    Corpus("bulletins", "platform", True),
    Corpus("guidelines", "platform", True),
    Corpus("faq", "platform", True),
    Corpus("declarations", "platform", True),
    Corpus("claim-files", "claims", False),
    Corpus("producer-guides", "agency", False),
    Corpus("letter-templates", "policyholder", False),
]

# Topics whose current form edition changed after the agency team's fork.
CHANGED_SINCE_FORK = (3, 8, 17, 22)
# Topics where the agency's producer guide paraphrases coverage loosely.
NOTE_TOPICS = tuple(range(0, 30, 3))


def topic(i):
    return f"T{i:02d}"


def build():
    rng = random.Random(SEED)
    spans, fork = [], []
    agency_of = {}
    for i in range(N_TOPICS):
        t = topic(i)
        cur = Span(f"form-{t}@2026", "forms", t, "governing", rng.randint(260, 420), ALL)
        old = Span(f"form-{t}@2024", "forms", t, "governing", cur.tokens - 12, ALL, current=False)
        spans += [cur, old]
        if i in CHANGED_SINCE_FORK:
            fork += [Span(old.sid, "forms", t, "governing", old.tokens, ALL, current=True)]
        else:
            fork += [cur]
        if i % 2 == 0:
            b = Span(f"bulletin-{t}", "bulletins", t, "governing", rng.randint(140, 220),
                     frozenset({"adjuster", "supervisor", "agent"}))
            spans.append(b)
            fork.append(b)
        g = Span(f"guide-{t}", "guidelines", t, "guidance", rng.randint(300, 480), STAFF)
        f = Span(f"faq-{t}", "faq", t, "public", rng.randint(90, 160), ALL)
        spans += [g, f]
        fork += [g, f]
        if i in NOTE_TOPICS:
            n = Span(f"pg-{t}", "producer-guides", t, "team-note", rng.randint(120, 200),
                     frozenset({"agent"}))
            spans.append(n)
            fork.append(n)
        spans.append(Span(f"lt-{t}", "letter-templates", t, "team-note", 110,
                          frozenset({"policyholder"})))
        c = Span(f"cf-{t}", "claim-files", t, "guidance", 220, frozenset({"adjuster"}))
        spans.append(c)
    for k in range(N_POLICIES):
        pid = f"P{k:04d}"
        ag = f"A{k % N_AGENCIES:02d}"
        agency_of[pid] = ag
        d = declarations(pid, ag)
        spans.append(d)
        fork.append(d)
    return spans, fork, agency_of


def declarations(pid, agency):
    return Span(f"dec-{pid}", "declarations", "*", "governing", 180, ALL,
                policy_id=pid, agency=agency)


def people(agency_of):
    d = Directory()
    d.agency_of = dict(agency_of)
    for a in range(N_AGENCIES):
        for j in range(5):
            d.grant(Principal("agency", "agent-assist", f"agent-{a:02d}-{j}", "agent",
                              agency=f"A{a:02d}"))
    for k in range(N_POLICIES):
        d.grant(Principal("policyholder", "ph-assist", f"ph-{k:04d}", "policyholder",
                          policy_ids=frozenset({f"P{k:04d}"})))
    for j in range(60):
        d.grant(Principal("claims", "det-assist", f"adj-{j:02d}", "adjuster"))
    return d


BUDGETS = {
    "claims": TeamBudget("claims", 60_000_000, 3_000, 3),
    "policyholder": TeamBudget("policyholder", 90_000_000, 1_200, 5),
    "agency": TeamBudget("agency", 50_000_000, 2_000, 3),
}
REQ_BUDGET = {"claims": 3_000, "policyholder": 1_200, "agency": 2_000}
DAILY_CALLS = {"claims": 1_200, "policyholder": 6_000, "agency": 2_500}
RENEWAL_BATCH = 3_000          # agency batch job, Mondays
PRICES = {"per_unit": 0.003, "per_token": 0.000004}   # US$, stand-in
SHARED_MONTHLY = 9_000.0       # US$ per month: index builds, storage, on-call
PRIVATE_STORAGE = {"claims": 300.0, "agency": 120.0, "policyholder": 60.0}
