"""Chapter 46 lab: the context layer as a platform for three teams.

Deterministic stand-ins (seed 46). Writes output.txt (via stdout) and
out/results.json. Section numbers match the chapter's.
Run from 04-code: python3 ch46/lab.py > ch46/output.txt
"""
import json
import os
import random
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

from contextlab.platform import (Platform, Principal, issue_token, entitled, render,
                                 ContractTest, run_contract_tests, simulate, pct,
                                 SLO, verify_packet, sign, TIERS, VERSION, Ledger, Charge)
from lakeshore46 import (SEED, CORPORA, build, people, BUDGETS, REQ_BUDGET, DAILY_CALLS,
                         RENEWAL_BATCH, PRICES, SHARED_MONTHLY, PRIVATE_STORAGE,
                         CHANGED_SINCE_FORK, NOTE_TOPICS, N_TOPICS, N_AGENCIES, topic)

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
R = {"version": VERSION}
SPANS, FORK, AGENCY_OF = build()
CORP = {c.name: c for c in CORPORA}


def say(*a):
    print(" ".join(str(x) for x in a))


def platform(directory=None, ttl=0.25):
    d = directory or people(AGENCY_OF)
    return Platform(SPANS, CORPORA, d, BUDGETS, ttl=ttl), d


def agency_calls(rng, n, outside=0.03, t0=8.0, t1=18.0):
    """One day of agent questions: (t, user, policy, topic)."""
    calls = []
    for _ in range(n):
        a = rng.randrange(N_AGENCIES)
        user = f"agent-{a:02d}-{rng.randrange(5)}"
        if rng.random() < outside:
            b = (a + rng.randrange(1, N_AGENCIES)) % N_AGENCIES
        else:
            b = a
        pid = f"P{rng.randrange(50) * N_AGENCIES + b:04d}"
        calls.append((rng.uniform(t0, t1), user, pid, topic(rng.randrange(N_TOPICS))))
    return sorted(calls)


# ---------------------------------------------------------------- 46.1
say("== 46.1 the agency team's forked copy of the claims pipeline")
rng = random.Random(SEED)
day = agency_calls(rng, DAILY_CALLS["agency"])
fork_by_topic = {}
for s in FORK:
    key = s.policy_id or s.topic
    fork_by_topic.setdefault(key, []).append(s)
pf, d = platform()
stale = staff = other = claimfile = 0
plat_bad = 0
for t, user, pid, tp in day:
    cands = sorted(fork_by_topic.get(tp, []) + fork_by_topic.get(pid, []),
                   key=lambda s: (TIERS.index(s.tier), s.sid))
    kept, used = [], 0
    for s in cands:
        if used + s.tokens <= REQ_BUDGET["agency"]:
            kept.append(s); used += s.tokens
    sids = {s.sid for s in kept}
    stale += any(s.corpus == "forms" and int(tp[1:]) in CHANGED_SINCE_FORK for s in kept)
    staff += any(s.corpus == "guidelines" for s in kept)
    claimfile += any(s.corpus == "claim-files" for s in kept)
    other += any(s.policy_id and AGENCY_OF[s.policy_id] != user.split("-")[1].join(["A", ""]) for s in kept)
    pk, st = pf.assemble(issue_token(user, "agent-assist"), pid, tp, REQ_BUDGET["agency"], t)
    p = d.users[user]
    plat_bad += any(not entitled(s, p, CORP, d.agency_of)[0] or not s.current for s in pk.spans)
n = len(day)
R["fork"] = {"calls": n, "stale_edition": stale, "staff_guidance": staff,
             "other_book": other, "platform_bad": plat_bad}
say(f"agency calls in one day: {n:,}")
say(f"fork: cited a superseded edition as current   {stale:5d} ({stale/n:.1%})")
say(f"fork: carried adjuster-only guidance           {staff:5d} ({staff/n:.1%})")
say(f"fork: carried another agency's declarations    {other:5d} ({other/n:.1%})")
say(f"platform, same calls: packets with any of these {plat_bad}")

# ---------------------------------------------------------------- 46.2
say("\n== 46.2 assemble, explain, feedback")
pf, d = platform()
tok = issue_token("agent-07-2", "agent-assist")
pk, st = pf.assemble(tok, "P0407", "T03", 2000, 9.0)
ex = pf.explain(pk.packet_id)
say("status", st, "packet", pk.packet_id, "epoch", pk.epoch, "expires", pk.expires_at)
for sid, tier, corpus in ex["included"]:
    say(f"  included {sid:22s} {tier:10s} {corpus}")
for sid, why in ex["withheld"]:
    say(f"  withheld {sid:22s} {why}")
fb = pf.feedback(pk.packet_id, "missing", "agent expected the 2026 endorsement schedule")
say("feedback", fb)
R["api_example"] = {"included": ex["included"], "withheld": ex["withheld"],
                    "packet_id": pk.packet_id}
# explain completeness over a day of three teams
rng = random.Random(SEED + 1)
complete = total = 0
for team, role_user in (("claims", lambda r: f"adj-{r.randrange(60):02d}"),
                        ("policyholder", lambda r: f"ph-{r.randrange(2000):04d}"),
                        ("agency", lambda r: f"agent-{r.randrange(40):02d}-{r.randrange(5)}")):
    svc = d.users[role_user(random.Random(0))].service
    for _ in range(DAILY_CALLS[team] // 10):
        u = role_user(rng)
        subj = (sorted(d.users[u].policy_ids)[0] if team == "policyholder"
                else f"P{rng.randrange(2000):04d}")
        tp = topic(rng.randrange(N_TOPICS))
        pk, st = pf.assemble(issue_token(u, svc), subj, tp, REQ_BUDGET[team], 10.0)
        ex = pf.explain(pk.packet_id)
        cands = len(pf.by_topic[tp]) + len(pf.by_subject.get(subj, []))
        total += 1
        complete += len(ex["included"]) + len(ex["withheld"]) == cands
R["explain_complete"] = [complete, total]
say(f"explain accounts for every candidate: {complete}/{total}")

# ---------------------------------------------------------------- 46.3 load
say("\n== 46.3 a Monday 09:00-10:00 under the agency renewal batch")
rng = random.Random(SEED + 2)
reqs = []
per_hour = {"claims": 150, "policyholder": 900, "agency": 300}
for team, n in per_hour.items():
    for _ in range(n):
        reqs.append((rng.uniform(0, 3600), team, "interactive", rng.lognormvariate(-1.6, 0.45)))
base = list(reqs)
for k in range(RENEWAL_BATCH):
    reqs.append((rng.uniform(0, 60), "agency", "batch", rng.lognormvariate(-1.3, 0.35)))
shares = {t: b.concurrency for t, b in BUDGETS.items()}
WORKERS = 6
WINDOW = 600.0      # p95 over requests arriving 09:00-09:10, when the batch lands
res = {}
for label, rq, mode in (("quiet hour", base, "fifo"), ("batch, first come first served", reqs, "fifo"),
                        ("batch, per-team shares", reqs, "fair")):
    done = simulate(rq, WORKERS, mode, shares)
    row = {}
    for team in ("claims", "policyholder", "agency"):
        xs = [l for tm, k, l, ta in done if tm == team and k == "interactive" and ta < WINDOW]
        row[team] = round(pct(xs, 0.95), 3)
    bx = [ta + l for tm, k, l, ta in done if k == "batch"]
    row["batch_done_s"] = round(max([0] + [l for l in bx]), 1)
    res[label] = row
    say(f"{label:32s} p95 s: claims {row['claims']:.2f}  policyholder {row['policyholder']:.2f}"
        f"  agency {row['agency']:.2f}  batch finished at {row['batch_done_s']:.0f} s")
R["load"] = res

# ---------------------------------------------------------------- 46.3 month
say("\n== 46.3 budgets, quotas and chargeback over a 30-day month")
pf, d = platform()
rng = random.Random(SEED + 3)
users = {"claims": [u for u in d.users if u.startswith("adj-")],
         "policyholder": [u for u in d.users if u.startswith("ph-")],
         "agency": [u for u in d.users if u.startswith("agent-")]}
day_tokens = {}
for team in ("claims", "policyholder", "agency"):
    svc = d.users[users[team][0]].service
    tot = 0
    for _ in range(DAILY_CALLS[team] // 10):
        u = rng.choice(users[team])
        subj = (sorted(d.users[u].policy_ids)[0] if team == "policyholder"
                else f"P{rng.randrange(2000):04d}")
        pk, st = pf.assemble(issue_token(u, svc), subj, topic(rng.randrange(N_TOPICS)),
                             REQ_BUDGET[team], 9.0)
        tot += sum(s.tokens for s in pk.spans)
    day_tokens[team] = tot * 10
batch_tokens = int(day_tokens["agency"] / DAILY_CALLS["agency"] * RENEWAL_BATCH)
ledger = Ledger()
cum = {"soft": [], "hard": []}
refused_batch = {"soft": 0, "hard": 0}
for mode in ("soft", "hard"):
    used = 0
    for dday in range(30):
        monday = dday % 7 == 0
        want = day_tokens["agency"] + (batch_tokens if monday else 0)
        if mode == "hard" and monday and used + want > BUDGETS["agency"].monthly_tokens:
            refused_batch[mode] += 1          # batch refused; interactive traffic reserved
            want = day_tokens["agency"]
        used += want
        cum[mode].append(used)
for team in ("claims", "policyholder", "agency"):
    calls = DAILY_CALLS[team] * 30 + (RENEWAL_BATCH * 5 if team == "agency" else 0)
    toks = day_tokens[team] * 30 + (batch_tokens * 5 if team == "agency" else 0)
    ledger.add(Charge(team, toks, calls, 0))
bill_u = ledger.bill(PRICES, SHARED_MONTHLY, "usage")
bill_e = ledger.bill(PRICES, SHARED_MONTHLY, "even")
for b in (bill_u, bill_e):
    for t in b:
        b[t]["private"] = PRIVATE_STORAGE[t]
        b[t]["total"] += PRIVATE_STORAGE[t]
R["month"] = {"day_tokens": day_tokens, "batch_tokens_per_pass": batch_tokens,
              "quota": BUDGETS["agency"].monthly_tokens, "cum": cum,
              "refused_batch_days": refused_batch, "bill_usage": bill_u, "bill_even": bill_e}
say("packet tokens per day:", {k: f"{v:,}" for k, v in day_tokens.items()})
say(f"agency renewal pass: {batch_tokens:,} tokens; one pass on each of five Mondays")
over = next((i + 1 for i, v in enumerate(cum['soft']) if v > BUDGETS['agency'].monthly_tokens), None)
say(f"agency quota {BUDGETS['agency'].monthly_tokens:,}; soft cap month total {cum['soft'][-1]:,}"
    f" (crossed on day {over}); hard cap total {cum['hard'][-1]:,}, batch cut on {refused_batch['hard']} day(s)")
R["month"]["soft_crossed_day"] = over
for label, b in (("usage share", bill_u), ("even split", bill_e)):
    say(f"bill by {label}:")
    for t in b:
        say(f"  {t:13s} direct US${b[t]['direct']:>9,.0f}  shared US${b[t]['shared']:>8,.0f}"
            f"  private US${b[t]['private']:>5,.0f}  total US${b[t]['total']:>9,.0f}")

# ---------------------------------------------------------------- 46.4
say("\n== 46.4 where entitlement is enforced: 600 probes")
rng = random.Random(SEED + 4)
pf, d = platform()
probes = []
for _ in range(200):   # agents asking about policies outside their book
    a = rng.randrange(N_AGENCIES); b = (a + 1 + rng.randrange(N_AGENCIES - 1)) % N_AGENCIES
    probes.append(("agency", f"agent-{a:02d}-{rng.randrange(5)}", f"P{rng.randrange(50)*N_AGENCIES+b:04d}",
                   topic(rng.randrange(N_TOPICS)), None))
for _ in range(200):   # policyholders naming someone else's policy
    k = rng.randrange(2000); j = (k + 1 + rng.randrange(1999)) % 2000
    probes.append(("policyholder", f"ph-{k:04d}", f"P{j:04d}", topic(rng.randrange(N_TOPICS)), None))
for _ in range(200):   # an assistant asserting a different user in the body
    a = rng.randrange(N_AGENCIES)
    probes.append(("agency", f"agent-{a:02d}-0", f"P{rng.randrange(2000):04d}",
                   topic(rng.randrange(N_TOPICS)), f"adj-{rng.randrange(60):02d}"))


def caller_filter(team, p, s):
    """What each team's own filter checked when filtering was the caller's job."""
    if team == "policyholder":
        return s.policy_id is None or s.policy_id in p.policy_ids
    if team == "agency":
        return p.role in s.roles      # role only: no book-of-business check
    return True


designs = {}
for design in ("caller filters", "platform, identity in body", "platform, identity from token"):
    leaks = 0
    leaked_spans = 0
    for team, user, pid, tp, claimed in probes:
        real = d.users[user]
        if design == "caller filters":
            cands = pf.by_topic[tp] + pf.by_subject.get(pid, [])
            got = [s for s in cands if s.current and (CORP[s.corpus].shared or CORP[s.corpus].owner == team)]
            got = [s for s in got if caller_filter(team, real, s)]
        elif design == "platform, identity in body":
            asserted = d.users[claimed] if claimed else real
            pk, _ = pf.assemble(issue_token(asserted.user, asserted.service), pid, tp,
                                REQ_BUDGET[asserted.team], 9.0)
            got = pk.spans
        else:
            pk, _ = pf.assemble(issue_token(user, real.service), pid, tp, REQ_BUDGET[team], 9.0)
            got = pk.spans
        bad = [s for s in got if not entitled(s, real, CORP, d.agency_of)[0]]
        leaks += bool(bad)
        leaked_spans += len(bad)
    designs[design] = {"probes_leaking": leaks, "spans": leaked_spans}
    say(f"{design:32s} probes that leaked {leaks:4d}/600  spans {leaked_spans}")
R["entitlement"] = designs

# ---------------------------------------------------------------- 46.4 injection
say("\n== 46.4 failure injection: the agency service caches packets")
rng = random.Random(SEED + 5)
calls = agency_calls(rng, DAILY_CALLS["agency"], outside=0.0)
# Reuse is common: agents in one agency ask about the same few policies.
calls = [(t, u, f"P{(int(p[1:]) % (5 * N_AGENCIES)):04d}", tp) for t, u, p, tp in calls]
REVOKE_AT, REVOKED = 11.0, [f"agent-{a:02d}-4" for a in range(0, N_AGENCIES, 4)]
MOVE_AT = 13.0
moved = [f"P{k:04d}" for k in range(0, 5 * N_AGENCIES, 7)]
inj = {}
for design in ("client cache, no check", "client cache, gateway verifies",
               "platform cache by entitlement class"):
    pf, d = platform()
    cache = {}
    leaks = leak_calls = hits = assembles = refused = 0
    revoked_done = moved_done = False
    reasons = {}
    for t, user, pid, tp in calls:
        if not revoked_done and t >= REVOKE_AT:
            for u in REVOKED:
                d.revoke(u)
            revoked_done = True
        if not moved_done and t >= MOVE_AT:
            for p_ in moved:
                a = int(d.agency_of[p_][1:])
                d.move_policy(p_, f"A{(a + 1) % N_AGENCIES:02d}")
            moved_done = True
        real = d.users.get(user)
        if design == "platform cache by entitlement class":
            if real is None:
                refused += 1; continue
            key = (real.agency, real.role, d.epoch, pid, tp)
            pk = cache.get(key)
            if pk is not None and t - pk.issued_at < 24:
                hits += 1     # reuse the entitled span list; mint a packet for this caller
                pk = replace(pk, audience=user, issued_at=t, expires_at=t + pf.ttl)
                pk.sig = sign(pk, pf.key)
            else:
                pk, st = pf.assemble(issue_token(user, "agent-assist"), pid, tp, 2000, t)
                assembles += 1
                cache[key] = pk
            got = pk.spans
        else:
            key = (pid, tp)
            pk = cache.get(key)
            if pk is not None and t - pk.issued_at < 24:
                hits += 1
                if design == "client cache, gateway verifies":
                    ok, why = (verify_packet(pk, real, pid, t, d.epoch, pf.key)
                               if real else (False, "revoked"))
                    if not ok:
                        reasons[why] = reasons.get(why, 0) + 1
                        hits -= 1
                        pk, st = pf.assemble(issue_token(user, "agent-assist"), pid, tp, 2000, t)
                        assembles += 1
                        if pk is None:
                            refused += 1; continue
                        cache[key] = pk
            else:
                pk, st = pf.assemble(issue_token(user, "agent-assist"), pid, tp, 2000, t)
                assembles += 1
                if pk is None:
                    refused += 1; continue
                cache[key] = pk
            got = pk.spans
        if real is None:
            bad = list(got)        # a revoked user should receive nothing
            if bad:
                reasons["leak: revoked user"] = reasons.get("leak: revoked user", 0) + 1
        else:
            bad = [s for s in got if not entitled(s, real, CORP, d.agency_of)[0]]
            if bad:
                reasons["leak: policy moved"] = reasons.get("leak: policy moved", 0) + 1
        leaks += len(bad)
        leak_calls += bool(bad)
    inj[design] = {"calls": len(calls), "leak_calls": leak_calls, "leaked_spans": leaks,
                   "cache_hits": hits, "assembles": assembles, "refused": refused,
                   "reasons": reasons}
    say(f"{design:38s} leaking calls {leak_calls:4d}  spans {leaks:5d}  hits {hits:5d}"
        f"  assembles {assembles:5d}  refused {refused:3d}  {reasons}")
R["injection"] = inj
R["injection_setup"] = {"revoked_users": len(REVOKED), "moved_policies": len(moved),
                        "revoke_at": REVOKE_AT, "move_at": MOVE_AT}

# ---------------------------------------------------------------- 46.5
say("\n== 46.5 shared and private corpora: producer-guide notes on ten topics")
pf, d = platform()
above_fork = above_plat = 0
claims_see_pg = 0
fork_score = random.Random(SEED + 6)
for i in NOTE_TOPICS:
    tp = topic(i)
    # the fork ranked by similarity score; notes paraphrase the question, so they score high
    fk = sorted([s for s in FORK if s.topic == tp],
                key=lambda s: -(0.9 if s.tier == "team-note" else fork_score.uniform(0.6, 0.95)))
    order = [s.corpus for s in fk]
    above_fork += order.index("producer-guides") < order.index("forms")
    pk, _ = pf.assemble(issue_token("agent-01-1", "agent-assist"), "P0041", tp, 2000, 9.0)
    order = [s.corpus for s in pk.spans]
    above_plat += ("producer-guides" in order and order.index("producer-guides") < order.index("forms"))
    pk, _ = pf.assemble(issue_token("adj-03", "det-assist"), "P0041", tp, 3000, 9.0)
    claims_see_pg += any(s.corpus == "producer-guides" for s in pk.spans)
R["private"] = {"note_topics": len(NOTE_TOPICS), "note_above_form_fork": above_fork,
                "note_above_form_platform": above_plat, "claims_packets_with_notes": claims_see_pg}
say(f"note ranked above the governing form: fork {above_fork}/10, platform {above_plat}/10")
say(f"claims packets carrying the agency's private notes: {claims_see_pg}/10")

# ---------------------------------------------------------------- 46.6
say("\n== 46.6 versioning the contract: consumer tests against v2")
pf, d = platform()
pk, _ = pf.assemble(issue_token("agent-07-2", "agent-assist"), "P0407", "T03", 2000, 9.0)
tests = [
    ContractTest("claims", "every span has sid and tokens", lambda r: all("sid" in s and "tokens" in s for s in r["spans"])),
    ContractTest("claims", "packet id present", lambda r: bool(r["id"])),
    ContractTest("claims", "span ids unique", lambda r: len({s['sid'] for s in r['spans']}) == len(r['spans'])),
    ContractTest("policyholder", "source names a corpus", lambda r: all(s["source"] for s in r["spans"])),
    ContractTest("policyholder", "packet id present", lambda r: bool(r["id"])),
    ContractTest("agency", "spans sorted by sid", lambda r: [s["sid"] for s in r["spans"]] == sorted(s["sid"] for s in r["spans"])),
    ContractTest("agency", "total tokens within 2,000", lambda r: sum(s["tokens"] for s in r["spans"]) <= 2000),
    ContractTest("agency", "packet id present", lambda r: bool(r["id"])),
]
matrix = {}
for v in ("v1", "v2"):
    rr = run_contract_tests(tests, render(pk, v))
    matrix[v] = rr
    fails = [(c, n) for c, n, ok in rr if not ok]
    say(f"{v}: {sum(ok for *_, ok in rr)}/{len(rr)} pass; failing: {fails}")
R["contract"] = {v: [[c, n, ok] for c, n, ok in rr] for v, rr in matrix.items()}
# migration: v1 served through an adapter for 90 days; share of v1 calls by week
weeks = list(range(0, 14))
v1_share = []
moved_at = {"claims": 2, "policyholder": 6, "agency": 10}
for w in weeks:
    tot = sum(DAILY_CALLS.values())
    v1 = sum(DAILY_CALLS[t] for t, m in moved_at.items() if w < m)
    v1_share.append(round(v1 / tot, 3))
R["contract"]["v1_share_by_week"] = v1_share
say("v1 share of calls by week:", v1_share)

# ---------------------------------------------------------------- 46.7
say("\n== 46.7 platform SLOs over a simulated 28 days")
rng = random.Random(SEED + 7)
slos = [SLO("assemble availability (no server error)", 0.999),
        SLO("interactive p95 under 0.8 s, per hour", 0.99),
        SLO("bulletin visible within 2 h of publication", 0.95),
        SLO("entitlement correctness", 1.0, budgeted=False),
        SLO("explain completeness", 1.0, budgeted=False)]
calls_28 = sum(DAILY_CALLS.values()) * 28
errors = 0
burn = []
for dday in range(28):
    e = rng.randint(0, 12) + (220 if dday == 17 else 0)   # day 18: an index rebuild overran
    errors += e
    burn.append(errors)
hours = 28 * 10
slow_hours = 1     # 09:00 on the first Monday, before per-team shares were set
bulletins = 26
late = 1
evals = [slos[0].evaluate(calls_28 - errors, calls_28),
         slos[1].evaluate(hours - slow_hours, hours),
         slos[2].evaluate(bulletins - late, bulletins),
         slos[3].evaluate(600, 600),
         slos[4].evaluate(complete, total)]
for ev in evals:
    say(f"{ev['slo']:46s} {ev['good']:>7,}/{ev['total']:<7,} attained {ev['attained']:.4%}"
        f"  budget left {ev['budget_left']:+.0%}  met {ev['met']}")
R["slo"] = {"evals": evals, "error_burn": burn, "calls_28": calls_28,
            "allowed_errors": int((1 - 0.999) * calls_28)}

json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
say("\nwrote out/results.json")
