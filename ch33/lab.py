"""Chapter 33 lab: compression without making the packet wrong.

Note: deterministic stand-in. No language model runs (see lakeshore33.py
and agent33.py for what stands in for what, and the stated constants).
Price constants are the series' stand-in values. Four characters per token.
Run: python3 lab.py > output.txt   (about a minute)
"""
import json
import pathlib
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import lakeshore33 as L          # noqa: E402
import agent33 as A              # noqa: E402
from contextlab import compress as C   # noqa: E402

PRICE_IN, PRICE_OUT = 3.00, 15.00        # US$ per million tokens (stand-in)
CALLS = 3                                 # model calls per determination that read the packet
RATIOS = (0.5, 0.33, 0.2)
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
R = {}
EVIDENCE = {"declarations", "policy", "endorsement"}
HISTORY = {"notes", "letters"}


def section(t):
    print(f"\n== {t}")


def texts(pk):
    return [s.text for s in pk.sents]


# ------------------------------------------------------------- compressors
def extractive(pk, r, protect=False):
    score = L.ext_score(pk, L.qterms(pk))
    if not protect:
        return [s.text for s in C.extract(pk.sents, score, r * pk.tokens)]
    return [s.text for s in C.extract(pk.sents, score, r * pk.tokens, guard(pk), link(pk))]


def guard(pk):
    p = pk.peril
    return lambda s: s.kind in EVIDENCE and (C.negation_profile(s.text) > 0 or L.money(s.text)
                                             or " days" in s.text) \
        and (p["form"] in s.text or p["kw"] in s.text.lower() or "notice" in s.text
             or "Coverage A" in s.text)


def link(pk):
    dev = pk.peril["dev"]
    return lambda a, b: b.kind == "notes" and (
        (dev in a.text and dev in b.text and "nspection" in b.text)
        or ("notice" in a.text and ("occurred" in b.text or "reported" in b.text))
        or ("agreed" in b.text and L.money(b.text) is not None))


def pruned(pk, r, force=False, items=None):
    items = items if items is not None else texts(pk)
    total = sum(C.tokens(x) for x in items)
    f = L.FORCE if force else ()
    lo, hi = 0.05, 1.0
    for _ in range(7):
        m = (lo + hi) / 2
        if sum(C.tokens(x) for x in C.prune_packet(items, m, L.token_score, f)) > r * total:
            hi = m
        else:
            lo = m
    return C.prune_packet(items, lo, L.token_score, f)


def abstractive(pk, r, guided=False, invert_rate=L.P_INVERT, seed="a"):
    return L.abstractive(pk, r, guided=guided, invert_rate=invert_rate, seed=seed)


def run(packets, fn):
    c, ratio, cites = Counter(), 0.0, 0
    for pk in packets:
        out = fn(pk)
        c[L.grade(pk, out)] += 1
        ratio += sum(C.tokens(x) for x in out) / pk.tokens
        orig = {s.text for s in pk.sents if s.handle in pk.facts.values()}
        cites += sum(x in orig for x in out) / len(orig)
    n = len(packets)
    return dict(ratio=round(ratio / n, 3), right=c["right"], wrong=c["wrong"],
                withheld=c["withheld"], cited=round(100 * cites / n, 1))


# ------------------------------------------------------------- survival and probes
FACTMAP = dict(exclusion="excl", exception="exc", limit="limit", deductible="ded",
               condition="cond", finding="maint", loss="loss", d_loss="d_loss",
               d_rep="d_rep", notice="notice")


def survival(pk, out):
    full, got = L.read_facts(texts(pk), pk.peril), L.read_facts(out, pk.peril)
    res = {}
    for kind, k in FACTMAP.items():
        if kind not in pk.facts:
            continue
        if got[k] is None or (k in ("exc", "notice") and not got[k] and full[k]):
            res[kind] = "dropped"
        elif got[k] != full[k]:
            res[kind] = "altered"
        else:
            res[kind] = "kept"
    return res


def probes(pk):
    p = pk.peril
    ask = lambda k: (lambda t: L.read_facts(t, p)[k])
    return [C.Probe(n, ask(k)) for n, k in [
        ("what is the limit", "limit"), ("what is the deductible", "ded"),
        ("is the peril excluded", "excl"), ("does an endorsement restore it", "exc"),
        ("does a maintenance failure void cover", "cond"), ("was the device maintained", "maint"),
        ("when did the loss occur", "d_loss"), ("when was it reported", "d_rep"),
        ("is there a notice period", "notice"), ("what amount was agreed", "loss"),
        ("what is the Coverage A limit", "cov_a")]]


def main():
    P = L.build(200)
    section("1. Two hundred packets (stand-in)")
    gold = Counter(p.gold()[0] if p.gold()[0] == "pay" else p.gold()[1] for p in P)
    mean = sum(p.tokens for p in P) / len(P)
    share = {k: round(100 * sum(s.tokens for p in P for s in p.sents if s.kind == k)
                      / sum(p.tokens for p in P), 1) for k in ("declarations", "policy",
                                                               "endorsement", "notes", "letters")}
    base = run(P, texts)
    print(f"mean packet {mean:.0f} tokens; gold {dict(gold)}; token share by source {share}")
    print(f"uncompressed: {base}")
    R["corpus"] = dict(mean=round(mean), gold=dict(gold), share=share, base=base)

    section("2. Three compressors at three ratios (whole packet)")
    R["compare"] = {}
    for name, fn in [("extractive", lambda p, r: extractive(p, r)),
                     ("token pruning", lambda p, r: pruned(p, r)),
                     ("abstractive", lambda p, r: abstractive(p, r))]:
        for r in RATIOS:
            res = run(P, lambda p: fn(p, r))
            R["compare"][f"{name}@{r}"] = res
            print(f"{name:14s} target {r:<4}: {res}")

    section("3. What survived, by fact (ratio 0.33)")
    R["survival"] = {}
    for name, fn in [("extractive", lambda p: extractive(p, 0.33)),
                     ("token pruning", lambda p: pruned(p, 0.33)),
                     ("abstractive", lambda p: abstractive(p, 0.33))]:
        tally = {k: Counter() for k in FACTMAP}
        for p in P:
            for k, v in survival(p, fn(p)).items():
                tally[k][v] += 1
        R["survival"][name] = {k: dict(v) for k, v in tally.items()}
        print(name)
        for k, v in tally.items():
            n = sum(v.values())
            print(f"   {k:11s} kept {v['kept']:3d}  altered {v['altered']:3d}  dropped {v['dropped']:3d}  (of {n})")

    section("4. Failure injection: a summary that inverts the negation (ratio 0.5)")
    clean = run(P, lambda p: abstractive(p, 0.5, guided=True, invert_rate=0.0, seed="inj"))
    inj = run(P, lambda p: abstractive(p, 0.5, guided=True, invert_rate=6.0, seed="inj"))
    print(f"guided summary, no inversion : {clean}")
    print(f"guided summary, inversion    : {inj}")
    inverted = wrong_inv = latent = by_q = by_pol = by_probe = 0
    fp_pol = fp_probe = 0
    for p in P:
        a = abstractive(p, 0.5, guided=True, invert_rate=0.0, seed="inj")
        if "condition" in p.facts:
            cond_o = [s.text for s in p.sents if s.handle == p.facts["condition"]][0]
            ca = [x for x in a if "fail" in x and p.peril["dev"] in x]
            fp_pol += bool(ca) and C.polarity_diff(cond_o, ca[0]) > 0
        fp_probe += bool(C.probe(texts(p), a, probes(p)))
    for p in P:
        if "condition" not in p.facts:
            continue
        a = abstractive(p, 0.5, guided=True, invert_rate=0.0, seed="inj")
        b = abstractive(p, 0.5, guided=True, invert_rate=6.0, seed="inj")
        if a == b:
            continue
        inverted += 1
        wrong = L.grade(p, b) == "wrong" and L.grade(p, a) == "right"
        wrong_inv += wrong
        latent += not wrong
        by_q += L.determine(texts(p), p.peril) != L.determine(b, p.peril)
        cond_o = [s.text for s in p.sents if s.handle == p.facts["condition"]][0]
        cond_b = [x for x in b if "fail" in x and p.peril["dev"] in x]
        by_pol += bool(cond_b) and C.polarity_diff(cond_o, cond_b[0]) > 0
        by_probe += bool(C.probe(texts(p), b, probes(p)))
    print(f"summaries with the condition inverted: {inverted}; determinations made wrong: {wrong_inv}; "
          f"latent (right today, wrong summary): {latent}")
    print(f"flagged by: re-asking the question at hand {by_q}; polarity check {by_pol}; probe suite {by_probe}")
    print(f"on the 200 clean summaries: polarity check flags {fp_pol}; probe suite flags {fp_probe} "
          f"(those summaries dropped or changed some other fact)")
    R["inject"] = dict(clean=clean, inj=inj, inverted=inverted, wrong=wrong_inv, latent=latent,
                       by_question=by_q, by_polarity=by_pol, by_probe=by_probe,
                       fp_polarity=fp_pol, fp_probe=fp_probe)

    section("5. The fixes (ratio 0.33)")
    R["fixes"] = {}
    for name, fn in [("extractive, clause groups", lambda p: extractive(p, 0.33, True)),
                     ("token pruning, force-keep", lambda p: pruned(p, 0.33, True)),
                     ("token pruning, force-keep @0.5", lambda p: pruned(p, 0.5, True)),
                     ("abstractive, guided", lambda p: abstractive(p, 0.33, guided=True))]:
        res = run(P, fn)
        R["fixes"][name] = res
        print(f"{name:32s}: {res}")
    for r in (0.5, 0.2):
        res = run(P, lambda p: extractive(p, r, True))
        R["fixes"][f"extractive, clause groups @{r}"] = res
        print(f"{'extractive, clause groups @' + str(r):32s}: {res}")

    section("6. Compressing history versus compressing evidence")
    R["scope"] = {}
    for name, fn in [("nothing compressed", texts),
                     ("everything abstractive 0.33", lambda p: abstractive(p, 0.33)),
                     ("history abstractive, evidence verbatim", lambda p: scoped(p, "abs")),
                     ("history pruned, evidence verbatim", lambda p: scoped(p, "prune")),
                     ("history: facts lifted + summary; evidence clause groups", lambda p: shipped(p))]:
        res = run(P, fn)
        R["scope"][name] = res
        print(f"{name:55s}: {res}")

    section("7. Testing a summary before trusting it (probe gate on the shipped policy)")
    gate = Counter()
    for p in P:
        out = shipped(p)
        fails = C.probe(texts(p), out, probes(p))
        gate["fail" if fails else "pass"] += 1
    abs_fail = sum(bool(C.probe(texts(p), scoped(p, "abs"), probes(p))) for p in P)
    abs_wrong = R["scope"]["history abstractive, evidence verbatim"]
    print(f"shipped policy: {dict(gate)}; history-abstractive policy: {abs_fail} of 200 fail probes "
          f"(wrong {abs_wrong['wrong']}, withheld {abs_wrong['withheld']})")
    R["gate"] = dict(shipped=dict(gate), abs_fail=abs_fail)

    section("8. Measuring what survived (shipped policy)")
    tally = Counter()
    for p in P:
        for k, v in survival(p, shipped(p)).items():
            tally[v] += 1
    ratio = sum(sum(C.tokens(x) for x in shipped(p)) / p.tokens for p in P) / len(P)
    print(f"facts: {dict(tally)}; decisions unchanged {R['scope'][list(R['scope'])[-1]]['right']} of 200; "
          f"ratio {ratio:.3f}")
    R["measure"] = dict(facts=dict(tally), ratio=round(ratio, 3))
    curve = {}
    for name, fn in [("extractive", lambda p, r: extractive(p, r)),
                     ("clause groups", lambda p, r: extractive(p, r, True)),
                     ("abstractive", lambda p, r: abstractive(p, r)),
                     ("guided abstractive", lambda p, r: abstractive(p, r, guided=True))]:
        curve[name] = []
        for r in RATIOS:
            k = n = 0
            for p in P:
                for v in survival(p, fn(p, r)).values():
                    k += v == "kept"; n += 1
            curve[name].append(round(100 * k / n, 1))
        print(f"fact survival % at {RATIOS}: {name:18s} {curve[name]}")
    R["curve"] = curve

    section("9. Long-running agent: compaction, trimming, isolation (200 episodes)")
    E = A.episodes(200)
    R["agent"] = []
    for fam, pin, off in [("full", False, False), ("threshold", False, False), ("threshold", True, False),
                          ("threshold", True, True), ("on_demand", True, False), ("on_demand", True, True), ("trim", False, False),
                          ("trim", True, False), ("isolate", False, False), ("isolate", True, False)]:
        s = A.summarise(E, fam, pin, off)
        R["agent"].append(s)
        print(f"{fam:10s} pin={pin!s:5s} offload={off!s:5s}: peak {s['peak']:6d}  input/episode "
              f"{s['total']:8d}  violations {s['violations']:3d}  success {s['success']}%")
    EA = A.episodes(200, attack_share=1.0)
    att, attp = A.summarise(EA, "threshold"), A.summarise(EA, "threshold", True)
    print(f"eviction attack in every episode: threshold violations {att['violations']}; pinned {attp['violations']}")
    R["attack"] = dict(plain=att["violations"], pinned=attp["violations"],
                       base=A.summarise(E, "threshold")["violations"])

    section("10. Lakeshore: the full 1,000-determination mix")
    M = L.build(1000, seed=3310, residual=("D0150", "D0412", "D0771", "D0938"))
    lake = {}
    for name, fn, summ in [("Chapter 32 packet (uncompressed)", texts, False),
                           ("shipped compression policy", shipped, True),
                           ("incident: endorsements over 400 tokens summarised", incident, True),
                           ("incident config with probe gate", lambda p: gated(p, incident, shipped), True),
                           ("shipped policy with probe gate", lambda p: gated(p, shipped), True)]:
        c, tok, out_tok, fallbacks = Counter(), 0, 0, 0
        for p in M:
            out = fn(p)
            c[L.grade(p, out)] += 1
            tok += sum(C.tokens(x) for x in out)
            if summ:
                hist = sum(s.tokens for s in p.sents if s.kind in HISTORY)
                out_tok += 0.2 * hist
        n = len(M)
        cost = (tok / n * CALLS * PRICE_IN + (out_tok / n) * PRICE_OUT
                + (sum(s.tokens for p in M for s in p.sents if s.kind in HISTORY) / n * PRICE_IN if summ else 0)) / 1e6
        lake[name] = dict(tokens=round(tok / n), wrong=c["wrong"], withheld=c["withheld"],
                          cost_per_1000=round(cost * 1000, 2))
        print(f"{name:52s}: {lake[name]}")
    R["lakeshore"] = lake
    summarised = [p for p in M if sum(s.tokens for s in p.sents if s.kind == "endorsement") > LONG_END]
    inv = sum(any("applies even if" in x for x in incident(p)) for p in summarised)
    new_wrong = [p.pid for p in summarised if L.grade(p, incident(p)) == "wrong"
                 and L.grade(p, shipped(p)) == "right"]
    why = Counter()
    for pid in new_wrong:
        p = next(q for q in M if q.pid == pid)
        sv = survival(p, incident(p))
        why[",".join(sorted(k for k, v in sv.items() if v != "kept"))] += 1
    pf = sum(bool(C.probe(texts(p), incident(p), probes(p))) for p in summarised)
    print(f"incident: {len(summarised)} endorsement spans summarised; {inv} conditions inverted; "
          f"{len(new_wrong)} newly wrong {new_wrong}; causes {dict(why)}; probe failures {pf}")
    R["incident"] = dict(summarised=len(summarised), inverted=inv, new_wrong=new_wrong,
                         causes=dict(why), probe_fail=pf)
    json.dump(R, open(OUT / "results.json", "w"), indent=1, default=str)


def scoped(p, how):
    ev = [s.text for s in p.sents if s.kind in EVIDENCE]
    hist = [s for s in p.sents if s.kind in HISTORY]
    if how == "abs":
        return L.abstractive(p, 0.33, only=HISTORY)
    return ev + pruned(p, 0.33, False, [s.text for s in hist])


FACT_CUES = ("occurred", "reported", "Inspection", "agreed")


def shipped(p):
    """Evidence: clause groups at 0.45 of evidence tokens. History: lift the
    sentences that carry facts verbatim; summarise the rest; drop letters."""
    evs = [s for s in p.sents if s.kind in EVIDENCE]
    score = L.ext_score(p, L.qterms(p))
    budget = 0.45 * sum(s.tokens for s in evs)
    ev = [s.text for s in C.extract(evs, score, budget, guard(p))]
    notes = [s for s in p.sents if s.kind == "notes"]
    lifted = [s.text for s in notes if any(c in s.text for c in FACT_CUES)]
    chatter = sum(s.tokens for s in notes) - sum(C.tokens(x) for x in lifted)
    summary, used = [], 0
    for t in L.FILLER_SUMMARY:
        if used >= 0.2 * chatter:
            break
        summary.append(t); used += C.tokens(t)
    return ev + lifted + summary


LONG_END = 195      # stand-in tokens (about the longest fifth); the production rule said 400


def incident(p):
    """The shipped policy plus a config change: any long endorsement span goes
    through the history summariser (guided prompt) instead of clause groups."""
    base = shipped(p)
    end = [s for s in p.sents if s.kind == "endorsement"]
    if sum(s.tokens for s in end) <= LONG_END:
        return base
    summ = L.abstractive(p, 0.4, guided=True, invert_rate=0.9, seed="inc", only={"endorsement"})
    end_txt = {s.text for s in end}
    new_end = [x for x in summ if x not in {s.text for s in p.sents if s.kind != "endorsement"}]
    return [x for x in base if x not in end_txt] + new_end


def gated(p, fn, fallback=None):
    """Probe gate: a compressed packet that changes any probe answer is not
    used; the packet falls back to the shipped policy, or to verbatim."""
    out = fn(p)
    if not C.probe(texts(p), out, probes(p)):
        return out
    if fallback is not None:
        alt = fallback(p)
        if not C.probe(texts(p), alt, probes(p)):
            return alt
    return texts(p)


if __name__ == "__main__":
    main()
