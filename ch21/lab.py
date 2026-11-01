"""Chapter 21 lab: give the context layer a past without poisoning its present.

Run from 04-code/:  python3 ch21/lab.py   (writes ch21/output.txt and ch21/out/*.json)
Stand-ins only (see standin21.py). The determination pipeline is Chapter 15's
Policy on the full mix of 1,000; Chapters 16 to 20 changed nothing that the
memory in this chapter touches, so the effect of memory is reported as a
change against that pipeline. Prices are illustrative, in US dollars (US$).
"""
from __future__ import annotations

import importlib.util, json, os, statistics, sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "ch15")]

import corpus15 as C15
import corpus21 as C
import standin21 as S
from contextlab.memory import (Candidate, Memory, MemoryRecord, ReadRequest, WriteEverything,
                               WritePolicy, read, tokens)
from contextlab.policy import Policy
from contextlab.retrieval.index import InvertedIndex, analyse, top

_spec = importlib.util.spec_from_file_location("lab15", os.path.join(ROOT, "ch15", "lab.py"))
L15 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(L15)
OUT = os.path.join(HERE, "out")
RATE = 0.00573 / 1609            # Chapter 9's blended US$ per token, held fixed
LINES = []


def say(s=""):
    LINES.append(s); print(s)


def pct(n, d=1000):
    return f"{100 * n / d:.1f}% ({n}/{d})"


# ---------------------------------------------------------------- the world
def world():
    dets = C15.full_mix()
    spans, ix, g = L15.build(dets)
    pol = Policy(list(spans.values()))
    people = C.claimants(dets)
    for i, d in enumerate(dets):                       # the replay: two weeks after go-live
        people[d.did].decided = C.GO_LIVE + timedelta(days=1 + i // 72)
    return dets, spans, ix, g, pol, people


def histories(people):
    eps, cands = {}, {}
    for did, c in people.items():
        if c.repeat:
            eps[did] = C.episodes(c)
            C.observe(c, eps[did])
            cands[did] = C.extract(c, eps[did])
    return eps, cands


def other_tenant(c, cand):
    """A claimant with policies under two brands: some history sits under the other one."""
    if c.other_tenant and C.h(cand.source_ref, "brand") < 0.4:
        return replace(cand, tenant=c.other_tenant)
    return cand


def build_memory(people, cands, policy):
    mem = Memory(policy=policy)
    for who, prefs in sorted(C.ADJUSTERS.items()):
        for fld, text in prefs:
            mem.write(Candidate(f"adjuster:{who}", "preference", f"{who}: {fld}: {text}", "adjuster",
                                f"settings:{who}", C.GO_LIVE - timedelta(days=7), "lakeshore", fld, text))
    for did in sorted(cands):
        for cd in sorted(cands[did], key=lambda x: (x.when, x.source_ref, x.ctype)):
            mem.write(other_tenant(people[did], cd))
    return mem


def lessons(mem):
    """Nightly consolidation at go-live: adjuster corrections become procedural lessons."""
    out = []
    for who, when, scope, text, cause in C.CORRECTIONS:
        t, sc, named = S.lesson(text, scope, cause)
        out.append((t, sc, named, who, when, text))
    for who, scope, text in C.GOOD:
        out.append((text, scope, None, who, C.GO_LIVE - timedelta(days=30), text))
    for t, sc, named, who, when, src in out:
        r = MemoryRecord(f"M{len(mem.records) + 1:05d}", "procedural", f"procedure:{'/'.join(sc)}",
                         t, "lakeshore", "consolidation", f"correction:{who}:{when.isoformat()}",
                         "consolidate-nightly", C.GO_LIVE, 0.85, scope=sc,
                         value=named)
        mem.put(r)
    return [r for r in mem.records.values() if r.kind == "procedural"]


def subjects(c, did=None):
    s = {f"claimant:{c.cid}", f"policy:{c.policy}"}
    return s | {f"adjuster:{C.adjuster_for(did)}"} if did else s


def relevance_for(question):
    q = set(analyse(question))
    def rel(r):
        if r.kind != "procedural":
            return 1.0
        return len(q & set(analyse(r.text))) / 4.0
    return rel


FLOOR = 0.5                      # a lesson must share two content terms with the question


def memory_block(mem, d, c):
    q = ReadRequest(subjects(c, d.did), (d.req.line, d.req.policy_state, C.family(d.req.cause)),
                    d.req.tenant, "adjuster", c.decided)
    rel = relevance_for(d.question)
    got = read(mem, q, rel)
    return [r for r in got if r.kind != "procedural" or rel(r) >= FLOOR]


# ---------------------------------------------------------------- sections
def s1(dets, spans, ix, g, pol, people, eps):
    say("== 1. The pipeline before memory, and the people behind the claims")
    base = L15.run(dets, spans, ix, g, "policy", policy=pol)
    sc = L15.score(dets, base, spans)
    rep = [d for d in dets if people[d.did].repeat]
    n_eps = sum(len(v) for v in eps.values())
    say(f"full mix, Chapter 15 Policy: wrong {pct(sc['wrong'])}, withheld {sc['withheld']}")
    say(f"determinations for repeat claimants: {len(rep)} of 1,000; prior claims "
        f"{sum(len(people[d.did].prior) for d in rep)}; episodes before go-live {n_eps}")
    kinds = Counter(e.kind for v in eps.values() for e in v)
    say("episodes by kind: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    return base, sc


def s2(cands):
    say("\n== 2. What the extractor proposes")
    allc = [x for v in cands.values() for x in v]
    by = Counter(x.ctype for x in allc)
    say(f"candidates: {len(allc)}; " + ", ".join(f"{k} {v}" for k, v in by.most_common()))
    return allc


def s3(dets, people, cands):
    say("\n== 3. Write everything, or write on purpose")
    res = {}
    for name, pol in (("write-everything", WriteEverything()), ("selective", WritePolicy())):
        mem = build_memory(people, cands, pol)
        stored = Counter(r.kind for r in mem.records.values())
        refused = Counter(x[5] for x in mem.write_log if x[4] is None)
        stale = infer = sens = toks = n = 0
        for d in dets:
            c = people[d.did]
            if not c.repeat:
                continue
            blk = memory_block(mem, d, c)
            n += 1; toks += sum(tokens(r.text) for r in blk)
            rem = {r.field: r.value for r in blk if r.field in ("deductible", "limit")}
            for fld, live in (("deductible", c.deductible[1]), ("limit", c.limit[1])):
                used = S.letter_number(d.did, fld, str(live), rem.get(fld))
                stale += used != str(live)
            infer += any("probably" in r.text or "likely" in r.text or "prone" in r.text
                         or "exaggerate" in r.text for r in blk)
            sens += any(w in r.text for r in blk for w in ("hospital", "money is tight", "divorce"))
        res[name] = {"records": len(mem.records), "by_kind": dict(stored), "refused": dict(refused),
                     "stale_numbers": stale, "packets_with_inference": infer,
                     "packets_with_sensitive": sens, "tokens": round(toks / n, 1)}
        say(f"{name}: records {len(mem.records)} {dict(sorted(stored.items()))}")
        if refused:
            say("  refused: " + "; ".join(f"{k}: {v}" for k, v in refused.most_common()))
        say(f"  over {n} repeat-claimant determinations: stale deductibles or limits in letters {stale}; "
            f"packets with a model inference {infer}; with a sensitive detail {sens}; "
            f"memory tokens per determination {toks / n:.1f}")
    return res


def s4(mem, people):
    say("\n== 4. Supersession: one current value per subject and field")
    sup = [r for r in mem.records.values() if r.superseded_by]
    by = Counter(r.field for r in sup)
    say(f"semantic records {sum(r.kind == 'semantic' for r in mem.records.values())}; "
        f"superseded {len(sup)}: " + ", ".join(f"{k} {v}" for k, v in by.most_common(4)))
    ex = next(r for r in sorted(sup, key=lambda r: r.mid) if r.field == "contact.channel")
    new = mem.records[ex.superseded_by]
    say(f"example: {ex.subject} {ex.written} '{ex.value}' ({ex.source_ref}) superseded by "
        f"{new.written} '{new.value}' ({new.source_ref})")
    return {"superseded": len(sup), "example": [ex.mid, ex.value, new.mid, new.value]}


def s5(mem):
    say("\n== 5. Consolidation: episodes into notes, corrections into lessons")
    per = defaultdict(list)
    for r in mem.records.values():
        if r.kind == "semantic":
            per[r.subject].append(r)
    notes = {s: S.consolidate_note(s, rs) for s, rs in per.items()}
    n_notes = sum(len(v) for v in notes.values())
    sample = sorted(f"{s}#{f}" for s, v in notes.items() for f in v)[:200]
    outcomes = Counter(S.review(n) for n in sample)
    les = lessons(mem)
    src_w = statistics.mean(len(c[3].split()) for c in C.CORRECTIONS)
    say(f"semantic records {sum(len(v) for v in per.values())} across {len(per)} subjects "
        f"-> {n_notes} current notes")
    say("review of 200 consolidated notes: " + ", ".join(f"{k} {v}" for k, v in outcomes.most_common()))
    say(f"procedural lessons written at go-live: {len(les)} (from {len(C.CORRECTIONS) + len(C.GOOD)} "
        f"corrections); not in the review sample")
    return {"notes": n_notes, "review": dict(outcomes), "lessons": len(les), "src_words": src_w}


def naive_read(mem, ix, d, c, k=8):
    sims = ix.bm25(c.name.split()[1] + " " + d.question)
    return [mem.records[m] for m in top(sims, k)]


def s6(dets, people, mem):
    say("\n== 6. Reading: route by kind and scope, then rank")
    ix = InvertedIndex()
    for r in mem.records.values():
        ix.add(r.mid, r.text)
    rows = {}
    for name in ("similarity only", "routed"):
        other = sup = brand = toks = n = 0; prec = []
        for d in dets:
            c = people[d.did]
            if not c.repeat:
                continue
            blk = naive_read(mem, ix, d, c) if name == "similarity only" else memory_block(mem, d, c)
            mine = [r for r in blk if r.subject in subjects(c, d.did) or r.kind == "procedural"]
            n += 1; toks += sum(tokens(r.text) for r in blk)
            other += len(mine) < len(blk); sup += any(r.superseded_by for r in blk)
            brand += any(r.tenant != d.req.tenant for r in blk)
            prec.append(len(mine) / len(blk) if blk else 1.0)
        rows[name] = {"other_claimant": other, "superseded": sup, "other_brand": brand,
                      "precision": round(statistics.mean(prec), 3), "tokens": round(toks / n, 1)}
        say(f"{name}: packets with another claimant's memory {other}; with a superseded value {sup}; "
            f"with another brand's memory {brand}; precision {statistics.mean(prec):.3f}; "
            f"tokens {toks / n:.1f}")
    return rows


def questions(people):
    qs = []
    for did in sorted(people):
        c = people[did]
        if not c.repeat:
            continue
        s = c.name.split()[1]
        for f in C.FIELDS:
            vals = c.facts.get(f)
            kind = "never-stored" if not vals else "update" if len(vals) > 1 else "stored"
            qs.append((kind, c, f, vals[-1][1] if vals else None, C.ASK[f].format(s=s)))
    pick = {"stored": 200, "never-stored": 60, "update": 40}
    out = []
    for kind, k in pick.items():
        pool = [q for q in qs if q[0] == kind]
        pool.sort(key=lambda q: C.h("q", q[1].cid, q[2]))
        out += pool[:k]
    return out


def s7(people, cands):
    say("\n== 7. Structured notes against a vector store of atoms (300 memory questions)")
    qs = questions(people)
    allc = [x for v in cands.values() for x in v]
    atoms = InvertedIndex()
    for i, x in enumerate(allc):
        atoms.add(str(i), x.text)
    routed = build_memory(people, cands, WritePolicy())
    rows = {}
    for backend in ("atoms, write everything", "atoms, routed", "structured notes"):
        tally, toks = Counter(), []
        for kind, c, f, truth, text in qs:
            subj = f"claimant:{c.cid}" if C.CTYPE_OF.get(f) else f"policy:{c.policy}"
            if backend == "atoms, write everything":
                sims = atoms.bm25(text); hits = top(sims, 3)
                toks.append(sum(tokens(allc[int(i)].text) for i in hits))
                best = allc[int(hits[0])] if hits and sims[hits[0]] > 6.0 else None
                ans = best.value if best and best.field == f and best.subject == subj else \
                    (best.text.split(": ")[-1] if best else None)
            elif backend == "atoms, routed":
                live = [r for r in routed.live() if r.subject == subj]
                ix = InvertedIndex()
                for r in live:
                    ix.add(r.mid, r.text)
                sims = ix.bm25(text) if live else {}
                hits = top(sims, 3)
                toks.append(sum(tokens(routed.records[m].text) for m in hits))
                best = routed.records[hits[0]] if hits and sims[hits[0]] > 1.5 else None
                ans = best.value if best and best.field else (best.text.split(": ")[-1] if best else None)
            else:
                page = [r for r in routed.live() if r.subject == subj and r.field]
                toks.append(sum(tokens(r.text) for r in page) + 12)
                cur = routed.current(subj, f)
                ans = cur.value if cur else None
                if cur and C.h("curator", c.cid, f) < 0.015:
                    ans = None                      # a curator mis-filed the fact on the page
            if kind == "never-stored":
                tally["abstained" if ans is None else "invented"] += 1
            else:
                old = [v for _, v, _ in (c.facts.get(f) or [])[:-1]]
                tally[f"{kind} " + ("right" if ans == truth else "stale" if ans in old else
                                    "missed" if ans is None else "wrong")] += 1
        right = tally["stored right"] + tally["update right"] + tally["abstained"]
        rows[backend] = dict(tally, score=round(100 * right / 300, 1),
                             tokens=round(statistics.mean(toks), 1))
        say(f"{backend}: score {100 * right / 300:.1f}; stored right {tally['stored right']}/200; "
            f"updates right {tally['update right']}/40 (stale {tally['update stale']}); "
            f"never-stored abstained {tally['abstained']}/60 (invented {tally['invented']}); "
            f"tokens per read {statistics.mean(toks):.1f}")
    pages = sum(tokens(r.text) for r in routed.live() if r.field) + 12 * len(
        {r.subject for r in routed.live() if r.field})
    hits = [r for r in routed.live() if r.field == "backflow_valve" and r.value == "backflow valve fitted"]
    say(f"across subjects ('which policies record a backflow valve'): reading every page costs "
        f"{pages:,} tokens; a field index returns {len(hits)} records, {sum(tokens(r.text) for r in hits):,} tokens")
    rows["cross_subject"] = {"pages_tokens": pages, "index_records": len(hits),
                             "index_tokens": sum(tokens(r.text) for r in hits)}
    return rows


BASE_MIN, SPREAD, HISTORY_MIN, MEMORY_MIN, WITHHELD_MIN = 0.6, 0.9, 1.2, 0.3, 10.0


def minutes(d, c, withheld, with_memory, blk):
    """Lookup-time proxy from illustrative constants; not a measurement of adjusters."""
    m = BASE_MIN + SPREAD * C.h("base", d.did)
    if c.repeat:
        m += MEMORY_MIN if with_memory and any(r.kind == "episodic" for r in blk) else HISTORY_MIN
    return m + (WITHHELD_MIN if withheld else 0.0)


def replay(dets, spans, ix, g, pol, people, mem, with_memory=True):
    new, moved, blocks = [], [], {}
    for d in dets:
        c = people[d.did]
        blk = memory_block(mem, d, c) if with_memory else []
        blocks[d.did] = blk
        named = [(r.mid, r.value) for r in blk if r.kind == "procedural"]
        cause, changed = S.cause_reading(d.did, d.req.cause, named)
        if changed:
            moved.append(d.did)
        new.append(replace(d, req=replace(d.req, cause=cause)) if changed else d)
    res = L15.run(new, spans, ix, g, "policy", policy=pol)
    return new, res, blocks, moved


def s8(dets, people, base, mem, res_on, blocks):
    say("\n== 9. Lookup time on repeat claimants (proxy from illustrative constants)")
    out = {}
    for label, rr, wm in (("before memory", base, False), ("with memory", res_on, True)):
        rep, first, allm = [], [], []
        for d, x in zip(dets, rr):
            c = people[d.did]
            m = minutes(d, c, x["status"] == "withheld", wm, blocks[d.did] if wm else [])
            (rep if c.repeat else first).append(m); allm.append(m)
        out[label] = {"repeat_median": round(statistics.median(rep), 2),
                      "first_median": round(statistics.median(first), 2),
                      "mean_all": round(statistics.mean(allm), 2),
                      "median_all": round(statistics.median(allm), 2)}
        say(f"{label}: repeat claimants median {statistics.median(rep):.1f} min; first-time "
            f"{statistics.median(first):.1f} min; full mix median {statistics.median(allm):.1f}, "
            f"mean {statistics.mean(allm):.2f}")
    return out


def s9(dets, spans, ix, g, pol, people, mem, base_sc):
    say("\n== 8. Two weeks of determinations with memory on")
    new, res, blocks, moved = replay(dets, spans, ix, g, pol, people, mem, True)
    sc = L15.score(new, res, spans)
    added = sorted(set(sc["wrong_ids"]) - set(base_sc["wrong_ids"]))
    gone = sorted(set(base_sc["wrong_ids"]) - set(sc["wrong_ids"]))
    off_new, off_res, _, _ = replay(dets, spans, ix, g, pol, people, mem, False)
    off = L15.score(off_new, off_res, spans)
    toks = statistics.mean(sum(tokens(r.text) for r in b) for b in blocks.values())
    with_mem = sum(1 for b in blocks.values() if b)
    say(f"memory on: wrong {pct(sc['wrong'])}, withheld {sc['withheld']}; change {sc['wrong'] - base_sc['wrong']:+d}"
        f" (newly wrong {len(added)}, newly right {len(gone)})")
    say(f"memory off: wrong {pct(off['wrong'])}, withheld {off['withheld']}")
    say(f"determinations with a memory block {with_mem}; memory tokens per determination {toks:.1f}; "
        f"cost US${toks * RATE * 1000:.3f} per 1,000")
    byc = Counter(next(d for d in dets if d.did == x).req.cause for x in added)
    bys = Counter(next(d for d in dets if d.did == x).req.policy_state for x in added)
    say("newly wrong by recorded cause at first notice: " + ", ".join(f"{k} {v}" for k, v in byc.most_common()))
    say("newly wrong by state: " + ", ".join(f"{k} {v}" for k, v in bys.most_common()))
    say("newly wrong: " + " ".join(added))
    return {"wrong": sc["wrong"], "withheld": sc["withheld"], "added": added, "gone": gone,
            "off_wrong": off["wrong"], "tokens": round(toks, 1), "with_memory": with_mem,
            "moved": moved, "res": res, "new": new, "blocks": blocks}


def s10(r9, pol):
    say("\n== 10. What the audit record says about memory (v1)")
    did = r9["added"][0]
    i = next(i for i, d in enumerate(r9["new"]) if d.did == did)
    d, x, blk = r9["new"][i], r9["res"][i], r9["blocks"][did]
    rec = {"determination": did, "cause": d.req.cause, "cited": x["cited"],
           "evidence": x["packet"][:3] + ["..."], "memory": {"records": len(blk),
           "tokens": sum(tokens(r.text) for r in blk)}}
    say(json.dumps(rec))
    say("the write log holds every record's source; nothing records which records were read")
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    dets, spans, ix, g, pol, people = world()
    eps, cands = histories(people)
    base, base_sc = s1(dets, spans, ix, g, pol, people, eps)
    allc = s2(cands)
    r3 = s3(dets, people, cands)
    mem = build_memory(people, cands, WritePolicy())
    r4 = s4(mem, people)
    r5 = s5(mem)
    r6 = s6(dets, people, mem)
    r7 = s7(people, cands)
    r9 = s9(dets, spans, ix, g, pol, people, mem, base_sc)
    r8 = s8(dets, people, base, mem, r9["res"], r9["blocks"])
    r10 = s10(r9, pol)
    keep = {k: v for k, v in r9.items() if k not in ("res", "new", "blocks")}
    json.dump({"s2": Counter(x.ctype for x in allc), "s3": r3, "s4": r4, "s5": r5, "s6": r6,
               "s7": r7, "s8": r8, "s9": keep, "s10": r10, "base_wrong": base_sc["wrong"]},
              open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, "output.txt"), "w").write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
