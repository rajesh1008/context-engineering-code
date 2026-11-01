"""Chapter 10 lab: parse 300 Lakeshore pages five ways, cut them nine ways,
and measure which units the retriever can rank and the adjuster can cite.

Run: python 04-code/ch10/make_corpus.py && python 04-code/ch10/ocr_all.py
     python 04-code/ch10/lab.py            (writes 04-code/ch10/out/results.json)
Real: the PDFs, the text-layer, layout and OCR parsers, every chunker, BM25,
the timings of the three real parsers. Stand-ins (labelled): the vision parser,
the vector retriever and late chunking, and the scoreboard link to Chapter 9.
"""
from __future__ import annotations
import json, os, re, sys, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from contextlab.parsing import (Block, ParsedDoc, parse_layout, parse_ocr, parse_text_layer,
                                parse_vision_standin, route_page, furniture_key, VISION_USD_PER_PAGE)
from contextlab.chunker import Chunker, count_tokens, elements
from contextlab.chunkeval import BM25, VectorStandIn, ranked, evidence_hit, cited, words, coverage
import questions as Q

CORPUS, OUT = os.path.join(HERE, "corpus"), os.path.join(HERE, "out")
TRUTH = json.load(open(os.path.join(CORPUS, "truth.json")))
DOCS = {d["doc_id"]: d for d in TRUTH["docs"]}
import hashlib
for _d in TRUTH["docs"]:                      # stage 1: checksum every source file
    _d["sha256"] = hashlib.sha256(open(os.path.join(CORPUS, _d["doc_id"] + ".pdf"), "rb").read()).hexdigest()
TPAGES = defaultdict(list)
for p in TRUTH["pages"]:
    TPAGES[p["doc_id"]].append(p)
_OC = os.path.join(CORPUS, "ocr_cache.json")
OCR = json.load(open(_OC)) if os.path.exists(_OC) else {"lines": {}, "seconds": {}}
QS = Q.build(TRUTH)
R: dict = {}


def pdf(doc_id):
    return os.path.join(CORPUS, doc_id + ".pdf")


# ------------------------------------------------------------------ 1. parse five ways
def parse_all():
    out = {m: {} for m in ("text_layer", "layout", "ocr", "vision", "routed")}
    for d in TRUTH["docs"]:
        i = d["doc_id"]
        out["text_layer"][i] = parse_text_layer(pdf(i), i)
        out["layout"][i] = parse_layout(pdf(i), i)
        o = parse_ocr(pdf(i), i, cached=OCR["lines"][i]); o.seconds = OCR["seconds"][i]
        out["ocr"][i] = o
        out["vision"][i] = parse_vision_standin(TPAGES[i], i)
        r = ParsedDoc(i, "routed")
        routes = []
        for tp in TPAGES[i]:
            has_table = any(l["kind"] == "table" for l in tp["lines"])
            m = route_page(not d["scanned"], has_table)
            routes.append(m)
            r.blocks += [b for b in out[m][i].blocks if b.page == tp["page"]]
            r.seconds += out[m][i].seconds / d["pages"]
            r.usd += VISION_USD_PER_PAGE if m == "vision" else 0.0
        r.pages = d["pages"]; r.routes = routes
        out["routed"][i] = r
    return out


def truth_page_text(tp, body_only=True):
    parts = []
    for l in tp["lines"]:
        parts.append(" ".join(" ".join(r) for r in l["rows"]) if l["kind"] == "table" else l["text"])
    return " ".join(parts)


def parse_quality(parsed):
    res = {}
    for m, docs in parsed.items():
        f1s, orders, leaks, rows_ok, rows_n, empty, secs, usd = [], [], 0, 0, 0, 0, 0.0, 0.0
        for d in TRUTH["docs"]:
            pd = docs[d["doc_id"]]
            secs += pd.seconds; usd += pd.usd
            for tp in TPAGES[d["doc_id"]]:
                got_blocks = [b for b in pd.blocks if b.page == tp["page"]]
                got = " ".join(b.text for b in got_blocks)
                if not got.strip():
                    empty += 1
                tw, gw = Counter(words(truth_page_text(tp))), Counter(words(got))
                inter = sum((tw & gw).values())
                p = inter / max(1, sum(gw.values())); r = inter / max(1, sum(tw.values()))
                f1s.append(2 * p * r / (p + r) if p + r else 0.0)
                tb = [tuple(x) for l in tp["lines"] if l["kind"] != "table" for x in zip(words(l["text"]), words(l["text"])[1:])]
                gws = words(got)
                gb = set(zip(gws, gws[1:]))
                orders.append(sum(b in gb for b in tb) / max(1, len(tb)))
                if furniture_key(tp["footer"]) in {furniture_key(b.text) for b in got_blocks} or \
                        tp["footer"].split()[0] in got[:80] and "Page" in got:
                    leaks += 1
                flat = re.sub(r"\s+", " ", " ".join(b.text for b in got_blocks))
                for l in tp["lines"]:
                    if l["kind"] == "table":
                        for row in l["rows"]:
                            rows_n += 1
                            pat = r"\W{0,6}".join(re.escape(c) for c in row)
                            rows_ok += re.search(pat, flat) is not None
        n = len(f1s)
        res[m] = dict(pages=n, word_f1=sum(f1s) / n, order=sum(orders) / n, footer_leak_pages=leaks,
                      empty_pages=empty, table_rows_intact=rows_ok / max(1, rows_n), table_rows=rows_n,
                      seconds=secs, sec_per_page=secs / n, usd=usd)
        if m == "routed":
            res[m]["routes"] = Counter(x for d in TRUTH["docs"] for x in docs[d["doc_id"]].routes)
    return res


# ------------------------------------------------------------------ 2. chunk and retrieve
def chunk_all(parsed_docs, chunker):
    out = []
    for d in TRUTH["docs"]:
        out += chunker.chunk(parsed_docs[d["doc_id"]], d)
    return out


def allowed_fn(policy):
    return lambda c: Q.policy_allows(policy, DOCS[c.doc_id])


def evaluate(chunks, retriever="bm25", ks=(1, 3, 5, 10), late=False, qs=None):
    index = BM25(chunks) if retriever == "bm25" else VectorStandIn(chunks, late=late)
    qs = qs or QS
    hits = {k: 0 for k in ks}
    cites = {k: 0 for k in ks}
    per_q, packet_tokens, group = {}, [], defaultdict(lambda: [0, 0, 0])
    for q in qs:
        r = ranked(index, q["text"], allowed_fn(q["policy"]))
        rank = next((i + 1 for i, c in enumerate(r[:50]) if evidence_hit(c, q["gold"])), None)
        crank = next((i + 1 for i, c in enumerate(r[:50]) if cited(c, q["gold"])), None)
        for k in ks:
            cites[k] += crank is not None and crank <= k
        per_q[q["qid"]] = dict(rank=rank, cite_rank=crank, top=r[0].citation() if r else None, top_id=r[0].chunk_id if r else None)
        for k in ks:
            hits[k] += rank is not None and rank <= k
        packet_tokens.append(sum(c.tokens + count_tokens(c.prefix) for c in r[:5]))
        group[q["group"]][0] += rank == 1; group[q["group"]][1] += 1
        group[q["group"]][2] += rank is not None and rank <= 10
    n = len(qs)
    return dict(hit={k: hits[k] / n for k in ks}, cite={k: cites[k] / n for k in ks}, n=n, packet5=sum(packet_tokens) / n,
                chunks=len(chunks), mean_tokens=sum(c.tokens for c in chunks) / max(1, len(chunks)),
                sectioned=sum(bool(c.path) for c in chunks) / max(1, len(chunks)),
                groups={g: v[0] / v[1] for g, v in group.items()},
                groups10={g: v[2] / v[1] for g, v in group.items()}, per_q=per_q)


CHUNKERS = {
    "fixed-64": Chunker("fixed", 64), "fixed-128": Chunker("fixed", 128), "fixed-256": Chunker("fixed", 256),
    "fixed-512": Chunker("fixed", 512), "fixed-1024": Chunker("fixed", 1024), "fixed-2048": Chunker("fixed", 2048),
    "fixed-256/o64": Chunker("fixed", 256, overlap=64),
    "recursive-256": Chunker("recursive", 256), "recursive-512": Chunker("recursive", 512),
    "semantic-512": Chunker("semantic", 512),
    "structural": Chunker("structural", 512, prefix=False), "structural+path": Chunker("structural", 512, prefix=True),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    parsed = parse_all()
    R["parse"] = parse_quality(parsed)
    print("== parse quality, 300 pages ==")
    for m, v in R["parse"].items():
        print(f"{m:10s} wordF1 {v['word_f1']:.3f} order {v['order']:.3f} footer-leak {v['footer_leak_pages']:3d} "
              f"empty {v['empty_pages']:3d} rows {v['table_rows_intact']:.2f} s/page {v['sec_per_page']:.3f} usd {v['usd']:.2f}")
    print("routes:", dict(R["parse"]["routed"]["routes"]))

    # downstream effect of each parser, same chunker
    R["parse_downstream"] = {}
    for m in parsed:
        ev = evaluate(chunk_all(parsed[m], CHUNKERS["structural+path"]))
        R["parse_downstream"][m] = {k: ev[k] for k in ("hit", "cite", "groups", "groups10", "chunks")}
        print(f"downstream {m:10s} hit@1 {ev['hit'][1]:.3f} hit@5 {ev['hit'][5]:.3f} cite@5 {ev['cite'][5]:.3f}")

    # chunking sweep on the routed parse
    R["sweep"] = {}
    routed = parsed["routed"]
    store = {}
    for name, ch in CHUNKERS.items():
        chunks = chunk_all(routed, ch); store[name] = chunks
        for ret in ("bm25", "vector"):
            ev = evaluate(chunks, ret)
            R["sweep"][f"{name}|{ret}"] = {k: ev[k] for k in ("hit", "cite", "packet5", "chunks", "mean_tokens", "sectioned", "groups", "groups10")}
            print(f"{name:16s} {ret:6s} hit@1 {ev['hit'][1]:.3f} @5 {ev['hit'][5]:.3f} @10 {ev['hit'][10]:.3f} cite@1 {ev['cite'][1]:.3f} @5 {ev['cite'][5]:.3f} "
                  f"chunks {ev['chunks']:5d} tok {ev['mean_tokens']:6.1f} pkt5 {ev['packet5']:7.1f}")
    for name in ("fixed-256", "structural+path"):
        ev = evaluate(store[name], "vector", late=True)
        R["sweep"][f"{name}|late"] = {k: ev[k] for k in ("hit", "cite", "packet5", "chunks", "mean_tokens", "sectioned", "groups", "groups10")}
        print(f"{name:16s} late   hit@1 {ev['hit'][1]:.3f} @5 {ev['hit'][5]:.3f} @10 {ev['hit'][10]:.3f}")

    frozen(parsed, store)
    footer_injection(parsed)
    table_incident(parsed)
    revision(parsed)
    scoreboard(parsed)
    R["questions"] = Counter(q["group"] for q in QS)
    R["seconds"] = time.time() - t0
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    print(f"lab done in {R['seconds']:.0f}s")


# ------------------------------------------------------------------ 3. the five frozen-pipe claims
def frozen(parsed, store):
    qs = [q for q in QS if q["group"] == "frozen-2022"]
    cases = {
        "chapter 2 import (text layer, cut at numbered headings)": chunk_all(parsed["text_layer"], Chunker("structural", 512, prefix=False)),
        "fixed-512 on layout parse": chunk_all(parsed["layout"], Chunker("fixed", 512)),
        "structural on layout parse": chunk_all(parsed["layout"], Chunker("structural", 512, prefix=False)),
        "structural+path on layout parse": chunk_all(parsed["layout"], Chunker("structural", 512, prefix=True)),
    }
    R["frozen"] = {}
    print("== the five frozen-pipe determinations ==")
    for name, chunks in cases.items():
        ev = evaluate(chunks, qs=qs)
        R["frozen"][name] = {q: v for q, v in ev["per_q"].items()}
        print(f"{name}: hit@1 {ev['hit'][1]:.1f} hit@5 {ev['hit'][5]:.1f} cite@5 {ev['cite'][5]:.1f}")
        for qid, v in ev["per_q"].items():
            print(f"   {qid:9s} gold rank {v['rank']}  top: {v['top']}")
    before = cases["chapter 2 import (text layer, cut at numbered headings)"]
    after = cases["structural+path on layout parse"]
    R["frozen_spans"] = {
        "before": [dict(cid=c.chunk_id, path=c.path, pages=c.pages, text=c.text) for c in before
                   if c.doc_id == "HO-3_2022-01" and ("Freezing" in c.text or "This peril includes the water" in c.text)],
        "after": [dict(cid=c.chunk_id, path=c.path, pages=c.pages, text=c.text, prefix=c.prefix) for c in after
                  if c.doc_id == "HO-3_2022-01" and c.path and c.path[-1].startswith("12.")],
    }


# ------------------------------------------------------------------ 4. a footer that becomes a clause
FURNITURE_Q = "The {name} was on last year's declarations but the customer is not sure it is on this year's. Does it still apply?"
ENDO_NAMES = {"HO 04 95": "water back-up endorsement", "HO 04 54": "earthquake endorsement", "HO 04 98": "refrigerated property endorsement",
              "HO 04 90": "replacement cost endorsement", "HO 04 61": "scheduled personal property endorsement",
              "HO 04 35": "loss assessment endorsement", "HO 04 16": "premises alarm endorsement", "LM HO 17": "roof surface payment endorsement"}


def furniture_questions():
    out = []
    for cid, c in TRUTH["clauses"].items():
        if cid.endswith("#other") and c["label"] in ENDO_NAMES:
            g = dict(c); g["id"] = cid
            pol = dict(form="2024-06" if c["edition"] >= "2024" else "2022-01", endorsements=[(c["label"], c["edition"])],
                       state="WI", dol="2025-05-01")
            out.append(dict(qid=f"decl-{c['label']}-{c['edition']}".replace(" ", ""), group="declarations",
                            text=FURNITURE_Q.format(name=ENDO_NAMES[c["label"]]), gold=g, policy=pol))
    return out


def footer_injection(parsed):
    """Keep the page furniture and index it as text: the endorsement footer is a
    sentence that reads like policy language, and it competes with the clause."""
    kept = {d: parse_layout(pdf(d), d, strip_furniture=False) for d in DOCS}
    ch = Chunker("structural", 512, prefix=True)
    kc, sc = chunk_all(kept, ch), chunk_all(parsed["layout"], ch)
    fq = furniture_questions()
    groups = ("endorsement", "form", "frozen-2022")
    qs = [q for q in QS if q["group"] in groups] + fq
    a, b = evaluate(kc, qs=qs), evaluate(sc, qs=qs)
    foot = "applies only when the endorsement is shown in the Declarations"
    R["footer"] = dict(kept=a["hit"], stripped=b["hit"], kept_cite=a["cite"], stripped_cite=b["cite"],
                       kept_groups=a["groups"], stripped_groups=b["groups"], n_decl=len(fq),
                       chunks_with_footer=sum(foot in c.text for c in kc),
                       glued=sum(foot in c.text and not c.text.strip().startswith(c.label) for c in kc),
                       endorsement_chunks=sum(c.doc_id.startswith(("HO04", "LM-HO17")) or "HO 04" in c.label for c in kc))
    flips = [(q, a["per_q"][q], b["per_q"][q]) for q in a["per_q"]
             if a["per_q"][q]["cite_rank"] != 1 and b["per_q"][q]["cite_rank"] == 1]
    R["footer"]["flips"] = [dict(qid=q, kept=x, stripped=y) for q, x, y in flips]
    kchunks = {c.chunk_id: c for c in kc}
    R["footer"]["examples"] = [dict(qid=q, text=kchunks[x["top_id"]].text, citation=kchunks[x["top_id"]].citation(),
                                    stripped_top=y["top"]) for q, x, y in flips[:4]]
    rep = {d: parse_layout(pdf(d), d, band_only=False) for d in DOCS}
    lost = split_rate(chunk_all(rep, ch)); kept_ok = split_rate(chunk_all(parsed["layout"], ch))
    R["footer"]["repetition_only"] = dict(split=lost, band=kept_ok,
        lines_removed=sum(len([b for b in parsed["layout"][d].blocks]) - len(rep[d].blocks) for d in DOCS))
    print(f"== furniture by repetition alone: {R['footer']['repetition_only']['lines_removed']} body lines deleted; "
          f"clauses no chunk holds whole {lost[1]}/{lost[2]} ({lost[0]:.1%}) vs margin band and repetition {kept_ok[1]}/{kept_ok[2]} ({kept_ok[0]:.1%})")
    print(f"== footer kept vs stripped ({len(qs)} questions, {len(fq)} about the declarations): "
          f"cite@1 {a['cite'][1]:.3f} -> {b['cite'][1]:.3f}; hit@1 {a['hit'][1]:.3f} -> {b['hit'][1]:.3f}; flips {len(flips)}; "
          f"chunks carrying the footer {R['footer']['chunks_with_footer']}")
    glued = next(c for c in kc if c.doc_id == "HO0495_2024-06" and foot in c.text)
    R["footer"]["glued_example"] = dict(citation=glued.citation(), pages=glued.pages, text=glued.text)
    print(f"   footer inside a clause: {glued.citation()} pages {glued.pages}")
    print("   " + glued.text.replace("\n\n", " / "))
    print("   declarations group cite@1 kept", round(a["groups"].get("declarations", 0), 3), "stripped", round(b["groups"].get("declarations", 0), 3))
    for f in R["footer"]["examples"]:
        print("  ", f["qid"], "| kept top:", f["citation"], "|", f["text"][-200:].replace("\n", " "), "| stripped top:", f["stripped_top"])


# ------------------------------------------------------------------ 5. the deductible table
def read_deductible(text: str, state: str) -> str | None:
    """Stand-in reader: the first 'N% of Coverage A' after the state's name."""
    i = text.find(state)
    if i < 0:
        return None
    m = re.search(r"\d+% of Coverage A", text[i:])
    return m.group() if m else None


def table_incident(parsed):
    d = "LM-DED_2024-06"
    res = {}
    for m in ("text_layer", "layout", "ocr"):
        ch = Chunker("structural", 512, prefix=True) if m == "layout" else Chunker("fixed", 512)
        chunks = ch.chunk(parsed[m][d], DOCS[d])
        out = {}
        for state, _a, wind, _mn in __import__("clauses").DEDUCTIBLE_ROWS:
            c = next((c for c in chunks if state in c.text), None)
            got = read_deductible(c.text, state) if c else None
            out[state] = dict(read=got, truth=wind, ok=got == wind, page=c.pages[0] if c else None)
        res[m] = out
    tl = " ".join(b.text for b in parsed["text_layer"][d].blocks if b.page == 3)
    i = tl.find("Minnesota")
    res["text_layer_excerpt"] = tl[max(0, i - 10): i + 260]
    lay = [b.rows for b in parsed["layout"][d].blocks if b.kind == "table"]
    res["layout_rows"] = lay
    R["table"] = res
    print("== deductible schedule, wind or hail by state ==")
    for m in ("text_layer", "layout", "ocr"):
        print(f"  {m:10s}", {s: (v['read'], 'ok' if v['ok'] else 'WRONG') for s, v in res[m].items()})
    print("  text layer, page 3:", res["text_layer_excerpt"][:200])


# ------------------------------------------------------------------ 6. a revised edition
def revision(parsed):
    d = "HO-3_2024-06"
    base = parsed["layout"][d]
    rev = ParsedDoc(d, "layout", [Block(b.kind, b.text, b.page, b.level, b.rows, b.continued) for b in base.blocks], 0, base.pages)
    k = next(i for i, b in enumerate(rev.blocks) if b.text.startswith("We insure for loss caused by windstorm or hail."))
    rev.blocks[k].text = rev.blocks[k].text.replace("windstorm or hail.", "windstorm or hail, including a derecho.")
    R["revision"] = {}
    print("== one sentence edited in Perils 2: chunks to re-embed ==")
    for name in ("fixed-256", "fixed-512", "fixed-256/o64", "recursive-512", "semantic-512", "structural+path"):
        ch = CHUNKERS[name]
        a = {c.digest() for c in ch.chunk(base, DOCS[d])}
        b = [c.digest() for c in ch.chunk(rev, DOCS[d])]
        changed = sum(x not in a for x in b)
        R["revision"][name] = dict(changed=changed, total=len(b), share=changed / len(b))
        print(f"  {name:16s} {changed:4d} of {len(b):4d} chunks ({changed / len(b):.1%})")


# ------------------------------------------------------------------ 7. the scoreboard (modelled link)
def split_rate(chunks):
    """Share of prose clauses that no single chunk holds whole on a cited page:
    the chunk-boundary failure behind Chapter 9's docstore misses."""
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)
    prose = [c for cid, c in TRUTH["clauses"].items() if "#table" not in cid and "#schedule" not in cid and len(c["text"]) > 40]
    split = sum(not any(evidence_hit(ch, g) for ch in by_doc[g["doc_id"]]) for g in prose)
    return split / len(prose), split, len(prose)


def scoreboard(parsed):
    sys.path.insert(0, os.path.join(HERE, "..", "ch09"))
    import fullmix as F
    fz = R["frozen"]
    # a frozen-pipe determination is fixed when its governing span reaches the
    # ten-span packet (Chapter 2 keeps the top ten); the reader then cites it
    fixed_n = sum(1 for v in fz["structural+path on layout parse"].values() if v["rank"] and v["rank"] <= 10)
    # docstore chunk-boundary misses scale with the share of clauses a chunker cuts
    before = split_rate(chunk_all(parsed["text_layer"], Chunker("structural", 512, prefix=False)))
    after = split_rate(chunk_all(parsed["routed"], Chunker("structural", 512, prefix=True)))
    ratio = after[0] / before[0] if before[0] else 1.0
    sp = chunk_all(parsed["routed"], Chunker("structural", 512, prefix=True))
    # the provenance header already carries document, edition and page (Chapter 9);
    # the section path is what this chapter adds to each of the ten spans
    path_tokens = sum(count_tokens(" > ".join(c.path)) for c in sp) / len(sp)
    extra = 10 * path_tokens
    dets = F.build()
    eval_wrong = [d for d in dets if d.split == "eval" and d.carried_wrong]

    def outcome10(d):
        if d.split == "eval" and d.carried_wrong and eval_wrong.index(d) < fixed_n:
            return "right"
        # the two holdout errors are the same mechanism (a 2022 clause cut at a
        # column break); they are fixed only if every frozen-pipe case is
        if d.split == "holdout" and d.carried_wrong and fixed_n == len(fz["structural+path on layout parse"]):
            return "right"
        if d.split == "new" and d.stratum == "docstore":
            return "wrong" if d.u < 0.06 * ratio else "right"
        return F.outcome(d, "ch09")
    res = [(d, outcome10(d)) for d in dets]
    n = len(res)
    wrong = sum(r == "wrong" for _, r in res)
    ev = [r for d, r in res if d.split == "eval"]
    ho = [r for d, r in res if d.split == "holdout"]
    ch9 = F.score(dets, "ch09")
    R["scoreboard"] = dict(ratio=ratio, split_before=before, split_after=after, frozen_fixed=fixed_n,
                           eval_wrong=ev.count("wrong"), eval_rate=ev.count("wrong") / len(ev),
                           holdout_wrong=ho.count("wrong"), full_wrong=wrong, full_rate=wrong / n,
                           withheld=sum(r == "withheld" for _, r in res),
                           tokens=ch9["tokens"] + extra, usd=F.USD_PER_DET_CH08 * (ch9["tokens"] + extra) / F.TOKENS_CH08,
                           ch9_full_rate=ch9["rate"], ch9_tokens=ch9["tokens"], ch9_usd=ch9["usd"], path_tokens_per_span=path_tokens,
                           by_stratum=Counter(f"{d.stratum if d.split == 'new' else d.split}:{r}" for d, r in res))
    s = R["scoreboard"]
    print(f"== clause split rate: chapter 2 import {before[1]}/{before[2]} ({before[0]:.1%}), structural+path routed {after[1]}/{after[2]} ({after[0]:.1%})")
    print(f"== scoreboard: eval {s['eval_wrong']}/200 ({s['eval_rate']:.1%}), holdout {s['holdout_wrong']}/300, "
          f"full mix {s['full_wrong']}/1000 ({s['full_rate']:.1%}) vs ch9 {s['ch9_full_rate']:.1%}; withheld {s['withheld']}; "
          f"tokens {s['tokens']:.0f} (ch9 {s['ch9_tokens']:.0f}; path {path_tokens:.1f}/span); usd {s['usd']:.5f}")
    print("   by stratum:", dict(sorted(s["by_stratum"].items())))


if __name__ == "__main__":
    main()
