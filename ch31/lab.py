"""Chapter 31 lab: assembling the packet, on Chapter 28's 300 determinations.

Run from 04-code:  python3 ch31/lab.py   (pip install tiktoken for real token counts)
The reader is the stand-in in lakeshore31.py; the checker is contextlab.citations
v6.4 unchanged. Numbers are properties of these stand-ins, except token counts,
which come from a real tokeniser when one is installed.
"""
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE, os.path.join(os.path.dirname(HERE), "ch28")]
import lakeshore31 as W                                              # noqa: E402
from contextlab.assembler import (Assembler, CollisionError, TOKENIZER, VERSION,   # noqa: E402
                                  count_tokens, encode_table)
from contextlab.citations import CitationChecker, parse, precision_recall          # noqa: E402

DETS = W.L.build()
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
RES = {"tokenizer": TOKENIZER, "version": VERSION}
PTR = re.compile(r"\[\[?[^\]]+\]?\]")
CHK = CitationChecker(threshold=0.6, reattribute=True)
CONFIGS = {   # name: (format, order, ids)
    "plain": ("plain", "retrieval", "none"),
    "json": ("json", "retrieval", "handle"),
    "markdown": ("markdown", "retrieval", "handle"),
    "xml": ("xml", "retrieval", "handle"),
    "schema": ("schema", "schema", "short"),
}


def section(t):
    print(f"\n== {t}")


def assemble(det, fmt, order, ids, packet=None, **kw):
    packet = packet or det.packet
    a = Assembler(fmt, order, ids, **kw)
    return a.assemble(W.INSTRUCTIONS, W.QUESTIONS[det.ctype], W.roles(det, packet))


def misresolved(intended, asm, man, src):
    """Pointers the reader meant for one span that resolve to another."""
    if not asm.ids or asm.ids[0] is None:
        return 0
    bad = 0
    for e in intended:
        ptr = parse(asm.to_handles(W.pointer(e, asm)))
        bad += bool(ptr) and all(src.get(x) != e for x in man.resolve(*ptr[0]))
    return bad


def score(det, asm, seed, packet=None):
    """One reading, checked. Returns per-sentence label dicts and counters."""
    sents = W.read(det, asm, seed, packet)
    man = asm.manifest()
    src = {v: k for k, v in asm.shown.items()}
    texts = [asm.to_handles(s["text"]) for s in sents]
    found = CHK.check(texts, man)
    g, m, c = [], [], Counter()
    for s, t, f in zip(sents, texts, found):
        ptrs = parse(t)
        sup = sum(any(src.get(x) in s["support"] for x in man.resolve(k, r)) for k, r in ptrs)
        wrong_text = misresolved(s["intended"], asm, man, src)
        need = s["kind"] != "note"
        g.append(dict(needs=need, total_cites=len(ptrs), supported_cites=sup))
        ok = f.status in ("supported", "reattributed")
        m.append(dict(needs=need, total_cites=len(ptrs), supported_cites=len(ptrs) if ok else 0))
        c["sentences"] += need
        c["faithful"] += need and W.faithful(s, asm)
        c["pointers"] += len(ptrs)
        c["ambiguous"] += sum(len(man.resolve(k, r)) > 1 for k, r in ptrs)
        c["wrong_text"] += wrong_text
        c[f"m_{s['mistake']}"] += 1
        c[f"st_{f.status}"] += need
        c["ptr_tokens"] += sum(count_tokens(x) for x in PTR.findall(s["text"]) if "no-source" not in x)
    bad = [f for s, f in zip(sents, found) if s["mistake"] in ("polarity", "exception")]
    c["wrong_decision"] += bool(bad)
    c["wrong_unflagged"] += bool(bad) and all(f.status in ("supported", "reattributed") for f in bad)
    return g, m, c


def run(cfg, packet_fn=None, **kw):
    fmt, order, ids = CONFIGS[cfg] if isinstance(cfg, str) else cfg
    G, M, C, toks, refused = [], [], Counter(), [], 0
    for d in DETS:
        packet = packet_fn(d.packet) if packet_fn else None
        try:
            asm = assemble(d, fmt, order, ids, packet, **kw)
        except CollisionError:
            refused += 1
            continue
        C["collisions"] += len(asm.collisions)
        toks.append(count_tokens(asm.text))
        for seed in range(W.SEEDS):
            g, m, c = score(d, asm, seed, packet)
            G += g
            M += m
            C.update(c)
    gp, gr = precision_recall(G)
    mp, mr = precision_recall(M)
    n = len(toks) * W.SEEDS
    return dict(tokens=sum(toks) / max(1, len(toks)), gold_p=gp, gold_r=gr, meas_p=mp, meas_r=mr,
                faith=C["faithful"] / max(1, C["sentences"]), wrong=C["wrong_decision"], readings=n,
                tpp=C["ptr_tokens"] / max(1, C["pointers"]), refused=refused,
                missing=C["m_missing"], copy=C["m_copy"], ambiguous=C["ambiguous"],
                unflagged=C["wrong_unflagged"], neighbour=C["m_neighbour"], polarity=C["m_polarity"],
                collisions=C["collisions"], wrong_text=C["wrong_text"], pointers=C["pointers"],
                sentences=C["sentences"])


def row(name, r):
    return (f"{name:<9} tok {r['tokens']:>5.0f}  ptr-tok {r['tpp']:>4.1f}  copy {r['copy']:>3}  gold P {r['gold_p']:.1%}  checked P {r['meas_p']:.1%} "
            f"R {r['meas_r']:.1%}  faith {r['faith']:.1%}  wrong {r['wrong']:>2}/{r['readings']}")


print(f"contextlab v{VERSION}; tokens counted with {TOKENIZER}; reader runs per determination: {W.SEEDS}")

section("1. One packet, five assemblies (DET28-0188, Emily Carter)")
emily = DETS[187]
one = {}
for name, (fmt, order, ids) in CONFIGS.items():
    a = assemble(emily, fmt, order, ids)
    one[name] = count_tokens(a.text)
    with open(os.path.join(OUT, f"det0188-{name}.txt"), "w") as f:
        f.write(a.text)
    print(f"{name:<9} {one[name]:>5} tokens  {len(a.text):>5} characters")
RES["one_packet"] = one
print("schema assembly, first lines:")
for line in open(os.path.join(OUT, "det0188-schema.txt")).read().splitlines()[3:9]:
    print("  " + (line if len(line) < 118 else line[:115] + "..."))

section("2. Five assemblies, 300 determinations x 5 readings")
five = {name: run(name) for name in CONFIGS}
for name, r in five.items():
    print(row(name, r))
for name, r in five.items():
    print(f"{name:<9} pointers {r['pointers']:>5}  resolving to more than one span {r['ambiguous']:>5}  "
          f"wrong conclusions the checker did not flag {r['unflagged']:>2}")
RES["five"] = five

section("3. Position: the schema format under four orders (missing pointers by third)")
orders = {}
for how in ("retrieval", "document", "edges", "schema"):
    r = run(("schema", how, "short"))
    orders[how] = r
    print(f"order={how:<10} missing pointers {r['missing']:>3}  checked R {r['meas_r']:.1%}  "
          f"polarity {r['polarity']:>2}  wrong {r['wrong']:>2}/{r['readings']}")
RES["orders"] = orders

section("4. Relationships: where the endorsement sits (back-up claims with HO 04 95)")
rel = {}
for name, cfg in (("markdown, retrieval", CONFIGS["markdown"]), ("xml, document", ("xml", "document", "handle")),
                  ("schema, no relation", ("xml", "schema", "short")), ("schema, modifies=", CONFIGS["schema"]),
                  ("markdown, reranked", ("markdown", "retrieval", "handle", "rerank"))):
    kinds, pol = Counter(), 0
    for d in [d for d in DETS if d.ctype == "backup_endorsed"]:
        pk = W.L.rerank(d.packet, "ch31-rerank") if len(cfg) == 4 else None
        a = assemble(d, *cfg[:3], packet=pk)
        kinds[W.relation(a)] += 1
        pol += sum(s["mistake"] == "polarity" for seed in range(W.SEEDS) for s in W.read(d, a, seed, pk))
    rel[name] = dict(kinds=dict(kinds), polarity=pol, readings=82 * W.SEEDS)
    print(f"{name:<22} placement {dict(sorted(kinds.items()))}  polarity mistakes {pol}/{82 * W.SEEDS}")
RES["relation"] = rel

section("5. Elastic snippets: shorten context spans, never governing ones")
el = {}
for label, kw in (("full", {}), ("context<=20 words", dict(trim_context=20)),
                  ("context<=12 words", dict(trim_context=12)), ("context<=8 words", dict(trim_context=8)),
                  ("everything<=12 words", dict(trim_context=12, trim_roles=("context", "governing", "modifier")))):
    r = run("schema", **kw)
    el[label] = r
    print(f"{label:<22} tok {r['tokens']:>5.0f}  faith {r['faith']:.1%}  checked R {r['meas_r']:.1%}  "
          f"checked P {r['meas_p']:.1%}")
RES["elastic"] = el

section("6. Tables: thirty rows of claim history in five encodings")
r = W.L.rng("ch31-history")
perils = ["hail", "water back-up", "frozen pipe", "wind", "theft", "mould"]
hist = [dict(date=f"20{18 + i // 4}-{1 + (i * 5) % 12:02d}-{1 + (i * 7) % 28:02d}",
             claim=f"CLM-20{18 + i // 4}-{10000 + r.randrange(90000):05d}", peril=r.choice(perils),
             paid=r.choice([0, 450, 1200, 2750, 5000, 8300]), status=r.choice(["closed", "closed", "denied"]))
        for i in range(30)]
tab = {}
for how in ("json", "json-compact", "toon", "csv", "markdown"):
    tab[how] = count_tokens(encode_table("history", hist, how))
    print(f"{how:<13} {tab[how]:>5} tokens  ({tab[how] / tab['json']:.0%} of indented JSON)")
spans = [dict(id=f"S{i + 1}", doc=s.entry.doc, edition=s.entry.edition, page=s.entry.page, text=s.entry.text)
         for i, s in enumerate(W.roles(emily))]
ev = {how: count_tokens(encode_table("spans", spans, how)) for how in ("json", "json-compact", "toon")}
print("the same comparison on DET28-0188's evidence spans: " +
      ", ".join(f"{k} {v}" for k, v in ev.items()) + f"  (TOON {ev['toon'] / ev['json']:.0%} of JSON)")
RES["tables"] = dict(history=tab, evidence=ev)
with open(os.path.join(OUT, "history-toon.txt"), "w") as f:
    f.write(encode_table("history", hist, "toon"))

section("7. Identifiers: what a pointer costs, and Chapter 28's reranker")
ids = {}
for name, cfg in (("handles", ("xml", "schema", "handle")), ("short ids", CONFIGS["schema"])):
    a = run(cfg)
    b = run(cfg, packet_fn=lambda p: W.L.rerank(p, "ch31-rerank"))
    ids[name] = dict(plain=a, rerank=b)
    print(f"{name:<10} ptr-tok {a['tpp']:>4.1f}  checked P {a['meas_p']:.1%} -> after rerank upstream "
          f"{b['meas_p']:.1%}")
RES["ids"] = ids

section("8. Failure injection: the re-chunk splits HO-3 page 7 span 1, positional ids collide")
inj = {}
for label, cfg, kw in (("positional ids, no check", ("schema", "schema", "positional"), dict(check_ids=False)),
                       ("positional ids, check", ("schema", "schema", "positional"), {}),
                       ("short ids (v6.7)", CONFIGS["schema"], {})):
    r0 = run(cfg, **kw) if label.startswith("short") else None
    r = run(cfg, packet_fn=W.rechunk, **kw)
    inj[label] = r
    print(f"{label:<25} refused {r['refused']:>3}/300  collisions {r['collisions']:>3}  "
          f"pointers to the wrong text {r['wrong_text']:>4}/{r['pointers']}  gold P {r['gold_p']:.1%}  "
          f"checked P {r['meas_p']:.1%}  R {r['meas_r']:.1%}")
RES["injection"] = inj

section("9. Lakeshore before and after")
before, after = five["markdown"], five["schema"]
for k, lab_ in (("tokens", "tokens per packet"), ("tpp", "tokens per pointer"), ("meas_p", "checked precision"),
                ("meas_r", "checked recall"), ("faith", "faithfulness"), ("wrong", "wrong conclusions drafted"),
                  ("unflagged", "  of which unflagged")):
    f = (lambda v: f"{v:.1%}") if k in ("meas_p", "meas_r", "faith") else (lambda v: f"{v:,.1f}")
    print(f"{lab_:<20} {f(before[k]):>8} -> {f(after[k]):>8}")
RES["before_after"] = dict(before=before, after=after)

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(RES, f, indent=1, default=float)
