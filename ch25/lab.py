"""Chapter 25 lab: a repository map for this book's own code, and what goes wrong.

Note: deterministic, no language model runs. The 'repository' is a pinned
snapshot of 04-code (contextlab plus the chapter labs for Chapters 4 to 19).
Packets are scored against jedi's find-references (04-code/.venv-ch25), an
analyser independent of contextlab.codemap. A change task's packet is
'sufficient' when it holds every line jedi says refers to the symbol; whether a
model would then make the change correctly is not measured here.
Token counts use the lab tokenizer (contextlab.chunker.count_tokens).
Run from 04-code/ch25: python3 lab.py > output.txt
"""
import json, pathlib, re, shutil, subprocess, sys, time
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from contextlab.chunker import Chunk, count_tokens
from contextlab.chunkeval import BM25, VectorStandIn, ranked, words
from contextlab import codemap as CM
import tasks25 as T

SNAP, OUT = HERE / "snapshot", HERE / "out"
JEDI = HERE.parent / ".venv-ch25" / "bin" / "python"
BUDGET, WIN = 3000, 8
RES = {}


def section(t):
    print(f"\n== {t}")


def dirs_of(root):
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "__pycache__")


def gold_refs(root, idx, qnames, tag):
    cache = OUT / f"gold_{tag}.json"
    if not cache.exists():
        t = [dict(qname=q, name=idx.symbols[q].name, kind=idx.symbols[q].kind,
                  path=idx.symbols[q].path, line=idx.symbols[q].line) for q in qnames]
        (OUT / f"targets_{tag}.json").write_text(json.dumps(t))
        subprocess.run([str(JEDI), str(HERE / "gold_jedi.py"), str(root),
                        str(OUT / f"targets_{tag}.json"), str(cache)], check=True,
                       stdout=subprocess.DEVNULL)
    return {q: [tuple(x) for x in v] for q, v in json.loads(cache.read_text()).items()}


class Repo:
    def __init__(self, root):
        self.root = root
        self.lines = {str(p.relative_to(root)): p.read_text().splitlines()
                      for d in dirs_of(root) for p in (root / d).rglob("*.py")
                      if "__pycache__" not in p.parts}

    def tokens(self, spans):
        return sum(count_tokens("\n".join(self.lines[p][a - 1:b])) for p, a, b in spans)


def merge(spans):
    out = []
    for p, a, b in sorted(spans):
        if out and out[-1][0] == p and a <= out[-1][2] + 1:
            out[-1] = (p, out[-1][1], max(b, out[-1][2]))
        else:
            out.append((p, a, b))
    return out


def fill(repo, candidates):
    """Add candidate spans in order until the budget is spent."""
    kept = []
    for sp in candidates:
        trial = merge(kept + [sp])
        if repo.tokens(trial) <= BUDGET:
            kept = trial
    return kept


def window(repo, p, line):
    return (p, max(1, line - WIN), min(len(repo.lines[p]), line + WIN))


def covered(spans, p, line):
    return any(sp == p and a <= line <= b for sp, a, b in spans)


# ---- four ways to build a packet -------------------------------------------
class Chunks:
    def __init__(self, repo, size=40):
        self.repo, self.chunks = repo, []
        for p, ls in sorted(repo.lines.items()):
            for a in range(0, len(ls), size):
                txt = "\n".join(ls[a:a + size])
                # split identifiers so rrf_score and RRF both reach the analyser
                txt = txt + "\n" + re.sub(r"[_.]", " ", txt)
                self.chunks.append(Chunk(f"{p}:{a + 1}", p, p, "-", "code", txt, [1]))
        self.kw, self.vec = BM25(self.chunks), VectorStandIn(self.chunks)

    def packet(self, query, which):
        idx = self.kw if which == "keyword" else self.vec
        cands = []
        for c in ranked(idx, query)[:60]:
            p, a = c.chunk_id.rsplit(":", 1)
            cands.append((p, int(a), min(int(a) + 39, len(self.repo.lines[p]))))
        return fill(self.repo, cands)


def grep_packet(repo, name):
    pat = re.compile(rf"\b{re.escape(name)}\b")
    hits = [(p, i + 1) for p, ls in sorted(repo.lines.items()) for i, l in enumerate(ls) if pat.search(l)]
    return fill(repo, [window(repo, p, l) for p, l in hits]), len(hits)


def grep_files(repo, name):
    pat = re.compile(rf"\b{re.escape(name)}\b")
    return sorted(p for p, ls in repo.lines.items() if any(pat.search(l) for l in ls))


def rarest_word(repo, query):
    df = Counter(w for ls in repo.lines.values() for w in set(words("\n".join(ls))))
    ws = [w for w in words(query) if len(w) > 3 and df.get(w)]
    return min(ws, key=lambda w: (df[w], w)) if ws else query.split()[0]


def symbol_packet(repo, idx, qname):
    s = idx.symbols.get(qname)
    if s is None:
        return []
    cands = [(s.path, s.line, min(s.end, s.line + 30))]
    order = {"exact": 0, "name-only": 1, "ambiguous": 2}
    for e in sorted(idx.callers(qname), key=lambda e: (order[e.confidence], e.path, e.line)):
        if e.path in repo.lines:
            cands.append(window(repo, e.path, e.line))
    return fill(repo, cands)


def card_index(idx):
    syms = sorted(idx.symbols.values(), key=lambda s: s.qname)
    cards = [Chunk(s.qname, s.path, s.qname, "-", "symbol",
                   f"{re.sub(r'[_.]', ' ', s.qname)} {s.signature} {s.doc}", [1]) for s in syms]
    return BM25(cards)


def score(spans, gold):
    hit = sum(covered(spans, p, l) for p, l in gold)
    return hit == len(gold), hit / max(1, len(gold))


def run_change_tasks(repo, idx, ch, gold, targets, label):
    rows = defaultdict(list)
    for q in targets:
        name = idx.symbols[q].name if q in idx.symbols else q.rsplit(".", 1)[1]
        text = T.change_text(name)
        packs = {"vector chunks": ch.packet(text, "vector"),
                 "keyword chunks": ch.packet(text, "keyword"),
                 "grep and read": grep_packet(repo, name)[0],
                 "symbol index + graph": symbol_packet(repo, idx, q)}
        for k, sp in packs.items():
            ok, rec = score(sp, gold[q])
            rows[k].append((ok, rec, repo.tokens(sp)))
    out = {}
    for k, v in rows.items():
        n = len(v)
        out[k] = {"sufficient": sum(o for o, _, _ in v) / n, "line_recall": sum(r for _, r, _ in v) / n,
                  "tokens": sum(t for _, _, t in v) / n, "n": n}
        print(f"  {label:<10} {k:<22} sufficient {out[k]['sufficient']:.2f}   reference-line recall "
              f"{out[k]['line_recall']:.2f}   tokens/packet {out[k]['tokens']:,.0f}")
    return out


def main():
    OUT.mkdir(exist_ok=True)
    section("1. Index the snapshot")
    t0 = time.time()
    idx = CM.SymbolIndex.build(SNAP, include=dirs_of(SNAP))
    repo = Repo(SNAP)
    conf = Counter(e.confidence for e in idx.edges)
    print(f"  {len(idx.files)} files, {sum(len(v) for v in repo.lines.values()):,} lines, "
          f"{len(idx.symbols)} symbols, {len(idx.edges):,} reference edges, revision {idx.revision}")
    print(f"  edges by confidence: exact {conf['exact']:,}, name-only {conf['name-only']:,}, "
          f"ambiguous {conf['ambiguous']:,}   (built in {time.time() - t0:.1f} s)")
    RES["index"] = {"files": len(idx.files), "symbols": len(idx.symbols), "edges": dict(conf),
                    "revision": idx.revision}

    section("2. Names that mean different things in different modules")
    dup = {n: sorted(q) for n, q in idx.by_name.items()
           if len({idx.symbols[x].path for x in q if idx.symbols[x].kind == "class"}) > 1}
    print(f"  {len(dup)} class names are defined in more than one file")
    for n in sorted(dup, key=lambda n: (-len(dup[n]), n))[:6]:
        print(f"  class {n:<14} defined {len(dup[n])} times: " + ", ".join(
            idx.symbols[q].path for q in dup[n]))
    RES["same_name_classes"] = {n: len(v) for n, v in dup.items()}

    section("3. Change tasks: which packet holds every line that refers to the symbol?")
    pool = sorted(q for q, s in idx.symbols.items() if s.path.startswith("contextlab/")
                  and not s.name.startswith("_") and s.kind in ("function", "class", "method"))
    gold = gold_refs(SNAP, idx, pool, "snapshot")
    targets = [q for q in pool if 3 <= len(gold[q]) <= 25
               and len({p for p, _ in gold[q]}) >= 2]
    print(f"  {len(targets)} change tasks: contextlab symbols referenced from 2+ files, "
          f"3 to 25 reference lines (jedi); budget {BUDGET:,} tokens")
    ch = Chunks(repo)
    RES["change"] = run_change_tasks(repo, idx, ch, gold, targets, "fresh")
    why = Counter()
    for q in targets:
        sp = symbol_packet(repo, idx, q)
        miss = [(p, l) for p, l in gold[q] if not covered(sp, p, l)]
        if miss:
            el = {(e.path, e.line): e.confidence for e in idx.callers(q)}
            if any(m not in el for m in miss):
                why["a missed line has no edge in the index"] += 1
            else:
                why["every missed line has an edge; budget spent first (" +
                    "/".join(sorted({el[m] for m in miss})) + ")"] += 1
    for k, v in sorted(why.items()):
        print(f"    symbol route misses: {v:>2} tasks, {k}")
    RES["change_miss_reasons"] = dict(why)
    RES["change_targets"] = len(targets)

    section("4. Locate questions in plain words (no identifiers)")
    cards = card_index(idx)
    rows = defaultdict(list)
    for qtext, q in T.LOCATE:
        s = idx.symbols[q]
        want = [(s.path, s.line)]
        top = [c.chunk_id for c in ranked(cards, qtext)[:3]]
        packs = {"vector chunks": ch.packet(qtext, "vector"), "keyword chunks": ch.packet(qtext, "keyword"),
                 "grep and read": grep_packet(repo, rarest_word(repo, qtext))[0],
                 "symbol cards (top 3)": fill(repo, [(idx.symbols[t].path, idx.symbols[t].line,
                                                     min(idx.symbols[t].end, idx.symbols[t].line + 30)) for t in top])}
        for k, sp in packs.items():
            rows[k].append((score(sp, want)[0], repo.tokens(sp)))
    RES["locate"] = {}
    for k, v in rows.items():
        RES["locate"][k] = {"found": sum(o for o, _ in v) / len(v), "tokens": sum(t for _, t in v) / len(v)}
        print(f"  {k:<22} definition in packet {sum(o for o, _ in v):>2}/{len(v)}   "
              f"tokens/packet {RES['locate'][k]['tokens']:,.0f}")

    section("5. The repository map: what fits in a budget")
    text, shown = CM.render_map(idx, 300, focus=["contextlab.retrieval.fusion.rrf"])
    print("\n".join("  " + l for l in text.splitlines()[:9]))
    sweep = []
    for b in (250, 500, 1000, 2000, 4000, 8000):
        cov = {"plain": [], "focused": []}
        for q in targets:
            need = {p for p, _ in gold[q]}
            for mode in cov:
                _, sh = CM.render_map(idx, b, focus=[q] if mode == "focused" else ())
                cov[mode].append(len(need & set(sh)) / len(need))
        row = {"budget": b, **{m: sum(v) / len(v) for m, v in cov.items()}}
        sweep.append(row)
        print(f"  map budget {b:>5,}  files a change touches that the map lists: plain {row['plain']:.2f}"
              f"   focused on the target {row['focused']:.2f}")
    full = count_tokens(CM.render_map(idx, 10 ** 7)[0])
    print(f"  whole map, every file: {full:,} tokens; whole snapshot: "
          f"{sum(count_tokens(chr(10).join(v)) for v in repo.lines.values()):,} tokens")
    RES["map_sweep"], RES["map_full_tokens"] = sweep, full

    section("6. A refactor lands; the index does not")
    after = OUT / "refactored"
    refactor(SNAP, after)
    old_files = dict(idx.files)
    repo2 = Repo(after)
    changed = sorted({p for p in repo2.lines if not (SNAP / p).exists()
                      or (after / p).read_text() != (SNAP / p).read_text()}
                     | {p for p in old_files if not (after / p).exists()})
    print(f"  commit touches {len(changed)} files: rename rrf -> reciprocal_rank_fusion (jedi rename), "
          f"move BM25/VectorStandIn/words/grams to contextlab/rankers.py with a re-export, "
          f"licence header on 10 modules")
    moved = {"contextlab.retrieval.fusion.rrf": "contextlab.retrieval.fusion.reciprocal_rank_fusion",
             **{f"contextlab.chunkeval.{n}": f"contextlab.rankers.{n}"
                for n in ("BM25", "VectorStandIn", "words", "grams")}}
    fresh = CM.SymbolIndex.build(after, include=dirs_of(after))
    pool2 = [moved.get(q, q) for q in targets]
    gold2 = gold_refs(after, fresh, pool2, "refactored")
    ch2 = Chunks(repo2)
    configs = {}
    stale = idx
    naive = CM.SymbolIndex.build(SNAP, include=dirs_of(SNAP)); naive.root = after
    log_n = naive.refresh(changed, relink_dependents=False)
    relink = CM.SymbolIndex.build(SNAP, include=dirs_of(SNAP)); relink.root = after
    log_r = relink.refresh(changed, relink_dependents=True)
    print(f"  incremental refresh re-indexed {len(log_n['reindexed'])} files; with dependents "
          f"re-linked, {len(log_r['relinked'])} files; both finished without error")
    for label, ix in (("stale", stale), ("naive incr", naive), ("relinked", relink), ("rebuild", fresh)):
        rows = []
        silent = 0
        for q0, q in zip(targets, pool2):
            key = q if q in ix.symbols else q0 if label == "stale" else q
            sp = symbol_packet(repo2, ix, key)
            ok, rec = score(sp, gold2[q])
            rows.append((ok, rec, repo2.tokens(sp)))
            s = ix.symbols.get(key)
            if s and s.path in repo2.lines and (s.line > len(repo2.lines[s.path]) or
                                                s.name not in repo2.lines[s.path][s.line - 1]):
                silent += 1
        n = len(rows)
        configs[label] = {"sufficient": sum(r[0] for r in rows) / n, "line_recall": sum(r[1] for r in rows) / n,
                          "wrong_definition_served": silent}
        print(f"  {label:<11} symbol index + graph: sufficient {configs[label]['sufficient']:.2f}   "
              f"line recall {configs[label]['line_recall']:.2f}   definition window points at other code: {silent}")
    lost = [q for q in pool2 if q.startswith("contextlab.rankers.")]
    for label, ix in (("naive incr", naive), ("relinked", relink)):
        miss = sum(1 for q in lost for p, l in gold2[q]
                   if not (ix.symbols.get(q) and ix.symbols[q].path == p and ix.symbols[q].line <= l <= ix.symbols[q].end)
                   and not any((e.path, e.line) == (p, l) for e in ix.callers(q)))
        print(f"  {label:<11} the four moved symbols: reference lines with no caller edge {miss}")
        RES.setdefault("moved_missing", {})[label] = miss
    RES["refactor"] = {"changed_files": len(changed), "configs": configs,
                       "grep_after": None}

    section("7. Resolve before use")
    st = Counter()
    for q in targets:
        s = stale.symbols[q]
        f = after / s.path
        if not f.exists():
            st["file-missing"] += 1; continue
        if CM.sha(f.read_text()) == stale.files[s.path]:
            st["fresh"] += 1; continue
        head = f.read_text().splitlines()[s.line - 1] if s.line <= len(f.read_text().splitlines()) else ""
        st["changed, name still at line" if s.name in head else "changed, name gone from line"] += 1
    print("  stale index, every target checked against the refactored tree by file hash:")
    for k, v in sorted(st.items()):
        print(f"    {k:<30} {v}")
    guard = CM.SymbolIndex.build(SNAP, include=dirs_of(SNAP)); guard.root = after
    lazily, rows, def_only = set(), [], []
    for q0, q in zip(targets, pool2):
        name = q.rsplit(".", 1)[1]
        if q not in guard.symbols:                  # unknown name: find files that mention it
            files = [p for p in grep_files(repo2, name) if guard.files.get(p) != CM.sha((after / p).read_text())]
            files += [p for p in guard.files if not (after / p).exists()]
            if files:
                lazily |= set(files); guard.refresh(files)
        r = guard.resolve(q) if q in guard.symbols else {"status": "unknown"}
        if r["status"] not in ("fresh", "unknown"):
            lazily.add(r["path"]); guard.refresh([r["path"]])
        def_only.append(score(symbol_packet(repo2, guard, q), gold2[q])[0])
        # every span the packet will read, not only the definition, is checked
        for p in sorted({e.path for e in guard.callers(q)}):
            if not (after / p).exists() or CM.sha((after / p).read_text()) != guard.files.get(p):
                lazily.add(p); guard.refresh([p])
        rows.append(score(symbol_packet(repo2, guard, q), gold2[q])[0])
    print(f"  stale index, definition checked before use:          sufficient {sum(def_only) / len(def_only):.2f}")
    print(f"  stale index, every span checked before use:          sufficient {sum(rows) / len(rows):.2f}; "
          f"{len(lazily)} files re-indexed on demand")
    RES["resolve"] = {**dict(st), "guard_sufficient": sum(rows) / len(rows), "def_only_sufficient": sum(def_only) / len(def_only), "lazy_files": len(lazily)}

    section("8. Second setting: a dispatcher the static graph cannot see")
    sd = HERE / "fixtures"
    six = CM.SymbolIndex.build(sd.parent, include=["fixtures"])
    st_calls = {q.rsplit('.', 1)[1]: len([e for e in six.callers(q) if e.src != q])
                for q in six.symbols if ".handle_" in q or ".escalate_" in q}
    sys.path.insert(0, str(sd)); import sellerdesk as S
    seen = Counter()
    def prof(frame, event, arg):
        if event == "call" and frame.f_code.co_name.startswith(("handle_", "escalate_")):
            seen[(frame.f_code.co_name, frame.f_back.f_code.co_name)] += 1
    sys.setprofile(prof)
    for a in ["refund", "reship", "fraud", "refund"]:
        S.dispatch({"action": a, "order": "A-1009", "seller": "S-77"})
    sys.setprofile(None)
    for name in sorted(st_calls):
        rt = sorted({c for (f, c) in seen if f == name})
        print(f"  {name:<15} static callers {st_calls[name]}   called at run time from: {', '.join(rt) or '-'}")
    added = registry_rules(six, sd / "sellerdesk.py")
    for e in added:
        print(f"  rule adds {e.src.rsplit('.', 1)[1]} -> {e.dst.rsplit('.', 1)[1]:<15} confidence {e.confidence}")
    RES["dispatch"] = {"static": st_calls, "runtime": {f"{f}<-{c}": n for (f, c), n in seen.items()},
                       "rule_edges": len(added)}

    section("9. Instruction file check (Listing 6.9 against 04-code)")
    agents = (HERE / "fixtures" / "AGENTS_ch06.md").read_text()
    code_root = SNAP
    refs = sorted({r.rstrip(".") for r in re.findall(r"(?<![\w/])((?:contextlab|fixtures|chNN|CODEOWNERS)[\w./*]*)", agents)})
    bad = []
    for r0 in refs:
        ok = any(code_root.glob(r0.replace("chNN", "ch*"))) if ("*" in r0 or "chNN" in r0) else (code_root / r0).exists()
        print(f"  {r0:<32} {'resolves' if ok else 'DOES NOT RESOLVE'}")
        bad += [] if ok else [r0]
    print(f"  {len(refs) - len(bad)} of {len(refs)} references resolve; file costs "
          f"{count_tokens(agents)} tokens per session")
    RES["agents_check"] = {"refs": len(refs), "broken": bad, "tokens": count_tokens(agents)}

    section("10. A context map for a notebook (sidebar)")
    RES["notebook"] = notebook_map(HERE / "fixtures" / "loss_ratio.ipynb")

    section("11. The map as a tool result")
    r = tool_result(idx, "contextlab.retrieval.fusion.rrf", SNAP)
    print("  " + json.dumps(r, indent=1).replace("\n", "\n  "))
    r2 = tool_result(idx, "contextlab.retrieval.fusion.rrf", after)
    print(f"  the same call after the refactor commit: checked_against_disk = {r2['checked_against_disk']}")
    RES["tool_result"] = r
    (OUT / "results.json").write_text(json.dumps(RES, indent=1, default=str))


def registry_rules(idx, path):
    """Two project rules a static analyser cannot infer: @register(...) handlers are
    called by whatever indexes HANDLERS, and getattr(obj, "escalate_" + x) reaches
    every escalate_* method. Edges they add are marked 'convention'."""
    import ast
    tree = ast.parse(path.read_text())
    mod = CM.module_of(str(path.relative_to(idx.root)))
    disp = [f"{mod}.{n.name}" for n in tree.body if isinstance(n, ast.FunctionDef)
            and "HANDLERS[" in ast.unparse(n) and "getattr(" in ast.unparse(n)]
    prefixes = {c.value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "getattr"
                for c in ast.walk(n) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    new = []
    for q, sym in sorted(idx.symbols.items()):
        node_src = path.read_text().splitlines()[sym.line - 2] if sym.line > 1 else ""
        if node_src.strip().startswith("@register(") or any(sym.name.startswith(p) for p in prefixes):
            for d in disp:
                new.append(CM.Edge(d, q, sym.path, sym.line, "convention"))
    idx.edges += new
    return new


def tool_result(idx, qname, root):
    s = idx.symbols[qname]
    f = root / s.path
    status = "fresh" if f.exists() and CM.sha(f.read_text()) == idx.files[s.path] else "stale"
    ex = [e for e in idx.callers(qname) if e.confidence == "exact"]
    return {"symbol": qname, "defined_at": f"{s.path}:{s.line}-{s.end}", "signature": s.signature,
            "callers": [f"{p}:{n}" for p, n in sorted((e.path, e.line) for e in ex)][:6], "callers_total": len(ex),
            "caller_confidence": "exact",
            "unresolved_callers_possible": True, "index_revision": idx.revision,
            "checked_against_disk": status, "requires_source_read": True}


def refactor(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    # copy without preserving mtimes: jedi's parser cache keys on path and mtime
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"), copy_function=shutil.copyfile)
    # 1. rename through jedi, the way an editor's rename command does it
    subprocess.run([str(JEDI), "-c", RENAME.format(root=str(dst))], check=True)
    # 2. move four names out of chunkeval into a new module, re-exported
    ce = dst / "contextlab" / "chunkeval.py"
    text = ce.read_text().splitlines(keepends=True)
    import ast
    tree = ast.parse("".join(text))
    keep, moved = [], []
    spans = {n.name: (n.lineno, n.end_lineno) for n in tree.body
             if getattr(n, "name", None) in ("BM25", "VectorStandIn", "words", "grams")}
    for i, line in enumerate(text, 1):
        (moved if any(a <= i <= b for a, b in spans.values()) else keep).append(line)
    head = ('"""contextlab.rankers: keyword and stand-in vector rankers (moved from chunkeval)."""\n'
            "from __future__ import annotations\n\nimport math\nimport re\nfrom collections import Counter\n\n"
            "from contextlab.chunker import Chunk\n\n"
            'WORD = re.compile(r"[a-z0-9$%]+")\n'
            'STOP = set("a an the of to and or in on for by is are be it this that with as at from we you your our "\n'
            '           "not do does any if when which will under only but".split())\n\n\n')
    (dst / "contextlab" / "rankers.py").write_text(head + "".join(moved))
    out = "".join(keep).replace("from contextlab.chunker import Chunk\n",
                                "from contextlab.chunker import Chunk\n"
                                "from contextlab.rankers import BM25, VectorStandIn, grams, words  # re-exported\n", 1)
    ce.write_text(out)
    # 3. a licence header on ten modules: every line number below it moves
    for m in ("agentic", "diagnostics", "docgate", "graph", "memguard", "memory", "multimodal",
              "parsing", "policy", "statestore"):
        f = dst / "contextlab" / f"{m}.py"
        f.write_text("# Copyright 2026 Lakeshore Mutual. Internal use only.\n"
                     "# SPDX-License-Identifier: LicenseRef-Lakeshore-Internal\n"
                     "# Generated header; do not edit by hand.\n\n" + f.read_text())


RENAME = """
import jedi, pathlib, tempfile
jedi.settings.cache_directory = tempfile.mkdtemp()
root = pathlib.Path({root!r})
dirs = [str(root)] + [str(p) for p in root.iterdir() if p.is_dir()]
proj = jedi.Project(path=str(root), added_sys_path=dirs, smart_sys_path=False)
f = root / 'contextlab/retrieval/fusion.py'
src = f.read_text().splitlines()
ln = next(i for i, l in enumerate(src, 1) if l.startswith('def rrf('))
jedi.Script(path=str(f), project=proj).rename(ln, 4, new_name='reciprocal_rank_fusion').apply()
"""


def notebook_map(path):
    nb = json.loads(path.read_text())
    import ast
    cells, defined_by = [], {}
    for i, c in enumerate(nb["cells"], 1):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        tree = ast.parse(src)
        defs = sorted({t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
                       if isinstance(t, ast.Name)}
                      | {t.value.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
                         if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)})
        uses = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)})
        ref = f"C{len(cells) + 1:02d}"
        ups = sorted({defined_by[u] for u in uses if u in defined_by} - {f"C{len(cells) + 1:02d}"})
        for d in defs:
            defined_by[d] = ref
        title = c.get("metadata", {}).get("title", "")
        cells.append({"ref": ref, "title": title, "defines": defs, "after": ups,
                      "runtime_s": c.get("metadata", {}).get("runtime_s")})
    lines = ["# notebook map: " + nb["metadata"]["title"]]
    for c in cells:
        lines.append(f"{c['ref']} {c['title']} | defines {','.join(c['defines']) or '-'} | "
                     f"after {','.join(c['after']) or '-'} | {c['runtime_s']}s")
    m = "\n".join(lines)
    raw = count_tokens(json.dumps(nb))
    src_only = count_tokens("\n".join("".join(c["source"]) for c in nb["cells"]))
    print("\n".join("  " + l for l in lines))
    print(f"  tokens: raw notebook JSON {raw:,}; all cell sources {src_only:,}; map {count_tokens(m):,}")
    return {"cells": len(cells), "raw": raw, "sources": src_only, "map": count_tokens(m)}


if __name__ == "__main__":
    main()
