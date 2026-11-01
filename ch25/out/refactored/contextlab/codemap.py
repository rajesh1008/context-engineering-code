"""contextlab.codemap: a repository as context (Chapter 25, v5.7).

SymbolIndex parses Python files with the standard-library `ast` module into
symbols (functions, classes, methods) and reference edges, each edge tagged
with how it was resolved: exact (through an import, the same module or
self), name-only (one symbol in the repository has that name) or ambiguous
(several do). render_map() ranks files and symbols with a personalised
PageRank and prints signatures until a token budget is spent. refresh()
re-indexes changed files by content hash and, unless told not to, re-resolves
the edges of every file that refers to them. resolve() checks a symbol's
recorded location against the file on disk before anyone reads it.
Static analysis only: string dispatch and getattr() leave no edge.
"""
from __future__ import annotations

import ast
import hashlib
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field

from contextlab.chunker import count_tokens


@dataclass(frozen=True)
class Symbol:
    qname: str          # contextlab.retrieval.fusion.rrf, contextlab.policy.Policy.admit
    name: str
    kind: str           # function | class | method
    path: str           # repository-relative
    line: int
    end: int
    signature: str
    doc: str = ""


@dataclass(frozen=True)
class Edge:
    src: str            # qname of the enclosing symbol, or "<path>" for module level
    dst: str
    path: str
    line: int
    confidence: str     # exact | name-only | ambiguous


# Attribute names that dict, list, str and file objects also have: never trusted alone.
COMMON = set("get add append items keys values update pop copy read write text name size "
             "search delete format split join strip lower upper count index sort open close".split())
WEIGHT = {"exact": 1.0, "name-only": 0.2, "ambiguous": 0.0}


def module_of(path: str) -> str:
    parts = list(pathlib.PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _sig(node) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    args = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"def {node.name}({args}){ret}"


def _defs(path: str, tree) -> list[Symbol]:
    mod, out = module_of(path), []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            doc = (ast.get_docstring(node) or "").split("\n")[0]
            out.append(Symbol(f"{mod}.{node.name}", node.name, kind, path, node.lineno,
                              node.end_lineno, _sig(node), doc))
            if kind == "class":
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        d = (ast.get_docstring(m) or "").split("\n")[0]
                        out.append(Symbol(f"{mod}.{node.name}.{m.name}", m.name, "method", path,
                                          m.lineno, m.end_lineno, _sig(m), d))
    return out


def _imports(path: str, tree) -> dict[str, str]:
    """Local alias -> fully qualified name, with relative imports resolved."""
    pkg = module_of(path).split(".")
    if not path.endswith("__init__.py"):
        pkg = pkg[:-1]
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            base = pkg[:len(pkg) - node.level + 1] if node.level else []
            mod = ".".join(base + ([node.module] if node.module else []))
            for a in node.names:
                out[a.asname or a.name] = f"{mod}.{a.name}"
    return out


@dataclass
class SymbolIndex:
    root: pathlib.Path
    files: dict = field(default_factory=dict)        # path -> sha
    symbols: dict = field(default_factory=dict)      # qname -> Symbol
    edges: list = field(default_factory=list)
    by_name: dict = field(default_factory=lambda: defaultdict(set))
    imports: dict = field(default_factory=dict)      # module -> {alias: qualified}

    def canon(self, name, hops=4):
        """Follow re-exports (from .b import f in module a) to the defining symbol."""
        for _ in range(hops):
            if name is None or name in self.symbols:
                return name
            mod, _, attr = name.rpartition(".")
            name = self.imports.get(mod, {}).get(attr)
        return name if name in self.symbols else None

    @classmethod
    def build(cls, root, include=("contextlab",)) -> "SymbolIndex":
        idx = cls(pathlib.Path(root))
        paths = sorted(str(p.relative_to(idx.root)) for d in include
                       for p in (idx.root / d).rglob("*.py")
                       if "__pycache__" not in p.parts and "tests" not in p.parts)
        idx._index_files(paths)
        idx._link(paths)
        return idx

    @property
    def revision(self) -> str:
        return sha("".join(f"{p}:{h}" for p, h in sorted(self.files.items())))

    def _index_files(self, paths):
        for p in paths:
            for q in [q for q, s in self.symbols.items() if s.path == p]:
                self.by_name[self.symbols[q].name].discard(q)
                del self.symbols[q]
            text = (self.root / p).read_text()
            self.files[p] = sha(text)
            tree = ast.parse(text)
            self.imports[module_of(p)] = self._local(p, _imports(p, tree))
            for s in _defs(p, tree):
                self.symbols[s.qname] = s
                self.by_name[s.name].add(s.qname)

    def _local(self, p, imp):
        """Scripts import siblings by bare name (import lakeshore19): qualify them."""
        here = module_of(p).rpartition(".")[0]
        roots = {pathlib.PurePosixPath(f).parts[0] for f in self.files} | {"contextlab"}
        return {a: (q if q.split(".")[0] in roots or not here else f"{here}.{q}")
                for a, q in imp.items()}

    def _link(self, paths):
        self.edges = [e for e in self.edges if e.path not in set(paths)]
        for p in paths:
            self.edges += self._edges_for(p)

    def _edges_for(self, p) -> list[Edge]:
        tree = ast.parse((self.root / p).read_text())
        mod, imp, out = module_of(p), self.imports[module_of(p)], []
        spans = sorted((s.line, s.end, s.qname, s.kind) for s in self.symbols.values() if s.path == p)

        def owner(line):
            best = f"<{p}>"
            for a, b, q, _ in spans:
                if a <= line <= b:
                    best = q                       # innermost wins: methods come after classes
            return best

        def cls_of(line):
            for a, b, q, k in spans:
                if k == "class" and a <= line <= b:
                    return q
            return None

        for node in ast.walk(tree):
            dst, conf = None, None
            if isinstance(node, ast.ImportFrom):        # an import is a use: a rename must edit it
                for a in node.names:
                    q = self.canon(imp.get(a.asname or a.name))
                    if q:
                        out.append(Edge(owner(node.lineno), q, p, node.lineno, "exact"))
                continue
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if f"{mod}.{node.id}" in self.symbols:
                    dst, conf = f"{mod}.{node.id}", "exact"
                elif self.canon(imp.get(node.id)):
                    dst, conf = self.canon(imp[node.id]), "exact"
            elif isinstance(node, ast.Attribute):
                base = node.value.id if isinstance(node.value, ast.Name) else None
                target = self.canon(f"{imp[base]}.{node.attr}") if base in imp else None
                c = cls_of(node.lineno)
                if target:
                    dst, conf = target, "exact"
                elif base in ("self", "cls") and c and f"{c}.{node.attr}" in self.symbols:
                    dst, conf = f"{c}.{node.attr}", "exact"
                elif self.by_name.get(node.attr) and not node.attr.startswith("__"):
                    cands = sorted(self.by_name[node.attr])
                    conf = "name-only" if len(cands) == 1 and node.attr not in COMMON else "ambiguous"
                    for q in cands:
                        out.append(Edge(owner(node.lineno), q, p, node.lineno, conf))
                    continue
            if dst:
                out.append(Edge(owner(node.lineno), dst, p, node.lineno, conf))
        return out

    def refresh(self, changed, relink_dependents=True) -> dict:
        """Re-index changed files; with relink_dependents, re-resolve every file that
        referred to a symbol in them. Returns what was done, for the log."""
        changed = sorted(set(changed))
        before = {q for q, s in self.symbols.items() if s.path in changed}
        gone = [p for p in changed if not (self.root / p).exists()]
        for p in gone:
            self.files.pop(p, None)
            self.imports.pop(module_of(p), None)
            for q in [q for q, s in self.symbols.items() if s.path == p]:
                self.by_name[self.symbols[q].name].discard(q)
                del self.symbols[q]
        live = [p for p in changed if p not in gone]
        self._index_files(live)
        relink = set(live) | set(gone)
        if relink_dependents:
            relink |= {e.path for e in self.edges if e.dst in before}
        relink = sorted(p for p in relink if p in self.files or p in gone)
        self.edges = [e for e in self.edges if e.path not in relink]
        for p in relink:
            if p in self.files:
                self.edges += self._edges_for(p)
        return {"reindexed": live, "removed": gone, "relinked": relink, "revision": self.revision}

    def callers(self, qname, min_conf="ambiguous") -> list[Edge]:
        rank = {"exact": 0, "name-only": 1, "ambiguous": 2}
        return [e for e in self.edges if e.dst == qname and rank[e.confidence] <= rank[min_conf]]

    def resolve(self, qname) -> dict:
        """Check the recorded location against the file on disk before it is read."""
        s = self.symbols.get(qname)
        if s is None:
            return {"qname": qname, "status": "unknown"}
        f = self.root / s.path
        if not f.exists():
            return {"qname": qname, "status": "file-missing", "path": s.path}
        text = f.read_text()
        if sha(text) == self.files.get(s.path):
            return {"qname": qname, "status": "fresh", "path": s.path, "lines": (s.line, s.end)}
        lines = text.splitlines()
        head = lines[s.line - 1] if s.line <= len(lines) else ""
        ok = f"def {s.name}(" in head or f"class {s.name}" in head
        return {"qname": qname, "status": "moved-within-file" if not ok else "shifted-but-present",
                "path": s.path, "lines": (s.line, s.end)}


def pagerank(nodes, weighted_edges, personal=None, d=0.85, iters=60) -> dict:
    nodes = sorted(nodes)
    p = {n: (personal or {}).get(n, 0.0) for n in nodes}
    tot = sum(p.values())
    p = {n: (v / tot if tot else 1 / len(nodes)) for n, v in p.items()}
    out_w = defaultdict(float)
    for (a, b), w in weighted_edges.items():
        out_w[a] += w
    r = dict(p)
    for _ in range(iters):
        nxt = {n: (1 - d) * p[n] for n in nodes}
        dangling = sum(r[n] for n in nodes if out_w[n] == 0)
        for (a, b), w in weighted_edges.items():
            nxt[b] += d * r[a] * w / out_w[a]
        for n in nodes:
            nxt[n] += d * dangling * p[n]
        r = nxt
    return r


def render_map(idx: SymbolIndex, budget: int, focus=()) -> tuple[str, list[str]]:
    """Rank files by personalised PageRank over the file reference graph, then print
    each file's most referenced signatures until `budget` tokens are spent."""
    w = defaultdict(float)
    for e in idx.edges:
        if e.dst in idx.symbols:
            a, b = e.path, idx.symbols[e.dst].path
            if a != b and WEIGHT[e.confidence]:
                w[(a, b)] += WEIGHT[e.confidence]
                w[(b, a)] += WEIGHT[e.confidence]     # a change flows back to its users
    personal = {idx.symbols[q].path: 1.0 for q in focus if q in idx.symbols}
    rank = pagerank(idx.files, w, personal or None)
    indeg = defaultdict(float)
    for src_file, dst, conf in {(e.path, e.dst, e.confidence) for e in idx.edges}:
        indeg[dst] += WEIGHT[conf]                 # distinct referring files, not raw counts
    lines, used, shown = [f"# repository map @ {idx.revision}"], 0, []
    used = count_tokens(lines[0])
    for path in sorted(idx.files, key=lambda f: (f not in personal, -rank[f], f)):
        syms = sorted((s for s in idx.symbols.values() if s.path == path),
                      key=lambda s: (-(indeg[s.qname] + (5 if s.qname in focus else 0)), s.line))
        if not syms:
            continue
        block = [f"{path}:"] + [f"  {s.line:>4}: {'  ' if s.kind == 'method' else ''}{s.signature}"
                                for s in syms[:6]]
        cost = count_tokens("\n".join(block))
        if used + cost > budget:
            continue
        lines += block; used += cost; shown.append(path)
    return "\n".join(lines), shown
