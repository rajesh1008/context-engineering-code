import pathlib, textwrap
from contextlab import codemap as CM
from contextlab.chunker import count_tokens

SNAP = pathlib.Path(__file__).parent / "snapshot"


def mini(tmp_path, files):
    for name, body in files.items():
        f = tmp_path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(textwrap.dedent(body))
    return tmp_path


def test_snapshot_index_finds_exact_callers():
    idx = CM.SymbolIndex.build(SNAP, include=["contextlab", "ch12"])
    paths = {e.path for e in idx.callers("contextlab.retrieval.fusion.rrf", "exact")}
    assert "contextlab/retrieval/pipeline.py" in paths and "ch12/lab.py" in paths


def test_map_respects_budget():
    idx = CM.SymbolIndex.build(SNAP, include=["contextlab"])
    for b in (200, 800):
        text, shown = CM.render_map(idx, b)
        assert count_tokens(text) <= b and shown


def test_focus_puts_target_file_first():
    idx = CM.SymbolIndex.build(SNAP, include=["contextlab"])
    _, shown = CM.render_map(idx, 400, focus=["contextlab.docgate.psi"])
    assert shown[0] == "contextlab/docgate.py"


def test_move_with_reexport_needs_dependents_relinked(tmp_path):
    root = mini(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "def f():\n    return 1\n",
                           "pkg/c.py": "from pkg.a import f\n\ndef g():\n    return f()\n"})
    stale = CM.SymbolIndex.build(root, include=["pkg"])
    (root / "pkg/b.py").write_text("def f():\n    return 1\n")
    (root / "pkg/a.py").write_text("from pkg.b import f  # re-exported\n")
    naive = CM.SymbolIndex.build(root, include=["pkg"]); naive.edges = []
    naive = stale; naive.refresh(["pkg/a.py", "pkg/b.py"], relink_dependents=False)
    assert not [e for e in naive.callers("pkg.b.f") if e.path == "pkg/c.py"]
    full = CM.SymbolIndex.build(root, include=["pkg"])
    assert [e for e in full.callers("pkg.b.f") if e.path == "pkg/c.py"]


def test_relink_repairs_the_moved_symbol(tmp_path):
    root = mini(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "def f():\n    return 1\n",
                           "pkg/c.py": "from pkg.a import f\n\ndef g():\n    return f()\n"})
    idx = CM.SymbolIndex.build(root, include=["pkg"])
    (root / "pkg/b.py").write_text("def f():\n    return 1\n")
    (root / "pkg/a.py").write_text("from pkg.b import f  # re-exported\n")
    idx.refresh(["pkg/a.py", "pkg/b.py"], relink_dependents=True)
    assert {e.path for e in idx.callers("pkg.b.f")} >= {"pkg/c.py"}


def test_resolve_flags_a_shifted_definition(tmp_path):
    root = mini(tmp_path, {"pkg/__init__.py": "", "pkg/a.py": "def f():\n    return 1\n"})
    idx = CM.SymbolIndex.build(root, include=["pkg"])
    assert idx.resolve("pkg.a.f")["status"] == "fresh"
    (root / "pkg/a.py").write_text("# header\n\ndef f():\n    return 1\n")
    assert idx.resolve("pkg.a.f")["status"] == "moved-within-file"


def test_dynamic_dispatch_leaves_no_static_edge():
    root = pathlib.Path(__file__).parent
    idx = CM.SymbolIndex.build(root, include=["fixtures"])
    assert idx.callers("fixtures.sellerdesk.handle_refund") == []
