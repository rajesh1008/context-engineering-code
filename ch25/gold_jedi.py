"""Independent reference oracle for the Chapter 25 lab: jedi's find-references.

Runs in 04-code/.venv-ch25 (jedi 0.20.1). For each target symbol, records every
reference location jedi finds in the snapshot. The lab never reads jedi's
answers while building packets; it only scores packets against them.
Run: ../.venv-ch25/bin/python gold_jedi.py SNAPSHOT_DIR TARGETS.json OUT.json
"""
import json, pathlib, sys, time
import jedi, tempfile
jedi.settings.cache_directory = tempfile.mkdtemp()  # no stale parses between runs

root = pathlib.Path(sys.argv[1]).resolve()
targets = json.load(open(sys.argv[2]))
dirs = [str(root)] + [str(p) for p in root.iterdir() if p.is_dir()]
project = jedi.Project(path=str(root), added_sys_path=dirs, smart_sys_path=False)
out, t0 = {}, time.time()
for t in targets:
    f = root / t["path"]
    line = f.read_text().splitlines()[t["line"] - 1]
    key = ("class " if t["kind"] == "class" else "def ") + t["name"]
    col = line.index(key) + len(key) - len(t["name"])
    refs = jedi.Script(path=str(f), project=project).get_references(t["line"], col, scope="project")
    out[t["qname"]] = sorted({(str(pathlib.Path(r.module_path).relative_to(root)), r.line)
                              for r in refs if r.module_path and str(r.module_path).startswith(str(root))})
json.dump(out, open(sys.argv[3], "w"), indent=0)
print(f"{len(out)} symbols, {sum(len(v) for v in out.values())} references, {time.time() - t0:.0f} s")
