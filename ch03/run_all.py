"""Run every Chapter 3 script in chapter order and collect the outputs in output.txt."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["tokens", "embeddings", "lab", "fix_ablation", "variance", "kv_memory", "cache_layout",
           "system_turn", "cag_snapshot", "order", "flip_probe"]

if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    for s in SCRIPTS:
        print(f"===== {s}.py", flush=True)
        runpy.run_path(str(HERE / f"{s}.py"), run_name="__main__")
    out = "".join(f"===== {p.stem}\n{p.read_text()}\n" for p in sorted((HERE / "out").glob("*.txt"),
                  key=lambda p: [int(x) if x.isdigit() else x for x in p.stem.split("_")[0].split(".")]))
    (HERE / "output.txt").write_text(out)
