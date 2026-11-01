"""Small helpers shared by the Chapter 3 scripts: output capture and number formats."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"


def tee(name: str, fn) -> str:
    """Run fn, print its output and save it to out/<name>.txt."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    text = buf.getvalue()
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{name}.txt").write_text(text)
    sys.stdout.write(text)
    return text
