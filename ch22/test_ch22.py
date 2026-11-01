"""Pin the Chapter 22 numbers quoted in the text. Run: pytest -q"""
import json
import pathlib
import subprocess
import sys

from contextlab import memguard as G
from contextlab.memguard import MemoryRecord as M, Origin as O
import replay22 as R

HERE = pathlib.Path(__file__).resolve().parent


def res():
    subprocess.run([sys.executable, "lab.py"], cwd=HERE, check=True, capture_output=True)
    return json.loads((HERE / "out/results.json").read_text())


def test_write_gate_holds_unverified_coverage_claims():
    m = M("x", "team", "WI", "t", None, (O("determination", "D1"),), asserts=("wi_backup", "excluded"))
    assert G.write_gate(m) == "held" or m.status == "held"
    g = M("y", "team", "WI", "t", None, (O("span", "HO0495#1"),), asserts=("wi_backup", "covered"))
    G.write_gate(g)
    assert g.status == "live"


def test_family_follows_parents():
    _, store, _, _, _ = R.run()
    assert G.family(store, "M02116") == {"M02116", "M-A01-D", "M-A02-D"}


def test_numbers_in_text():
    r = res()
    assert (r["ch20"], r["ch21_cause_path"], r["ch21"], r["ch22"]) == (7, 13, 15, 9)
    assert r["causes"] == {"over-remembering": 1, "poison": 6, "stale": 1}
    assert r["readers"] == 64
    assert r["outcome"]["wrong (family)"] == 6 and r["outcome"]["outranked"] == 30
    assert r["remediation"] == {"rerun": 64, "changed": 6, "fixed": 6}
    assert r["removal"]["text search"]["legit_removed"] == 6
    assert r["injection"]["no guard"]["wrong"] == 4
    assert r["injection"]["read-time monitor"]["wrong"] == 0
    assert len(r["injection"]["write gate"]["held"]) == 8
    assert r["leak"]["address"]["violations"] == 144 and r["leak"]["policy"]["violations"] == 0


def test_chapter21_lesson_readers():
    dets, _, log, _, _ = R.run()
    assert sum("M02116" in r.mids for r in log.reads) == 37
