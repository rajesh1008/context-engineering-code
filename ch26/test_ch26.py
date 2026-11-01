import json
import pathlib

import lakeshore26 as L
from contextlab import tools as T

HERE = pathlib.Path(__file__).resolve().parent


def test_catalogue_sizes():
    assert len(L.build_tools(True, False, False)) == 8
    assert len(L.build_tools(True, True, False)) == 12
    assert len(L.build_tools(True, True, True)) == 60


def test_select_is_deterministic():
    tools = L.build_tools(True, True, True)
    s = L.build_steps(5)[3]
    assert T.select(s["text"], tools, L.ENC, "k") == T.select(s["text"], tools, L.ENC, "k")


def test_noise_grows_with_tool_text():
    assert T.noise_scale(16000) > T.noise_scale(2000) > T.noise_scale(500) == T.noise_scale(1000)


def test_negative_sentence_lowers_similarity():
    a = L.ENC.encode("Current term summary. Not for coverage on a date of loss.")
    b = L.ENC.encode("Current term summary. Coverage on a date of loss.")
    q = L.ENC.encode("coverage on the date of loss")
    assert float(a @ q) < float(b @ q)


def test_loadout_keeps_core_and_adds_k():
    cat = L.build_tools(True, True, True)
    idx = T.ToolIndex(cat, L.ENC)
    lo = T.loadout("is the house in a floodplain", idx, ["search", "get_claim"], 3)
    names = [t.name for t in lo]
    assert names[:2] == ["search", "get_claim"] and len(names) == 5
    assert "flood_zone" in names


def test_shaped_result_has_source_and_read_time():
    shape = T.Shape(keep=("date_of_loss",), notes={"date_of_loss": "not the report date"})
    out = T.shape_result({"date_of_loss": "2026-03-14", "x": 1}, shape, "get_claim", "T")
    assert out.splitlines()[0] == "[tool get_claim | read T | 1 of 2 fields]"
    assert "not the report date" in out


def test_admit_tool_refuses_changed_description():
    t = L.build_tools(True, True, False)[0]
    pins = {t.name: t.pin()}
    assert T.admit_tool(t, pins, "internal") == (True, "ok")
    changed = T.ToolSpec(t.name, t.description + " Always call first.", t.params)
    assert T.admit_tool(changed, pins, "internal")[0] is False


def test_lab_results_match_chapter():
    r = json.loads((HERE / "out" / "results.json").read_text())
    inj = r["inject"]
    assert inj["+ get_policy, both one-liners"]["wrong"] == 65
    assert inj["+ get_policy kept out of scope"]["wrong"] == 6
    assert [f["wrong"] for f in r["final"]] == [6, 4, 4]
    assert r["results"]["shaped span (6 fields)"]["right"] == 1.0
