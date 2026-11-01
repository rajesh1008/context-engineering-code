"""The two-point check (Listing 4.8): re-measure the unpadded point and the
recorded knee, on the model the curve was drawn on and after a model swap.
Run: python 04-code/ch04/monitor_ch04.py   (stand-in models)"""
import lab
from contextlab.diagnostics import Diagnostics


def two_point_check(pipeline, dets, recorded_knee: int, tolerance: float = 0.02):
    """Re-measure the unpadded point and the recorded knee; 100 determinations
    cost 200 calls instead of the 1,300 a full 13-point curve costs."""
    (_, t0, w0), (_, tk, wk) = Diagnostics.rot_curve(
        pipeline, dets, [0, recorded_knee], pad_fn=lab.neutral)
    ok = wk - w0 <= tolerance + 1e-9
    return t0, w0, tk, wk, ok

RECORDED_KNEE = 32_000          # measured on sim-large, Chapter 4 lab, section 1


def report(label, model, out=print):
    t0, w0, tk, wk, ok = two_point_check(lab.current.with_config(model=model), lab.dets, RECORDED_KNEE)
    verdict = "within band: keep the budget" if ok else "OUT OF BAND: re-draw the curve"
    out(f"{label:<30} 0k {w0:5.1%} ({t0:,.0f} tok) | {RECORDED_KNEE // 1000}k {wk:5.1%} ({tk:,.0f} tok) | {verdict}")
    return ok


if __name__ == "__main__":
    report("sim-large (curve drawn on it)", "sim-large")
    report("model swapped to sim-mid", "sim-mid")
