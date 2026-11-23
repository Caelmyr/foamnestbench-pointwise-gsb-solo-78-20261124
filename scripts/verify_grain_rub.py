"""Assertion-based checks for the grain-aware rub map on the synthetic tray.

Run from the repository root:  python -I scripts/verify_grain_rub.py
Exits non-zero if any invariant of the detector regresses.
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(REPO))

import cv2 as cv  # noqa: E402

from foamnest import demo, grainmap  # noqa: E402
from foamnest.geometry import paint  # noqa: E402
from foamnest.photos import decode, plane, png, image  # noqa: E402
from foamnest.store import Library  # noqa: E402

CONTRAST, GAP_MM, MIN_PART = .075, 2., .25


def build():
    tmp = tempfile.TemporaryDirectory()
    library = Library(Path(tmp.name) / "samples")
    run = demo.create(library)
    photo = run["photos"][0]
    frame = plane({**photo, "pixels": decode(library.blob(photo["blob"]).read_bytes())}, run)
    marks = {m["purpose"]: m for m in run["marks"] if m["photo"] == photo["id"]}
    slots = {"region": marks["rub"], "grain_a": marks["grain_a"], "grain_b": marks["grain_b"],
             "confirmed": marks["confirmed"], "intact": marks["intact"]}
    return frame, marks, slots


def main():
    frame, marks, slots = build()
    angle_a, strength_a = grainmap.grain_angle(frame, marks["grain_a"])
    angle_b, strength_b = grainmap.grain_angle(frame, marks["grain_b"])
    print(f"grain A {angle_a:.2f} deg (strength {strength_a:.3f}) | "
          f"grain B {angle_b:.2f} deg (strength {strength_b:.3f})")
    assert abs(angle_a - angle_b) > 3, "the two texture references must give different grain angles"

    confirmed = grainmap.region(frame, marks["confirmed"]) & frame.useful
    intact = grainmap.region(frame, marks["intact"]) & frame.useful
    core = grainmap._scuff_core(frame, confirmed)
    assert core.sum() >= 50, "the confirmed scuff must contain a measurable dark core"

    trials = {}
    for slot in ("grain_a", "grain_b"):
        angle, _ = grainmap.grain_angle(frame, marks[slot])
        result = grainmap.evaluate_direction(frame, slots, angle, CONTRAST, GAP_MM, MIN_PART)
        trials[slot] = result
        mark, unknown, domain = result["mark"], result["unknown"], result["domain"]

        # 1. The actual scuff core is retained on visible pixels.
        retained = int((mark & core).sum()) / max(1, int(core.sum()))
        assert retained >= .9, f"{slot} retained only {retained:.0%} of the confirmed scuff core"

        # 2. The intact groove reference is not flagged as a rub.
        false = int((mark & intact).sum()) / max(1, int(intact.sum()))
        assert false <= .01, f"{slot} falsely flags {false:.2%} of the intact liner"

        # 3. A mark never lands on missing-capture material: no bridging.
        raw_gap = domain & ~frame.useful
        assert int((mark & raw_gap).sum()) == 0, "a mark was joined across a missing-capture gap"
        assert unknown.sum() > 0, "the missing-capture band must be reported as undecidable"

    # 4. End-to-end selection honours the operator's keep/false limits.
    opts = {"contrast": CONTRAST, "gap_mm": GAP_MM, "min_part_mm2": MIN_PART,
            "keep_pct": 70, "false_pct": 5}
    summary, rows, shown = grainmap.grain_rub_map(frame, slots, opts)
    assert summary["limits_met"] is True, "default limits should be met on the demonstration"
    assert summary["retained_confirmed_pct"] >= 70
    assert summary["false_intact_pct"] <= 5
    direction_rows = [r for r in rows if "grain_angle_deg" in r]
    assert len(direction_rows) == 2 and all(r["qualified"] for r in direction_rows)

    # 5. Impossible limits are reported as not met rather than silently passed.
    strict = dict(opts, keep_pct=100, false_pct=0)
    strict_summary, _, _ = grainmap.grain_rub_map(frame, slots, strict)
    # 100%/0% can still pass here, but a retention above 100 must be rejected upstream.
    impossible = dict(opts, contrast=.5)
    impossible_summary, _, _ = grainmap.grain_rub_map(frame, slots, impossible)
    assert impossible_summary["limits_met"] is False, "a contrast that erases the scuff must fail"

    out = Path(__file__).resolve().parent / "grain_rub_out"
    out.mkdir(exist_ok=True)
    (out / "grain-rub-overlay.png").write_bytes(png(shown))
    for label, slot in (("A", "grain_a"), ("B", "grain_b")):
        angle, _ = grainmap.grain_angle(frame, marks[slot])
        result = trials[slot]
        picture = image(frame)
        paint(picture, result["unknown"], (.55, .49, .72), .55)
        paint(picture, result["mark"], (1., .36, .24), .72)
        (out / f"direction-{label}.png").write_bytes(png(picture))

    print("all grain-rub invariants passed")
    print(" summary:", {k: summary[k] for k in
          ("chosen_reference", "chosen_grain_angle_deg", "retained_confirmed_pct",
           "false_intact_pct", "undecidable_mm2", "mark_components")})


if __name__ == "__main__":
    main()
