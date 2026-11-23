"""Trial-pad selection and lift-access re-reads with an adopted pad in place."""
import cv2 as cv
import numpy as np

from image_lab.validation import Invalid

from .analysis import square_mm
from .geometry import area, draw_polygon, image, line, paint, paint_line
from .model import number
from .photos import region

# BGR colors used on the archived overlay images.
COLOR_CLEAR = (58, 213, 173)
COLOR_BLOCK = (247, 141, 94)
COLOR_UNSEEN = (175, 162, 227)
COLOR_PAD = (240, 196, 84)
COLOR_BEARING = (52, 168, 133)
COLOR_KEEPOUT = (88, 96, 238)
COLOR_TEXT = (236, 238, 240)

STATE_LABELS = {
    "clear": "clear",
    "blocked": "original-photo block",
    "unseen": "missing evidence",
    "pad": "trial pad",
}


def pad_mask(frame, mark):
    mask = region(frame, mark)
    if (mask & frame.useful).sum() < 8:
        raise Invalid(f"Trial pad “{mark['title']}” needs a larger visible footprint.")
    return mask


def approach_to_pad(probe, covered):
    """Continuous clear length from the line head before the pad outline is reached."""
    if not covered.any():
        return float(probe.coordinate_mm[-1])
    first = int(np.flatnonzero(covered)[0])
    if first == 0:
        return 0.0
    return float((probe.coordinate_mm[first - 1] + probe.coordinate_mm[first]) / 2)


def score_pad(frame, probe, mark, bearing, keepout, options):
    """Measure one candidate pad against every adoption rule."""
    footprint = pad_mask(frame, mark)
    overlap = footprint & bearing
    keepout_hit = footprint & keepout
    on_line = probe.sample(footprint)
    approach = approach_to_pad(probe, on_line)
    bearing_area = square_mm(frame, overlap & frame.useful)
    keepout_area = square_mm(frame, keepout_hit)
    minimum_bearing = number(options.get("minimum_bearing_mm2"), "minimum bearing coverage area (mm²)", 0, 2_000_000)
    minimum_approach = number(options.get("minimum_approach_mm"), "minimum continuous approach (mm)", 0, 5000)
    passes = (not keepout_hit.any() and bearing_area >= minimum_bearing
              and approach >= minimum_approach)
    return {
        "pad_id": mark["id"], "pad_title": mark["title"],
        "bearing_coverage_mm2": round(bearing_area, 3),
        "thin_lip_contact_mm2": round(keepout_area, 3),
        "approach_to_pad_mm": round(approach, 3),
        "crosses_lift_line": bool(on_line.any()),
        "meets_minimum_bearing": bool(bearing_area >= minimum_bearing),
        "meets_minimum_approach": bool(approach >= minimum_approach),
        "touches_thin_lip": bool(keepout_hit.any()),
        "eligible": bool(passes),
        "_overlap": overlap, "_keepout_hit": keepout_hit,
    }


def recommendation(rows, line_length):
    """Adopt the strongest eligible pad; never adopt a disqualified one."""
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return None, "No pad adopted: every candidate was eliminated by the thin lip or a minimum."
    best_bearing = max(row["bearing_coverage_mm2"] for row in eligible)
    best_approach = max(row["approach_to_pad_mm"] for row in eligible)
    for row in eligible:
        bearing_score = row["bearing_coverage_mm2"] / best_bearing if best_bearing else 0.0
        approach_score = row["approach_to_pad_mm"] / best_approach if best_approach else 0.0
        row["bearing_score"] = round(bearing_score, 4)
        row["approach_score"] = round(approach_score, 4)
        row["combined_score"] = round((bearing_score + approach_score) / 2, 4)
    ranked = sorted(eligible, key=lambda row: (-row["combined_score"],
                                               -row["bearing_coverage_mm2"],
                                               -row["approach_to_pad_mm"], row["pad_title"]))
    chosen = ranked[0]
    alternatives = [row for row in rows if row["pad_id"] != chosen["pad_id"]]
    notes = []
    for other in alternatives:
        if not other["eligible"]:
            reasons = []
            if other["touches_thin_lip"]:
                reasons.append("it intersects the thin-lip keep-out")
            if not other["meets_minimum_bearing"]:
                reasons.append("its bearing coverage is below the minimum")
            if not other["meets_minimum_approach"]:
                reasons.append("its continuous approach is below the minimum")
            notes.append(f"“{other['pad_title']}” is eliminated because {' and '.join(reasons)}.")
            continue
        if other["bearing_coverage_mm2"] > chosen["bearing_coverage_mm2"]:
            notes.append(
                f"“{other['pad_title']}” bears {other['bearing_coverage_mm2']:g} mm², "
                f"{other['bearing_coverage_mm2'] - chosen['bearing_coverage_mm2']:g} mm² more than the adopted pad, "
                f"but only allows {other['approach_to_pad_mm']:g} mm of reach versus "
                f"{chosen['approach_to_pad_mm']:g} mm.")
        elif other["approach_to_pad_mm"] > chosen["approach_to_pad_mm"]:
            notes.append(
                f"“{other['pad_title']}” lets the hand reach {other['approach_to_pad_mm']:g} mm, "
                f"{other['approach_to_pad_mm'] - chosen['approach_to_pad_mm']:g} mm deeper, "
                f"but bears only {other['bearing_coverage_mm2']:g} mm² versus "
                f"{chosen['bearing_coverage_mm2']:g} mm² for the adopted pad.")
        else:
            notes.append(
                f"“{other['pad_title']}” matches the adopted pad without improving bearing or reach.")
    notes.insert(0,
                 f"Adopted “{chosen['pad_title']}”: it combines {chosen['bearing_coverage_mm2']:g} mm² of "
                 f"load-bearing coverage with {chosen['approach_to_pad_mm']:g} mm of continuous approach "
                 f"along the {line_length:g} mm lift line, without touching the thin lip.")
    return chosen, " ".join(notes)


def pad_trial(frame, marks, options):
    minimum_bearing = number(options.get("minimum_bearing_mm2"),
                             "minimum bearing coverage area (mm²)", 0, 2_000_000)
    minimum_approach = number(options.get("minimum_approach_mm"),
                              "minimum continuous approach (mm)", 0, 5000)
    validated = {"minimum_bearing_mm2": minimum_bearing, "minimum_approach_mm": minimum_approach}
    _, bearing = area(frame, marks["bearing_zone"])
    keepout = region(frame, marks["keepout"])
    probe = line(frame, marks["line"])
    if marks["pad_a"]["id"] == marks["pad_b"]["id"]:
        raise Invalid("Choose two different trial pad outlines.")
    rows = [score_pad(frame, probe, marks["pad_a"], bearing, keepout, validated),
            score_pad(frame, probe, marks["pad_b"], bearing, keepout, validated)]
    chosen, rationale = recommendation(rows, float(probe.coordinate_mm[-1]))
    pad_marks = {marks["pad_a"]["id"]: marks["pad_a"], marks["pad_b"]["id"]: marks["pad_b"]}
    shown = (255 * image(frame)).astype(np.uint8)
    paint(shown, bearing, COLOR_BEARING, .32)
    paint(shown, keepout, COLOR_KEEPOUT, .32)
    for row in rows:
        paint(shown, row["_overlap"], COLOR_BEARING, .55)
        paint(shown, row["_keepout_hit"], (70, 70, 240), .7)
        outline_color = COLOR_PAD if chosen and row["pad_id"] == chosen["pad_id"] else (150, 158, 168)
        draw_polygon(shown, pad_marks[row["pad_id"]], outline_color)
    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    summary = {
        "adopted_pad": chosen["pad_title"] if chosen else "none",
        "eligible_pads": sum(row["eligible"] for row in rows),
        "recommendation_rationale": rationale,
        "lift_line_length_mm": round(float(probe.coordinate_mm[-1]), 3),
        "minimum_bearing_mm2": minimum_bearing,
        "minimum_approach_mm": minimum_approach,
    }
    return summary, public_rows, shown


def lift_states(probe, pad_on_line, threshold):
    """Per-sample state with the trial pad taking precedence over photo evidence."""
    open_luma = probe.luma >= threshold
    return np.where(pad_on_line, "pad",
                    np.where(probe.known, np.where(open_luma, "clear", "blocked"), "unseen"))


def lift_access_pad(frame, marks, options):
    threshold = number(options.get("obstacle_luma", .31), "dark stop threshold", .05, .85)
    probe = line(frame, marks["line"])
    footprint = pad_mask(frame, marks["pad"])
    pad_on_line = probe.sample(footprint)

    states = lift_states(probe, pad_on_line, threshold)
    spans = probe.runs(states)
    first = spans[0]
    continuous = first.last_mm if first.state == "clear" else 0.0
    stop_state = first.state if first.state != "clear" else (
        spans[1].state if len(spans) > 1 else "end of marked line")

    baseline_open = probe.known & (probe.luma >= threshold)
    baseline_spans = probe.classify(baseline_open, "clear")
    baseline_first = baseline_spans[0]
    baseline_clear = baseline_first.last_mm if baseline_first.state == "clear" else 0.0
    baseline_stop = baseline_first.state if baseline_first.state != "clear" else (
        baseline_spans[1].state if len(baseline_spans) > 1 else "end of marked line")

    colors = np.where(states[:, None] == "clear", COLOR_CLEAR,
                      np.where(states[:, None] == "pad", COLOR_PAD,
                               np.where(states[:, None] == "blocked", COLOR_BLOCK, COLOR_UNSEEN)))
    shown = paint_line(frame, probe, colors)
    draw_polygon(shown, marks["pad"], COLOR_PAD)
    if stop_state not in {"clear", "end of marked line"}:
        stop_index = next((index for index, state in enumerate(states) if state == stop_state), None)
        if stop_index is not None:
            point = tuple(np.rint(probe.positions[stop_index]).astype(int))
            cv.drawMarker(shown, point, COLOR_TEXT, cv.MARKER_TILTED_CROSS, 14, 2, cv.LINE_AA)
    legend = [("clear approach", COLOR_CLEAR), ("trial pad", COLOR_PAD),
              ("original-photo block", COLOR_BLOCK), ("missing evidence", COLOR_UNSEEN)]
    for index, (label, color) in enumerate(legend):
        origin = (12, 22 + 18 * index)
        cv.rectangle(shown, origin, (origin[0] + 14, origin[1] + 10), color, -1, cv.LINE_AA)
        cv.putText(shown, label, (origin[0] + 20, origin[1] + 10),
                   cv.FONT_HERSHEY_SIMPLEX, .42, COLOR_TEXT, 1, cv.LINE_AA)

    summary = {
        "continuous_approach_mm": round(continuous, 3),
        "first_stop": STATE_LABELS.get(stop_state, stop_state),
        "first_stop_state": stop_state,
        "baseline_first_stop": STATE_LABELS.get(baseline_stop, baseline_stop),
        "baseline_continuous_approach_mm": round(baseline_clear, 3),
        "line_length_mm": round(float(probe.coordinate_mm[-1]), 3),
        "pad_encountered": bool(pad_on_line.any()),
    }
    return summary, [span.row() for span in spans], shown
