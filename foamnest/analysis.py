"""Image measurements for fitted tray liners and the objects they support."""
import math

import cv2 as cv
import numpy as np

from image_lab.validation import Invalid

from .geometry import area, components, footprint_rows, largest_rectangle, line, mark_line, paint
from .model import number
from .photos import coordinates, image


def rounded(values, digits=3):
    return np.round(values, digits).tolist()


def square_mm(frame, mask):
    return float(mask.sum() * frame.step[0] * frame.step[1])


def fraction(part, total):
    return round(100 * part / max(1, total), 3)


def object_outline(frame, marks, opts):
    _, seen = area(frame, marks["region"])
    threshold = number(opts.get("luma", .53), "object brightness threshold", .05, .95)
    body = seen & (frame.luma >= threshold)
    if not body.any():
        raise Invalid("No visible object pixels meet that threshold.")
    yy, xx = np.nonzero(body)
    dims = [(np.ptp(xx) + 1) * frame.step[0], (np.ptp(yy) + 1) * frame.step[1]]
    _, rows = footprint_rows(frame, body)
    shown = image(frame)
    paint(shown, body, (.93, .65, .34), .53)
    return {"observed_area_mm2": round(square_mm(frame, body), 3), "bounding_mm": rounded(dims),
            "observed_components": len(rows)}, rows[:40], shown


def lift_access(frame, marks, opts):
    probe = line(frame, marks["line"])
    threshold = number(opts.get("obstacle_luma", .31), "dark stop threshold", .05, .85)
    open_samples = probe.luma >= threshold
    spans = probe.classify(open_samples, "clear")
    first = spans[0]
    clear = first.last_mm if first.state == "clear" else 0.
    stop = spans[1].state if len(spans) > 1 and first.state == "clear" else first.state
    if stop == "clear":
        stop = "end of marked line"
    summary = {"continuous_approach_mm": round(clear, 3), "first_stop": stop,
               "line_length_mm": round(float(probe.coordinate_mm[-1]), 3)}
    return summary, [span.row() for span in spans], mark_line(frame, probe, open_samples)


def pressure(frame, threshold):
    red, green, blue = np.moveaxis(frame.rgb, 2, 0)
    return red - np.maximum(green, blue) >= threshold


def bearing_patches(frame, marks, opts):
    _, visible = area(frame, marks["region"])
    cutoff = number(opts.get("red_excess", .19), "contact red excess", .02, .9)
    contact = visible & pressure(frame, cutoff)
    _, rows = footprint_rows(frame, contact)
    shown = image(frame)
    paint(shown, contact, (1., .57, .28))
    return {"contact_mm2": round(square_mm(frame, contact), 3),
            "observed_contact_pct": fraction(contact.sum(), visible.sum()),
            "contact_patches": len(rows)}, rows[:40], shown


def cavity_section(frame, marks, opts):
    probe = line(frame, marks["line"])
    threshold = number(opts.get("floor_luma", .61), "bright floor threshold", .05, .95)
    floor = probe.luma >= threshold
    spans = probe.classify(floor, "floor")
    gaps = [s.last_mm - s.first_mm for s in spans if s.state == "floor"]
    summary = {"floor_crossings": len(gaps), "longest_floor_span_mm": round(max(gaps, default=0), 3),
               "visible_line_pct": fraction(probe.known.sum(), len(probe.known))}
    return summary, [span.row() for span in spans], mark_line(frame, probe, floor)


def grain_direction(frame, marks, _opts):
    _, mask = area(frame, marks["region"], 90)
    yy, xx = np.nonzero(mask)
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    patch = frame.luma[y0:y1, x0:x1].astype(np.float32)
    inside = mask[y0:y1, x0:x1]
    surrounded = cv.erode(inside.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)).astype(bool)
    if not surrounded.any():
        raise Invalid("Mark a patch with interior pixels for the liner texture.")
    gx = cv.Sobel(patch, cv.CV_32F, 1, 0)[surrounded] / frame.step[0]
    gy = cv.Sobel(patch, cv.CV_32F, 0, 1)[surrounded] / frame.step[1]
    tensor = np.array([[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]])
    eigenvalues, direction = np.linalg.eigh(tensor)
    if eigenvalues.sum() <= 1e-9:
        raise Invalid("This patch has too little resolved texture.")
    dx, dy = direction[:, 0]
    angle = math.degrees(math.atan2(dy, dx)) % 180
    shown = (image(frame) * 255).astype(np.uint8)
    center = np.array([xx.mean(), yy.mean()])
    radius = .33 * min(x1 - x0, y1 - y0)
    arrow = np.array([dx / frame.step[0], dy / frame.step[1]]) * radius * min(frame.step)
    start, end = (tuple(np.rint(p).astype(int)) for p in (center - arrow, center + arrow))
    cv.line(shown, start, end, (240, 186, 88), 3, cv.LINE_AA)
    ratio = float((eigenvalues[1] - eigenvalues[0]) / eigenvalues.sum())
    rows = [{"component": k, "gradient_energy": round(float(v), 4)}
            for k, v in zip(("along", "across"), eigenvalues)]
    return {"grain_angle_deg": round(angle, 2), "directional_strength": round(ratio, 4)}, rows, shown


def landmark_offsets(frame, other, marks, _opts):
    fixed = coordinates(marks["reference"]["points"], frame.rgb.shape)
    moving = coordinates(marks["comparison"]["points"], other.rgb.shape)
    if len(fixed) != len(moving):
        raise Invalid("Mark the same number of landmarks in the same order.")
    vectors = fixed * frame.step - moving * other.step
    center = np.median(vectors, axis=0)
    residual = np.linalg.norm(vectors - center, axis=1)
    shown = (image(frame) * 255).astype(np.uint8)
    rows = []
    for index, (point, second, shift, error) in enumerate(zip(fixed, moving, vectors, residual), 1):
        start = tuple(np.rint(second).astype(int))
        end = tuple(np.rint(point).astype(int))
        cv.arrowedLine(shown, start, end, (236, 180, 108), 2, cv.LINE_AA, tipLength=.3)
        cv.circle(shown, end, 4, (240, 194, 122), 1, cv.LINE_AA)
        rows.append({"landmark": index, "reference_mm": rounded(point * frame.step),
                     "comparison_mm": rounded(second * other.step), "offset_mm": rounded(shift),
                     "residual_mm": round(float(error), 3)})
    return {"median_offset_mm": rounded(center), "largest_residual_mm": round(float(residual.max()), 3),
            "landmarks": len(rows)}, rows, shown


def rub_marks(frame, marks, opts):
    _, visible = area(frame, marks["region"])
    contrast = number(opts.get("contrast", .075), "local contrast", .02, .35)
    luminance = frame.luma
    estimated = np.where(frame.useful, luminance, float(np.median(luminance[visible]))).astype(np.float32)
    baseline = cv.GaussianBlur(estimated, (0, 0), sigmaX=9)
    mark = visible & (np.abs(estimated - baseline) >= contrast)
    _, rows = footprint_rows(frame, mark)
    shown = image(frame)
    paint(shown, mark, (1., .4, .25))
    return {"visible_mark_pct": fraction(mark.sum(), visible.sum()), "components": len(rows)}, rows[:40], shown


def liner_tone(frame, marks, _opts):
    _, visible = area(frame, marks["region"])
    values = cv.cvtColor(frame.rgb[visible].astype(np.float32).reshape((-1, 1, 3)),
                         cv.COLOR_RGB2LAB).reshape((-1, 3))
    low, mid, high = np.percentile(values, [10, 50, 90], axis=0)
    shown = image(frame)
    paint(shown, visible, (.33, .73, .75), .22)
    return {"observed_pixels": int(visible.sum()), "median_lab": rounded(mid)}, [
        {"channel": channel, "median": round(float(mid[i]), 3),
         "p10": round(float(low[i]), 3), "p90": round(float(high[i]), 3)}
        for i, channel in enumerate(("L*", "a*", "b*"))], shown


def coverage_states(first, second):
    return first & second, first & ~second, second & ~first


def paired_coverage(frame, other, marks, _opts):
    domain, visible = area(frame, marks["region"])
    known = other.useful
    masks = (*coverage_states(visible, domain & known), domain & ~visible & ~known)
    shown = image(frame)
    rows = []
    for mask, label, color in zip(masks, ("shared", "first only", "second only", "neither"),
                                  ((.3, .85, .69), (.96, .73, .32), (.46, .62, .94), (.56, .49, .72))):
        paint(shown, mask, color)
        rows.append({"visibility": label, "area_mm2": round(square_mm(frame, mask), 3),
                     "region_pct": fraction(mask.sum(), domain.sum())})
    return {"shared_pct": rows[0]["region_pct"], "region_mm2": round(square_mm(frame, domain), 3)}, rows, shown


def bearing_change(frame, other, marks, opts):
    domain, visible = area(frame, marks["region"])
    common = visible & other.useful
    cutoff = number(opts.get("red_excess", .19), "contact red excess", .02, .9)
    first = common & pressure(frame, cutoff)
    second = common & pressure(other, cutoff)
    carried, lost, gained = coverage_states(first, second)
    shown = image(frame)
    rows = []
    for mask, label, color in zip((carried, lost, gained, domain & ~common),
                                  ("retained", "lost", "gained", "unseen"),
                                  ((.31, .77, .67), (.92, .38, .26), (.42, .55, .96), (.56, .49, .72))):
        paint(shown, mask, color)
        rows.append({"contact_change": label, "area_mm2": round(square_mm(frame, mask), 3)})
    return {"observed_both_pct": fraction(common.sum(), domain.sum()),
            "retained_contact_mm2": rows[0]["area_mm2"]}, rows, shown


def label_window(frame, marks, opts):
    _, visible = area(frame, marks["region"])
    threshold = number(opts.get("floor_luma", .55), "clear label floor threshold", .05, .95)
    clear = visible & (frame.luma >= threshold)
    yy, xx = np.nonzero(visible)
    left, top, right, bottom = xx.min(), yy.min(), xx.max() + 1, yy.max() + 1
    x0, y0, x1, y1 = largest_rectangle(clear[top:bottom, left:right])
    shown = image(frame)
    paint(shown, clear, (.22, .72, .74), .22)
    chosen = np.zeros_like(clear)
    chosen[top+y0:top+y1, left+x0:left+x1] = True
    paint(shown, chosen, (.95, .66, .29), .74)
    dims = [(x1 - x0) * frame.step[0], (y1 - y0) * frame.step[1]]
    bounds = {"x_mm": round(float((left+x0)*frame.step[0]), 3),
              "y_mm": round(float((top+y0)*frame.step[1]), 3),
              "width_mm": round(float(dims[0]), 3), "height_mm": round(float(dims[1]), 3)}
    summary = {"largest_label_area_mm2": round(square_mm(frame, chosen), 3),
               "label_available": bool(chosen.any()), "label_dimensions_mm": rounded(dims)}
    return summary, [bounds] if chosen.any() else [], shown


def liner_bridges(frame, marks, opts):
    _, visible = area(frame, marks["region"])
    threshold = number(opts.get("foam_luma", .4), "foam brightness threshold", .05, .95)
    wear = number(opts.get("wear_mm", 1.5), "edge wear", .1, 15)
    raw = visible & (frame.luma >= threshold)
    xrad, yrad = [max(1, round(wear / step)) for step in frame.step]
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * xrad + 1, 2 * yrad + 1))
    reduced = cv.erode(raw.astype(np.uint8), kernel, borderType=cv.BORDER_CONSTANT, borderValue=0).astype(bool)
    _, before = components(frame, raw, .1)
    _, rows = footprint_rows(frame, reduced)
    shown = image(frame)
    paint(shown, raw & ~reduced, (.94, .45, .27))
    paint(shown, reduced, (.31, .77, .67))
    return {"visible_pieces": len(before), "pieces_under_wear": len(rows),
            "remaining_mm2": round(square_mm(frame, reduced), 3), "edge_wear_mm": wear}, rows[:40], shown
