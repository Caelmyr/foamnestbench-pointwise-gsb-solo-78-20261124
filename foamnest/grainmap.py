"""Grain-aware rub detection on diagonal foam-liner grooves.

The original rub map (``analysis.rub_marks``) subtracts an isotropic blur, so a
regular diagonal groove is as dark as a real scuff and gets flagged as a mark.
This module instead builds the background from the groove structure itself:

1. Each circled texture reference yields a candidate grain-line angle from its
   structure tensor (the eigenvector of weakest gradient energy).
2. Across the inspection window the groove brightness is a periodic function of
   the coordinate perpendicular to that line.  A robust phase-binned median
   estimates that periodic baseline directly in calibrated-frame coordinates,
   so no rotated grid can fragment the evidence or bridge missing-capture gaps.
3. A rub is material darker than the groove baseline, so only the *negative*
   residual is a candidate.  Missing-capture pixels never enter the template and
   are reported separately with a safety halo; a mark is never joined across
   unseen material.
"""
import math

import cv2 as cv
import numpy as np

from image_lab.validation import Invalid

from .geometry import area, footprint_rows, paint
from .model import number
from .photos import image, region


def _mask_pixels(frame, mask):
    yy, xx = np.nonzero(mask)
    if len(xx) < 32:
        raise Invalid("Mark a larger visible part of the tray.")
    return xx * frame.step[0], yy * frame.step[1]


def grain_angle(frame, mark):
    """Groove line direction (degrees, 0-180) from the structure tensor.

    The groove line follows the smallest-eigenvalue eigenvector because the
    gradients concentrate across the grooves.
    """
    _, mask = area(frame, mark, 32)
    x_mm, y_mm = _mask_pixels(frame, mask)
    x0, x1 = int(x_mm.min() / frame.step[0]), int(x_mm.max() / frame.step[0]) + 1
    y0, y1 = int(y_mm.min() / frame.step[1]), int(y_mm.max() / frame.step[1]) + 1
    patch = frame.luma[y0:y1, x0:x1].astype(np.float32)
    inside = mask[y0:y1, x0:x1]
    surrounded = cv.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not surrounded.any():
        raise Invalid("Mark a texture patch with interior pixels.")
    gx = cv.Sobel(patch, cv.CV_32F, 1, 0)[surrounded] / frame.step[0]
    gy = cv.Sobel(patch, cv.CV_32F, 0, 1)[surrounded] / frame.step[1]
    tensor = np.array([[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]])
    values, vectors = np.linalg.eigh(tensor)
    if values.sum() <= 1e-9:
        raise Invalid("This texture patch has too little resolved structure.")
    dx, dy = vectors[:, 0]
    angle = math.degrees(math.atan2(dy, dx)) % 180
    strength = float((values[1] - values[0]) / values.sum())
    return angle, strength


def _pitch_pixels(frame, visible, theta_deg):
    """Groove pitch in calibrated pixels along the across-grain normal."""
    rad = math.radians(theta_deg)
    nx, ny = -math.sin(rad), math.cos(rad)
    yy, xx = np.nonzero(visible)
    across = xx * nx + yy * ny
    order = np.argsort(across)
    coord, values = across[order], frame.luma[yy, xx][order]
    bins = 240
    edges = np.linspace(coord.min(), coord.max(), bins)
    profile = np.array([values[(coord >= edges[i]) & (coord < edges[i + 1])].mean()
                        if ((coord >= edges[i]) & (coord < edges[i + 1])).any() else np.nan
                        for i in range(bins - 1)])
    filled = np.isnan(profile)
    if filled.all():
        raise Invalid("Not enough visible liner to estimate the groove pitch.")
    profile = np.interp(np.arange(len(profile)), np.flatnonzero(~filled), profile[~filled])
    detrended = profile - cv.GaussianBlur(profile, (21, 1), 0).ravel()
    cell = (coord.max() - coord.min()) / (bins - 1)
    correlation = np.correlate(detrended, detrended, "full")[len(detrended) - 1:]
    correlation /= max(correlation[0], 1e-9)
    for lag in range(3, (bins - 1) // 3):
        if correlation[lag - 1] < correlation[lag] >= correlation[lag + 1] and correlation[lag] > .2:
            return float(lag * cell)
    spectrum = np.abs(np.fft.rfft(detrended))
    peak = 1 + int(np.argmax(spectrum[1:]))
    return float((coord.max() - coord.min()) / peak)


def _groove_background(frame, visible, theta_deg, pitch_px):
    """Periodic groove baseline as a function of the across-grain coordinate."""
    rad = math.radians(theta_deg)
    nx, ny = -math.sin(rad), math.cos(rad)
    yy, xx = np.nonzero(visible)
    across = xx * nx + yy * ny
    phase = across % pitch_px
    values = frame.luma[yy, xx]
    phase_bins = 24
    edges = np.linspace(0, pitch_px, phase_bins + 1)
    level = float(np.median(values))
    template = np.full(phase_bins, level, dtype=np.float32)
    for i in range(phase_bins):
        members = (phase >= edges[i]) & (phase < edges[i + 1])
        if members.any():
            template[i] = float(np.median(values[members]))
    index = np.clip((phase / pitch_px * phase_bins).astype(int), 0, phase_bins - 1)
    background = np.full(frame.luma.shape, level, dtype=np.float32)
    background[yy, xx] = template[index]
    return background


def _area_mm2(frame, mask):
    return round(float(mask.sum() * frame.step[0] * frame.step[1]), 3)


def evaluate_direction(frame, marks, angle_deg, contrast, gap_mm, min_part_mm2):
    """Run the grain-aware detector for one candidate groove direction."""
    domain, visible = area(frame, marks["region"])
    pitch_px = _pitch_pixels(frame, visible, angle_deg)
    background = _groove_background(frame, visible, angle_deg, pitch_px)
    # A rub is darker than the periodic baseline; bright object edges are ignored.
    dark_residual = background - frame.luma
    candidate = visible & (dark_residual >= contrast)

    # Missing-capture pixels inside the window, dilated so a mark touching a
    # gap cannot be declared continuous across unseen material.  A minimum
    # one-cell halo applies even when the operator sets gap_mm to zero.
    gap = domain & ~frame.useful
    gap_radius = max(1, round(gap_mm / min(frame.step)))
    gap_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * gap_radius + 1, 2 * gap_radius + 1))
    excluded = cv.dilate(gap.astype(np.uint8), gap_kernel).astype(bool) & domain
    mark = candidate & ~excluded

    # Drop specks below the minimum component size.
    unit = frame.step[0] * frame.step[1]
    min_pixels = max(1, round(min_part_mm2 / unit))
    count, labels, stats, _ = cv.connectedComponentsWithStats(mark.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(mark)
    for label in range(1, count):
        if stats[label, cv.CC_STAT_AREA] >= min_pixels:
            kept[labels == label] = True
    mark = kept
    unknown = excluded & domain & ~visible
    return {
        "angle_deg": round(angle_deg, 2),
        "pitch_px": round(pitch_px, 3),
        "mark": mark,
        "unknown": unknown,
        "domain": domain,
        "visible": visible,
        "background": background,
        "dark_residual": dark_residual,
    }


def _scuff_core(frame, confirmed):
    """Actual marked scuff inside the loose confirmed polygon (dark pixels)."""
    groove_level = float(np.median(frame.luma[confirmed]))
    core = confirmed & (frame.luma < groove_level * .8)
    return core if core.sum() >= 8 else confirmed


def _score(frame, result, confirmed, intact):
    """Retention against the confirmed-scuff core, false area on intact liner.

    The operator's confirmed polygon is a loose box around the scuff, so its
    full area includes untouched groove.  The true-mark reference is the dark
    scuff core; retention measures how much of the actual scuff survives.
    """
    mark = result["mark"]
    scuff_core = _scuff_core(frame, confirmed)
    confirmed_area = max(1, int(scuff_core.sum()))
    intact_area = max(1, int(intact.sum()))
    retained = int((mark & scuff_core).sum())
    false_pixels = int((mark & intact).sum())
    components = cv.connectedComponents(mark.astype(np.uint8))[0] - 1 if mark.any() else 0
    return {
        "retained_confirmed_pct": round(100 * retained / confirmed_area, 3),
        "retained_confirmed_mm2": _area_mm2(frame, mark & scuff_core),
        "confirmed_scuff_mm2": _area_mm2(frame, scuff_core),
        "false_intact_pct": round(100 * false_pixels / intact_area, 3),
        "false_intact_mm2": _area_mm2(frame, mark & intact),
        "mark_mm2": _area_mm2(frame, mark),
        "unknown_mm2": _area_mm2(frame, result["unknown"]),
        "components": int(components),
    }


def grain_rub_map(frame, marks, opts):
    """Try both candidate grain directions and keep the one meeting the limits.

    ``marks`` carries the inspection window as ``region``, the two texture
    references ``grain_a``/``grain_b``, a ``confirmed`` scuff polygon and an
    ``intact`` control polygon.

    Options:
      contrast     dark-residual threshold against the groove baseline
      gap_mm       safety halo around missing-capture pixels
      min_part_mm2 smallest accepted mark component
      keep_pct     retain at least this share of the confirmed-scuff area
      false_pct    flag at most this share of the intact reference area
    """
    contrast = number(opts.get("contrast", .075), "dark residual threshold", .01, .6)
    gap_mm = number(opts.get("gap_mm", 2), "missing-capture halo", 0., 20)
    min_part_mm2 = number(opts.get("min_part_mm2", .25), "minimum mark component", .01, 50)
    keep_limit = number(opts.get("keep_pct", 70), "minimum retained scuff area", 0., 100)
    false_limit = number(opts.get("false_pct", 5), "maximum false intact area", 0., 100)

    angles = []
    for slot in ("grain_a", "grain_b"):
        angle, strength = grain_angle(frame, marks[slot])
        angles.append((slot, angle, strength))
    confirmed = region(frame, marks["confirmed"]) & frame.useful
    intact = region(frame, marks["intact"]) & frame.useful
    if not confirmed.any():
        raise Invalid("The confirmed rub mark needs visible pixels.")
    if not intact.any():
        raise Invalid("The intact reference patch needs visible pixels.")

    trials = []
    for slot, angle, strength in angles:
        result = evaluate_direction(frame, marks, angle, contrast, gap_mm, min_part_mm2)
        score = _score(frame, result, confirmed, intact)
        score.update({"reference": slot, "grain_angle_deg": round(angle, 2),
                      "groove_pitch_px": result["pitch_px"],
                      "directional_strength": round(strength, 4)})
        trials.append((score, result))

    qualified = [(score, result) for score, result in trials
                 if score["retained_confirmed_pct"] >= keep_limit
                 and score["false_intact_pct"] <= false_limit]
    if qualified:
        # Both pass: prefer the map that mis-flags the least intact liner;
        # break a tie by the one that retains more of the shallow scuffs.
        chosen_score, chosen = sorted(
            qualified, key=lambda pair: (pair[0]["false_intact_pct"], -pair[0]["retained_confirmed_pct"]))[0]
        choice_reason = "meets both limits; lowest false area on intact liner"
    else:
        chosen_score, chosen = sorted(
            trials, key=lambda pair: (pair[0]["false_intact_pct"], -pair[0]["retained_confirmed_pct"]))[0]
        choice_reason = "no direction met both limits; closest candidate shown for review"

    shown = image(frame)
    scuff_core = _scuff_core(frame, confirmed)
    paint(shown, chosen["unknown"], (.55, .49, .72), .55)
    paint(shown, chosen["mark"], (1., .36, .24), .72)
    paint(shown, scuff_core & ~chosen["mark"], (.95, .75, .25), .55)
    summary = {
        "chosen_reference": chosen_score["reference"],
        "chosen_grain_angle_deg": chosen_score["grain_angle_deg"],
        "groove_pitch_px": chosen_score["groove_pitch_px"],
        "choice": choice_reason,
        "overlay_red": "retained rub mark",
        "overlay_violet": "undecidable: missing capture plus safety halo",
        "overlay_amber": "confirmed scuff not retained",
        "keep_limit_pct": keep_limit,
        "false_limit_pct": false_limit,
        "retained_confirmed_pct": chosen_score["retained_confirmed_pct"],
        "false_intact_pct": chosen_score["false_intact_pct"],
        "mark_mm2": chosen_score["mark_mm2"],
        "undecidable_mm2": chosen_score["unknown_mm2"],
        "mark_components": chosen_score["components"],
        "limits_met": bool(qualified),
    }
    rows = []
    for score, _ in trials:
        rows.append({**score, "qualified": (score["retained_confirmed_pct"] >= keep_limit
                                            and score["false_intact_pct"] <= false_limit)})
    _, component_rows = footprint_rows(frame, chosen["mark"])
    rows.extend({"component_detail": f"mark {row['component']}", **row} for row in component_rows[:40])
    return summary, rows, shown
