"""Validate runs and calibrated annotations for foam tray measurements."""
import math
import re

from image_lab.validation import Invalid


ROLES = {"face": "Tray face", "pressure": "Pressure transfer", "reseated": "After reseating"}
MARKS = {
    "outline": "Object search area",
    "lane": "Lift approach line",
    "bearing": "Pressure witness patch",
    "gap": "Cavity cross-section",
    "grain": "Liner grain patch",
    "register": "Matching tray landmarks",
    "rub": "Surface rub patch",
    "inspect": "Inspection patch",
    "label": "Label search area",
    "bridge": "Foam bridge patch",
}
READINGS = {
    "outline": ("Object outline", "Find and measure the visible object footprint in its search area."),
    "lift-access": ("Lift access", "Read the clear approach along a marked lift line and locate the first block or missing evidence."),
    "bearing": ("Bearing patches", "Locate visible contact patches on the transfer witness, with their area and centroids."),
    "gap-section": ("Cavity section", "Measure uninterrupted spans of the bright cavity floor along a line."),
    "grain-direction": ("Liner grain direction", "Estimate the dominant orientation of the texture in a liner patch."),
    "landmark-offset": ("Tray landmark offsets", "Measure displacement between ordered pairs of marked landmarks."),
    "rub-marks": ("Rub mark map", "Find connected local brightness anomalies on the liner."),
    "liner-tone": ("Liner tone", "Check color and variation in the visible liner."),
    "paired-coverage": ("Paired tray visibility", "Compare the physically visible areas of two registered photographs."),
    "bearing-change": ("Bearing change after reseating", "Measure where the contact footprint persists, is gained or is lost."),
    "label-window": ("Label window", "Find the largest rectangular clear patch for an identification label."),
    "liner-bridges": ("Liner bridges", "Check how narrow passages in a visible foam outline separate under edge wear."),
}
POINT_MARKS = {"register", "lane", "gap"}
IDENTITY = re.compile(r"^[0-9a-f]{24}$")


def identifier(value):
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise Invalid("The selected item is unavailable.")
    return value


def locate(items, value, label):
    wanted = identifier(value)
    for item in items:
        if item["id"] == wanted:
            return item
    raise Invalid(f"{label} is no longer available.")


def obj(value):
    if not isinstance(value, dict):
        raise Invalid("Expected a set of named fields.")
    return value


def text(value, label, limit=100, optional=False):
    if not isinstance(value, str) or len(value.strip()) > limit or (not optional and not value.strip()):
        raise Invalid(f"Enter {label} using at most {limit} characters.")
    return value.strip()


def number(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise Invalid(f"Enter a valid {label}.")
    if not minimum <= value <= maximum:
        raise Invalid(f"{label.capitalize()} must be between {minimum:g} and {maximum:g}.")
    return value


def choice(value, values, label):
    if not isinstance(value, str) or value not in values:
        raise Invalid(f"Choose a valid {label}.")
    return value


def points(raw, minimum, maximum, label):
    if not isinstance(raw, list) or not minimum <= len(raw) <= maximum:
        raise Invalid(f"{label.capitalize()} needs between {minimum} and {maximum} ordered points.")
    result = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            raise Invalid("Positions must be coordinate pairs.")
        result.append([number(point[0], "position", 0, 1), number(point[1], "position", 0, 1)])
    return result


def run_fields(raw):
    data = obj(raw)
    return {
        "title": text(data.get("title"), "a run title"),
        "stock": text(data.get("stock", ""), "the liner description", 100, optional=True),
        "width_mm": number(data.get("width_mm"), "tray width (mm)", 20, 1500),
        "height_mm": number(data.get("height_mm"), "tray height (mm)", 20, 1500),
    }


def mark_fields(raw):
    data = obj(raw)
    purpose = choice(data.get("purpose"), MARKS, "mark purpose")
    if purpose in {"lane", "gap"}:
        minimum, maximum = 2, 2
    elif purpose == "register":
        minimum, maximum = 2, 16
    else:
        minimum, maximum = 3, 32
    return {"title": text(data.get("title"), "a mark title"), "purpose": purpose,
            "points": points(data.get("points"), minimum, maximum, MARKS[purpose])}
