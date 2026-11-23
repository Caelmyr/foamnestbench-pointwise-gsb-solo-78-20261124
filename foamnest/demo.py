"""Produce labeled synthetic foam tray captures and transfer-witness marks."""
import math

import cv2 as cv
import numpy as np

from .photos import png
from .store import identity, stamp

WIDTH, HEIGHT = 888, 592
CORNERS = [[0, 0], [1, 0], [1, 1], [0, 1]]
LANDMARKS = [(102, 116), (346, 413), (730, 101), (770, 463)]
REGIONS = (
    ("Bronze mount silhouette", "outline", (370, 148, 670, 398)),
    ("Pressure witness plate", "bearing", (674, 165, 807, 360)),
    ("Grain under the lip", "grain", (120, 402, 318, 473)),
    ("Rub marks on the left wall", "rub", (134, 178, 281, 286)),
    ("Shared inspection window", "inspect", (339, 120, 800, 428)),
    ("Foam pads and bridge", "bridge", (340, 423, 685, 515)),
)
# Two label candidate panels plus the old lettering and raised bump that rule out part of them.
LABEL_CANDIDATES = (
    ("Label candidate near the entry", "label", (96, 288, 318, 392)),
    ("Label candidate by the lower wall", "label", (575, 300, 690, 500)),
)
KEEPOUTS = (
    ("Old printed lettering", "keepout", ((252, 326), (296, 326), (296, 360), (252, 360))),
    ("Raised locating bump", "keepout", ((652, 452), (684, 452), (684, 490), (652, 490))),
)
LINES = (
    ("Entry from the left wall", "lane", [(108, 261), (504, 261)]),
    ("Across the grip channel", "gap", [(170, 256), (496, 256)]),
)


def normalized(points):
    return [[x / (WIDTH - 1), y / (HEIGHT - 1)] for x, y in points]


def bounds(box):
    x0,y0,x1,y1 = box
    return normalized([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])


def photograph(role, shifts, seed):
    rng = np.random.default_rng(seed)
    y, x = np.indices((HEIGHT, WIDTH), dtype=np.float32)
    noise = rng.normal(0, 2.6, (HEIGHT, WIDTH))
    light = 1. - .16 * ((x - 453) / 600)**2 - .2 * ((y - 239)/500)**2
    rgb = light[:, :, None] * np.array((53, 59, 69)) + noise[:, :, None]
    tray = (x >= 64) & (x < 830) & (y >= 62) & (y < 530)
    grain = 6 * np.cos((y - .31 * x) * (2 * math.pi / 6.9))
    rgb[tray] = np.array((78, 92, 104)) + (grain[tray]+noise[tray])[:, None]
    rgba = cv.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv.COLOR_RGB2RGBA)
    cv.rectangle(rgba, (64, 62), (830, 530), (159, 175, 184, 255), 4)
    cv.rectangle(rgba, (78, 78), (814, 513), (59, 71, 81, 255), 2)

    cv.rectangle(rgba, (108, 242), (390, 276), (178, 200, 203, 255), -1)
    cv.rectangle(rgba, (300, 243), (339, 274), (49, 51, 50, 255), -1)
    for points in (
        [(176,196),(208,198),(218,209),(202,216),(178,207)],
        [(240,207),(268,210),(272,224),(252,229),(244,219)],
    ):
        cv.fillPoly(rgba, [np.array(points, dtype=np.int32)], (150, 158, 154, 255), cv.LINE_AA)

    dx, dy = shifts
    center = (528 + dx, 269 + dy)
    cv.ellipse(rgba, center, (131, 106), -13, 0, 360, (40, 43, 46, 255), -1, cv.LINE_AA)
    contour = np.array([(407,265),(442,203),(506,178),(585,178),(637,237),(624,298),(565,347),
                        (490,356),(453,326),(419,304)], dtype=np.int32) + (dx,dy)
    cv.fillPoly(rgba, [contour.astype(np.int32)], (182, 152, 100, 255), cv.LINE_AA)
    cv.polylines(rgba, [contour.astype(np.int32)], True, (219, 190, 127, 255), 2, cv.LINE_AA)
    for cx,cy,rx,ry in ((452,261,19,28),(518,229,23,35),(580,247,19,16),(540,304,44,18)):
        cv.ellipse(rgba, (cx+dx,cy+dy), (rx,ry), -18, 0, 360, (211, 176, 117, 255), 3, cv.LINE_AA)

    cv.rectangle(rgba, (680, 165), (802, 360), (165, 166, 161, 255), -1)
    witnesses = [(708,212,16),(769,263,17),(718,305,17)]
    if role == "reseated":
        witnesses = [(712,215,16),(764,266,17),(741,323,18)]
    for wx,wy,radius in witnesses:
        cv.circle(rgba, (wx,wy), radius, (214, 87, 73, 255), -1, cv.LINE_AA)
        cv.circle(rgba, (wx,wy), radius//2, (227, 108, 88, 255), -1, cv.LINE_AA)

    # Two clear label candidate panels; dark old lettering and a raised bump are no-stick zones.
    cv.rectangle(rgba, (96, 288), (318, 392), (172, 187, 190, 255), -1)
    cv.rectangle(rgba, (252, 326), (296, 360), (68, 80, 92, 255), -1)
    cv.putText(rgba, "OLD", (255, 350), cv.FONT_HERSHEY_SIMPLEX, .55, (44, 52, 60, 255), 2, cv.LINE_AA)
    cv.rectangle(rgba, (575, 300), (690, 500), (176, 190, 192, 255), -1)
    cv.circle(rgba, (668, 471), 16, (96, 108, 116, 255), -1, cv.LINE_AA)
    cv.circle(rgba, (668, 471), 16, (150, 162, 168, 255), 2, cv.LINE_AA)
    cv.rectangle(rgba, (345, 438), (481, 496), (170, 171, 157, 255), -1)
    cv.rectangle(rgba, (527, 433), (667, 498), (170, 171, 157, 255), -1)
    cv.rectangle(rgba, (480, 464), (528, 471), (170, 171, 157, 255), -1)
    for index, (cx,cy) in enumerate(LANDMARKS):
        point = (cx+dx+index % 2, cy+dy)
        cv.drawMarker(rgba, point, (205, 179, 111, 255), markerSize=16, thickness=2, line_type=cv.LINE_AA)
    if role == "face":
        rgba[248:272, 345:369] = 0
    if role == "reseated":
        rgba[318:357, 711:727] = 0
    cv.putText(rgba, "SYNTHETIC TRAY", (94, 553), cv.FONT_HERSHEY_SIMPLEX, .66,
               (225, 224, 212, 255), 1, cv.LINE_AA)
    return rgba


def create(library):
    captures, marks = [], []
    for title, role, shifts, seed in (
        ("Tray face / standard seat", "face", (0, 0), 81),
        ("Tray face / after reseating", "reseated", (4, -2), 82),
    ):
        capture = {"id": identity(), "title": title, "role": role, "filename": role + "-tray.png",
                   "blob": library.keep(png(photograph(role, shifts, seed))), "corners": CORNERS,
                   "synthetic": True, "uploaded": stamp()}
        captures.append(capture)
        items = [(name, purpose, bounds(box)) for name, purpose, box in REGIONS]
        items.extend((name, purpose, bounds(box)) for name, purpose, box in LABEL_CANDIDATES)
        items.extend((name, purpose, normalized(points)) for name, purpose, points in KEEPOUTS)
        items.extend((name, purpose, normalized(points)) for name, purpose, points in LINES)
        ox, oy = shifts
        matching = normalized([(x+ox+i % 2,y+oy) for i,(x,y) in enumerate(LANDMARKS)])
        items.append(("Corresponding fixture landmarks", "register", matching))
        for name, purpose, ordered in items:
            marks.append({"id": identity(), "title": name, "purpose": purpose,
                          "points": ordered, "photo": capture["id"]})
    return library.new_run({"title": "Bronze mount tray", "stock": "Synthetic foam liner and pressure witness",
                            "width_mm": 222, "height_mm": 148}, captures, marks)
