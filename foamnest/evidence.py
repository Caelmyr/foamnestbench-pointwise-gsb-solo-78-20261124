"""Create readable, self-contained local proof evidence downloads."""
import csv
import io
import json
import math
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw

from .store import encode_doc

PROOF_LIMIT = 40_000_000


def field(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def rows_csv(reading):
    rows = reading["rows"]
    output = io.StringIO(newline="")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    if keys:
        report = csv.DictWriter(output, fieldnames=keys)
        report.writeheader()
        report.writerows({key: field(value) for key, value in row.items()} for row in rows)
    return output.getvalue().encode("utf-8")


def label_proof(reading, px_per_mm):
    """Full-tray transparent PNG of the adopted label, aligned to physical mm coordinates."""
    plan = reading.get("summary", {}).get("label_plan")
    if not plan or px_per_mm is None:
        return None, None
    tray_w, tray_h = plan["tray_mm"]
    width = max(1, round(tray_w * px_per_mm))
    height = max(1, round(tray_h * px_per_mm))
    if width * height > PROOF_LIMIT:
        raise ValueError("That proof scale would create an image larger than 40 megapixels.")
    scale_x, scale_y = width / tray_w, height / tray_h
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    orange = (242, 168, 74, 255)
    amber = (245, 211, 97, 235)
    arrow_fill = (250, 252, 248, 255)
    arrow_edge = (38, 58, 56, 255)
    line_w = max(2, round(px_per_mm))

    def physical(rect):
        x0, y0, x1, y1 = rect
        return [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]

    def dashed_rectangle(box, color, width, dash):
        x0, y0, x1, y1 = box
        for start in range(round(x0), round(x1), dash * 2):
            draw.line([(start, y0), (min(start + dash, x1), y0)], fill=color, width=width)
            draw.line([(start, y1), (min(start + dash, x1), y1)], fill=color, width=width)
        for start in range(round(y0), round(y1), dash * 2):
            draw.line([(x0, start), (x0, min(start + dash, y1))], fill=color, width=width)
            draw.line([(x1, start), (x1, min(start + dash, y1))], fill=color, width=width)

    # Clear-margin envelope (label plus its required gap), drawn outside the label as a dashed ring.
    dashed_rectangle(physical(plan["envelope_mm"]), amber, max(1, line_w - 1),
                     max(4, round(3 * px_per_mm)))
    corners = [physical([x, y, x, y])[:2] for x, y in plan["corners_mm"]]
    lx0, ly0, lx1, ly1 = physical(
        [min(p[0] for p in plan["corners_mm"]), min(p[1] for p in plan["corners_mm"]),
         max(p[0] for p in plan["corners_mm"]), max(p[1] for p in plan["corners_mm"])])
    draw.rectangle([lx0, ly0, lx1, ly1], outline=orange, width=line_w)
    radius = max(3, round(2.2 * px_per_mm))
    for point in corners:
        draw.ellipse([point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
                     fill=orange)

    # Reading direction arrow: label text runs along this vector from the label centre.
    dx, dy = plan["reading_direction"]
    cx, cy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
    reach = .3 * ((lx1 - lx0) * abs(dx) + (ly1 - ly0) * abs(dy))
    end = (cx + dx * reach, cy + dy * reach)
    head = max(5, round(3 * px_per_mm))
    angle = math.atan2(dy, dx)
    barbs = [(end[0] + head * math.cos(angle + sweep),
              end[1] + head * math.sin(angle + sweep))
             for sweep in (math.radians(150), math.radians(-150))]
    draw.line([cx, cy, end[0], end[1]], fill=arrow_edge, width=line_w + 2)
    for barb in barbs:
        draw.line([end, barb], fill=arrow_edge, width=line_w + 2)
    draw.line([cx, cy, end[0], end[1]], fill=arrow_fill, width=line_w)
    for barb in barbs:
        draw.line([end, barb], fill=arrow_fill, width=line_w)

    output = io.BytesIO()
    canvas.save(output, "PNG")
    metadata = {
        "px_per_mm": px_per_mm,
        "tray_size_mm": plan["tray_mm"],
        "proof_size_px": [width, height],
        "coordinate_origin": "calibrated tray upper-left, x right and y down, in millimetres",
        "chosen_candidate": plan["candidate"],
        "candidate_title": plan["candidate_title"],
        "orientation": plan["orientation"],
        "label_corners_mm": plan["corners_mm"],
        "clear_margin_envelope_mm": plan["envelope_mm"],
        "label_size_mm": plan["label_mm"],
        "required_gap_mm": plan["gap_mm"],
        "measured_margins_mm": plan["margins_mm"],
        "minimum_clear_margin_mm": plan["minimum_margin_mm"],
        "reading_direction_xy": plan["reading_direction"],
    }
    return output.getvalue(), (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode()


def packet(library, reading, px_per_mm=None):
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("reading.json", encode_doc(reading))
        archive.writestr("measured-values.csv", rows_csv(reading))
        archive.writestr("proof-overlay.png", library.blob(reading["overlay"]).read_bytes())
        if reading.get("mode") == "label-window":
            proof, metadata = label_proof(reading, px_per_mm)
            if proof is not None:
                archive.writestr("label-proof.png", proof)
                archive.writestr("label-proof.json", metadata)
        for index, image in enumerate(reading["photos"], 1):
            archive.writestr(f"source/{index}-{image['filename']}", library.blob(image["blob"]).read_bytes())
    return output.getvalue()
