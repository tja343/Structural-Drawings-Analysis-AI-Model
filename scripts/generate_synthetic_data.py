"""
Synthetic dataset generator for training YOLO object-detection models.

Produces images on a plain white canvas containing randomized geometric shapes
(rectangles, L-shapes, T-shapes) drawn as thick colored marker lines, each paired
with a random reinforcement-style text label (e.g. "H25 100 T1").

For every image a YOLO-format annotation file is written with two classes:
    class 0 -> geometric shape
    class 1 -> accompanying text

This module is UI-agnostic so it can be driven from Streamlit, a desktop app,
or the command line.
"""

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont
from app.core.config import yaml_config

# --------------------------------------------------------------------------- #
# Class ids (kept here so the UI and the writer agree on the contract)
# --------------------------------------------------------------------------- #
CLASS_SHAPE = 0
CLASS_TEXT = 1
CLASS_NAMES = ["shape", "text"]

# Default palette for the "engineering marker" look. Any number of colors may
# be supplied via GenConfig.colors; each entry can be a hex string ("#dc1e1e")
# or an (R, G, B) tuple — both are accepted directly by Pillow.
DEFAULT_COLORS = ["#dc1e1e", "#1e46dc"]  # red, blue

SHAPE_TYPES = (
    "rectangle",
    "l_shape",
    "t_shape",
    "cross",
    "u_shape",
    "h_shape",
    "z_shape",
    "stairs",
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class GenConfig:
    """All tunable parameters for a generation run."""

    # Output
    output_dir: str = "output"
    num_images: int = 50
    image_format: str = "png"          # "png" or "jpg"
    filename_prefix: str = "img"

    # Canvas
    canvas_width: int = 1024
    canvas_height: int = 768

    # Which shapes are allowed
    use_rectangle: bool = True
    use_l_shape: bool = True
    use_t_shape: bool = True
    use_cross: bool = True
    use_u_shape: bool = True
    use_h_shape: bool = True
    use_z_shape: bool = True
    use_stairs: bool = True

    # How many shapes per image
    min_shapes: int = 2
    max_shapes: int = 5

    # Overlap: by default shapes are spread out; set allow_overlap to let a
    # fraction of them (overlap_prob) ignore collision and pile on top.
    allow_overlap: bool = False
    overlap_prob: float = 0.3

    # Visual parameters
    min_line_thickness: int = 4
    max_line_thickness: int = 9
    min_font_size: int = 22
    max_font_size: int = 40

    # Colors to draw with: any number of hex strings or (R, G, B) tuples.
    colors: list = field(default_factory=lambda: list(DEFAULT_COLORS))

    # Font file (truetype). Falls back to a bundled bitmap font if unreadable.
    font_path: str | None = None

    # Reproducibility (None -> random each run)
    seed: int | None = None

    def enabled_shapes(self) -> list[str]:
        flags = {
            "rectangle": self.use_rectangle,
            "l_shape": self.use_l_shape,
            "t_shape": self.use_t_shape,
            "cross": self.use_cross,
            "u_shape": self.use_u_shape,
            "h_shape": self.use_h_shape,
            "z_shape": self.use_z_shape,
            "stairs": self.use_stairs,
        }
        # Preserve the canonical SHAPE_TYPES ordering.
        return [name for name in SHAPE_TYPES if flags[name]]


# --------------------------------------------------------------------------- #
# Font handling
# --------------------------------------------------------------------------- #
# Common locations to look for a usable TrueType font on Windows / Linux / Mac.
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
]


def find_default_font() -> str | None:
    """Return the first TrueType font path that exists, or None."""
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont:
    """Load a sizeable font, falling back gracefully."""
    candidate = font_path or find_default_font()
    if candidate:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    # Last resort: PIL's built-in bitmap font (size is fixed, but it works).
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _bbox_of_points(points: list[tuple[float, float]], pad: float) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _maybe_flip(points, w, h, flip_x, flip_y):
    out = []
    for x, y in points:
        if flip_x:
            x = w - x
        if flip_y:
            y = h - y
        out.append((x, y))
    return out


def _shape_polygon(shape_type: str, w: float, h: float) -> list[tuple[float, float]]:
    """Return polygon points (origin at 0,0) describing the shape outline."""
    if shape_type == "rectangle":
        return [(0, 0), (w, 0), (w, h), (0, h)]

    if shape_type == "l_shape":
        # Thickness of each arm of the L.
        t = max(w, h) * random.uniform(0.28, 0.45)
        t = min(t, w * 0.9, h * 0.9)
        return [
            (0, 0),
            (t, 0),
            (t, h - t),
            (w, h - t),
            (w, h),
            (0, h),
        ]

    if shape_type == "t_shape":
        t = h * random.uniform(0.25, 0.45)          # thickness of top bar
        s = w * random.uniform(0.28, 0.45)          # stem width
        left = (w - s) / 2.0
        right = (w + s) / 2.0
        return [
            (0, 0),
            (w, 0),
            (w, t),
            (right, t),
            (right, h),
            (left, h),
            (left, t),
            (0, t),
        ]

    if shape_type == "cross":
        # A plus sign: a vertical bar and a horizontal bar, both centered.
        aw = w * random.uniform(0.28, 0.45)        # vertical-bar width
        ah = h * random.uniform(0.28, 0.45)        # horizontal-bar height
        vx0, vx1 = (w - aw) / 2, (w + aw) / 2       # vertical-bar x edges
        hy0, hy1 = (h - ah) / 2, (h + ah) / 2       # horizontal-bar y edges
        return [
            (vx0, 0), (vx1, 0),                     # top of vertical bar
            (vx1, hy0), (w, hy0),                   # step out to right arm
            (w, hy1), (vx1, hy1),                   # back in below right arm
            (vx1, h), (vx0, h),                     # bottom of vertical bar
            (vx0, hy1), (0, hy1),                   # step out to left arm
            (0, hy0), (vx0, hy0),                   # back in above left arm
        ]

    if shape_type == "u_shape":
        # A U: two vertical legs joined by a bar across the bottom.
        t = w * random.uniform(0.22, 0.35)          # leg width
        b = h * random.uniform(0.22, 0.35)          # bottom-bar height
        return [
            (0, 0), (t, 0),                         # top of left leg
            (t, h - b), (w - t, h - b),             # across the inner bottom
            (w - t, 0), (w, 0),                     # up & top of right leg
            (w, h), (0, h),                         # outer bottom edge
        ]

    if shape_type == "h_shape":
        # An H: two vertical legs joined by a bar across the middle.
        t = w * random.uniform(0.22, 0.34)          # leg width
        mt = h * random.uniform(0.22, 0.36)         # middle-bar thickness
        my0, my1 = (h - mt) / 2, (h + mt) / 2       # middle-bar y edges
        return [
            (0, 0), (t, 0),                         # top of left leg
            (t, my0), (w - t, my0),                 # inner top of crossbar
            (w - t, 0), (w, 0),                     # top of right leg
            (w, h), (w - t, h),                     # bottom of right leg
            (w - t, my1), (t, my1),                 # inner bottom of crossbar
            (t, h), (0, h),                         # bottom of left leg
        ]

    if shape_type == "z_shape":
        # A Z / S offset block (like the Z-tetromino): an upper bar shifted
        # right and a lower bar shifted left, overlapping in the middle.
        t = w * random.uniform(0.25, 0.40)          # horizontal offset
        h2 = h * random.uniform(0.40, 0.60)         # height of the split line
        return [
            (t, 0), (w, 0),                         # top bar (right-shifted)
            (w, h2), (w - t, h2),                   # down to the middle ledge
            (w - t, h), (0, h),                     # bottom bar (left-shifted)
            (0, h2), (t, h2),                       # back up the middle ledge
        ]

    if shape_type == "stairs":
        # A staircase rising left-to-right with `n` equal steps.
        n = random.choice((2, 3, 4))
        sw, sh = w / n, h / n
        pts = []
        for i in range(n):                          # trace the stepped top edge
            y = h - (i + 1) * sh
            pts.append((i * sw, y))                 # riser then tread
            pts.append(((i + 1) * sw, y))
        pts.append((w, h))                          # down the right side
        pts.append((0, h))                          # back along the bottom
        return pts

    raise ValueError(f"unknown shape type: {shape_type}")


# --------------------------------------------------------------------------- #
# Reinforcement text
# --------------------------------------------------------------------------- #
_BAR_SIZES = [10, 12, 13, 16, 20, 25, 32]
_SPACINGS = [100, 125, 150, 175, 200, 250, 300]
_POSITIONS = ["T1", "T2", "B1", "B2", "B3", "T", "B"]


def random_reinforcement_text() -> str:
    """Generate text like 'H25 100 T1' or 'H16 150 B2'."""
    size = random.choice(_BAR_SIZES)
    spacing = random.choice(_SPACINGS)
    pos = random.choice(_POSITIONS)
    return f"H{size} {spacing} {pos}"


def parse_reinforcement_text(text: str) -> dict:
    """Return semantic fields that match the rendered reinforcement label."""
    parts = text.strip().upper().split()
    bar = parts[0]
    layer_dir = parts[2] if len(parts) > 2 else ""
    direction = layer_dir[1:] if len(layer_dir) > 1 else ""
    return {
        "raw": text,
        "bar_type": bar[0],
        "diameter": int(bar[1:]),
        "spacing": int(parts[1]),
        "layer": layer_dir[:1] or None,
        "direction": int(direction) if direction.isdigit() else None,
    }


# --------------------------------------------------------------------------- #
# Overlap testing
# --------------------------------------------------------------------------- #
def _intersects(a, b, margin=0.0):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (
        ax1 + margin < bx0
        or bx1 + margin < ax0
        or ay1 + margin < by0
        or by1 + margin < ay0
    )


def _in_bounds(bbox, cw, ch):
    x0, y0, x1, y1 = bbox
    return x0 >= 0 and y0 >= 0 and x1 <= cw and y1 <= ch


def _fits(bbox, placed, cw, ch, margin):
    if not _in_bounds(bbox, cw, ch):
        return False
    return all(not _intersects(bbox, p, margin) for p in placed)


# --------------------------------------------------------------------------- #
# Single image generation
# --------------------------------------------------------------------------- #
@dataclass
class Annotation:
    cls: int
    bbox: tuple[float, float, float, float]   # pixel coords (x0, y0, x1, y1)
    id: str | None = None
    shape_type: str | None = None
    text: str | None = None
    semantic: dict | None = None
    color: str | tuple | None = None
    related_id: str | None = None


def generate_image(cfg: GenConfig):
    """
    Build one synthetic image.

    Returns (PIL.Image, list[Annotation]).
    """
    cw, ch = cfg.canvas_width, cfg.canvas_height
    image = Image.new("RGB", (cw, ch), "white")
    draw = ImageDraw.Draw(image)

    enabled = cfg.enabled_shapes()
    if not enabled:
        return image, []

    palette = cfg.colors or list(DEFAULT_COLORS)
    n_shapes = random.randint(cfg.min_shapes, cfg.max_shapes)

    placed_boxes: list[tuple] = []
    annotations: list[Annotation] = []

    for shape_index in range(n_shapes):
        shape_type = random.choice(enabled)
        color = random.choice(palette)
        # Some shapes may be allowed to overlap others (piled on top).
        overlap_this = cfg.allow_overlap and random.random() < cfg.overlap_prob
        thickness = random.randint(cfg.min_line_thickness, cfg.max_line_thickness)
        font_size = random.randint(cfg.min_font_size, cfg.max_font_size)
        font = load_font(cfg.font_path, font_size)

        text = random_reinforcement_text()
        # Measure text so we can reserve room for it.
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

        placed = False
        for _attempt in range(40):
            # Shape size: a fraction of the canvas, never larger than it.
            sw = random.uniform(cw * 0.12, cw * 0.30)
            sh = random.uniform(ch * 0.12, ch * 0.30)
            sw = min(sw, cw * 0.45)
            sh = min(sh, ch * 0.45)

            # Reserve a horizontal band wide enough for shape + text label.
            block_w = sw + 20 + tw
            block_h = max(sh, th)
            if block_w > cw or block_h > ch:
                continue

            ox = random.uniform(0, cw - block_w)
            oy = random.uniform(0, ch - block_h)

            pad = thickness / 2.0 + 2
            shape_bbox = (ox - pad, oy - pad, ox + sw + pad, oy + sh + pad)

            # Text sits to the right of the shape, vertically centered on it.
            tx = ox + sw + 20
            ty = oy + (sh - th) / 2.0
            text_bbox = (tx - 2, ty - 2, tx + tw + 2, ty + th + 2)

            if overlap_this:
                # Only require staying on-canvas; collisions are allowed.
                if not (_in_bounds(shape_bbox, cw, ch) and _in_bounds(text_bbox, cw, ch)):
                    continue
            else:
                if not _fits(shape_bbox, placed_boxes, cw, ch, margin=6):
                    continue
                if not _fits(text_bbox, placed_boxes + [shape_bbox], cw, ch, margin=4):
                    continue

            # --- draw the shape ---
            poly = _shape_polygon(shape_type, sw, sh)
            if shape_type != "rectangle":
                poly = _maybe_flip(
                    poly, sw, sh,
                    flip_x=random.random() < 0.5,
                    flip_y=random.random() < 0.5,
                )
            poly = [(ox + px, oy + py) for px, py in poly]
            # Closed outline with rounded joints -> marker-pen look.
            draw.line(poly + [poly[0]], fill=color, width=thickness, joint="curve")

            # --- draw the text (same color as its shape) ---
            # textbbox can report a non-zero origin offset; compensate so the
            # recorded bbox matches the rendered glyphs.
            draw.text((tx - tb[0], ty - tb[1]), text, fill=color, font=font)

            shape_id = f"beam_{shape_index + 1:03d}"
            text_id = f"text_{shape_index + 1:03d}"
            semantic = parse_reinforcement_text(text)
            semantic["color"] = color
            annotations.append(
                Annotation(
                    CLASS_SHAPE,
                    shape_bbox,
                    id=shape_id,
                    shape_type=shape_type,
                    semantic={"type": shape_type, "color": color},
                    color=color,
                )
            )
            annotations.append(
                Annotation(
                    CLASS_TEXT,
                    text_bbox,
                    id=text_id,
                    text=text,
                    semantic=semantic,
                    color=color,
                    related_id=shape_id,
                )
            )
            placed_boxes.extend([shape_bbox, text_bbox])
            placed = True
            break
        # if not placed after all attempts, silently skip this shape

    return image, annotations


# --------------------------------------------------------------------------- #
# YOLO writing
# --------------------------------------------------------------------------- #
def to_yolo_lines(annotations: list[Annotation], cw: int, ch: int) -> list[str]:
    """Convert pixel bboxes to normalized YOLO lines: 'cls xc yc w h'."""
    lines = []
    for a in annotations:
        x0, y0, x1, y1 = a.bbox
        # clamp to canvas
        x0, x1 = max(0, x0), min(cw, x1)
        y0, y1 = max(0, y0), min(ch, y1)
        bw = x1 - x0
        bh = y1 - y0
        if bw <= 0 or bh <= 0:
            continue
        xc = (x0 + x1) / 2.0 / cw
        yc = (y0 + y1) / 2.0 / ch
        nw = bw / cw
        nh = bh / ch
        lines.append(f"{a.cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
    return lines


def _round_bbox(bbox: tuple[float, float, float, float]) -> list[int]:
    return [int(round(v)) for v in bbox]


def build_semantic_metadata(image_id: str, annotations: list[Annotation]) -> dict:
    """Create image-level JSON whose beam/text relationships match the drawing."""
    shapes = {a.id: a for a in annotations if a.cls == CLASS_SHAPE and a.id}
    texts = [a for a in annotations if a.cls == CLASS_TEXT]

    elements = []
    legacy_annotations = []
    for text_ann in texts:
        shape = shapes.get(text_ann.related_id)
        if shape is None:
            continue
        parsed = dict(text_ann.semantic or {})
        elements.append(
            {
                "id": shape.id,
                "type": "beam",
                "shape_type": shape.shape_type,
                "bbox": _round_bbox(shape.bbox),
                "color": shape.color,
                "annotations": [
                    {
                        "id": text_ann.id,
                        "bbox": _round_bbox(text_ann.bbox),
                        "text": text_ann.text,
                        "semantic": parsed,
                    }
                ],
            }
        )
        legacy_annotations.append(
            {
                "bbox": _round_bbox(text_ann.bbox),
                "text": text_ann.text,
                "semantic": parsed,
                "associated_element_id": shape.id,
            }
        )

    return {
        "schema_version": "1.1",
        "image_id": image_id,
        "summary": {
            "element_count": len(elements),
            "annotation_count": len(legacy_annotations),
        },
        "elements": elements,
        "annotations": legacy_annotations,
    }


# --------------------------------------------------------------------------- #
# Full dataset run
# --------------------------------------------------------------------------- #
def generate_dataset(cfg: GenConfig, progress=None):
    """
    Generate cfg.num_images images + annotation files into cfg.output_dir.

    `progress`, if given, is called as progress(done, total, last_image_path).
    Returns the number of images written.
    """
    if cfg.seed is not None:
        random.seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    images_dir = os.path.join(cfg.output_dir, "images")
    labels_dir = os.path.join(cfg.output_dir, "labels")
    semantics_dir = os.path.join(cfg.output_dir, "semantics")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(semantics_dir, exist_ok=True)

    # Write a classes.txt so the dataset is self-describing.
    with open(os.path.join(cfg.output_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")

    ext = "jpg" if cfg.image_format.lower() in ("jpg", "jpeg") else "png"
    pad = max(4, len(str(cfg.num_images)))

    last_path = None
    for i in range(cfg.num_images):
        image, anns = generate_image(cfg)
        stem = f"{cfg.filename_prefix}_{i:0{pad}d}"
        img_path = os.path.join(images_dir, f"{stem}.{ext}")
        txt_path = os.path.join(labels_dir, f"{stem}.txt")
        json_path = os.path.join(semantics_dir, f"{stem}.json")

        if ext == "jpg":
            image.save(img_path, quality=95)
        else:
            image.save(img_path)

        lines = to_yolo_lines(anns, cfg.canvas_width, cfg.canvas_height)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        metadata = build_semantic_metadata(stem, anns)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        last_path = img_path
        if progress:
            progress(i + 1, cfg.num_images, last_path)
            
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{cfg.num_images}")

    return cfg.num_images


def draw_debug_boxes(image: Image.Image, annotations: list[Annotation]) -> Image.Image:
    """Return a copy of `image` with annotation boxes drawn (for previews)."""
    preview = image.copy()
    d = ImageDraw.Draw(preview)
    colors = {CLASS_SHAPE: (0, 180, 0), CLASS_TEXT: (255, 140, 0)}
    for a in annotations:
        d.rectangle(a.bbox, outline=colors.get(a.cls, (0, 0, 0)), width=2)
    return preview


def main():
    output_dir = yaml_config.get("paths", {}).get("data_synthetic", "data/synthetic")
    print(f"Generating 200 synthetic structural drawings into {output_dir}...")
    
    cfg = GenConfig(
        output_dir=output_dir,
        num_images=200,
        filename_prefix="synthetic_draft"
    )
    
    generate_dataset(cfg)
    print("Synthetic dataset generation complete!")

if __name__ == "__main__":
    main()
