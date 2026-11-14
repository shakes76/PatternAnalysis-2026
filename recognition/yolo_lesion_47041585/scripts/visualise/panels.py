"""Side-by-side failure-case panels: two models on the same image, truth boxed in blue.

Not run directly: scripts/autopsy.py picks the cases and calls `draw_panel`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from yolo_lesion_47041585.helpers.visualisation import draw_detections
from yolo_lesion_47041585.predict import Detection

# ---- settings -------------------------------------------------------------------------------
CLASS_NAMES = ("Benign", "Malignant")
PANEL_WIDTH = 760  # width of each image in pixels
GAP = 12
CAPTION_HEIGHT = 170
LINE_HEIGHT = 34
FONT_SIZE = 17
TRUTH_COLOUR = "blue"
BACKGROUND = "white"
TEXT_COLOUR = "black"
MAX_BOXES = 5  # draw at most this many detections, highest confidence first
MIN_CONFIDENCE = 0.05  # and none below this
NOTE = "blue box: the annotated lesion. Boxes drawn are the 5 highest-confidence detections at 0.05 or more."
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelSide:
    """One model's view of the image."""

    detections: Sequence[Detection]
    truth: Detection
    caption: str


def draw_panel(image_path: Path, left: PanelSide, right: PanelSide, title: str, out: Path) -> None:
    """Draw `left` and `right` over the same image with a caption underneath."""
    with Image.open(image_path) as original:
        scale = PANEL_WIDTH / original.width
        small = original.convert("RGB").resize((PANEL_WIDTH, round(original.height * scale)))
    tiles: list[Image.Image] = []
    for side in (left, right):
        shown = [d for d in sorted(side.detections, key=lambda d: -d.confidence)[:MAX_BOXES] if d.confidence >= MIN_CONFIDENCE]
        tile = draw_detections(small, shown, CLASS_NAMES)
        t = side.truth.box
        ImageDraw.Draw(tile).rectangle(
            ((t.x_center - t.width / 2) * tile.width, (t.y_center - t.height / 2) * tile.height,
             (t.x_center + t.width / 2) * tile.width, (t.y_center + t.height / 2) * tile.height),
            outline=TRUTH_COLOUR, width=3,
        )  # fmt: skip
        tiles.append(tile)
    canvas = Image.new("RGB", (PANEL_WIDTH * 2 + GAP, small.height + CAPTION_HEIGHT), BACKGROUND)
    canvas.paste(tiles[0], (0, 0))
    canvas.paste(tiles[1], (PANEL_WIDTH + GAP, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=FONT_SIZE)
    y = small.height + 8
    for line in (title, left.caption, right.caption, NOTE):
        draw.text((10, y), line, fill=TEXT_COLOUR, font=font)
        y += LINE_HEIGHT
    canvas.save(out)
