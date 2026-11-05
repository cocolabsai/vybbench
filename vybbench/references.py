"""The prompts, and the color ground truth.

The font ground truths ship as rendered images under ``references/`` -- one
PIL render of the case's word in its pinned font file (provenance per case in
``data/fonts.jsonl``: google/fonts commit, file path, sha256). The color
ground truth is defined here: a solid rectangle on white, which scores exactly
zero through the color metric.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from . import data
from .cases import FontCase

CANVAS = 1024


def font_prompt(case: FontCase) -> str:
    template = data.read_prompt("font.txt").strip()
    return template.format(
        family=case.family, variant=case.variant, word=case.word
    )


def color_prompt(hex_color: str) -> str:
    return data.read_prompt("color.txt").strip().format(hex=hex_color.upper())


RECT_FRACTION = 0.5  # rectangle side, as a fraction of the canvas


def render_color_reference(hex_color: str, canvas: int = CANVAS) -> Image.Image:
    image = Image.new("RGB", (canvas, canvas), "#FFFFFF")
    side = round(canvas * RECT_FRACTION)
    corner = (canvas - side) // 2
    ImageDraw.Draw(image).rectangle(
        (corner, corner, corner + side, corner + side), fill=hex_color
    )
    return image
