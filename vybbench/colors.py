"""Colors: how far off is the hex.

Prompt: "A plain white (#FFFFFF) image with a single solid rectangle of {hex}
in the middle. No border, no shadow, no texture, no gradient."

Scoring finds the non-white pixels, keeps only the largest connected group,
shrinks 4px inward to drop the antialiased edge, takes the per-channel median
of what remains, and reports the largest of the three channel gaps out of 255.
Our own rendered reference scores exactly zero through this same function, so
every point of error in a real run belongs to the model.

Validity is "one solid rectangle and nothing else": the region must fill its
own bounding box (extent), be flat (no texture or gradient), and cover a sane
share of the canvas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.measure import label
from skimage.morphology import isotropic_erosion

# A pixel counts as "not background" if it differs from white by more than
# this per channel; the 4px shrink then removes the antialiased edge entirely.
NON_WHITE_THRESHOLD = 8
SHRINK_PX = 4

MIN_BOX_EXTENT = 0.90        # region area / its own bbox; 1.0 is a rectangle
MIN_FLAT_FRACTION = 0.95     # "no texture, no gradient", in pixels
FLATNESS_TOLERANCE = 12
MIN_BOX_CANVAS_FRACTION = 0.02   # outside this range it covers nothing
MAX_BOX_CANVAS_FRACTION = 0.80   # or the whole canvas -- not "a rectangle"


@dataclass(frozen=True)
class ColorResult:
    channel_error: int  # max(|dR|, |dG|, |dB|), 0-255, the scorecard number
    measured_hex: str
    valid: bool  # did it draw one flat rectangle at all


def color_error(image: Image.Image, expected_hex: str) -> ColorResult:
    rgb = np.asarray(image.convert("RGB"))
    non_white = (
        np.abs(rgb.astype(np.int16) - 255).max(axis=-1) > NON_WHITE_THRESHOLD
    )
    labeled = label(non_white)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    box = labeled == int(np.argmax(counts)) if labeled.max() else non_white
    interior = isotropic_erosion(box, SHRINK_PX)
    region = rgb[interior]

    box_pixels = int(box.sum())
    canvas_fraction = box_pixels / box.size
    extent = 0.0
    if box_pixels:
        ys, xs = np.where(box)
        extent = box_pixels / (
            (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        )

    if region.size == 0:
        # Nothing survived shrinking: no rectangle, or one too thin to have
        # an interior.
        measured = (255, 255, 255)
        flat_fraction = 0.0
    else:
        measured = tuple(int(v) for v in np.median(region, axis=0))
        flat = np.abs(region.astype(np.int16) - measured).max(axis=-1)
        flat_fraction = float((flat <= FLATNESS_TOLERANCE).mean())

    expected = tuple(int(expected_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return ColorResult(
        channel_error=max(abs(m - e) for m, e in zip(measured, expected)),
        measured_hex="#{:02x}{:02x}{:02x}".format(*measured),
        valid=(
            region.size > 0
            and extent >= MIN_BOX_EXTENT
            and flat_fraction >= MIN_FLAT_FRACTION
            and MIN_BOX_CANVAS_FRACTION <= canvas_fraction <= MAX_BOX_CANVAS_FRACTION
        ),
    )
