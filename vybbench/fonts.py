"""Fonts: can the model render a named font.

Prompt: 'A plain white image. The word "<word>" in solid black, centered, in
the font <Family> Regular. Nothing else in the image.' -- one pinned word per
family (data/fonts.jsonl).

Scoring: both images are binarized at 128 and speckle under 0.02% of canvas is
dropped; each ink mask is cropped to its bounding box; the output is then
searched over 21 candidate scales (+/-15% around the height ratio), placed
both centered and at the best phase-correlation shift, and the score is the
minimum mismatch found -- one XOR over one OR, the fraction of the ink union
where the masks disagree. The reported number is IoU = 1 - mismatch.

Validity is structural: a real attempt inks a small share of the canvas in a
handful of components (scaled up for longer words), so a blank page, a filled
page, or a photograph fails without needing the word to be read.

Each family carries its own noise floor -- the reference rendered at two
sizes and positions, scored against itself through this same function --
recorded per case in ``data/fonts.jsonl`` beside the shipped reference render.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.measure import label
from skimage.registration import phase_cross_correlation

BINARIZE_THRESHOLD = 128
SPECKLE_FRACTION = 0.0002
SCALE_SEARCH_RANGE = 0.15
SCALE_SEARCH_STEPS = 21

MIN_INK_FRACTION = 0.001
MAX_INK_FRACTION = 0.40
MAX_INK_COMPONENTS = 8

def max_ink_components(word: str) -> int:
    """`MAX_INK_COMPONENTS`, scaled up for a longer word."""
    return max(MAX_INK_COMPONENTS, len(word) + 4)


@dataclass(frozen=True)
class FontResult:
    mismatch_fraction: float  # the scorecard number: XOR / OR over the ink union
    valid: bool  # did it draw the requested word (structurally)


def ink_mask(image: Image.Image) -> np.ndarray:
    """Binarized ink with sub-speckle components dropped."""
    gray = np.asarray(image.convert("RGB")).mean(axis=-1)
    mask = gray < BINARIZE_THRESHOLD
    labeled = label(mask)
    if labeled.max() == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = np.nonzero(counts >= max(1, int(SPECKLE_FRACTION * mask.size)))[0]
    return np.isin(labeled, keep)


def _crop_to_ink(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.resize((width, height), Image.LANCZOS)) >= 128


def _place(mask: np.ndarray, shape: tuple[int, int], y0: int, x0: int) -> np.ndarray:
    canvas = np.zeros(shape, dtype=bool)
    h, w = mask.shape
    src_y0, src_x0 = max(0, -y0), max(0, -x0)
    dst_y0, dst_x0 = max(0, y0), max(0, x0)
    copy_h = min(h - src_y0, shape[0] - dst_y0)
    copy_w = min(w - src_x0, shape[1] - dst_x0)
    if copy_h > 0 and copy_w > 0:
        canvas[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = (
            mask[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
        )
    return canvas


def _mismatch(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a ^ b).sum()) / union if union else 0.0


def font_mismatch(
    reference: Image.Image, output: Image.Image,
    max_components: int = MAX_INK_COMPONENTS,
) -> FontResult:
    ref = ink_mask(reference)
    out = ink_mask(output)
    if not ref.any() or not out.any():
        # Nothing to align against, so a total mismatch rather than a crash.
        return FontResult(mismatch_fraction=1.0, valid=False)

    fraction = float(out.sum()) / out.size
    valid = (
        MIN_INK_FRACTION <= fraction <= MAX_INK_FRACTION
        and 1 <= int(label(out).max()) <= max_components
    )

    ref, out = _crop_to_ink(ref), _crop_to_ink(out)
    base = ref.shape[0] / out.shape[0]
    scales = np.linspace(
        base * (1 - SCALE_SEARCH_RANGE), base * (1 + SCALE_SEARCH_RANGE),
        SCALE_SEARCH_STEPS,
    )
    largest_h = max(1, round(out.shape[0] * scales[-1]))
    largest_w = max(1, round(out.shape[1] * scales[-1]))
    margin = max(8, round(0.08 * max(*ref.shape, largest_h, largest_w)))
    shape = (max(ref.shape[0], largest_h) + 2 * margin,
             max(ref.shape[1], largest_w) + 2 * margin)
    ref_canvas = _place(ref, shape, (shape[0] - ref.shape[0]) // 2,
                        (shape[1] - ref.shape[1]) // 2)
    ref_float = ref_canvas.astype(np.float32)

    best = 1.0
    for scale in scales:
        h = max(1, round(out.shape[0] * scale))
        w = max(1, round(out.shape[1] * scale))
        resized = _resize_mask(out, w, h)
        y0, x0 = (shape[0] - h) // 2, (shape[1] - w) // 2
        centered = _place(resized, shape, y0, x0)
        best = min(best, _mismatch(ref_canvas, centered))
        shift, _, _ = phase_cross_correlation(
            ref_float, centered.astype(np.float32), upsample_factor=1,
        )
        dy, dx = int(round(shift[0])), int(round(shift[1]))
        if dy or dx:
            shifted = _place(resized, shape, y0 + dy, x0 + dx)
            best = min(best, _mismatch(ref_canvas, shifted))
    return FontResult(mismatch_fraction=best, valid=valid)
