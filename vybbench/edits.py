"""The editing metric: region-free whole-image change, zero tolerance.

An edit's score is the fraction of pixels that changed AT ALL between the
shared starting design and the model's result -- max per-channel delta > 0
counts the pixel as changed. No region masks and no tolerance band: a
non-destructive editor leaves untouched pixels byte-identical, and anything
else is change the user did not ask for.

Whether the edit was actually performed is judged separately (an independent
LLM reviewer; see ``prompts/judge.txt``) and reported alongside -- a model
that returns its input unchanged scores 0 here but fails the judge.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def whole_change(start: Image.Image, result: Image.Image) -> float:
    """Fraction of pixels changed between ``start`` and ``result``. An arm
    that returns another canvas size is resized onto the start's grid first."""
    a = np.asarray(start.convert("RGB"))
    b = np.asarray(result.convert("RGB"))
    if b.shape != a.shape:
        resized = Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.LANCZOS)
        b = np.asarray(resized)
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1)
    return float((delta > 0).mean())
