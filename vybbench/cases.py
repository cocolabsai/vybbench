"""Case loaders for the released benchmark data.

Each loader reads one file under ``data/`` (see ``data.py`` for where that lives) so a case list is identical across
runs and across re-scoring: 145 fonts (one pinned word each), 100 colors, and
200 single-instruction edits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import data


@dataclass(frozen=True)
class FontCase:
    """One Google font and one word to render. The ground truth ships as a
    rendered image under ``references/``; ``noise_floor`` is that reference
    scored against itself rendered at another size and position."""

    family: str
    word: str
    noise_floor: float
    variant: str = "Regular"

    @property
    def case_id(self) -> str:
        return f"{self.family}/{self.word}"


@dataclass(frozen=True)
class ColorCase:
    hex: str

    @property
    def case_id(self) -> str:
        return self.hex.lstrip("#").lower()


@dataclass(frozen=True)
class EditCase:
    """One design and the single edit asked of it. ``start_png``/``start_svg``
    name the shared starting design under ``starts/``; every arm edits that
    same design."""

    design_id: str
    width: int
    height: int
    category: str
    instruction: str
    start_svg: str
    start_png: str


def _read_jsonl(name: str) -> list[dict]:
    path = data.path(f"data/{name}", *data.RUN)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def font_cases() -> list[FontCase]:
    return [FontCase(**row) for row in _read_jsonl("fonts.jsonl")]


def color_cases() -> list[ColorCase]:
    return [ColorCase(**row) for row in _read_jsonl("colors.jsonl")]


def edit_cases() -> list[EditCase]:
    return [EditCase(**row) for row in _read_jsonl("edits.jsonl")]
