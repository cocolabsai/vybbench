"""The interface a model implements to be benchmarked.

    class MyModel(Model):
        name = "mymodel"
        def generate(self, prompt, width, height): ...
        def edit(self, instruction, start): ...

Both methods return PNG bytes. Producing the PNG is the model's job: an
SVG-native model rasterizes with its own renderer, because the harness never
rasterizes and so no rasterizer choice leaks into the scores.

If your API has no size parameter, append the aspect ratio to the prompt
yourself (the "Vyb aspect hint" in the paper's disclosure is exactly this); it
is an adapter concern, not the harness's.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

# Built-in adapters, by the name ``vybbench run --model`` takes.
BUILTIN = {
    "gpt": "vybbench.models.gpt:GPT",
    "recraft": "vybbench.models.recraft:Recraft",
}


@dataclass(frozen=True)
class Start:
    """The shared starting design of an edit case, in both representations.
    Use whichever your model works on: an image-edit model reads ``png``, an
    SVG-native model reads ``svg``."""

    png: bytes
    svg: str
    width: int
    height: int


class Model:
    """Base class. ``name`` names the model's folder under ``outputs/``."""

    name: str

    def generate(self, prompt: str, width: int, height: int) -> bytes:
        """Text to image. Return PNG bytes."""
        raise NotImplementedError

    def edit(self, instruction: str, start: Start) -> bytes:
        """Apply one instruction to ``start``. Return PNG bytes. Only the
        instruction and the start are passed: no case metadata."""
        raise NotImplementedError


def load_model(spec: str) -> Model:
    """``gpt``/``recraft``, or a dotted import path ``pkg.module:Class``."""
    target = BUILTIN.get(spec, spec)
    module_name, _, class_name = target.partition(":")
    if not class_name:
        raise ValueError(
            f"unknown model {spec!r}: use one of {sorted(BUILTIN)} "
            "or a dotted path like mypkg.mymodel:MyModel"
        )
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls()
