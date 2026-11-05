"""GPT Image through the OpenAI API (``OPENAI_API_KEY``).

The published v1.0 arm ran through an Azure OpenAI deployment; this adapter
calls the OpenAI API directly. ``VYBBENCH_GPT_MODEL`` overrides the model id.
"""

from __future__ import annotations

import base64
import os

from ..model import Model, Start
from ._sizing import dimensions

DEFAULT_MODEL = "gpt-image-2.5-sunburst"
QUALITY = "high"  # a benchmark reports a model's best attempt


class GPT(Model):
    name = "gpt"

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI()  # reads OPENAI_API_KEY
        self.model = os.environ.get("VYBBENCH_GPT_MODEL", DEFAULT_MODEL)

    def generate(self, prompt: str, width: int, height: int) -> bytes:
        w, h = dimensions(width, height)
        response = self.client.images.generate(
            model=self.model, prompt=prompt, quality=QUALITY,
            size=f"{w}x{h}", output_format="png",
        )
        return _png(response)

    def edit(self, instruction: str, start: Start) -> bytes:
        w, h = dimensions(start.width, start.height)
        response = self.client.images.edit(
            model=self.model,
            image=("start.png", start.png, "image/png"),
            prompt=instruction, quality=QUALITY,
            size=f"{w}x{h}", output_format="png",
        )
        return _png(response)


def _png(response) -> bytes:
    encoded = response.data[0].b64_json if response.data else None
    if not encoded:
        raise RuntimeError("empty image response")
    return base64.b64decode(encoded)
