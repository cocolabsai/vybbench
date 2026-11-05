"""Recraft through its own API (``RECRAFT_API_KEY``).

Text-to-image uses ``/v1/images/generations``. Recraft's image-to-image
endpoint takes no size and only supports V3, so edits go through
``/v1/images/imageToImage`` (strength 0.5, prompt capped at 1000 characters),
as in the published run. ``VYBBENCH_RECRAFT_MODEL`` overrides the generation
model id.
"""

from __future__ import annotations

import io
import os

import httpx
from PIL import Image

from ..model import Model, Start
from ._sizing import dimensions

BASE_URL = "https://external.api.recraft.ai/v1/images"
DEFAULT_MODEL = "recraftv4_1"
EDIT_STRENGTH = 0.5
PROMPT_MAX_CHARS = 1000


class Recraft(Model):
    name = "recraft"

    def __init__(self) -> None:
        self.headers = {"Authorization": f"Bearer {os.environ['RECRAFT_API_KEY']}"}
        self.model = os.environ.get("VYBBENCH_RECRAFT_MODEL", DEFAULT_MODEL)

    def generate(self, prompt: str, width: int, height: int) -> bytes:
        w, h = dimensions(width, height)
        response = httpx.post(
            f"{BASE_URL}/generations", headers=self.headers, timeout=120,
            json={"prompt": prompt[:PROMPT_MAX_CHARS], "model": self.model,
                  "size": f"{w}x{h}"},
        )
        return self._fetch(response)

    def edit(self, instruction: str, start: Start) -> bytes:
        response = httpx.post(
            f"{BASE_URL}/imageToImage", headers=self.headers, timeout=120,
            files={"image": ("start.png", start.png, "image/png")},
            data={"prompt": instruction[:PROMPT_MAX_CHARS],
                  "strength": str(EDIT_STRENGTH)},
        )
        return self._fetch(response)

    @staticmethod
    def _fetch(response: httpx.Response) -> bytes:
        response.raise_for_status()
        image = httpx.get(response.json()["data"][0]["url"], timeout=120)
        image.raise_for_status()
        # Recraft returns WebP by default; the harness expects real PNG bytes.
        buffer = io.BytesIO()
        with Image.open(io.BytesIO(image.content)) as decoded:
            decoded.save(buffer, format="PNG")
        return buffer.getvalue()
