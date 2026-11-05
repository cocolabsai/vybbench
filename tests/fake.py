import io

from PIL import Image

from vybbench.model import Model, Start

CALLS: list = []


def _png(color: str, size=(64, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class Fake(Model):
    name = "fake"

    def generate(self, prompt: str, width: int, height: int) -> bytes:
        CALLS.append(("generate", prompt, width, height))
        return _png("white", (width // 16, height // 16))

    def edit(self, instruction: str, start: Start) -> bytes:
        CALLS.append(("edit", instruction, start))
        return start.png
