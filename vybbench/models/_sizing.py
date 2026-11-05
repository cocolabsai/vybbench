"""Size policy shared by the built-in adapters, so GPT and Recraft render at
comparable shapes (the same policy the published run used)."""

BASE = 1024


def dimensions(width: int, height: int) -> tuple[int, int]:
    """Pixel size near the width:height ratio: ratio clamped to [1/3, 3], the
    longer side scaled from ``BASE``, the off-base side a multiple of 16."""
    ratio = min(3.0, max(1 / 3, width / height))
    if ratio >= 1:
        return round(BASE * ratio / 16) * 16, BASE
    return BASE, round(BASE / ratio / 16) * 16
