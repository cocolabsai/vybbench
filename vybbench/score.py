"""The scorer: every benchmark number, computed from the artifacts alone.

    vybbench score

Reads ``data/`` (the cases), ``starts/`` (each edit's shared starting design),
``references/``, ``outputs/`` (every model's generations), and ``verdicts/`` (the recorded
judge verdicts), and prints the three score tables. No model APIs and no keys:
the one non-deterministic step, the edit judge, ships as recorded verdicts.

The benchmark runner scores with these same functions, so the published scorer
is the code that produced the numbers.
"""

from __future__ import annotations

import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from PIL import Image

from . import colors, data, edits, fonts
from .cases import color_cases, edit_cases, font_cases

ARMS = ("ours", "gpt", "muse", "recraft")


def root() -> Path:
    return data.root(*data.SCORE)


def outputs_dir() -> Path:
    return root() / "outputs"


def verdicts_dir() -> Path:
    return root() / "verdicts"


def slug(case_id: str) -> str:
    return case_id.replace("/", "_").replace("#", "").lower()


def _attempt_paths(arm: str, subtest: str, case_id: str,
                   published_root: Path | None = None) -> list[Path]:
    """This machine's own run of ``arm`` (``data.out_dir()``) wins over the
    published one, so a model you ran is scored from your outputs.
    ``published_root`` lets worker processes reuse the root the parent already
    resolved instead of re-triggering the download."""
    published = (published_root or root()) / "outputs"
    for base in (data.out_dir() / "outputs", published):
        directory = base / arm / subtest / slug(case_id)
        if directory.is_dir():
            return sorted(directory.glob("a*.png"))
    return []


def _verdict_paths(arm: str, design_id: str) -> list[Path]:
    pattern = f"{slug(design_id)}_{arm}_a*.json"
    for base in (data.out_dir() / "verdicts", verdicts_dir()):
        found = sorted(base.glob(pattern))
        if found:
            return found
    return []


def _percentiles(values: list[float], points: tuple[int, ...]) -> dict:
    if not values:
        return {}
    ranked = sorted(values)
    out = {"mean": statistics.fmean(ranked), "median": statistics.median(ranked)}
    for p in points:
        out[f"p{p}"] = ranked[min(len(ranked) - 1, int(round(p / 100 * (len(ranked) - 1))))]
    return out


def _font_case_ious(case, arms: tuple[str, ...], root_dir: Path) -> dict:
    """One case's per-arm attempt IoUs; alignment search is CPU-heavy, so
    score_fonts fans cases out over processes. ``root_dir`` is resolved by the
    parent: workers must never resolve it themselves, or each would race to
    download the dataset."""
    reference = Image.open(root_dir / "references" / f"{slug(case.case_id)}.png")
    components = fonts.max_ink_components(case.word)
    out = {}
    for arm in arms:
        ious = [
            1.0 - result.mismatch_fraction
            for path in _attempt_paths(arm, "fonts", case.case_id,
                                       published_root=root_dir)
            if (result := fonts.font_mismatch(
                reference, Image.open(path), components)).valid
        ]
        if ious:
            out[arm] = ious
    return out


def score_fonts(arms: tuple[str, ...] = ARMS) -> dict:
    """Per-arm IoU distribution over every scored attempt, plus the recorded
    measurement noise floor over the same word set."""
    root_dir = root()  # resolve (and download) once, before the pool forks
    cases = font_cases()
    with ProcessPoolExecutor() as pool:
        case_scores = list(pool.map(
            partial(_font_case_ious, arms=arms, root_dir=root_dir), cases))
    per_arm = {
        arm: [iou for scores in case_scores for iou in scores.get(arm, [])]
        for arm in arms
    }
    summary = {arm: _percentiles(vals, (20, 10, 5)) for arm, vals in per_arm.items()}
    summary["noise_floor"] = _percentiles(
        [1.0 - case.noise_floor for case in cases], (20, 10, 5),
    )
    return summary


def score_colors(arms: tuple[str, ...] = ARMS) -> dict:
    """Per-arm max-channel error distribution over every scored attempt."""
    per_arm: dict = {arm: [] for arm in arms}
    for case in color_cases():
        for arm in arms:
            per_arm[arm] += [
                float(result.channel_error)
                for path in _attempt_paths(arm, "colors", case.case_id)
                if (result := colors.color_error(Image.open(path), case.hex)).valid
            ]
    return {arm: _percentiles(vals, (80, 90, 95)) for arm, vals in per_arm.items()}


def score_edits(arms: tuple[str, ...] = ARMS) -> dict:
    """Two numbers per arm, each over every generation: the whole-image change,
    and the rate at which the judge accepted the requested change as made."""
    change: dict = {arm: [] for arm in arms}
    attempts: dict = {arm: [] for arm in arms}
    for case in edit_cases():
        start = Image.open(root() / case.start_png)
        for arm in arms:
            change[arm] += [
                edits.whole_change(start, Image.open(path))
                for path in _attempt_paths(arm, "edits", case.design_id)
            ]
            attempts[arm] += [
                json.loads(path.read_text())["satisfied"]
                for path in _verdict_paths(arm, case.design_id)
            ]
    summary = {}
    for arm in arms:
        summary[arm] = {
            **_percentiles(change[arm], (80, 90, 95)),
            "satisfied_rate": (
                statistics.fmean(attempts[arm]) if attempts[arm] else None
            ),
        }
    return summary


def print_summary(label: str, summary: dict) -> None:
    print(f"\n=== {label} ===")
    for arm, stats in summary.items():
        line = "  ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in stats.items()
        )
        print(f"  {arm:12s} {line}")


def main() -> None:
    print_summary("fonts (IoU, higher is better)", score_fonts())
    print_summary("colors (max channel error, lower is better)", score_colors())
    print_summary("edits (fraction changed, lower is better)", score_edits())


if __name__ == "__main__":
    main()
