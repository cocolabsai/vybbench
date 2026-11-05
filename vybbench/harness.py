"""Run a model over the benchmark.

Loads the cases, fills the prompt templates, calls the model ``attempts``
times per case, and writes

    <out>/outputs/<model>/<subtest>/<case>/a<N>.png

Idempotent: an output already on disk is skipped, so an interrupted run
resumes, and a call that failed (logged, left empty) retries, by re-running
the same command.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import data, references
from .cases import color_cases, edit_cases, font_cases
from .model import Model, Start
from .references import CANVAS
from .score import slug

logger = logging.getLogger("vybbench")

SUBTESTS = ("fonts", "colors", "edits")


def output_path(model: str, subtest: str, case_id: str, attempt: int) -> Path:
    return (data.out_dir() / "outputs" / model / subtest / slug(case_id)
            / f"a{attempt}.png")


def _jobs(model: Model, subtests, attempts: int, limit: int):
    """Every (path, call) still to make, in a stable order."""
    for subtest in subtests:
        if subtest == "fonts":
            cases = font_cases()[:limit or None]
            calls = [(c.case_id, references.font_prompt(c)) for c in cases]
        elif subtest == "colors":
            cases = color_cases()[:limit or None]
            calls = [(c.case_id, references.color_prompt(c.hex)) for c in cases]
        else:
            for case in edit_cases()[:limit or None]:
                for attempt in range(attempts):
                    out = output_path(model.name, "edits", case.design_id, attempt)
                    if not out.exists():
                        yield out, _edit_call(model, case)
            continue
        for case_id, prompt in calls:
            for attempt in range(attempts):
                out = output_path(model.name, subtest, case_id, attempt)
                if not out.exists():
                    yield out, _generate_call(model, prompt)


def _generate_call(model: Model, prompt: str):
    return lambda: model.generate(prompt, CANVAS, CANVAS)


def _edit_call(model: Model, case):
    def call() -> bytes:
        start = Start(
            png=data.path(case.start_png, *data.RUN).read_bytes(),
            svg=data.path(case.start_svg, *data.RUN).read_text(),
            width=case.width, height=case.height,
        )
        return model.edit(case.instruction, start)
    return call


def _one(out: Path, call) -> bool:
    try:
        png = call()
    except Exception as error:  # a failed call leaves its slot empty
        logger.error("%s failed: %s", out, error)
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    return True


def run(model: Model, subtests=SUBTESTS, attempts: int = 3, limit: int = 0,
        workers: int = 8) -> tuple[int, int]:
    """Returns (written, failed)."""
    jobs = list(_jobs(model, subtests, attempts, limit))
    logger.info("%s: %d outputs to make", model.name, len(jobs))
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda job: _one(*job), jobs))
    return sum(results), len(results) - sum(results)
