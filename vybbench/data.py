"""Where the benchmark's data lives, and how the code reaches it.

The benchmark's inputs and artifacts (cases, prompts, starting designs,
references, every model's outputs, judge verdicts, results) are published as a
Hugging Face dataset; this package is only code. Every read goes through
:func:`root`, which has one branch:

* an ``hf/`` folder next to this package (the layout of the source repo) is
  read in place, so the same code works inside the repo with no download;
* otherwise the needed files are fetched with ``snapshot_download``, pinned to
  :data:`REVISION`, and only the patterns the command asks for.

``VYBBENCH_DATA`` overrides both with an explicit directory.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

REPO_ID = "cocolabs/vybbench"

# The HF commit this release of the code reproduces. Set once per release,
# after the upload that creates that commit, so later uploads cannot change
# what a release computes. ``None`` follows the dataset's latest revision.
REVISION: str | None = "2907a6860693f811f98bea13e9b6cca55d226151"

_LOCAL = Path(__file__).resolve().parents[2] / "hf"

# What each command pulls. ``run`` needs the inputs (~360M); ``score`` needs
# the recorded verdicts (~10M) plus every output it measures, because fonts,
# colors and the edit change-fraction are computed from the images
# themselves (~5G); the judge needs only the edit inputs, prompt and recorded verdicts
# (the 4.8G of published outputs comes down only to score or re-judge them).
RUN = ("data/*", "prompts/*", "starts/**", "references/*")
SCORE = ("data/*", "starts/**", "references/*", "verdicts/*", "outputs/**")
JUDGE = ("data/*", "prompts/*", "starts/**", "verdicts/*")


def out_dir() -> Path:
    """Where this machine's own runs write (``outputs/``, ``verdicts/``).

    The dataset root may be a read-only download cache, so a run never writes
    into it. ``VYBBENCH_OUT`` (set by ``--out``) overrides the default."""
    return Path(os.environ.get("VYBBENCH_OUT", "vybbench_out"))


@cache
def root(*patterns: str) -> Path:
    """Directory holding the dataset, with at least ``patterns`` present."""
    if override := os.environ.get("VYBBENCH_DATA"):
        return Path(override)
    if _LOCAL.is_dir():
        return _LOCAL
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=list(patterns) or None,
    ))


def path(relative: str, *patterns: str) -> Path:
    """``relative`` under the dataset root (fetching ``patterns`` if remote)."""
    return root(*patterns) / relative


def read_prompt(name: str) -> str:
    """A prompt file's text, e.g. ``read_prompt("judge.txt")``."""
    return path(f"prompts/{name}", "prompts/*").read_text()


def revision() -> str:
    """The data revision the numbers were computed from: the pinned HF commit
    when downloaded, ``local`` when read from an ``hf/`` checkout."""
    if os.environ.get("VYBBENCH_DATA") or _LOCAL.is_dir():
        return "local"
    return REVISION or root().name  # snapshot dirs are named by commit hash
