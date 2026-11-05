"""Where the benchmark's data lives, and how the code reaches it.

The benchmark's inputs and artifacts (cases, prompts, starting designs,
references, every model's outputs, judge verdicts, results) are published as a
Hugging Face dataset; this package is only code. Every read goes through
:func:`root`, which has one branch:

* an ``hf/`` folder next to this package (the layout of the source repo) is
  read in place, so the same code works inside the repo with no download;
* otherwise the needed files are fetched from the Hub's resolver endpoint,
  pinned to :data:`REVISION`, into ``~/.cache/vybbench/<revision>/`` -- and
  only the patterns the command asks for.

Files are fetched as plain ``/resolve/`` GETs rather than through
``snapshot_download`` because the Hub meters its API endpoints far tighter
than its resolver endpoint; for this dataset's ~10k files that is the
difference between minutes and hours.

``VYBBENCH_DATA`` overrides both with an explicit directory;
``VYBBENCH_CACHE`` moves the download cache.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache, partial
from pathlib import Path

import httpx

logger = logging.getLogger("vybbench")

REPO_ID = "cocolabs/vybbench"

# The HF commit this release of the code reproduces. Set once per release,
# after the upload that creates that commit, so later uploads cannot change
# what a release computes. ``None`` follows the dataset's latest revision.
REVISION: str | None = "2907a6860693f811f98bea13e9b6cca55d226151"

HUB = "https://huggingface.co"

_LOCAL = Path(__file__).resolve().parents[2] / "hf"

# What each command pulls. ``run`` needs the inputs (~360M); ``score`` needs
# the recorded verdicts (~10M) plus every output it measures, because fonts,
# colors and the edit change-fraction are computed from the images
# themselves (~5G); the judge needs only the edit inputs, prompt and recorded verdicts
# (the published outputs come down only to score or re-judge them).
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
    return _download(patterns)


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
    return REVISION or "main"


def _prefixes(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """The glob patterns' fixed prefixes, which is all the filter needs."""
    return tuple(p.split("*", 1)[0] for p in patterns) or ("",)


def _headers() -> dict:
    token = os.environ.get("HF_TOKEN", "")
    return {"authorization": f"Bearer {token}"} if token else {}


def _tree(client: httpx.Client, rev: str) -> list[tuple[str, int]]:
    """Every (path, size) in the dataset at ``rev``, via the paginated tree API."""
    files = []
    url = f"{HUB}/api/datasets/{REPO_ID}/tree/{rev}?recursive=true"
    while url:
        response = client.get(url)
        response.raise_for_status()
        files += [(e["path"], e["size"]) for e in response.json()
                  if e["type"] == "file"]
        url = response.links.get("next", {}).get("url")
    return files


def _fetch(client: httpx.Client, rev: str, base: Path,
           item: tuple[str, int]) -> bool:
    """Download one file unless it is already complete. Honors Retry-After."""
    relative, size = item
    dest = base / relative
    if dest.exists() and dest.stat().st_size == size:
        return False
    url = f"{HUB}/datasets/{REPO_ID}/resolve/{rev}/{relative}"
    while True:
        with client.stream("GET", url) as response:
            if response.status_code == 429:
                wait = float(response.headers.get("retry-after", 60))
                logger.info("rate limited (%s); waiting %.0fs",
                            response.headers.get("ratelimit-policy", "no policy header"),
                            wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_suffix(dest.suffix + ".part")
            with open(part, "wb") as file:
                for chunk in response.iter_bytes(1 << 20):
                    file.write(chunk)
            part.rename(dest)
            return True


def _download(patterns: tuple[str, ...]) -> Path:
    rev = REVISION or "main"
    base = Path(os.environ.get(
        "VYBBENCH_CACHE", Path.home() / ".cache" / "vybbench")) / rev
    prefixes = _prefixes(patterns)
    with httpx.Client(follow_redirects=True, headers=_headers(),
                      timeout=120) as client:
        files = [f for f in _tree(client, rev) if f[0].startswith(prefixes)]
        logger.info("%s: fetching %d files into %s", REPO_ID, len(files), base)
        with ThreadPoolExecutor(4) as pool:
            fetched = sum(pool.map(partial(_fetch, client, rev, base), files))
    logger.info("%d downloaded, %d already present", fetched,
                len(files) - fetched)
    return base
