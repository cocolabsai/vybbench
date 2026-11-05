"""The edit judge, runnable: judges every edit output that has no recorded
verdict yet and writes verdicts in the released format. Already-recorded
verdicts are never re-judged, so the official rows stand on the shipped run;
this exists to verify the method and to judge new models' outputs.

Implements ``prompts/judge.txt`` exactly: ``gemini-3.5-flash`` at temperature
0, structured output ``{satisfied, reason}``.

    pip install google-genai
    GEMINI_API_KEY=... vybbench judge [model ...]   # default: every model you ran under --out
"""

from __future__ import annotations

import hashlib
import json

from google import genai
from google.genai import types

from . import data
from .cases import edit_cases
from .score import slug

JUDGE_MODEL = "gemini-3.5-flash"

VERDICT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "satisfied": {"type": "BOOLEAN"},
        "reason": {"type": "STRING"},
    },
    "required": ["satisfied", "reason"],
}


def _edit_outputs(arm: str, case_slug: str) -> list:
    return sorted((data.out_dir() / "outputs" / arm / "edits" / case_slug).glob("a*.png"))


def local_arms() -> list[str]:
    """Models with outputs under ``--out`` (the ones this machine ran)."""
    base = data.out_dir() / "outputs"
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


def pending_judgments(arms: list[str]) -> list[tuple]:
    """Every (case, arm, attempt, output path, verdict path) still unjudged."""
    pending = []
    for case in edit_cases():
        case_slug = slug(case.design_id)
        for arm in arms:
            for out in _edit_outputs(arm, case_slug):
                attempt = int(out.stem[1:])
                name = f"{case_slug}_{arm}_a{attempt}.json"
                local = data.out_dir() / "verdicts" / name
                if not local.exists() and not (
                    data.path("verdicts", *data.JUDGE) / name
                ).exists():
                    pending.append((case, arm, attempt, out, local))
    return pending


def judge_one(client: genai.Client, case, start_png: bytes, out_png: bytes) -> dict:
    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=[
            "Image 1 = ORIGINAL:",
            types.Part.from_bytes(data=start_png, mime_type="image/png"),
            "Image 2 = RESULT after instruction: " + case.instruction,
            types.Part.from_bytes(data=out_png, mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(
            system_instruction=data.read_prompt("judge.txt"),
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=VERDICT_SCHEMA,
        ),
    )
    return json.loads(response.text)


def prompt_sha256() -> str:
    return hashlib.sha256(data.read_prompt("judge.txt").encode()).hexdigest()


def main(arms: list[str] | None = None) -> None:
    arms = arms or local_arms()
    if not arms:
        print("no outputs under", data.out_dir() / "outputs", "- run a model first")
        return
    pending = pending_judgments(arms)
    if not pending:
        print("every edit output already has a recorded verdict; nothing to judge")
        return
    print(f"judging {len(pending)} edit outputs with {JUDGE_MODEL}")
    client = genai.Client()  # reads GEMINI_API_KEY
    (data.out_dir() / "verdicts").mkdir(parents=True, exist_ok=True)
    for case, arm, attempt, out, verdict_path in pending:
        verdict = judge_one(
            client, case, data.path(case.start_png, *data.JUDGE).read_bytes(), out.read_bytes(),
        )
        verdict_path.write_text(json.dumps({
            "design_id": case.design_id, "arm": arm, "attempt": attempt,
            "model": JUDGE_MODEL, **verdict,
        }, indent=2))
        print(f"{verdict_path.name}: satisfied={verdict['satisfied']}")


if __name__ == "__main__":
    main()
