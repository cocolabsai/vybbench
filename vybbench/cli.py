"""``vybbench run | judge | score``."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path


def _run(args) -> None:
    from . import harness
    from .model import load_model

    model = load_model(args.model)
    subtests = tuple(args.subtests.split(","))
    written, failed = harness.run(
        model, subtests, args.attempts, args.limit, args.workers,
    )
    print(f"{model.name}: wrote {written}, failed {failed}"
          + (" (re-run the same command to retry)" if failed else ""))


def _judge(args) -> None:
    from . import judge

    judge.main(args.models or None)


def _score(args) -> None:
    from . import data, judge, score

    arms = tuple(dict.fromkeys(
        (args.arms.split(",") if args.arms else list(score.ARMS)) + judge.local_arms()
    ))
    results = {
        "fonts": score.score_fonts(arms),
        "colors": score.score_colors(arms),
        "edits": score.score_edits(arms),
    }
    score.print_summary("fonts (IoU, higher is better)", results["fonts"])
    score.print_summary("colors (max channel error, lower is better)",
                        results["colors"])
    score.print_summary("edits (fraction changed, lower is better)",
                        results["edits"])
    if args.write:
        # From v1.1 on the results carry how they were produced. v1.0.json
        # predates this and stays frozen without it.
        document = {
            "judge_model": judge.JUDGE_MODEL,
            "judge_prompt_sha256": judge.prompt_sha256(),
            "data_revision": data.revision(),
            **results,
        }
        Path(args.write).write_text(json.dumps(document, indent=2) + "\n")
        print(f"\nwrote {args.write}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vybbench", description=__doc__)
    parser.add_argument(
        "--out", help="where your runs write outputs/ and verdicts/ "
                      "(default: ./vybbench_out)")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a model over the benchmark")
    run.add_argument("--model", required=True,
                     help="gpt, recraft, or a dotted path pkg.module:Class")
    run.add_argument("--subtests", default="fonts,colors,edits")
    run.add_argument("--attempts", type=int, default=3)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--limit", type=int, default=0,
                     help="first N cases per subtest (a smoke slice)")
    run.set_defaults(func=_run)

    judge = commands.add_parser(
        "judge", help="judge your edit outputs (needs GEMINI_API_KEY)")
    judge.add_argument("models", nargs="*",
                       help="default: every model you ran under --out")
    judge.set_defaults(func=_judge)

    score = commands.add_parser(
        "score", help="score the published models and any you ran (no keys)")
    score.add_argument("--arms", help="comma-separated; default: the published four")
    score.add_argument("--write", metavar="PATH",
                       help="also write a results JSON with its provenance")
    score.set_defaults(func=_score)

    args = parser.parse_args(argv)
    if args.out:
        os.environ["VYBBENCH_OUT"] = args.out
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
