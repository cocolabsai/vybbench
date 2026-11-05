# Changelog

## 1.0 (2026-10-20)

Initial release: 145 fonts (one pinned word each), 100 colors, 200
single-instruction edits on shared starting designs. Scoring: font IoU against
programmatic references with per-case noise floors; max per-channel color
error; zero-tolerance whole-image change for edits, with an independent
reviewer's acceptance rate reported alongside.

### 1.0 provenance

`results/v1.0.json` is frozen; it carries no provenance block, so it is
recorded here. Releases from 1.1 on write the judge model, judge prompt hash
and data revision into the results JSON itself.

- Judge model: `gemini-3.5-flash` (recorded in all 2,400 verdicts), temperature 0.
- Judge prompt: `prompts/judge.txt`, sha256
  `30b68b6812ce17a8901107d81d19481812a844bada8cf39bf40ebe3f841f4f97`.
- Data revision: set when the dataset is re-uploaded in its final layout.
