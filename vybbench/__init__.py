"""VybBench scoring: standalone, offline, no model APIs.

Every number in the benchmark recomputes from the released artifacts alone:

* fonts  -- IoU against a programmatic (PIL) render of the pinned font file,
            after crop/scale alignment (``fonts.font_mismatch``).
* colors -- max per-channel delta of the produced rectangle against the
            requested hex (``colors.color_error``).
* edits  -- fraction of pixels changed at all between the shared starting
            design and the result (``edits.whole_change``), with the recorded
            LLM reviewer's acceptance rate reported alongside.

``vybbench score`` recomputes everything from the released artifacts.
"""
