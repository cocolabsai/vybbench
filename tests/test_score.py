import json

from vybbench import data, judge, score


def test_published_arm_scores_are_unchanged_by_a_local_run(tmp_path, monkeypatch):
    monkeypatch.setenv("VYBBENCH_OUT", str(tmp_path))
    assert score._attempt_paths("gpt", "colors", "000000") == score._attempt_paths(
        "gpt", "colors", "000000")
    assert judge.local_arms() == []


def test_local_outputs_win_over_published(tmp_path, monkeypatch):
    monkeypatch.setenv("VYBBENCH_OUT", str(tmp_path))
    case = tmp_path / "outputs" / "gpt" / "colors" / "000000"
    case.mkdir(parents=True)
    (case / "a0.png").write_bytes(b"x")
    assert [p.parent for p in score._attempt_paths("gpt", "colors", "#000000")] == [case]


def test_judge_skips_verdicts_it_already_wrote(tmp_path, monkeypatch):
    from vybbench.cases import edit_cases

    monkeypatch.setenv("VYBBENCH_OUT", str(tmp_path))
    case_slug = score.slug(edit_cases()[0].design_id)
    out = tmp_path / "outputs" / "mymodel" / "edits" / case_slug
    out.mkdir(parents=True)
    (out / "a0.png").write_bytes(b"png")
    assert len(judge.pending_judgments(["mymodel"])) == 1

    verdicts = tmp_path / "verdicts"
    verdicts.mkdir()
    (verdicts / f"{case_slug}_mymodel_a0.json").write_text("{}")
    assert judge.pending_judgments(["mymodel"]) == []


def test_judge_prompt_hash_is_stable():
    assert len(judge.prompt_sha256()) == 64
    assert judge.prompt_sha256() == judge.prompt_sha256()


def test_revision_is_local_inside_the_repo():
    assert data.revision() == "local"


def test_v1_0_results_have_no_provenance_block():
    results = json.loads((data.root() / "results" / "v1.0.json").read_text())
    assert set(results) == {"fonts", "colors", "edits"}
