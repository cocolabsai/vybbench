import pytest

from tests import fake
from vybbench import data, harness
from vybbench.model import Start, load_model


@pytest.fixture(autouse=True)
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("VYBBENCH_OUT", str(tmp_path))
    fake.CALLS.clear()
    return tmp_path


@pytest.fixture(autouse=True)
def stand_in_starts(tmp_path, monkeypatch):
    """The starting designs (``hf/starts/``) are ~256M and not in git, so the
    tests serve a tiny stand-in for any ``starts/`` path and never touch the
    real ones. Cases and prompts still come from the tracked ``hf/`` files."""
    (tmp_path / "start.png").write_bytes(fake._png("white"))
    (tmp_path / "start.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    real_path = data.path

    def path(relative, *patterns):
        if relative.startswith("starts/"):
            return tmp_path / relative.rsplit("/", 1)[-1]
        return real_path(relative, *patterns)

    monkeypatch.setattr(data, "path", path)


def test_load_model_dotted_path_and_unknown():
    assert load_model("tests.fake:Fake").name == "fake"
    with pytest.raises(ValueError):
        load_model("nonsense")


def test_run_layout_attempts_and_idempotence(out):
    model = load_model("tests.fake:Fake")
    written, failed = harness.run(
        model, ("colors", "edits"), attempts=3, limit=2, workers=2)
    assert (written, failed) == (2 * 3 * 2, 0)  # 2 cases x 3 attempts x 2 subtests
    pngs = sorted(p.relative_to(out) for p in (out / "outputs").rglob("a*.png"))
    assert len(pngs) == 12
    assert all(p.parts[:2] == ("outputs", "fake") for p in pngs)
    assert {p.parts[2] for p in pngs} == {"colors", "edits"}
    assert {p.name for p in pngs} == {"a0.png", "a1.png", "a2.png"}

    fake.CALLS.clear()
    assert harness.run(model, ("colors", "edits"), 3, 2, 2) == (0, 0)  # skips existing
    assert fake.CALLS == []


def test_edit_gets_only_instruction_and_start():
    harness.run(load_model("tests.fake:Fake"), ("edits",), 1, 1, 1)
    kind, instruction, start = fake.CALLS[0]
    assert kind == "edit" and isinstance(instruction, str)
    assert isinstance(start, Start)
    assert start.png[:8] == b"\x89PNG\r\n\x1a\n" and start.svg.lstrip().startswith("<")


def test_failed_call_leaves_slot_empty_and_retries(out):
    class Flaky(fake.Fake):
        name = "flaky"
        fail = True

        def generate(self, prompt, width, height):
            if Flaky.fail:
                raise RuntimeError("429")
            return super().generate(prompt, width, height)

    model = Flaky()
    assert harness.run(model, ("colors",), 1, 1, 1) == (0, 1)
    Flaky.fail = False
    assert harness.run(model, ("colors",), 1, 1, 1) == (1, 0)
