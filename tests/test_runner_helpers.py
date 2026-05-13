from __future__ import annotations

from replay_kit.runner import make_run_id, safe_name


def test_safe_name_strips_unsafe_characters() -> None:
    assert safe_name("a/b c:d") == "a_b_c_d"


def test_make_run_id_contains_commit_or_placeholder() -> None:
    run_id = make_run_id()
    assert len(run_id.split("_")) >= 3
