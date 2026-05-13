from __future__ import annotations

import json

from replay_kit.metadata import initial_metadata, write_metadata
from replay_kit.runner import make_run_id, record_postprocess_error, safe_name


def test_safe_name_strips_unsafe_characters() -> None:
    assert safe_name("a/b c:d") == "a_b_c_d"


def test_make_run_id_contains_commit_or_placeholder() -> None:
    run_id = make_run_id()
    assert len(run_id.split("_")) >= 3


def test_record_postprocess_error_appends_metadata_field(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metadata = initial_metadata(
        run_id="test-run",
        config={"project_name": "replay-kit", "method_name": "toy_torch", "experiment_name": "debug"},
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        log_path=run_dir / "train.log",
        metrics_path=run_dir / "metrics.json",
        summary_path=run_dir / "summary.md",
        command="pytest",
    )
    metadata_path = run_dir / "metadata.json"
    write_metadata(metadata_path, metadata)

    record_postprocess_error(metadata_path, "notification", RuntimeError("boom"))

    updated = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert updated["postprocess_errors"][0]["stage"] == "notification"
    assert updated["postprocess_errors"][0]["message"] == "boom"
