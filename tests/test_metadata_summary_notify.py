from __future__ import annotations

import json
from pathlib import Path

import yaml

from replay_kit.metadata import initial_metadata, write_metadata
from replay_kit.notifier import notify
from replay_kit.summary import generate_summary


def make_run_dir(tmp_path: Path) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {
        "project_name": "replay-kit",
        "method_name": "toy_torch",
        "experiment_name": "debug",
        "device": "auto",
        "gpu_id": 0,
        "seed": 7,
        "summary": {"log_tail_lines": 5},
    }
    config_path = run_dir / "config.yaml"
    log_path = run_dir / "train.log"
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.md"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    log_path.write_text("line1\nline2\n", encoding="utf-8")
    metrics_path.write_text(
        json.dumps({"device": "cpu", "final_loss": 0.1, "best_loss": 0.1, "reached_target": True}),
        encoding="utf-8",
    )
    metadata = initial_metadata(
        run_id="test-run",
        config=config,
        run_dir=run_dir,
        config_path=config_path,
        log_path=log_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        command="pytest",
    )
    metadata["status"] = "finished"
    metadata["end_time"] = metadata["start_time"]
    write_metadata(run_dir / "metadata.json", metadata)
    return run_dir, metadata


def test_generate_summary_from_run_artifacts(tmp_path: Path) -> None:
    run_dir, _ = make_run_dir(tmp_path)
    summary_path = generate_summary(run_dir)
    text = summary_path.read_text(encoding="utf-8")
    assert "# 实验总结" in text
    assert "actual_device：cpu" in text
    assert "best_loss" in text


def test_notifier_dry_run_without_webhook(tmp_path: Path, monkeypatch) -> None:
    run_dir, metadata = make_run_dir(tmp_path)
    summary_path = generate_summary(run_dir)
    metadata["summary_path"] = str(summary_path)
    monkeypatch.setenv("FEISHU_WEBHOOK", "")
    monkeypatch.delenv("REPLAY_KIT_NOTIFY_DRY_RUN", raising=False)
    payload_path = notify(run_dir, metadata)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["webhook_configured"] is False
    assert payload["event"] == "finished"


def test_notifier_dry_run_when_real_send_disabled(tmp_path: Path, monkeypatch) -> None:
    run_dir, metadata = make_run_dir(tmp_path)
    summary_path = generate_summary(run_dir)
    metadata["summary_path"] = str(summary_path)
    monkeypatch.setenv("FEISHU_WEBHOOK", "https://open.feishu.cn/open-apis/bot/v2/hook/test")
    monkeypatch.delenv("REPLAY_KIT_NOTIFY_DRY_RUN", raising=False)
    payload_path = notify(run_dir, metadata, {"enabled": True, "real_send": False})
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["webhook_configured"] is True
    assert payload["real_send"] is False
