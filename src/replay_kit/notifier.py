from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT, repo_path
from .summary import conclusion_lines


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(REPO_ROOT / ".env")


DEFAULT_MAX_TEXT_CHARS = 1200


def read_json_if_exists(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def summary_excerpt(summary_path: str | Path, max_chars: int = 600) -> str:
    path = repo_path(summary_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n..."


def format_float(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.4g}"
    return str(value)


def aggregate_brief(metrics: dict[str, Any]) -> list[str]:
    aggregate = metrics.get("aggregate")
    if not isinstance(aggregate, dict) or not aggregate:
        return []
    arm_labels = {
        "random_init_train": "rand",
        "random_init_noise_train": "rand+noise",
        "pretrained_train": "pre",
        "pretrained_noise_train": "pre+noise",
    }
    lines = ["结果摘要(acc mean):"]
    for domain, arms in aggregate.items():
        if not isinstance(arms, dict):
            continue
        parts = []
        for arm, label in arm_labels.items():
            summary = arms.get(arm, {})
            value = (
                summary.get("metrics", {})
                .get("accuracy", {})
                .get("mean")
                if isinstance(summary, dict)
                else None
            )
            if value is not None:
                parts.append(f"{label}={format_float(value)}")
        if parts:
            lines.append(f"- {domain}: " + ", ".join(parts))
    return lines if len(lines) > 1 else []


def metrics_brief(metrics: dict[str, Any]) -> list[str]:
    aggregate_lines = aggregate_brief(metrics)
    if aggregate_lines:
        return aggregate_lines
    if not metrics:
        return []
    scalar_keys = [
        "final_loss",
        "best_loss",
        "reached_target",
        "accuracy",
        "nll",
        "ece",
        "device",
    ]
    lines = []
    for key in scalar_keys:
        if key in metrics:
            lines.append(f"- {key}: {format_float(metrics[key])}")
    return ["结果摘要:", *lines] if lines else []


def conclusion_brief(metrics: dict[str, Any], status: str) -> list[str]:
    lines = conclusion_lines(metrics, status)
    if not lines:
        return []
    return ["结论:", *[f"- {line}" for line in lines]]


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n...（通知已截短，完整内容见 summary/log 路径）"
    keep = max(0, max_chars - len(marker))
    return text[:keep].rstrip() + marker


def build_text(
    metadata: dict[str, Any],
    result_lines: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    status = metadata.get("status")
    title = "实验完成" if status == "finished" else "实验失败"
    lines = [
        f"{title}: {metadata.get('method_name')}/{metadata.get('experiment_name')}",
        f"run_id: {metadata.get('run_id')}",
        f"status: {status}",
        f"branch: {metadata.get('branch')}",
        f"commit: {metadata.get('short_commit') or metadata.get('commit')}",
        f"host: {metadata.get('hostname')}",
        f"device: {metadata.get('device')} gpu_id={metadata.get('gpu_id')}",
        f"start: {metadata.get('start_time')}",
        f"end: {metadata.get('end_time')}",
        f"summary: {metadata.get('summary_path')}",
        f"log: {metadata.get('log_path')}",
    ]
    if metadata.get("error_message"):
        lines.append(f"error: {metadata.get('error_message')}")
    if result_lines:
        lines.extend(["", *result_lines])
    return truncate_text("\n".join(lines), max_chars)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def notify(
    run_dir: str | Path,
    metadata: dict[str, Any],
    notify_config: dict[str, Any] | None = None,
) -> Path:
    load_dotenv_if_available()
    notify_config = notify_config or {}
    run_dir = repo_path(run_dir)
    webhook = os.getenv("FEISHU_WEBHOOK", "").strip()
    explicit_dry_run = os.getenv("REPLAY_KIT_NOTIFY_DRY_RUN", "").strip().lower()
    enabled = notify_config.get("enabled", True)
    real_send = truthy(notify_config.get("real_send", False))
    dry_run = (
        not enabled
        or not real_send
        or explicit_dry_run in {"1", "true", "yes", "on"}
        or not webhook
    )
    max_chars = int(notify_config.get("max_text_chars", DEFAULT_MAX_TEXT_CHARS))
    metrics = read_json_if_exists(metadata.get("metrics_path", run_dir / "metrics.json"))
    result_lines = [*metrics_brief(metrics), *conclusion_brief(metrics, str(metadata.get("status", "")))]
    if not result_lines and metadata.get("status") == "failed":
        result_lines = ["日志尾部:", summary_excerpt(metadata.get("summary_path", ""))]
    text = build_text(metadata, result_lines, max_chars=max_chars)
    feishu_payload = {
        "msg_type": "text",
        "content": {"text": text},
    }
    payload = {
        "dry_run": dry_run,
        "enabled": bool(enabled),
        "real_send": real_send,
        "webhook_configured": bool(webhook),
        "event": "finished" if metadata.get("status") == "finished" else "failed",
        "feishu_payload": feishu_payload,
    }
    payload_path = run_dir / "notification_payload.json"
    payload_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if dry_run:
        print(f"[notifier] dry-run payload written to {payload_path}")
        return payload_path

    try:
        import requests

        response = requests.post(webhook, json=feishu_payload, timeout=10)
        response.raise_for_status()
    except Exception as exc:
        payload["send_error"] = str(exc)
        payload_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise

    print(f"[notifier] feishu notification sent; payload saved to {payload_path}")
    return payload_path
