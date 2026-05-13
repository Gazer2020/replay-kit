from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT, repo_path


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(REPO_ROOT / ".env")


def summary_excerpt(summary_path: str | Path, max_chars: int = 1200) -> str:
    path = repo_path(summary_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n..."


def build_text(metadata: dict[str, Any], summary_text: str) -> str:
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
    if summary_text:
        lines.extend(["", summary_text])
    return "\n".join(lines)


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
    text = build_text(metadata, summary_excerpt(metadata.get("summary_path", "")))
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
