from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT, repo_path


Metadata = dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    value = result.stdout.strip()
    return value or None


def git_branch() -> str:
    return run_git(["branch", "--show-current"]) or "unknown"


def git_commit(short: bool = False) -> str:
    args = ["rev-parse"]
    if short:
        args.append("--short")
    args.append("HEAD")
    return run_git(args) or "no-commit"


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cuda_device_count": 0,
        "gpu_name": None,
        "mps_available": False,
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        snapshot["torch_import_error"] = str(exc)
        return snapshot

    snapshot["torch_available"] = True
    snapshot["torch_version"] = getattr(torch, "__version__", None)
    snapshot["cuda_available"] = bool(torch.cuda.is_available())
    snapshot["cuda_version"] = getattr(torch.version, "cuda", None)
    snapshot["cuda_device_count"] = int(torch.cuda.device_count())
    if snapshot["cuda_available"]:
        try:
            snapshot["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover - device specific
            snapshot["gpu_name_error"] = str(exc)
    try:
        snapshot["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception:
        snapshot["mps_available"] = False
    return snapshot


def read_metadata(path: str | Path) -> Metadata:
    path = repo_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metadata(path: str | Path, metadata: Metadata) -> None:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
    tmp_path.replace(path)


def update_metadata(path: str | Path, **updates: Any) -> Metadata:
    metadata = read_metadata(path)
    metadata.update(updates)
    write_metadata(path, metadata)
    return metadata


def initial_metadata(
    *,
    run_id: str,
    config: dict[str, Any],
    run_dir: Path,
    config_path: Path,
    log_path: Path,
    metrics_path: Path,
    summary_path: Path,
    command: str,
) -> Metadata:
    checkpoint_policy = config.get("checkpoint_policy", {})
    checkpoint_dir = run_dir / str(checkpoint_policy.get("directory", "checkpoints"))
    return {
        "run_id": run_id,
        "project_name": config.get("project_name"),
        "method_name": config.get("method_name"),
        "experiment_name": config.get("experiment_name"),
        "branch": git_branch(),
        "commit": git_commit(short=False),
        "short_commit": git_commit(short=True),
        "command": command,
        "working_directory": str(REPO_ROOT),
        "hostname": socket.gethostname(),
        "gpu_id": config.get("gpu_id"),
        "device": config.get("device"),
        "start_time": utc_now(),
        "end_time": None,
        "status": "running",
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "resolved_config_path": str(config_path),
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "checkpoint_path": str(checkpoint_dir) if checkpoint_policy.get("save") else None,
        "checkpoint_dir": str(checkpoint_dir),
        "error_message": None,
        "postprocess_errors": [],
        "screen_session": None,
        "environment": environment_snapshot(),
    }
