from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", "replay_kit.runner", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )


def extract_run_dir(stdout: str) -> Path:
    first_line = stdout.splitlines()[0]
    return Path(first_line)


@pytest.mark.skipif(shutil.which("screen") is None, reason="screen is not installed")
def test_screen_launch_debug_generates_artifacts(tmp_path: Path) -> None:
    result = run_cli(
        [
            "launch",
            "--method",
            "toy_torch",
            "--experiment",
            "debug",
            "--output-root",
            str(tmp_path),
            "--wait",
            "--timeout",
            "120",
        ],
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run_dir = extract_run_dir(result.stdout)
    for name in ["metadata.json", "config.yaml", "train.log", "metrics.json", "summary.md", "notification_payload.json"]:
        assert (run_dir / name).exists(), name
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "finished"
    assert metadata["device"] in {"cuda", "mps", "cpu"}


@pytest.mark.skipif(shutil.which("screen") is None, reason="screen is not installed")
def test_screen_launch_failure_generates_failure_summary(tmp_path: Path) -> None:
    result = run_cli(
        [
            "launch",
            "--method",
            "toy_torch",
            "--experiment",
            "fail",
            "--output-root",
            str(tmp_path),
            "--wait",
            "--timeout",
            "120",
        ],
        tmp_path,
    )
    assert result.returncode == 1
    run_dir = extract_run_dir(result.stdout)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "notification_payload.json").exists()
    log_text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    assert "force_fail=true" in log_text
