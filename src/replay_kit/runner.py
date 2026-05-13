from __future__ import annotations

import argparse
import errno
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import compose_config, write_yaml
from .metadata import (
    environment_snapshot,
    git_commit,
    initial_metadata,
    read_metadata,
    update_metadata,
    utc_now,
    write_metadata,
)
from .notifier import notify
from .paths import REPO_ROOT, SRC_ROOT, repo_path
from .summary import generate_summary


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_commit = safe_name(git_commit(short=True))
    suffix = f"{os.getpid():x}"
    return f"{timestamp}_{short_commit}_{suffix}"


def shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def build_screen_command(run_dir: Path, log_path: Path) -> str:
    python = shlex.quote(sys.executable)
    repo = shlex.quote(str(REPO_ROOT))
    src = shlex.quote(str(SRC_ROOT))
    run_dir_q = shlex.quote(str(run_dir))
    log_q = shlex.quote(str(log_path))
    return (
        f"cd {repo} && "
        f"export PYTHONPATH={src}:\"${{PYTHONPATH:-}}\" && "
        f"export REPLAY_KIT_RUN_DIR={run_dir_q} && "
        f"{python} -u -m replay_kit.runner worker --run-dir {run_dir_q} "
        f">> {log_q} 2>&1"
    )


def launch(args: argparse.Namespace) -> int:
    screen_bin = shutil.which("screen")
    if not screen_bin:
        raise RuntimeError("screen is required but was not found on PATH")

    config = compose_config(
        method=args.method,
        experiment=args.experiment,
        overrides=args.overrides,
        output_root=args.output_root,
    )
    run_id = make_run_id()
    output_root = repo_path(config.get("output_root", "outputs/runs"))
    run_dir = output_root / config["method_name"] / config["experiment_name"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config_path = run_dir / "config.yaml"
    log_path = run_dir / config.get("log_policy", {}).get("train_log_name", "train.log")
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.md"
    write_yaml(config, config_path)
    log_path.touch()

    launch_command = shell_join([sys.executable, "-m", "replay_kit.runner", *sys.argv[1:]])
    metadata = initial_metadata(
        run_id=run_id,
        config=config,
        run_dir=run_dir,
        config_path=config_path,
        log_path=log_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        command=launch_command,
    )
    session = safe_name(f"rk_{config['method_name']}_{config['experiment_name']}_{run_id}")[:80]
    metadata["screen_session"] = session
    metadata_path = run_dir / "metadata.json"
    write_metadata(metadata_path, metadata)

    worker_command = build_screen_command(run_dir, log_path)
    update_metadata(metadata_path, worker_command=worker_command)
    subprocess.run(
        [screen_bin, "-dmS", session, "/bin/bash", "-lc", worker_command],
        cwd=REPO_ROOT,
        check=True,
    )
    print(str(run_dir))
    print(f"screen session: {session}")

    if args.wait:
        status = wait_for_completion(metadata_path, args.timeout)
        print(f"final status: {status}")
        return 0 if status == "finished" else 1
    return 0


def wait_for_completion(metadata_path: Path, timeout: int) -> str:
    deadline = time.time() + timeout
    last_status = "unknown"
    while time.time() < deadline:
        metadata = read_metadata(metadata_path)
        last_status = metadata.get("status", "unknown")
        if last_status in {"finished", "failed"}:
            return last_status
        time.sleep(0.5)
    return last_status


def run_training_command(config: dict[str, Any], run_dir: Path) -> None:
    command_config = config.get("command", {})
    entrypoint = command_config.get("entrypoint")
    if not entrypoint:
        raise ValueError("config.command.entrypoint is required")
    entrypoint_path = repo_path(entrypoint)
    if not entrypoint_path.exists():
        raise FileNotFoundError(f"Training entrypoint not found: {entrypoint_path}")

    config_path = run_dir / "config.yaml"
    cmd = [
        sys.executable,
        "-u",
        str(entrypoint_path),
        "--config",
        str(config_path),
        "--run-dir",
        str(run_dir),
    ]
    cmd.extend(str(item) for item in command_config.get("args", []))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["REPLAY_KIT_RUN_DIR"] = str(run_dir)
    gpu_id = config.get("gpu_id")
    requested_device = str(config.get("device", "auto")).lower()
    if gpu_id is not None and requested_device in {"auto", "cuda"}:
        env.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))

    print(f"[runner] training command: {shell_join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def load_config_snapshot(run_dir: Path) -> dict[str, Any]:
    import yaml

    with (run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("config.yaml must contain a mapping")
    return data


def read_metrics_device(run_dir: Path) -> str | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except Exception:
        return None
    value = metrics.get("device")
    return str(value) if value else None


def maybe_shutdown(config: dict[str, Any], metadata: dict[str, Any]) -> None:
    system_config = config.get("system", {})
    if not system_config.get("shutdown_on_finish", False):
        return
    command = system_config.get("shutdown_command", ["shutdown", "-h", "now"])
    if isinstance(command, str):
        command = shlex.split(command)
    if not isinstance(command, list) or not command:
        raise ValueError("system.shutdown_command must be a non-empty string or list")
    print(
        f"[runner] shutdown_on_finish enabled after status={metadata.get('status')}: "
        f"{shell_join([str(item) for item in command])}",
        flush=True,
    )
    try:
        subprocess.Popen([str(item) for item in command], cwd=REPO_ROOT)
    except OSError as exc:
        if exc.errno != errno.ENOEXEC:
            raise
        executable = shutil.which(str(command[0]))
        if not executable:
            raise
        fallback = ["/bin/bash", executable, *[str(item) for item in command[1:]]]
        print(
            f"[runner] shutdown command is a shell script without shebang; "
            f"retrying with {shell_join(fallback)}",
            flush=True,
        )
        subprocess.Popen(fallback, cwd=REPO_ROOT)


def worker(args: argparse.Namespace) -> int:
    run_dir = repo_path(args.run_dir)
    metadata_path = run_dir / "metadata.json"
    return_code = 0
    try:
        config = load_config_snapshot(run_dir)
        print(f"[runner] worker started at {utc_now()}", flush=True)
        update_metadata(
            metadata_path,
            worker_start_time=utc_now(),
            environment=environment_snapshot(),
        )
        run_training_command(config, run_dir)
        actual_device = read_metrics_device(run_dir)
        updates: dict[str, Any] = {
            "status": "finished",
            "end_time": utc_now(),
            "error_message": None,
            "environment": environment_snapshot(),
        }
        if actual_device:
            updates["device"] = actual_device
        update_metadata(metadata_path, **updates)
    except subprocess.CalledProcessError as exc:
        return_code = exc.returncode or 1
        print("[runner] training command failed", flush=True)
        traceback.print_exc()
        update_metadata(
            metadata_path,
            status="failed",
            end_time=utc_now(),
            error_message=f"Training command failed with exit code {return_code}",
            environment=environment_snapshot(),
        )
    except Exception as exc:
        return_code = 1
        print("[runner] worker failed", flush=True)
        traceback.print_exc()
        update_metadata(
            metadata_path,
            status="failed",
            end_time=utc_now(),
            error_message=str(exc),
            environment=environment_snapshot(),
        )
    finally:
        try:
            summary_path = generate_summary(run_dir)
            config = load_config_snapshot(run_dir)
            metadata = read_metadata(metadata_path)
            metadata["summary_path"] = str(summary_path)
            write_metadata(metadata_path, metadata)
            notify(run_dir, metadata, config.get("notify", {}))
            maybe_shutdown(config, metadata)
        except Exception:
            print("[runner] failed while generating summary or notification", flush=True)
            traceback.print_exc()
            return_code = return_code or 1
    print(f"[runner] worker finished with code {return_code}", flush=True)
    return return_code


def status(args: argparse.Namespace) -> int:
    metadata = read_metadata(Path(args.run_dir) / "metadata.json")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


def summarize(args: argparse.Namespace) -> int:
    summary_path = generate_summary(args.run_dir)
    print(str(summary_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="replay-kit")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    launch_parser = subparsers.add_parser("launch", help="Launch an experiment in screen")
    launch_parser.add_argument("--method", required=True)
    launch_parser.add_argument("--experiment", required=True)
    launch_parser.add_argument("--output-root", default=None)
    launch_parser.add_argument("--wait", action="store_true")
    launch_parser.add_argument("--timeout", type=int, default=120)
    launch_parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional dotted key=value config overrides, e.g. device=mps toy_torch.epochs=2",
    )
    launch_parser.set_defaults(func=launch)

    worker_parser = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--run-dir", required=True)
    worker_parser.set_defaults(func=worker)

    status_parser = subparsers.add_parser("status", help="Print run metadata")
    status_parser.add_argument("--run-dir", required=True)
    status_parser.set_defaults(func=status)

    summarize_parser = subparsers.add_parser("summarize", help="Regenerate summary.md")
    summarize_parser.add_argument("--run-dir", required=True)
    summarize_parser.set_defaults(func=summarize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
