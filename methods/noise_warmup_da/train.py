from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

# Set before torch is imported through the method package so deterministic CUDA
# runs can use reproducible CuBLAS kernels when requested by config.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from noise_warmup_da.config import from_replay_config
from noise_warmup_da.engine import run_experiment


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("config.yaml must contain a mapping")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    replay_config = load_config(Path(args.config))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = from_replay_config(replay_config, run_dir)
    run_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
