from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .paths import REPO_ROOT, repo_path


Config = dict[str, Any]


def load_yaml(path: str | Path) -> Config:
    path = repo_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a mapping: {path}")
    return data


def write_yaml(data: Config, path: str | Path) -> None:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def deep_merge(base: Config, override: Config) -> Config:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError(f"Override must use key=value syntax: {raw}")
    key, value = raw.split("=", 1)
    key_parts = [part for part in key.split(".") if part]
    if not key_parts:
        raise ValueError(f"Override key is empty: {raw}")
    return key_parts, yaml.safe_load(value)


def set_nested(config: Config, key_parts: list[str], value: Any) -> None:
    cursor: Config = config
    for part in key_parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[key_parts[-1]] = value


def compose_config(
    method: str,
    experiment: str,
    overrides: list[str] | None = None,
    output_root: str | None = None,
) -> Config:
    base = load_yaml("configs/base.yaml")
    method_default_path = REPO_ROOT / "configs" / "methods" / method / "default.yaml"
    experiment_path = REPO_ROOT / "configs" / "methods" / method / f"{experiment}.yaml"

    config = deep_merge(base, load_yaml(method_default_path))
    config = deep_merge(config, load_yaml(experiment_path))
    config["method_name"] = method
    config["experiment_name"] = experiment
    if output_root:
        config["output_root"] = output_root

    for raw_override in overrides or []:
        key_parts, value = parse_override(raw_override)
        set_nested(config, key_parts, value)

    return config
