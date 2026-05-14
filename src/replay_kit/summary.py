from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from .metadata import read_metadata, update_metadata
from .paths import REPO_ROOT, repo_path


def read_json_if_exists(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml_if_exists(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def tail_text(path: str | Path, lines: int = 80) -> str:
    path = repo_path(path)
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def format_metrics(metrics: dict[str, Any]) -> str:
    formatter = method_hook(metrics.get("method"), "format_metrics")
    if formatter:
        return str(formatter(metrics))
    if not metrics:
        return "- metrics: 未生成"
    rows = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            rows.append(f"- {key}: {value:.6g}")
        else:
            rows.append(f"- {key}: {value}")
    return "\n".join(rows)


_METHOD_MODULE_CACHE: dict[str, ModuleType | None] = {}


def load_method_summary_module(method_name: Any) -> ModuleType | None:
    if not isinstance(method_name, str) or not method_name:
        return None
    if method_name in _METHOD_MODULE_CACHE:
        return _METHOD_MODULE_CACHE[method_name]
    module_path = REPO_ROOT / "methods" / method_name / "summary.py"
    if not module_path.exists():
        _METHOD_MODULE_CACHE[method_name] = None
        return None
    spec = importlib.util.spec_from_file_location(f"replay_kit_method_summary_{method_name}", module_path)
    if spec is None or spec.loader is None:
        _METHOD_MODULE_CACHE[method_name] = None
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _METHOD_MODULE_CACHE[method_name] = module
    return module


def method_hook(method_name: Any, hook_name: str):
    module = load_method_summary_module(method_name)
    if module is None:
        return None
    hook = getattr(module, hook_name, None)
    return hook if callable(hook) else None


def conclusion_lines(metrics: dict[str, Any], status: str) -> list[str]:
    if status == "failed":
        return ["实验失败，需先处理错误。"]
    hook = method_hook(metrics.get("method"), "conclusion_lines")
    if hook:
        return list(hook(metrics, status))
    reached_target = metrics.get("reached_target")
    if reached_target is True:
        return ["实验完成，并达到 toy target loss。"]
    if reached_target is False:
        return ["实验完成，但未达到 toy target loss。"]
    if metrics.get("hypothesis_supported") is True:
        return ["实验完成，当前单次结果支持配置中的复现假设。"]
    if metrics.get("hypothesis_supported") is False:
        return ["实验完成，当前单次结果未支持配置中的复现假设。"]
    return ["实验完成。"]


def conclusion_text(metrics: dict[str, Any], status: str) -> str:
    return "\n".join(f"- {line}" for line in conclusion_lines(metrics, status))


def reproduction_goal(config: dict[str, Any]) -> str:
    goal = config.get("reproduction_goal")
    if goal:
        return str(goal).strip()
    return "验证轻量复现仓库的运行、记录、总结和通知链路。"


def generate_summary(run_dir: str | Path) -> Path:
    run_dir = repo_path(run_dir)
    metadata_path = run_dir / "metadata.json"
    metadata = read_metadata(metadata_path)
    config = read_yaml_if_exists(metadata.get("resolved_config_path", run_dir / "config.yaml"))
    metrics = read_json_if_exists(metadata.get("metrics_path", run_dir / "metrics.json"))
    log_tail = tail_text(
        metadata.get("log_path", run_dir / "train.log"),
        int(config.get("summary", {}).get("log_tail_lines", 80)),
    )

    status = metadata.get("status", "unknown")
    summary_path = repo_path(metadata.get("summary_path", run_dir / "summary.md"))
    conclusion = conclusion_text(metrics, status)

    error_message = metadata.get("error_message")
    if not error_message and status == "failed":
        error_message = "错误详情见 train.log。"

    body = [
        "# 实验总结",
        "",
        "## 基本信息",
        f"- 方法：{metadata.get('method_name')}",
        f"- 实验名：{metadata.get('experiment_name')}",
        f"- run_id：{metadata.get('run_id')}",
        f"- status：{status}",
        f"- branch：{metadata.get('branch')}",
        f"- commit：{metadata.get('commit')}",
        f"- 配置：{metadata.get('resolved_config_path')}",
        f"- 日志：{metadata.get('log_path')}",
        "",
        "## 复现目标",
        reproduction_goal(config),
        "",
        "## 实验设置",
        f"- project：{metadata.get('project_name')}",
        f"- requested_device：{config.get('device')}",
        f"- actual_device：{metrics.get('device', metadata.get('device'))}",
        f"- seed：{config.get('seed')}",
        f"- Python：{metadata.get('environment', {}).get('python_version')}",
        f"- Torch：{metadata.get('environment', {}).get('torch_version')}",
        "",
        "## 结果",
        format_metrics(metrics),
        "",
        "## 结论",
        conclusion,
        "",
        "## 问题与备注",
        f"- error_message：{error_message or '无'}",
        "",
        "## 日志尾部",
        "```text",
        log_tail or "train.log 为空或不存在。",
        "```",
        "",
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(body), encoding="utf-8")
    update_metadata(metadata_path, summary_path=str(summary_path))
    return summary_path
