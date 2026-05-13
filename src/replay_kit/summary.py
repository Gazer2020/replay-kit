from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .metadata import read_metadata, update_metadata
from .paths import repo_path


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
    if not metrics:
        return "- metrics: 未生成"
    if isinstance(metrics.get("aggregate"), dict):
        return format_aggregate_metrics(metrics)
    if isinstance(metrics.get("arms"), dict):
        return format_multi_arm_metrics(metrics)
    rows = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            rows.append(f"- {key}: {value:.6g}")
        else:
            rows.append(f"- {key}: {value}")
    return "\n".join(rows)


def format_aggregate_metrics(metrics: dict[str, Any]) -> str:
    rows = [
        f"- dataset: {metrics.get('dataset')}",
        f"- domains: {metrics.get('domains')}",
        f"- seeds: {metrics.get('seeds')}",
        f"- model: {metrics.get('model')}",
        f"- epochs: {metrics.get('epochs')}",
        f"- warmup_epochs: {metrics.get('warmup_epochs')}",
        "",
        "| domain | arm | n | acc | nll | ece | final_train_loss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, arms in metrics["aggregate"].items():
        for arm, summary in arms.items():
            metric_summary = summary.get("metrics", {})
            rows.append(
                "| "
                f"{domain} | "
                f"{arm} | "
                f"{summary.get('n', '')} | "
                f"{_fmt_mean_std(metric_summary.get('accuracy'))} | "
                f"{_fmt_mean_std(metric_summary.get('nll'))} | "
                f"{_fmt_mean_std(metric_summary.get('ece'))} | "
                f"{_fmt_mean_std(metric_summary.get('final_train_loss'))} |"
            )
    return "\n".join(rows)


def _fmt_mean_std(summary: Any) -> str:
    if not isinstance(summary, dict):
        return ""
    mean_value = summary.get("mean")
    std_value = summary.get("std")
    if mean_value is None:
        return ""
    if isinstance(mean_value, float) and isinstance(std_value, float):
        return f"{mean_value:.4g} +/- {std_value:.3g}"
    return str(mean_value)


def format_multi_arm_metrics(metrics: dict[str, Any]) -> str:
    rows = []
    scalar_keys = [
        "dataset",
        "source_domain",
        "target_domain",
        "model",
        "baseline_target_accuracy",
        "noise_all_target_delta",
        "noise_head_target_delta",
        "noise_all_probe_target_delta",
        "hypothesis_supported",
    ]
    for key in scalar_keys:
        if key not in metrics:
            continue
        value = metrics[key]
        rows.append(f"- {key}: {value:.6g}" if isinstance(value, float) else f"- {key}: {value}")

    rows.append("")
    rows.append("| arm | source_acc | target_acc | target_nll | target_ece | probe_target_acc |")
    rows.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for arm, arm_metrics in metrics["arms"].items():
        source_eval = arm_metrics.get("source_eval", {})
        target_eval = arm_metrics.get("target_eval", {})
        probe_target = arm_metrics.get("linear_probe", {}).get("target_eval", {})
        rows.append(
            "| "
            f"{arm} | "
            f"{_fmt_metric(source_eval.get('accuracy'))} | "
            f"{_fmt_metric(target_eval.get('accuracy'))} | "
            f"{_fmt_metric(target_eval.get('nll'))} | "
            f"{_fmt_metric(target_eval.get('ece'))} | "
            f"{_fmt_metric(probe_target.get('accuracy'))} |"
        )
    return "\n".join(rows)


def _fmt_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


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
    reached_target = metrics.get("reached_target")
    conclusion = "实验失败，需先处理错误。" if status == "failed" else "实验完成。"
    if reached_target is True:
        conclusion = "实验完成，并达到 toy target loss。"
    elif reached_target is False and status == "finished":
        conclusion = "实验完成，但未达到 toy target loss。"
    elif metrics.get("hypothesis_supported") is True and status == "finished":
        conclusion = "实验完成，当前单次结果支持配置中的复现假设。"
    elif metrics.get("hypothesis_supported") is False and status == "finished":
        conclusion = "实验完成，当前单次结果未支持配置中的复现假设。"

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
