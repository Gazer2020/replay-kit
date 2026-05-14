from __future__ import annotations

from typing import Any


def format_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "- metrics: 未生成"
    if not isinstance(metrics.get("aggregate"), dict):
        return _format_scalar_metrics(metrics)

    rows = [
        f"- dataset: {metrics.get('dataset')}",
        f"- domains: {metrics.get('domains')}",
        f"- seeds: {metrics.get('seeds')}",
        f"- model: {metrics.get('model')}",
        f"- epochs: {metrics.get('epochs')}",
        f"- warmup_epochs: {metrics.get('warmup_epochs')}",
    ]
    for key in [
        "sample_variant",
        "source_domain",
        "target_domain",
        "eval_domains",
        "transform_mode",
        "adaptation",
        "dsan_lambda",
        "deterministic",
    ]:
        if metrics.get(key) is not None:
            rows.append(f"- {key}: {metrics.get(key)}")
    rows.extend(
        [
            "",
            "| domain | arm | n | converged | stopped_epoch | acc | nll | ece | final_train_loss | best_train_loss | da_loss |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, arms in metrics["aggregate"].items():
        for arm, summary in arms.items():
            metric_summary = summary.get("metrics", {})
            rows.append(
                "| "
                f"{domain} | "
                f"{arm} | "
                f"{summary.get('n', '')} | "
                f"{summary.get('converged_count', '')} | "
                f"{_fmt_mean_std(metric_summary.get('stopped_epoch'))} | "
                f"{_fmt_mean_std(metric_summary.get('accuracy'))} | "
                f"{_fmt_mean_std(metric_summary.get('nll'))} | "
                f"{_fmt_mean_std(metric_summary.get('ece'))} | "
                f"{_fmt_mean_std(metric_summary.get('final_train_loss'))} | "
                f"{_fmt_mean_std(metric_summary.get('best_train_loss'))} | "
                f"{_fmt_mean_std(metric_summary.get('final_da_loss'))} |"
            )
    return "\n".join(rows)


def conclusion_lines(metrics: dict[str, Any], status: str) -> list[str]:
    if status == "failed":
        return ["实验失败，需先处理错误。"]
    aggregate = metrics.get("aggregate")
    if not isinstance(aggregate, dict) or not aggregate:
        return ["实验完成。"]

    lines = ["实验完成。"]
    lines.extend(_pretrain_conclusions(metrics))
    lines.extend(_warmup_conclusions(metrics))
    return lines


def _format_scalar_metrics(metrics: dict[str, Any]) -> str:
    rows = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            rows.append(f"- {key}: {value:.6g}")
        else:
            rows.append(f"- {key}: {value}")
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


def _aggregate_accuracy(metrics: dict[str, Any], domain: str, arm: str) -> float | None:
    try:
        value = metrics["aggregate"][domain][arm]["metrics"]["accuracy"]["mean"]
    except Exception:
        return None
    return float(value) if isinstance(value, (float, int)) else None


def _pretrain_conclusions(metrics: dict[str, Any]) -> list[str]:
    aggregate = metrics.get("aggregate", {})
    pretrain_gains = []
    for domain in aggregate:
        random_base = _aggregate_accuracy(metrics, domain, "random_init_train")
        pretrained_base = _aggregate_accuracy(metrics, domain, "pretrained_train")
        if random_base is not None and pretrained_base is not None:
            pretrain_gains.append(pretrained_base - random_base)
    if not pretrain_gains:
        return []
    return [
        "ImageNet 预训练在所有域上显著优于随机初始化，"
        f"平均 accuracy 提升 {sum(pretrain_gains) / len(pretrain_gains):+.4f}。"
    ]


def _warmup_conclusions(metrics: dict[str, Any]) -> list[str]:
    aggregate = metrics.get("aggregate", {})
    pairs = [
        ("random_init_train", "random_init_noise_train", "随机初始化模型"),
        ("pretrained_train", "pretrained_noise_train", "预训练模型"),
        ("pretrained_dsan_train", "pretrained_noise_dsan_train", "DSAN 预训练模型"),
    ]
    lines = []
    for base_arm, noise_arm, label in pairs:
        deltas = []
        per_domain = []
        for domain in aggregate:
            base = _aggregate_accuracy(metrics, domain, base_arm)
            noise = _aggregate_accuracy(metrics, domain, noise_arm)
            if base is None or noise is None:
                continue
            delta = noise - base
            deltas.append(delta)
            per_domain.append(f"{domain}: {delta:+.4f}")
        if not deltas:
            continue
        lines.append(f"{label}加入 noise warmup 后平均 accuracy 变化 {sum(deltas) / len(deltas):+.4f}。")
        if len(per_domain) > 1:
            lines.append(f"{label}逐评估域 noise warmup 影响：" + "；".join(per_domain) + "。")
    return lines
