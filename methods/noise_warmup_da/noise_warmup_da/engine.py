from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from noise_warmup_da.config import ExperimentConfig, config_to_dict
from noise_warmup_da.data import DomainData, make_domain_data
from noise_warmup_da.metrics import evaluate
from noise_warmup_da.models import make_classifier, trainable_parameters


@dataclass(frozen=True)
class ArmSpec:
    name: str
    pretrained: bool
    noise_warmup: bool


ARMS = (
    ArmSpec("random_init_train", pretrained=False, noise_warmup=False),
    ArmSpec("random_init_noise_train", pretrained=False, noise_warmup=True),
    ArmSpec("pretrained_train", pretrained=True, noise_warmup=False),
    ArmSpec("pretrained_noise_train", pretrained=True, noise_warmup=True),
)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    device = resolve_device(config.device)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    arms = selected_arms(config)

    histories: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "method": "noise_warmup_da",
        "protocol": "officehome_domain_random_vs_pretrained_noise_warmup",
        "device": str(device),
        "requested_device": config.device,
        "dataset": config.dataset,
        "domains": list(config.domains or []),
        "seeds": list(config.seeds or []),
        "model": config.model,
        "epochs": config.epochs,
        "min_epochs": config.min_epochs,
        "convergence_patience": config.convergence_patience,
        "convergence_min_delta": config.convergence_min_delta,
        "target_train_loss": config.target_train_loss,
        "warmup_epochs": config.warmup_epochs,
        "arms": [arm.name for arm in arms],
        "config": config_to_dict(config),
        "results": results,
        "aggregate": {},
    }

    print(
        "[noise_warmup_da] "
        f"dataset={config.dataset} domains={metrics['domains']} seeds={metrics['seeds']} "
        f"model={config.model} requested_device={config.device} actual_device={device}",
        flush=True,
    )

    for domain in config.domains or []:
        for seed in config.seeds or []:
            set_seed(int(seed))
            data = make_domain_data(config, domain, int(seed))
            print(
                f"[domain={domain} seed={seed}] "
                f"train_size={data.train_size} test_size={data.test_size} classes={data.num_classes}",
                flush=True,
            )
            for arm in arms:
                result = run_arm(
                    arm=arm,
                    data=data,
                    config=config,
                    device=device,
                    seed=int(seed),
                    histories=histories,
                )
                results.append(result)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    metrics["aggregate"] = aggregate_results(results)
    _write_outputs(config, histories, metrics)
    return metrics


def selected_arms(config: ExperimentConfig) -> tuple[ArmSpec, ...]:
    if not config.run_arms:
        return ARMS
    by_name = {arm.name: arm for arm in ARMS}
    unknown = sorted(set(config.run_arms) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown run_arms entries: {unknown}")
    return tuple(by_name[name] for name in config.run_arms)


def run_arm(
    arm: ArmSpec,
    data: DomainData,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
    histories: list[dict[str, Any]],
) -> dict[str, Any]:
    set_seed(seed)
    uses_pretrained_weights = arm.pretrained and config.pretrained_weights
    model = make_classifier(config.model, data.num_classes, pretrained=uses_pretrained_weights).to(device)

    print(
        f"[domain={data.domain} seed={seed} arm={arm.name}] "
        f"start pretrained={arm.pretrained} weights={uses_pretrained_weights} "
        f"noise_warmup={arm.noise_warmup}",
        flush=True,
    )
    if arm.noise_warmup:
        _train_noise_model(model, data, config, device, seed, arm.name, histories)

    train_result = _train_supervised_model(model, data, config, device, seed, arm.name, histories)
    test_metrics = evaluate(model, data.test_loader, device, config.ece_bins).to_dict()
    result = {
        "domain": data.domain,
        "seed": seed,
        "arm": arm.name,
        "pretrained": arm.pretrained,
        "pretrained_weights": uses_pretrained_weights,
        "noise_warmup": arm.noise_warmup,
        "train_size": data.train_size,
        "test_size": data.test_size,
        "final_train_loss": train_result["final_train_loss"],
        "best_train_loss": train_result["best_train_loss"],
        "stopped_epoch": train_result["stopped_epoch"],
        "converged": train_result["converged"],
        "convergence_reason": train_result["convergence_reason"],
        "test": test_metrics,
    }
    print(
        f"[domain={data.domain} seed={seed} arm={arm.name}] "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"nll={test_metrics['nll']:.4f} ece={test_metrics['ece']:.4f}",
        flush=True,
    )
    _save_checkpoint(config, data.domain, seed, arm.name, model)
    del model
    return result


def _train_supervised_model(
    model: nn.Module,
    data: DomainData,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
    arm: str,
    histories: list[dict[str, Any]],
) -> dict[str, Any]:
    optimizer = AdamW(trainable_parameters(model), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    losses: list[float] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    convergence_reason = "max_epochs"
    for epoch in range(1, config.epochs + 1):
        loss = train_supervised_epoch(model, data.train_loader, optimizer, scaler, device, config.amp)
        losses.append(loss)
        improved = best_loss - loss > config.convergence_min_delta
        if improved:
            best_loss = loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        histories.append(
            {
                "domain": data.domain,
                "seed": seed,
                "arm": arm,
                "phase": "train",
                "epoch": epoch,
                "loss": loss,
                "best_loss": best_loss,
                "stale_epochs": stale_epochs,
            }
        )
        print(
            f"[domain={data.domain} seed={seed} arm={arm}] "
            f"train epoch={epoch}/{config.epochs} loss={loss:.4f} "
            f"best={best_loss:.4f} stale={stale_epochs}",
            flush=True,
        )
        can_stop = epoch >= max(1, config.min_epochs)
        if can_stop and config.target_train_loss is not None and loss <= config.target_train_loss:
            convergence_reason = f"target_train_loss<={config.target_train_loss}"
            break
        if can_stop and config.convergence_patience > 0 and stale_epochs >= config.convergence_patience:
            convergence_reason = (
                f"plateau_patience={config.convergence_patience},"
                f"min_delta={config.convergence_min_delta}"
            )
            break
    final_loss = losses[-1] if losses else None
    stopped_epoch = len(losses)
    converged = convergence_reason != "max_epochs"
    print(
        f"[domain={data.domain} seed={seed} arm={arm}] "
        f"train_stop epoch={stopped_epoch}/{config.epochs} reason={convergence_reason} "
        f"final_loss={final_loss:.4f} best_loss={best_loss:.4f} best_epoch={best_epoch}",
        flush=True,
    )
    return {
        "final_train_loss": final_loss,
        "best_train_loss": best_loss,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "converged": converged,
        "convergence_reason": convergence_reason,
    }


def _train_noise_model(
    model: nn.Module,
    data: DomainData,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
    arm: str,
    histories: list[dict[str, Any]],
) -> None:
    optimizer = AdamW(trainable_parameters(model), lr=config.warmup_lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    steps = config.noise_steps_per_epoch or len(data.train_loader)
    for epoch in range(1, config.warmup_epochs + 1):
        loss = train_noise_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=config.amp,
            num_steps=steps,
            batch_size=config.batch_size,
            input_shape=data.input_shape,
            num_classes=data.num_classes,
        )
        histories.append(
            {
                "domain": data.domain,
                "seed": seed,
                "arm": arm,
                "phase": "noise_warmup",
                "epoch": epoch,
                "loss": loss,
            }
        )
        print(
            f"[domain={data.domain} seed={seed} arm={arm}] "
            f"noise_warmup epoch={epoch}/{config.warmup_epochs} loss={loss:.4f}",
            flush=True,
        )


def train_supervised_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * inputs.size(0)
        total_seen += inputs.size(0)
    return total_loss / max(total_seen, 1)


def train_noise_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    num_steps: int,
    batch_size: int,
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0
    for _ in range(num_steps):
        inputs = torch.randn((batch_size, *input_shape), device=device)
        targets = torch.randint(num_classes, (batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch_size
        total_seen += batch_size
    return total_loss / max(total_seen, 1)


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    metric_keys = ["accuracy", "avg_confidence", "nll", "ece"]
    for result in results:
        domain = result["domain"]
        arm = result["arm"]
        aggregate.setdefault(domain, {}).setdefault(arm, {"n": 0, "metrics": {}})

    for domain, arms in aggregate.items():
        for arm, summary in arms.items():
            matching = [item for item in results if item["domain"] == domain and item["arm"] == arm]
            summary["n"] = len(matching)
            for key in metric_keys:
                values = [float(item["test"][key]) for item in matching]
                summary["metrics"][key] = {
                    "mean": mean(values),
                    "std": pstdev(values) if len(values) > 1 else 0.0,
                }
            for key in ["final_train_loss", "best_train_loss", "stopped_epoch"]:
                values = [float(item[key]) for item in matching if item.get(key) is not None]
                summary["metrics"][key] = {
                    "mean": mean(values) if values else None,
                    "std": pstdev(values) if len(values) > 1 else 0.0,
                }
            summary["converged_count"] = sum(1 for item in matching if item.get("converged"))
    return aggregate


def _save_checkpoint(
    config: ExperimentConfig,
    domain: str,
    seed: int,
    arm: str,
    model: nn.Module,
) -> None:
    if not config.save_checkpoints:
        return
    checkpoint_dir = config.run_dir / "checkpoints" / domain / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / f"{arm}.pt")


def _write_outputs(
    config: ExperimentConfig,
    histories: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    (config.run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    history_path = config.run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["domain", "seed", "arm", "phase", "epoch", "loss", "best_loss", "stale_epochs"],
        )
        writer.writeheader()
        writer.writerows(histories)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    device = (device or "auto").lower()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested, but CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device=mps was requested, but MPS is not available")
    return torch.device(device)
