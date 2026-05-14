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
    adaptation: str = "none"


ARMS = (
    ArmSpec("random_init_train", pretrained=False, noise_warmup=False),
    ArmSpec("random_init_noise_train", pretrained=False, noise_warmup=True),
    ArmSpec("pretrained_train", pretrained=True, noise_warmup=False),
    ArmSpec("pretrained_noise_train", pretrained=True, noise_warmup=True),
    ArmSpec("pretrained_dsan_train", pretrained=True, noise_warmup=False, adaptation="dsan"),
    ArmSpec("pretrained_noise_dsan_train", pretrained=True, noise_warmup=True, adaptation="dsan"),
)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    device = resolve_device(config.device)
    configure_determinism(config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    arms = selected_arms(config)

    histories: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "method": "noise_warmup_da",
        "protocol": _protocol_name(config),
        "device": str(device),
        "requested_device": config.device,
        "dataset": config.dataset,
        "domains": list(config.domains or []),
        "sample_variant": config.sample_variant,
        "source_domain": config.source_domain,
        "target_domain": config.target_domain,
        "eval_domains": list(config.eval_domains or []),
        "transform_mode": config.transform_mode,
        "random_horizontal_flip": config.random_horizontal_flip,
        "seeds": list(config.seeds or []),
        "model": config.model,
        "epochs": config.epochs,
        "min_epochs": config.min_epochs,
        "convergence_patience": config.convergence_patience,
        "convergence_min_delta": config.convergence_min_delta,
        "target_train_loss": config.target_train_loss,
        "warmup_epochs": config.warmup_epochs,
        "adaptation": sorted({arm.adaptation for arm in arms}),
        "dsan_lambda": config.dsan_lambda,
        "deterministic": config.deterministic,
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
            set_seed(int(seed), config.deterministic)
            data = make_domain_data(config, domain, int(seed))
            print(
                f"[domain={domain} seed={seed}] "
                f"train_size={data.train_size} eval_sizes={data.eval_sizes} "
                f"classes={data.num_classes} input_shape={data.input_shape}",
                flush=True,
            )
            for arm in arms:
                arm_results = run_arm(
                    arm=arm,
                    data=data,
                    config=config,
                    device=device,
                    seed=int(seed),
                    histories=histories,
                )
                results.extend(arm_results)
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


def _protocol_name(config: ExperimentConfig) -> str:
    if config.dataset == "sample_sar":
        return "sample_sar_synth_train_source_and_sourceonly_real_eval"
    return "officehome_domain_random_vs_pretrained_noise_warmup"


def run_arm(
    arm: ArmSpec,
    data: DomainData,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
    histories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    set_seed(seed, config.deterministic)
    uses_pretrained_weights = arm.pretrained and config.pretrained_weights
    model = make_classifier(config.model, data.num_classes, pretrained=uses_pretrained_weights).to(device)

    print(
        f"[domain={data.domain} seed={seed} arm={arm.name}] "
        f"start pretrained={arm.pretrained} weights={uses_pretrained_weights} "
        f"noise_warmup={arm.noise_warmup} adaptation={arm.adaptation}",
        flush=True,
    )
    if arm.noise_warmup:
        _train_noise_model(model, data, config, device, seed, arm.name, histories)

    if arm.adaptation == "dsan":
        train_result = _train_dsan_model(model, data, config, device, seed, arm.name, histories)
    else:
        train_result = _train_supervised_model(model, data, config, device, seed, arm.name, histories)
    results = []
    for eval_domain, loader in data.eval_loaders.items():
        test_metrics = evaluate(model, loader, device, config.ece_bins).to_dict()
        result = {
            "domain": eval_domain,
            "train_domain": data.domain,
            "eval_domain": eval_domain,
            "seed": seed,
            "arm": arm.name,
            "pretrained": arm.pretrained,
            "pretrained_weights": uses_pretrained_weights,
            "noise_warmup": arm.noise_warmup,
            "adaptation": arm.adaptation,
            "train_size": data.train_size,
            "test_size": data.eval_sizes[eval_domain],
            "final_train_loss": train_result["final_train_loss"],
            "best_train_loss": train_result["best_train_loss"],
            "final_da_loss": train_result.get("final_da_loss"),
            "best_da_loss": train_result.get("best_da_loss"),
            "stopped_epoch": train_result["stopped_epoch"],
            "converged": train_result["converged"],
            "convergence_reason": train_result["convergence_reason"],
            "test": test_metrics,
        }
        results.append(result)
        print(
            f"[domain={data.domain} eval={eval_domain} seed={seed} arm={arm.name}] "
            f"test_acc={test_metrics['accuracy']:.4f} "
            f"nll={test_metrics['nll']:.4f} ece={test_metrics['ece']:.4f}",
            flush=True,
        )
    _save_checkpoint(config, data.domain, seed, arm.name, model)
    del model
    return results


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


def _train_dsan_model(
    model: nn.Module,
    data: DomainData,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
    arm: str,
    histories: list[dict[str, Any]],
) -> dict[str, Any]:
    if data.target_train_loader is None:
        raise ValueError("DSAN adaptation requires target_train_loader")
    optimizer = AdamW(trainable_parameters(model), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    losses: list[float] = []
    da_losses: list[float] = []
    best_loss = float("inf")
    best_da_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    convergence_reason = "max_epochs"
    target_iterator = iter(data.target_train_loader)
    for epoch in range(1, config.epochs + 1):
        loss, da_loss, target_iterator = train_dsan_epoch(
            model=model,
            source_loader=data.train_loader,
            target_loader=data.target_train_loader,
            target_iterator=target_iterator,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=config.amp,
            num_classes=data.num_classes,
            dsan_lambda=config.dsan_lambda,
        )
        losses.append(loss)
        da_losses.append(da_loss)
        improved = best_loss - loss > config.convergence_min_delta
        if improved:
            best_loss = loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        best_da_loss = min(best_da_loss, da_loss)
        histories.append(
            {
                "domain": data.domain,
                "seed": seed,
                "arm": arm,
                "phase": "dsan_train",
                "epoch": epoch,
                "loss": loss,
                "best_loss": best_loss,
                "stale_epochs": stale_epochs,
                "da_loss": da_loss,
            }
        )
        print(
            f"[domain={data.domain} seed={seed} arm={arm}] "
            f"dsan epoch={epoch}/{config.epochs} loss={loss:.4f} "
            f"da_loss={da_loss:.4f} best={best_loss:.4f} stale={stale_epochs}",
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
    final_da_loss = da_losses[-1] if da_losses else None
    stopped_epoch = len(losses)
    converged = convergence_reason != "max_epochs"
    print(
        f"[domain={data.domain} seed={seed} arm={arm}] "
        f"dsan_stop epoch={stopped_epoch}/{config.epochs} reason={convergence_reason} "
        f"final_loss={final_loss:.4f} best_loss={best_loss:.4f} "
        f"final_da_loss={final_da_loss:.4f} best_epoch={best_epoch}",
        flush=True,
    )
    return {
        "final_train_loss": final_loss,
        "best_train_loss": best_loss,
        "best_epoch": best_epoch,
        "final_da_loss": final_da_loss,
        "best_da_loss": best_da_loss,
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


def train_dsan_epoch(
    model: nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    target_iterator,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    num_classes: int,
    dsan_lambda: float,
):
    model.train()
    total_loss = 0.0
    total_da_loss = 0.0
    total_seen = 0
    for source_inputs, source_targets in source_loader:
        try:
            target_inputs, _ = next(target_iterator)
        except StopIteration:
            target_iterator = iter(target_loader)
            target_inputs, _ = next(target_iterator)
        source_inputs = source_inputs.to(device, non_blocking=True)
        source_targets = source_targets.to(device, non_blocking=True)
        target_inputs = target_inputs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            source_logits, source_features = forward_logits_and_features(model, source_inputs)
            target_logits, target_features = forward_logits_and_features(model, target_inputs)
            classification_loss = F.cross_entropy(source_logits, source_targets)
            da_loss = label_aware_mmd(
                source_features=source_features,
                target_features=target_features,
                source_targets=source_targets,
                target_probabilities=target_logits.softmax(dim=1).detach(),
                num_classes=num_classes,
            )
            loss = classification_loss + dsan_lambda * da_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += classification_loss.item() * source_inputs.size(0)
        total_da_loss += da_loss.item() * source_inputs.size(0)
        total_seen += source_inputs.size(0)
    return (
        total_loss / max(total_seen, 1),
        total_da_loss / max(total_seen, 1),
        target_iterator,
    )


def forward_logits_and_features(model: nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    resnet_parts = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    if not all(hasattr(model, name) for name in resnet_parts):
        logits = model(inputs)
        return logits, logits
    features = model.conv1(inputs)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)
    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)
    features = model.avgpool(features)
    features = torch.flatten(features, 1)
    logits = model.fc(features)
    return logits, features


def label_aware_mmd(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_targets: torch.Tensor,
    target_probabilities: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    source_one_hot = F.one_hot(source_targets, num_classes=num_classes).to(source_features.dtype)
    losses = []
    for class_index in range(num_classes):
        source_weights = source_one_hot[:, class_index]
        target_weights = target_probabilities[:, class_index].to(target_features.dtype)
        if source_weights.sum() <= 0 or target_weights.sum() <= 1e-6:
            continue
        source_mean = (source_features * source_weights[:, None]).sum(dim=0) / source_weights.sum().clamp_min(1e-6)
        target_mean = (target_features * target_weights[:, None]).sum(dim=0) / target_weights.sum().clamp_min(1e-6)
        losses.append(F.mse_loss(source_mean, target_mean, reduction="mean"))
    if not losses:
        return source_features.new_tensor(0.0)
    return torch.stack(losses).mean()


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
            for key in ["final_train_loss", "best_train_loss", "final_da_loss", "best_da_loss", "stopped_epoch"]:
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
            fieldnames=["domain", "seed", "arm", "phase", "epoch", "loss", "best_loss", "stale_epochs", "da_loss"],
        )
        writer.writeheader()
        writer.writerows(histories)


def configure_determinism(config: ExperimentConfig) -> None:
    if not config.deterministic:
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False


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
