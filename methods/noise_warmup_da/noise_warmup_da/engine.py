from __future__ import annotations

import csv
import json
import random
from copy import deepcopy
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
from noise_warmup_da.models import (
    FrozenBackboneLinearProbe,
    make_classifier,
    set_backbone_trainable,
    trainable_parameters,
)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    data = make_domain_data(config)

    histories: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "method": "noise_warmup_da",
        "device": str(device),
        "requested_device": config.device,
        "dataset": config.dataset,
        "source_domain": config.source_domain,
        "target_domain": config.target_domain,
        "model": config.model,
        "seed": config.seed,
        "source_epochs": config.source_epochs,
        "warmup_epochs": config.warmup_epochs,
        "linear_probe_epochs": config.linear_probe_epochs,
        "config": config_to_dict(config),
        "arms": {},
    }

    print(
        "[noise_warmup_da] "
        f"dataset={config.dataset} source={config.source_domain} target={config.target_domain} "
        f"model={config.model} requested_device={config.device} actual_device={device}",
        flush=True,
    )

    source_model = make_classifier(config.model, data.num_classes, config.pretrained).to(device)
    _train_supervised_model(
        model=source_model,
        loader=data.source_train,
        config=config,
        device=device,
        phase="source_pretrain",
        arm="source_pretrained",
        histories=histories,
    )
    source_state = deepcopy(source_model.state_dict())
    _finish_arm("source_pretrained", source_model, data, config, device, metrics)
    _save_checkpoint(config, "source_pretrained", source_model)

    noise_all_model = make_classifier(config.model, data.num_classes, pretrained=False).to(device)
    noise_all_model.load_state_dict(source_state)
    _train_noise_model(
        model=noise_all_model,
        data=data,
        config=config,
        device=device,
        arm="pretrained_noise_all",
        histories=histories,
    )
    _finish_arm("pretrained_noise_all", noise_all_model, data, config, device, metrics)
    _save_checkpoint(config, "pretrained_noise_all", noise_all_model)

    noise_head_model = make_classifier(config.model, data.num_classes, pretrained=False).to(device)
    noise_head_model.load_state_dict(source_state)
    set_backbone_trainable(noise_head_model, trainable=False)
    _train_noise_model(
        model=noise_head_model,
        data=data,
        config=config,
        device=device,
        arm="pretrained_noise_head",
        histories=histories,
    )
    _finish_arm("pretrained_noise_head", noise_head_model, data, config, device, metrics)
    _save_checkpoint(config, "pretrained_noise_head", noise_head_model)

    random_model = make_classifier(config.model, data.num_classes, pretrained=False).to(device)
    _train_noise_model(
        model=random_model,
        data=data,
        config=config,
        device=device,
        arm="random_init_noise_before_source",
        histories=histories,
    )
    _train_supervised_model(
        model=random_model,
        loader=data.source_train,
        config=config,
        device=device,
        phase="source_train_after_noise",
        arm="random_init_noise_before_source",
        histories=histories,
    )
    _finish_arm("random_init_noise_before_source", random_model, data, config, device, metrics)
    _save_checkpoint(config, "random_init_noise_before_source", random_model)

    metrics.update(_diagnostics(metrics))
    _write_outputs(config, histories, metrics)
    return metrics


def _train_supervised_model(
    model: nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    phase: str,
    arm: str,
    histories: list[dict[str, Any]],
) -> None:
    optimizer = AdamW(trainable_parameters(model), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    for epoch in range(1, config.source_epochs + 1):
        loss = train_supervised_epoch(model, loader, optimizer, scaler, device, config.amp)
        histories.append({"arm": arm, "phase": phase, "epoch": epoch, "loss": loss})
        print(f"[{arm}] {phase} epoch={epoch}/{config.source_epochs} loss={loss:.4f}", flush=True)


def _train_noise_model(
    model: nn.Module,
    data: DomainData,
    config: ExperimentConfig,
    device: torch.device,
    arm: str,
    histories: list[dict[str, Any]],
) -> None:
    optimizer = AdamW(trainable_parameters(model), lr=config.warmup_lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    freeze_backbone_stats = arm == "pretrained_noise_head"
    for epoch in range(1, config.warmup_epochs + 1):
        loss = train_noise_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=config.amp,
            num_steps=len(data.source_train),
            batch_size=config.batch_size,
            input_shape=data.input_shape,
            num_classes=data.num_classes,
            freeze_backbone_stats=freeze_backbone_stats,
        )
        histories.append({"arm": arm, "phase": "noise_warmup", "epoch": epoch, "loss": loss})
        print(f"[{arm}] noise_warmup epoch={epoch}/{config.warmup_epochs} loss={loss:.4f}", flush=True)


def _finish_arm(
    arm: str,
    model: nn.Module,
    data: DomainData,
    config: ExperimentConfig,
    device: torch.device,
    metrics: dict[str, Any],
) -> None:
    source_metrics = evaluate(model, data.source_eval, device, config.ece_bins)
    target_metrics = evaluate(model, data.target_eval, device, config.ece_bins)
    probe_metrics = run_linear_probe(model, data, config, device)
    metrics["arms"][arm] = {
        "source_eval": source_metrics.to_dict(),
        "target_eval": target_metrics.to_dict(),
        "linear_probe": probe_metrics,
    }
    print(
        f"[{arm}] source_acc={source_metrics.accuracy:.4f} "
        f"target_acc={target_metrics.accuracy:.4f} "
        f"probe_target_acc={probe_metrics['target_eval']['accuracy']:.4f}",
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
    freeze_backbone_stats: bool = False,
) -> float:
    model.train()
    if freeze_backbone_stats:
        _set_non_fc_modules_eval(model)
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


def _set_non_fc_modules_eval(model: nn.Module) -> None:
    for name, module in model.named_children():
        if name != "fc":
            module.eval()


def run_linear_probe(
    model: nn.Module,
    data: DomainData,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    rng_state = _capture_rng_state()
    try:
        set_seed(config.seed)
        probe = FrozenBackboneLinearProbe(deepcopy(model).cpu(), data.num_classes).to(device)
        optimizer = AdamW(probe.head.parameters(), lr=config.probe_lr, weight_decay=config.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
        for _ in range(config.linear_probe_epochs):
            train_supervised_epoch(probe, data.source_train, optimizer, scaler, device, config.amp)
        return {
            "source_eval": evaluate(probe, data.source_eval, device, config.ece_bins).to_dict(),
            "target_eval": evaluate(probe, data.target_eval, device, config.ece_bins).to_dict(),
        }
    finally:
        _restore_rng_state(rng_state)


def _diagnostics(metrics: dict[str, Any]) -> dict[str, float | bool]:
    arms = metrics["arms"]
    baseline = arms["source_pretrained"]["target_eval"]["accuracy"]
    noise_all = arms["pretrained_noise_all"]["target_eval"]["accuracy"]
    noise_head = arms["pretrained_noise_head"]["target_eval"]["accuracy"]
    probe_base = arms["source_pretrained"]["linear_probe"]["target_eval"]["accuracy"]
    probe_noise_all = arms["pretrained_noise_all"]["linear_probe"]["target_eval"]["accuracy"]
    return {
        "baseline_target_accuracy": baseline,
        "noise_all_target_accuracy": noise_all,
        "noise_head_target_accuracy": noise_head,
        "noise_all_target_delta": noise_all - baseline,
        "noise_head_target_delta": noise_head - baseline,
        "noise_all_probe_target_delta": probe_noise_all - probe_base,
        "hypothesis_supported": noise_all <= baseline and probe_noise_all <= probe_base,
    }


def _save_checkpoint(config: ExperimentConfig, name: str, model: nn.Module) -> None:
    if not config.save_checkpoints:
        return
    checkpoint_dir = config.run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / f"{name}.pt")


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
        writer = csv.DictWriter(handle, fieldnames=["arm", "phase", "epoch", "loss"])
        writer.writeheader()
        writer.writerows(histories)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


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
