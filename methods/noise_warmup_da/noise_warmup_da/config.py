from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    dataset: str = "officehome"
    data_root: Path = Path("data")
    run_dir: Path = Path("outputs/runs/noise_warmup_da/manual")
    source_domain: str = "Art"
    target_domain: str = "Clipart"
    image_size: int = 224
    source_val_fraction: float = 0.2
    fake_train_size: int = 64
    fake_eval_size: int = 32
    num_classes: int = 65
    model: str = "resnet18"
    pretrained: bool = True
    seed: int = 7
    batch_size: int = 64
    num_workers: int = 8
    source_epochs: int = 20
    warmup_epochs: int = 5
    linear_probe_epochs: int = 10
    lr: float = 1e-4
    warmup_lr: float = 1e-4
    probe_lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"
    amp: bool = True
    ece_bins: int = 15
    save_checkpoints: bool = False


def from_replay_config(replay_config: dict[str, Any], run_dir: Path) -> ExperimentConfig:
    values = dict(replay_config.get("noise_warmup_da", {}))
    if "seed" not in values and replay_config.get("seed") is not None:
        values["seed"] = replay_config["seed"]
    if "device" not in values and replay_config.get("device") is not None:
        values["device"] = replay_config["device"]
    checkpoint_policy = replay_config.get("checkpoint_policy", {})
    if not checkpoint_policy.get("save", False):
        values["save_checkpoints"] = False
    values["run_dir"] = run_dir

    path_keys = {"data_root", "run_dir"}
    for key in path_keys & values.keys():
        values[key] = Path(values[key])

    known = set(ExperimentConfig.__dataclass_fields__)
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown noise_warmup_da config keys: {unknown}")
    return ExperimentConfig(**values)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    values = asdict(config)
    for key in {"data_root", "run_dir"}:
        values[key] = str(values[key])
    return values
