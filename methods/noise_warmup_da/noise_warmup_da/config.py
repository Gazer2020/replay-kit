from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    dataset: str = "officehome"
    data_root: Path = Path("data")
    run_dir: Path = Path("outputs/runs/noise_warmup_da/manual")
    domains: list[str] | None = None
    seeds: list[int] | None = None
    image_size: int = 224
    test_fraction: float = 0.2
    fake_train_size: int = 64
    fake_test_size: int = 32
    num_classes: int = 65
    model: str = "resnet50"
    pretrained_weights: bool = True
    run_arms: list[str] | None = None
    batch_size: int = 64
    num_workers: int = 8
    epochs: int = 20
    min_epochs: int = 0
    convergence_patience: int = 0
    convergence_min_delta: float = 1e-3
    target_train_loss: float | None = None
    warmup_epochs: int = 5
    noise_steps_per_epoch: int | None = None
    lr: float = 1e-4
    warmup_lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "auto"
    amp: bool = True
    ece_bins: int = 15
    save_checkpoints: bool = False

    def __post_init__(self) -> None:
        if self.domains is None:
            self.domains = ["Art", "Clipart", "Product", "Real World"]
        if self.seeds is None:
            self.seeds = [7, 13, 21]


def from_replay_config(replay_config: dict[str, Any], run_dir: Path) -> ExperimentConfig:
    values = dict(replay_config.get("noise_warmup_da", {}))
    if "device" not in values and replay_config.get("device") is not None:
        values["device"] = replay_config["device"]
    if "seeds" not in values and replay_config.get("seed") is not None:
        values["seeds"] = [int(replay_config["seed"])]
    checkpoint_policy = replay_config.get("checkpoint_policy", {})
    if not checkpoint_policy.get("save", False):
        values["save_checkpoints"] = False
    values["run_dir"] = run_dir

    for key in {"data_root", "run_dir"} & values.keys():
        values[key] = Path(values[key])
    for key in {"domains", "seeds"} & values.keys():
        if isinstance(values[key], tuple):
            values[key] = list(values[key])

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
