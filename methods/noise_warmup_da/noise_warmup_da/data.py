from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from noise_warmup_da.config import ExperimentConfig


OFFICEHOME_DOMAINS = ("Art", "Clipart", "Product", "Real World")


@dataclass(frozen=True)
class DomainData:
    source_train: DataLoader
    source_eval: DataLoader
    target_eval: DataLoader
    num_classes: int
    class_names: list[str]
    input_shape: tuple[int, int, int]


def make_domain_data(config: ExperimentConfig) -> DomainData:
    if config.dataset == "officehome":
        return _make_officehome(config)
    if config.dataset == "fake":
        return _make_fake(config)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def _make_officehome(config: ExperimentConfig) -> DomainData:
    root = config.data_root / "OfficeHome"
    source_dir = root / config.source_domain
    target_dir = root / config.target_domain
    _validate_officehome_path(root, source_dir, target_dir)

    train_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    full_source = datasets.ImageFolder(source_dir, transform=train_transform)
    source_eval_full = datasets.ImageFolder(source_dir, transform=eval_transform)
    target_eval = datasets.ImageFolder(target_dir, transform=eval_transform)
    if full_source.class_to_idx != target_eval.class_to_idx:
        raise ValueError(
            "Source and target OfficeHome class mappings differ. "
            "Ensure both domains contain the same class folders."
        )

    source_train, source_eval = _split_source(
        full_source=full_source,
        source_eval_full=source_eval_full,
        val_fraction=config.source_val_fraction,
        seed=config.seed,
    )
    return DomainData(
        source_train=_loader(source_train, config, shuffle=True),
        source_eval=_loader(source_eval, config, shuffle=False),
        target_eval=_loader(target_eval, config, shuffle=False),
        num_classes=len(full_source.classes),
        class_names=list(full_source.classes),
        input_shape=(3, config.image_size, config.image_size),
    )


def _make_fake(config: ExperimentConfig) -> DomainData:
    transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    source_train = datasets.FakeData(
        size=config.fake_train_size,
        image_size=(3, config.image_size, config.image_size),
        num_classes=config.num_classes,
        transform=transform,
        random_offset=0,
    )
    source_eval = datasets.FakeData(
        size=config.fake_eval_size,
        image_size=(3, config.image_size, config.image_size),
        num_classes=config.num_classes,
        transform=transform,
        random_offset=10_000,
    )
    target_eval = datasets.FakeData(
        size=config.fake_eval_size,
        image_size=(3, config.image_size, config.image_size),
        num_classes=config.num_classes,
        transform=transform,
        random_offset=20_000,
    )
    class_names = [str(index) for index in range(config.num_classes)]
    return DomainData(
        source_train=_loader(source_train, config, shuffle=True),
        source_eval=_loader(source_eval, config, shuffle=False),
        target_eval=_loader(target_eval, config, shuffle=False),
        num_classes=config.num_classes,
        class_names=class_names,
        input_shape=(3, config.image_size, config.image_size),
    )


def _validate_officehome_path(root, source_dir, target_dir) -> None:
    if not root.exists():
        raise FileNotFoundError(
            f"OfficeHome root not found: {root}. "
            "Expected data/OfficeHome/{Art,Clipart,Product,Real World}."
        )
    missing = [domain for domain in OFFICEHOME_DOMAINS if not (root / domain).exists()]
    if missing:
        raise FileNotFoundError(f"Missing OfficeHome domain directories under {root}: {missing}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source domain not found: {source_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Target domain not found: {target_dir}")


def _split_source(
    full_source: Dataset,
    source_eval_full: Dataset,
    val_fraction: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    if not 0.0 < val_fraction < 1.0:
        return full_source, source_eval_full
    val_size = max(1, int(len(full_source) * val_fraction))
    train_size = len(full_source) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, eval_subset = random_split(
        range(len(full_source)),
        [train_size, val_size],
        generator=generator,
    )
    return (
        torch.utils.data.Subset(full_source, list(train_subset)),
        torch.utils.data.Subset(source_eval_full, list(eval_subset)),
    )


def _loader(dataset: Dataset, config: ExperimentConfig, shuffle: bool) -> DataLoader:
    loader_kwargs: dict[str, int | bool] = {}
    if config.num_workers > 0:
        loader_kwargs = {"persistent_workers": True, "prefetch_factor": 4}
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )
