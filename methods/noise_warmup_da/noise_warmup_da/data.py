from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from noise_warmup_da.config import ExperimentConfig


OFFICEHOME_DOMAINS = ("Art", "Clipart", "Product", "Real World")


@dataclass(frozen=True)
class DomainData:
    domain: str
    train_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    class_names: list[str]
    input_shape: tuple[int, int, int]
    train_size: int
    test_size: int


def make_domain_data(config: ExperimentConfig, domain: str, seed: int) -> DomainData:
    if config.dataset == "officehome":
        return _make_officehome(config, domain, seed)
    if config.dataset == "fake":
        return _make_fake(config, domain, seed)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def _make_officehome(config: ExperimentConfig, domain: str, seed: int) -> DomainData:
    root = config.data_root / "OfficeHome"
    domain_dir = root / domain
    _validate_officehome_path(root, domain_dir)

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

    train_full = datasets.ImageFolder(domain_dir, transform=train_transform)
    test_full = datasets.ImageFolder(domain_dir, transform=eval_transform)
    train_indices, test_indices = stratified_split_indices(
        train_full,
        test_fraction=config.test_fraction,
        seed=seed,
    )
    train_dataset = Subset(train_full, train_indices)
    test_dataset = Subset(test_full, test_indices)
    return DomainData(
        domain=domain,
        train_loader=_loader(train_dataset, config, shuffle=True),
        test_loader=_loader(test_dataset, config, shuffle=False),
        num_classes=len(train_full.classes),
        class_names=list(train_full.classes),
        input_shape=(3, config.image_size, config.image_size),
        train_size=len(train_dataset),
        test_size=len(test_dataset),
    )


def _make_fake(config: ExperimentConfig, domain: str, seed: int) -> DomainData:
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
    domain_offset = sum((index + 1) * ord(char) for index, char in enumerate(domain)) % 10_000
    train_dataset = datasets.FakeData(
        size=config.fake_train_size,
        image_size=(3, config.image_size, config.image_size),
        num_classes=config.num_classes,
        transform=transform,
        random_offset=domain_offset + seed,
    )
    test_dataset = datasets.FakeData(
        size=config.fake_test_size,
        image_size=(3, config.image_size, config.image_size),
        num_classes=config.num_classes,
        transform=transform,
        random_offset=domain_offset + 20_000 + seed,
    )
    class_names = [str(index) for index in range(config.num_classes)]
    return DomainData(
        domain=domain,
        train_loader=_loader(train_dataset, config, shuffle=True),
        test_loader=_loader(test_dataset, config, shuffle=False),
        num_classes=config.num_classes,
        class_names=class_names,
        input_shape=(3, config.image_size, config.image_size),
        train_size=len(train_dataset),
        test_size=len(test_dataset),
    )


def stratified_split_indices(
    dataset: datasets.ImageFolder,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    per_class: dict[int, list[int]] = {}
    for index, (_, target) in enumerate(dataset.samples):
        per_class.setdefault(int(target), []).append(index)

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for indices in per_class.values():
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[index] for index in permutation]
        test_size = max(1, int(round(len(shuffled) * test_fraction)))
        test_indices.extend(shuffled[:test_size])
        train_indices.extend(shuffled[test_size:])
    train_indices.sort()
    test_indices.sort()
    return train_indices, test_indices


def _validate_officehome_path(root: Path, domain_dir: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(
            f"OfficeHome root not found: {root}. "
            "Expected data/OfficeHome/{Art,Clipart,Product,Real World}."
        )
    missing = [domain for domain in OFFICEHOME_DOMAINS if not (root / domain).exists()]
    if missing:
        raise FileNotFoundError(f"Missing OfficeHome domain directories under {root}: {missing}")
    if not domain_dir.exists():
        raise FileNotFoundError(f"OfficeHome domain not found: {domain_dir}")


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
