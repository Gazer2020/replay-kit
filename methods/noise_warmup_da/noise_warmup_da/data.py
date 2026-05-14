from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image, ImageOps
from torchvision import datasets, transforms

from noise_warmup_da.config import ExperimentConfig


OFFICEHOME_DOMAINS = ("Art", "Clipart", "Product", "Real World")
SAMPLE_DOMAINS = ("synth", "real")


@dataclass(frozen=True)
class DomainData:
    domain: str
    train_loader: DataLoader
    target_train_loader: DataLoader | None
    test_loader: DataLoader
    eval_loaders: dict[str, DataLoader]
    eval_sizes: dict[str, int]
    num_classes: int
    class_names: list[str]
    input_shape: tuple[int, int, int]
    train_size: int
    test_size: int


def make_domain_data(config: ExperimentConfig, domain: str, seed: int) -> DomainData:
    if config.dataset == "officehome":
        return _make_officehome(config, domain, seed)
    if config.dataset == "sample_sar":
        return _make_sample_sar(config, domain, seed)
    if config.dataset == "fake":
        return _make_fake(config, domain, seed)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def _make_officehome(config: ExperimentConfig, domain: str, seed: int) -> DomainData:
    root = config.data_root / "OfficeHome"
    domain_dir = root / domain
    _validate_officehome_path(root, domain_dir)

    train_transform = _image_transform(config, train=True, output_size=config.image_size)
    eval_transform = _image_transform(config, train=False, output_size=config.image_size)

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
        train_loader=_loader(train_dataset, config, shuffle=True, seed=seed),
        target_train_loader=None,
        test_loader=_loader(test_dataset, config, shuffle=False, seed=seed + 1),
        eval_loaders={domain: _loader(test_dataset, config, shuffle=False, seed=seed + 2)},
        eval_sizes={domain: len(test_dataset)},
        num_classes=len(train_full.classes),
        class_names=list(train_full.classes),
        input_shape=(3, config.image_size, config.image_size),
        train_size=len(train_dataset),
        test_size=len(test_dataset),
    )


def _make_sample_sar(config: ExperimentConfig, domain: str, seed: int) -> DomainData:
    source_domain = config.source_domain or domain
    target_domain = config.target_domain
    variant_root = config.data_root / "SAMPLE_dataset_public" / "png_images" / config.sample_variant
    source_dir = variant_root / source_domain
    target_dir = variant_root / target_domain
    _validate_sample_path(variant_root, source_dir, target_dir)

    output_size = _pad_output_size(config, [source_dir, target_dir])
    train_transform = _image_transform(config, train=True, output_size=output_size)
    eval_transform = _image_transform(config, train=False, output_size=output_size)
    train_full = datasets.ImageFolder(source_dir, transform=train_transform)
    source_eval_full = datasets.ImageFolder(source_dir, transform=eval_transform)
    target_eval_dataset = datasets.ImageFolder(target_dir, transform=eval_transform)
    target_train_dataset = datasets.ImageFolder(target_dir, transform=train_transform)
    _validate_matching_classes(train_full, target_eval_dataset, source_dir, target_dir)
    train_indices, source_eval_indices = stratified_split_indices(
        train_full,
        test_fraction=config.test_fraction,
        seed=seed,
    )
    train_dataset = Subset(train_full, train_indices)
    source_eval_dataset = Subset(source_eval_full, source_eval_indices)

    eval_datasets = {
        source_domain: source_eval_dataset,
        target_domain: target_eval_dataset,
    }
    selected_eval_domains = config.eval_domains or [source_domain, target_domain]
    unknown = sorted(set(selected_eval_domains) - set(eval_datasets))
    if unknown:
        raise ValueError(f"Unknown SAMPLE eval_domains entries: {unknown}")
    eval_loaders = {
        f"{source_domain}->{eval_domain}": _loader(
            eval_datasets[eval_domain],
            config,
            shuffle=False,
            seed=seed + 10 + index,
        )
        for index, eval_domain in enumerate(selected_eval_domains)
    }
    eval_sizes = {
        f"{source_domain}->{eval_domain}": len(eval_datasets[eval_domain])
        for eval_domain in selected_eval_domains
    }
    primary_eval_key = f"{source_domain}->{selected_eval_domains[0]}"
    return DomainData(
        domain=source_domain,
        train_loader=_loader(train_dataset, config, shuffle=True, seed=seed),
        target_train_loader=_loader(target_train_dataset, config, shuffle=True, seed=seed + 1),
        test_loader=eval_loaders[primary_eval_key],
        eval_loaders=eval_loaders,
        eval_sizes=eval_sizes,
        num_classes=len(train_full.classes),
        class_names=list(train_full.classes),
        input_shape=(3, output_size, output_size),
        train_size=len(train_dataset),
        test_size=eval_sizes[primary_eval_key],
    )


def _make_fake(config: ExperimentConfig, domain: str, seed: int) -> DomainData:
    transform = _image_transform(config, train=False, output_size=config.image_size)
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
        train_loader=_loader(train_dataset, config, shuffle=True, seed=seed),
        target_train_loader=None,
        test_loader=_loader(test_dataset, config, shuffle=False, seed=seed + 1),
        eval_loaders={domain: _loader(test_dataset, config, shuffle=False, seed=seed + 2)},
        eval_sizes={domain: len(test_dataset)},
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


def _validate_sample_path(variant_root: Path, source_dir: Path, target_dir: Path) -> None:
    if not variant_root.exists():
        raise FileNotFoundError(
            f"SAMPLE SAR variant root not found: {variant_root}. "
            "Expected data/SAMPLE_dataset_public/png_images/{decibel,qpm}/{synth,real}."
        )
    missing = [domain for domain in SAMPLE_DOMAINS if not (variant_root / domain).exists()]
    if missing:
        raise FileNotFoundError(f"Missing SAMPLE SAR domain directories under {variant_root}: {missing}")
    if not source_dir.exists():
        raise FileNotFoundError(f"SAMPLE SAR source domain not found: {source_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"SAMPLE SAR target domain not found: {target_dir}")


def _validate_matching_classes(
    source: datasets.ImageFolder,
    target: datasets.ImageFolder,
    source_dir: Path,
    target_dir: Path,
) -> None:
    if source.class_to_idx != target.class_to_idx:
        raise ValueError(
            "SAMPLE SAR source and target classes do not match: "
            f"{source_dir} vs {target_dir}"
        )


def _image_transform(config: ExperimentConfig, train: bool, output_size: int) -> transforms.Compose:
    transform_steps: list[object] = []
    if config.transform_mode == "resize":
        transform_steps.append(transforms.Resize((output_size, output_size)))
    elif config.transform_mode == "pad":
        transform_steps.append(PadToSquare(output_size, fill=config.padding_fill))
    else:
        raise ValueError(f"Unsupported transform_mode: {config.transform_mode}")
    if train and config.random_horizontal_flip:
        transform_steps.append(transforms.RandomHorizontalFlip())
    if config.dataset == "sample_sar":
        transform_steps.append(transforms.Grayscale(num_output_channels=3))
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(transform_steps)


def _pad_output_size(config: ExperimentConfig, roots: list[Path]) -> int:
    if config.transform_mode != "pad":
        return config.image_size
    max_side = 0
    for root in roots:
        dataset = datasets.ImageFolder(root)
        for path, _ in dataset.samples:
            with Image.open(path) as raw_image:
                image = ImageOps.exif_transpose(raw_image)
                max_side = max(max_side, *image.size)
    return max(config.image_size, max_side)


class PadToSquare:
    def __init__(self, size: int, fill: int = 0) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        target = max(self.size, width, height)
        horizontal = target - width
        vertical = target - height
        left = horizontal // 2
        top = vertical // 2
        right = horizontal - left
        bottom = vertical - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)


def _loader(dataset: Dataset, config: ExperimentConfig, shuffle: bool, seed: int) -> DataLoader:
    loader_kwargs: dict[str, int | bool] = {}
    if config.num_workers > 0:
        loader_kwargs = {"persistent_workers": True, "prefetch_factor": 4}
    generator = torch.Generator().manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=seed_worker,
        **loader_kwargs,
    )
