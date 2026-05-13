from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def load_toy_module():
    path = Path("methods/toy_torch/train.py").resolve()
    spec = importlib.util.spec_from_file_location("toy_torch_train", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_choose_device_auto_prefers_available_accelerator() -> None:
    module = load_toy_module()
    device = module.choose_device("auto")
    if torch.cuda.is_available():
        assert str(device) == "cuda"
    elif torch.backends.mps.is_available():
        assert str(device) == "mps"
    else:
        assert str(device) == "cpu"


def test_choose_device_cpu_is_always_supported() -> None:
    module = load_toy_module()
    assert str(module.choose_device("cpu")) == "cpu"
