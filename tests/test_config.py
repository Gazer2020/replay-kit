from __future__ import annotations

import pytest

from replay_kit.config import compose_config, parse_override


def test_compose_config_merges_base_default_and_experiment() -> None:
    config = compose_config("toy_torch", "debug")
    assert config["project_name"] == "replay-kit"
    assert config["method_name"] == "toy_torch"
    assert config["experiment_name"] == "debug"
    assert config["command"]["entrypoint"] == "methods/toy_torch/train.py"
    assert config["toy_torch"]["epochs"] == 3


def test_compose_config_applies_dotted_overrides(tmp_path) -> None:
    config = compose_config(
        "toy_torch",
        "debug",
        overrides=["device=cpu", "toy_torch.epochs=2"],
        output_root=str(tmp_path),
    )
    assert config["device"] == "cpu"
    assert config["toy_torch"]["epochs"] == 2
    assert config["output_root"] == str(tmp_path)


def test_parse_override_requires_key_value() -> None:
    with pytest.raises(ValueError):
        parse_override("device")
