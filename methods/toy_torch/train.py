from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml


def choose_device(requested: str):
    import torch

    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device=mps was requested, but MPS is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device value: {requested}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("config.yaml must contain a mapping")
    return data


def make_dataset(config: dict[str, Any], device):
    import torch

    method_config = config["toy_torch"]
    num_samples = int(method_config["num_samples"])
    input_dim = int(method_config["input_dim"])
    output_dim = int(method_config["output_dim"])

    x = torch.randn(num_samples, input_dim, device=device)
    true_w = torch.randn(input_dim, output_dim, device=device)
    true_b = torch.randn(output_dim, device=device)
    noise = 0.02 * torch.randn(num_samples, output_dim, device=device)
    y = x @ true_w + true_b + noise
    return x, y


def train(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import torch

    seed = int(config.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)

    method_config = config["toy_torch"]
    if bool(method_config.get("force_fail", False)):
        raise RuntimeError("toy_torch force_fail=true requested")

    device = choose_device(str(config.get("device", "auto")))
    print(f"[toy_torch] requested_device={config.get('device')} actual_device={device}", flush=True)

    input_dim = int(method_config["input_dim"])
    hidden_dim = int(method_config["hidden_dim"])
    output_dim = int(method_config["output_dim"])
    epochs = int(method_config["epochs"])
    batch_size = int(method_config["batch_size"])
    learning_rate = float(method_config["learning_rate"])

    x, y = make_dataset(config, device)
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    best_loss = float("inf")
    num_samples = x.shape[0]
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(num_samples, device=device)
        total_loss = 0.0
        for start in range(0, num_samples, batch_size):
            idx = permutation[start : start + batch_size]
            batch_x = x[idx]
            batch_y = y[idx]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * batch_x.shape[0]
        epoch_loss = total_loss / num_samples
        best_loss = min(best_loss, epoch_loss)
        print(f"[toy_torch] epoch={epoch} loss={epoch_loss:.6f}", flush=True)

    final_loss = epoch_loss
    target_loss = float(method_config.get("target_loss", 0.0))
    metrics = {
        "method": "toy_torch",
        "device": str(device),
        "requested_device": str(config.get("device", "auto")),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "target_loss": target_loss,
        "reached_target": best_loss <= target_loss,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[toy_torch] metrics written to {metrics_path}", flush=True)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train(config, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
