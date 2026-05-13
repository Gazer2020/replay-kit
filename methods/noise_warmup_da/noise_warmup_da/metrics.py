from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvalMetrics:
    accuracy: float
    avg_confidence: float
    nll: float
    ece: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, ece_bins: int) -> EvalMetrics:
    model.eval()
    logits_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits_parts.append(model(inputs).detach().cpu())
        target_parts.append(targets.detach().cpu())

    logits = torch.cat(logits_parts)
    targets = torch.cat(target_parts)
    probabilities = logits.softmax(dim=1)
    confidences, predictions = probabilities.max(dim=1)
    correct = predictions.eq(targets)
    return EvalMetrics(
        accuracy=correct.float().mean().item(),
        avg_confidence=confidences.mean().item(),
        nll=F.cross_entropy(logits, targets).item(),
        ece=expected_calibration_error(confidences, correct, ece_bins),
    )


def expected_calibration_error(confidences: Tensor, correct: Tensor, num_bins: int) -> float:
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    total = confidences.numel()
    ece = torch.tensor(0.0)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        confidence = confidences[in_bin].mean()
        accuracy = correct[in_bin].float().mean()
        ece += (count / total) * torch.abs(accuracy - confidence)
    return ece.item()
