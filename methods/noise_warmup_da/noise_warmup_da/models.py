from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


def make_classifier(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class FrozenBackboneLinearProbe(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int) -> None:
        super().__init__()
        if not hasattr(model, "fc"):
            raise ValueError("Linear probe expects a torchvision ResNet-like model.")
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        in_features = model.fc.in_features
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(inputs).flatten(1)
        return self.head(features)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("fc."):
            parameter.requires_grad = trainable


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
