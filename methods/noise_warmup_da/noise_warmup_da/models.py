from __future__ import annotations

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


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
