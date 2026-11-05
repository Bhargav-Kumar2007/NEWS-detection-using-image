"""Pretrained MobileNetV2 / ResNet18 classifiers + feature extraction."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

SUPPORTED = ("resnet18", "mobilenet_v2")
FEATURE_DIMS = {"resnet18": 512, "mobilenet_v2": 1280}


def build_model(arch: str = "resnet18", pretrained: bool = True, num_classes: int = 2,
                dropout: float = 0.2, freeze_backbone: bool = False) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
        head_prefix = "fc"
    elif arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(model.last_channel, num_classes))
        head_prefix = "classifier"
    else:
        raise ValueError(f"Unsupported arch '{arch}'. Choose from {SUPPORTED}.")

    model.arch = arch
    if freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(head_prefix)
    return model


def forward_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Penultimate-layer (globally pooled) features: 512-d (ResNet18) / 1280-d (MobileNetV2)."""
    arch = getattr(model, "arch", None)
    if arch == "resnet18":
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
        return torch.flatten(model.avgpool(x), 1)
    if arch == "mobilenet_v2":
        x = model.features(x)
        return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
    raise ValueError("Model has no 'arch' attribute; build it with build_model().")
