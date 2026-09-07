"""
Model factory covering the tiers from the project plan:

  - mobilenet   : fast/mobile baseline (MobileNetV3-Large)
  - resnet50    : strong CNN baseline
  - efficientnet: modern CNN (EfficientNet-B4)
  - vit         : transformer (ViT-B/16)
  - hybrid      : EfficientNet-B0 feature extractor + ViT-B16, features
                  concatenated before the classifier head

All models use ImageNet-pretrained weights and replace the final head for
the number of disease classes. This is a deliberately small, comparable
set (4 models) rather than 5+ arbitrary variants — add more only if a
crop's results call for it.
"""

from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as tvm

ModelName = Literal["mobilenet", "resnet50", "efficientnet", "vit", "hybrid"]


def build_model(name: ModelName, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Construct and return a model with a fresh classification head.

    The backbone is loaded with ImageNet-pretrained weights (unless
    *pretrained* is False) and the final layer is replaced to output
    *num_classes* logits.
    """
    if name == "mobilenet":
        weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = tvm.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif name == "resnet50":
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        model = tvm.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "efficientnet":
        weights = tvm.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = tvm.efficientnet_b4(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif name == "vit":
        weights = tvm.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = tvm.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.hidden_dim, num_classes)

    elif name == "hybrid":
        model = HybridCNNViT(num_classes=num_classes, pretrained=pretrained)

    else:
        raise ValueError(f"Unknown model name: {name!r}")

    return model


class HybridCNNViT(nn.Module):
    """EfficientNet-B0 (local texture/lesion features) + ViT-B16 (global
    context), features concatenated and passed through a small classifier
    head. Heavier than any single backbone — expect slower inference,
    reserve for crops where the single-model tiers plateau."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        eff_weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        effnet = tvm.efficientnet_b0(weights=eff_weights)
        self.cnn_features = effnet.features
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        cnn_out_dim = 1280  # EfficientNet-B0 final channel count

        vit_weights = tvm.ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = tvm.vit_b_16(weights=vit_weights)
        vit.heads = nn.Identity()  # keep the [CLS] embedding, drop the head
        self.vit = vit
        vit_out_dim = 768  # ViT-B/16 hidden dim

        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + vit_out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_pool(self.cnn_features(x)).flatten(1)
        vit_feat = self.vit(x)
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.classifier(combined)


MODEL_REGISTRY = {
    "mobilenet": "MobileNetV3-Large (fast/mobile baseline)",
    "resnet50": "ResNet-50 (strong CNN baseline)",
    "efficientnet": "EfficientNet-B4 (modern CNN)",
    "vit": "ViT-B/16 (transformer)",
    "hybrid": "EfficientNet-B0 + ViT-B/16 (hybrid)",
}
