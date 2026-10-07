"""Model definitions: a custom CNN baseline and a fine-tuned ResNet50."""
import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """4 conv blocks + classifier head, trained from scratch."""

    def __init__(self, num_classes: int = 2):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_resnet50(num_classes: int = 2, pretrained: bool = True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    # Freeze early layers, fine-tune layer3, layer4 and the head.
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(("layer3", "layer4", "fc"))
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
    return model


def get_model(name: str, num_classes: int = 2, pretrained: bool = True):
    if name == "baseline_cnn":
        return BaselineCNN(num_classes)
    if name == "resnet50":
        return build_resnet50(num_classes, pretrained)
    raise ValueError(f"Unknown model: {name}")


def gradcam_target_layer(model, name: str):
    """Last conv layer used for Grad-CAM."""
    if name == "baseline_cnn":
        return model.features[-1][3]  # last Conv2d of final block
    if name == "resnet50":
        return model.layer4[-1]
    raise ValueError(name)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
