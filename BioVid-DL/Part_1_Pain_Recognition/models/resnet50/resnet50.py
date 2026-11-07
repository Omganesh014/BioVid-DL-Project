import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class VideoResNet50(nn.Module):
    """
    ResNet-50 Visual Feature Extractor with Temporal Pooling for Video Pain Recognition.
    Takes video tensor [B, T, C, H, W], extracts frame features via ResNet-50 backbone,
    applies temporal pooling across T frames, and projects to 5 pain intensity classes.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        # Backbone without final FC layer
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features # 2048

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x_flat) # [B*T, 2048, 1, 1]
        feats = feats.view(B, T, -1) # [B, T, 2048]

        # Temporal average pooling
        vid_repr = torch.mean(feats, dim=1) # [B, 2048]
        logits = self.classifier(vid_repr) # [B, num_classes]
        return logits
