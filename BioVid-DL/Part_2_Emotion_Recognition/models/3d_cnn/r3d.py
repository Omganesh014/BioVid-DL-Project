import torch
import torch.nn as nn
import torchvision.models.video as video_models
from torchvision.models.video import R3D_18_Weights

class VideoR3D18(nn.Module):
    """
    3D CNN (R3D-18) for Spatio-Temporal Video Emotion Recognition.
    Processes video tensor [B, T, C, H, W] -> permutes to [B, C, T, H, W],
    extracts spatiotemporal representations via 3D convolutions,
    and classifies into task-specific emotion categories.
    """
    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        base = video_models.r3d_18(weights=weights)
        in_features = base.fc.in_features # 512

        # Replace classification head
        base.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [B, T, C, H, W] -> permute to [B, C, T, H, W]
        x_vid = x.permute(0, 2, 1, 3, 4)
        return self.model(x_vid)
