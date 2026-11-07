import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class Physiological1DCNN(nn.Module):
    """
    1D CNN feature extractor for multi-channel biomedical signals (GSR, ECG, EMG).
    Input: [B, num_channels, time_samples] (e.g. [B, 3, 2816] or [B, 5, 2816])
    Output: Latent physiological vector of shape [B, out_dim]
    """
    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.conv_net = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv_net(x) # [B, 128, 1]
        feats = feats.squeeze(-1) # [B, 128]
        return self.fc(feats)

class MultimodalCNN(nn.Module):
    """
    Main Proposed Architecture: Multimodal CNN for Pain and Emotion Recognition.
    - Visual Branch: Pretrained ResNet-18 spatial encoder + temporal pooling -> 512-dim
    - Physiological Branch: Multi-channel 1D CNN -> 256-dim
    - Fusion Layer: Projection + Normalization -> 256-dim joint representation
    - Task Heads: Head 1 (Pain 5-class) and Head 2 (Emotion 7-class or 5-class)
    """
    def __init__(
        self,
        num_bio_channels: int = 3,
        num_pain_classes: int = 5,
        num_emotion_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        # 1. Video Branch
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        self.video_cnn = nn.Sequential(*list(base.children())[:-1])
        video_dim = base.fc.in_features # 512

        # 2. Physiological Branch
        bio_dim = 256
        self.bio_cnn = Physiological1DCNN(in_channels=num_bio_channels, out_dim=bio_dim)

        # 3. Fusion Layer
        fused_dim = 256
        self.fusion = nn.Sequential(
            nn.Linear(video_dim + bio_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Separate unimodal projection heads for ablation testing
        self.video_only_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(video_dim, num_pain_classes)
        )
        self.bio_only_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bio_dim, num_pain_classes)
        )

        # 4. Joint Multimodal Classification Heads
        self.pain_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_pain_classes)
        )
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_emotion_classes)
        )

    def extract_video_features(self, video: torch.Tensor) -> torch.Tensor:
        # video: [B, T, C, H, W]
        B, T, C, H, W = video.shape
        v_flat = video.view(B * T, C, H, W)
        feats = self.video_cnn(v_flat).view(B, T, -1)
        return torch.mean(feats, dim=1) # [B, 512]

    def extract_bio_features(self, bio: torch.Tensor) -> torch.Tensor:
        # bio: [B, num_channels, time_samples]
        return self.bio_cnn(bio) # [B, 256]

    def forward(
        self,
        video: torch.Tensor = None,
        bio: torch.Tensor = None,
        task: str = "pain",
        modality_mode: str = "multimodal",
    ) -> torch.Tensor:
        """
        Forward pass with flexible modality ablation support:
        modality_mode: 'multimodal' | 'video_only' | 'bio_only'
        task: 'pain' | 'emotion'
        """
        if modality_mode == "video_only":
            assert video is not None, "Video tensor required for video_only mode"
            v_feat = self.extract_video_features(video)
            return self.video_only_head(v_feat)

        if modality_mode == "bio_only":
            assert bio is not None, "Bio tensor required for bio_only mode"
            b_feat = self.extract_bio_features(bio)
            return self.bio_only_head(b_feat)

        # Complete Multimodal Fusion
        v_feat = self.extract_video_features(video)
        b_feat = self.extract_bio_features(bio)
        fused = self.fusion(torch.cat([v_feat, b_feat], dim=-1))

        if task == "pain":
            return self.pain_head(fused)
        elif task == "emotion":
            return self.emotion_head(fused)
        else:
            raise ValueError(f"Unknown task {task}!")
