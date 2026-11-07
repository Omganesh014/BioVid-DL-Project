import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class VideoCNNLSTM(nn.Module):
    """
    CNN-LSTM Architecture for Pain Intensity Recognition.
    - Spatial feature extractor: Pretrained ResNet-18 backbone (extracts 512-dim vectors per frame)
    - Temporal sequence model: 2-layer Bidirectional LSTM
    - Classification head: Linear projection from final temporal hidden states to 5 pain classes
    """
    def __init__(
        self,
        num_classes: int = 5,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        self.cnn = nn.Sequential(*list(base.children())[:-1])
        cnn_out_dim = base.fc.in_features # 512

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        cnn_feats = self.cnn(x_flat) # [B*T, 512, 1, 1]
        cnn_feats = cnn_feats.view(B, T, -1) # [B, T, 512]

        lstm_out, _ = self.lstm(cnn_feats) # [B, T, hidden_dim * 2]
        # Temporal pooling over LSTM sequence
        temporal_repr = torch.mean(lstm_out, dim=1) # [B, hidden_dim * 2]
        logits = self.classifier(temporal_repr) # [B, num_classes]
        return logits
