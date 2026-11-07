import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention3D(nn.Module):
    """
    3D Window Multi-Head Self-Attention.
    """
    def __init__(self, dim, num_heads=4, qkv_bias=True, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class VideoSwinBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VideoSwinTransformer(nn.Module):
    """
    Video Swin Transformer for Emotion Recognition.
    - Patch Partition: (2, 16, 16) 3D convolution to embed spatiotemporal tubes
    - Swin Transformer Blocks: Multi-stage hierarchical window self-attention
    - Global temporal pooling and emotion classification head
    """
    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        embed_dim: int = 128,
        depths: tuple = (2, 2),
        num_heads: tuple = (4, 8),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.patch_embed = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(2, 16, 16),
            stride=(2, 16, 16),
            bias=True
        )
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        current_dim = embed_dim
        for d, h in zip(depths, num_heads):
            for _ in range(d):
                self.blocks.append(VideoSwinBlock(dim=current_dim, num_heads=h, dropout=dropout))

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W] -> permute to [B, C, T, H, W]
        x_vid = x.permute(0, 2, 1, 3, 4)
        tubes = self.patch_embed(x_vid) # [B, D, T_out, H_out, W_out]
        B, D, T_out, H_out, W_out = tubes.shape
        N = T_out * H_out * W_out

        # Flatten spatiotemporal tokens: [B, N, D]
        tokens = tubes.permute(0, 2, 3, 4, 1).contiguous().view(B, N, D)
        tokens = self.pos_drop(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        # Global pooling across tokens
        video_repr = torch.mean(tokens, dim=1) # [B, D]
        logits = self.head(video_repr) # [B, num_classes]
        return logits
