from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ViViTConfig:
    """
    Explicit specification of ViViT (Factorized Encoder) Architecture.
    """
    num_frames: int = 16              # Number of temporal frames T
    image_size: int = 224             # Image spatial dimension H = W = 224
    patch_size: int = 16              # Spatial patch size P
    tubelet_t: int = 1                # Temporal tubelet size t
    in_channels: int = 3              # RGB channels C
    embed_dim: int = 256              # Token embedding dimension D
    spatial_layers: int = 4           # Number of spatial transformer layers
    temporal_layers: int = 4          # Number of temporal transformer layers
    num_heads: int = 4                # Number of attention heads
    mlp_dim: int = 512                # Feedforward MLP dimension
    dropout: float = 0.1              # Dropout probability
    num_classes: int = 5              # Number of classes (BL1, PA1..PA4)


class TubeletEmbedding(nn.Module):
    """
    Extracts spatio-temporal tubelets from video tensors using 3D convolution.
    Input shape:  [B, C, T, H, W]
    Output shape: [B, D, T_out, H_out, W_out]
    """
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        tubelet_t: int = 1,
        patch_size: int = 16,
    ):
        super().__init__()
        self.projection = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(tubelet_t, patch_size, patch_size),
            stride=(tubelet_t, patch_size, patch_size),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        return self.projection(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention block with query, key, value projections and dropout.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Encoder Block.
    x -> LayerNorm -> Self-Attention + Residual -> LayerNorm -> MLP + Residual
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViViT(nn.Module):
    """
    ViViT: Video Vision Transformer (Model 2: Factorized Encoder Architecture).
    Reference: Arnab et al., "ViViT: A Video Vision Transformer", ICCV 2021.

    Pipeline:
    1. Input video [B, T, C, H, W] -> permute to [B, C, T, H, W]
    2. Tubelet Embedding -> [B, D, T_out, H_p, W_p]
    3. Spatial Transformer: processes patches within each frame along with a Spatial [CLS] token.
       Extracts frame-level representations of shape [B, T_out, D].
    4. Temporal Transformer: processes the sequence of frame representations along with a Temporal [CLS] token.
       Extracts video-level representation of shape [B, D].
    5. Classification Head: projects video representation to class logits [B, num_classes].
    """

    def __init__(self, cfg: Optional[ViViTConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = ViViTConfig()
        self.cfg = cfg

        # Compute token counts
        self.num_spatial_patches = (cfg.image_size // cfg.patch_size) * (cfg.image_size // cfg.patch_size)
        self.num_temporal_tubes = cfg.num_frames // cfg.tubelet_t

        # 1. Tubelet Embedding
        self.tubelet_embed = TubeletEmbedding(
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            tubelet_t=cfg.tubelet_t,
            patch_size=cfg.patch_size,
        )

        # 2. Spatial Transformer
        self.spatial_cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spatial_patches + 1, cfg.embed_dim)
        )
        self.spatial_pos_drop = nn.Dropout(cfg.dropout)
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_dim=cfg.mlp_dim,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.spatial_layers)
        ])
        self.spatial_norm = nn.LayerNorm(cfg.embed_dim)

        # 3. Temporal Transformer
        self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_temporal_tubes + 1, cfg.embed_dim)
        )
        self.temporal_pos_drop = nn.Dropout(cfg.dropout)
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_dim=cfg.mlp_dim,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(cfg.embed_dim)

        # 4. Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, cfg.num_classes),
        )

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spatial_cls_token, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract video representation vector of shape [B, D].
        Input x: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        # Tubelet embedding expects [B, C, T, H, W]
        x_vid = x.permute(0, 2, 1, 3, 4)
        tubes = self.tubelet_embed(x_vid)  # [B, D, T_out, H_p, W_p]

        B, D, T_out, H_p, W_p = tubes.shape
        num_patches = H_p * W_p

        # Reshape to [B * T_out, num_patches, D]
        # First permute to [B, T_out, H_p, W_p, D]
        tubes = tubes.permute(0, 2, 3, 4, 1).contiguous()
        spatial_tokens = tubes.view(B * T_out, num_patches, D)

        # Append spatial CLS token: [B * T_out, num_patches + 1, D]
        spatial_cls = self.spatial_cls_token.expand(B * T_out, -1, -1)
        spatial_tokens = torch.cat([spatial_cls, spatial_tokens], dim=1)
        spatial_tokens = self.spatial_pos_drop(spatial_tokens + self.spatial_pos_embed)

        # Pass through Spatial Transformer blocks
        for block in self.spatial_blocks:
            spatial_tokens = block(spatial_tokens)
        spatial_tokens = self.spatial_norm(spatial_tokens)

        # Extract spatial CLS token representation: [B * T_out, D]
        frame_features = spatial_tokens[:, 0, :]  # [B * T_out, D]

        # Reshape to temporal sequence: [B, T_out, D]
        temporal_tokens = frame_features.view(B, T_out, D)

        # Append temporal CLS token: [B, T_out + 1, D]
        temp_cls = self.temporal_cls_token.expand(B, -1, -1)
        temporal_tokens = torch.cat([temp_cls, temporal_tokens], dim=1)
        temporal_tokens = self.temporal_pos_drop(temporal_tokens + self.temporal_pos_embed)

        # Pass through Temporal Transformer blocks
        for block in self.temporal_blocks:
            temporal_tokens = block(temporal_tokens)
        temporal_tokens = self.temporal_norm(temporal_tokens)

        # Extract video-level temporal CLS representation: [B, D]
        video_repr = temporal_tokens[:, 0, :]
        return video_repr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass producing class logits.
        Input x: [B, T, C, H, W]
        Returns: logits of shape [B, num_classes]
        """
        video_repr = self.forward_features(x)
        logits = self.head(video_repr)
        return logits

    def trace_pipeline(self, x: torch.Tensor) -> Dict[str, Tuple[int, ...]]:
        """
        Executes forward pass while recording exact shapes at every single layer.
        Used for Step 15 tracing verification.
        """
        shapes = {}
        shapes["01_input_video_batch"] = tuple(x.shape)  # [B, T, C, H, W]

        # Permute for 3D conv
        x_vid = x.permute(0, 2, 1, 3, 4)
        shapes["02_permuted_for_tubelet"] = tuple(x_vid.shape)  # [B, C, T, H, W]

        # Tubelet projection
        tubes = self.tubelet_embed(x_vid)
        shapes["03_tubelet_projection"] = tuple(tubes.shape)  # [B, D, T_out, H_p, W_p]

        B, D, T_out, H_p, W_p = tubes.shape
        num_patches = H_p * W_p

        # Spatial tokens
        tubes_perm = tubes.permute(0, 2, 3, 4, 1).contiguous()
        spatial_tokens = tubes_perm.view(B * T_out, num_patches, D)
        shapes["04_spatial_tokens_flattened"] = tuple(spatial_tokens.shape)

        # Add Spatial CLS
        spatial_cls = self.spatial_cls_token.expand(B * T_out, -1, -1)
        spatial_tokens = torch.cat([spatial_cls, spatial_tokens], dim=1)
        shapes["05_spatial_tokens_with_cls"] = tuple(spatial_tokens.shape)

        # Add Spatial Pos Embed
        spatial_tokens = self.spatial_pos_drop(spatial_tokens + self.spatial_pos_embed)
        shapes["06_spatial_tokens_pos_encoded"] = tuple(spatial_tokens.shape)

        # Spatial Transformer
        for i, block in enumerate(self.spatial_blocks):
            spatial_tokens = block(spatial_tokens)
            shapes[f"07_spatial_block_{i+1}"] = tuple(spatial_tokens.shape)

        spatial_tokens = self.spatial_norm(spatial_tokens)
        shapes["08_spatial_norm"] = tuple(spatial_tokens.shape)

        # Extract frame CLS
        frame_features = spatial_tokens[:, 0, :]
        shapes["09_frame_cls_extracted"] = tuple(frame_features.shape)

        # Reshape to temporal sequence
        temporal_tokens = frame_features.view(B, T_out, D)
        shapes["10_temporal_sequence"] = tuple(temporal_tokens.shape)

        # Add Temporal CLS
        temp_cls = self.temporal_cls_token.expand(B, -1, -1)
        temporal_tokens = torch.cat([temp_cls, temporal_tokens], dim=1)
        shapes["11_temporal_with_cls"] = tuple(temporal_tokens.shape)

        # Add Temporal Pos Embed
        temporal_tokens = self.temporal_pos_drop(temporal_tokens + self.temporal_pos_embed)
        shapes["12_temporal_pos_encoded"] = tuple(temporal_tokens.shape)

        # Temporal Transformer
        for i, block in enumerate(self.temporal_blocks):
            temporal_tokens = block(temporal_tokens)
            shapes[f"13_temporal_block_{i+1}"] = tuple(temporal_tokens.shape)

        temporal_tokens = self.temporal_norm(temporal_tokens)
        shapes["14_temporal_norm"] = tuple(temporal_tokens.shape)

        # Video representation
        video_repr = temporal_tokens[:, 0, :]
        shapes["15_video_cls_representation"] = tuple(video_repr.shape)

        # Classification Head
        logits = self.head(video_repr)
        shapes["16_output_logits"] = tuple(logits.shape)

        return shapes
