import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ProjectConfig:
    # --- Paths ---
    db_root: Path = Path("/run/media/acernitro/Ventoy/DB")
    part_a_root: Path = Path("/run/media/acernitro/Ventoy/DB/PartA")
    samples_csv_path: Path = Path("/run/media/acernitro/Ventoy/DB/PartA/starting_point/samples.csv")
    project_root: Path = Path("/home/acernitro/Desktop/X3D/ACDMICS/DL/proj")
    output_dir: Path = Path("/home/acernitro/Desktop/X3D/ACDMICS/DL/proj/outputs")
    manifest_dir: Path = Path("/home/acernitro/Desktop/X3D/ACDMICS/DL/proj/outputs/manifest")
    checkpoints_dir: Path = Path("/home/acernitro/Desktop/X3D/ACDMICS/DL/proj/outputs/checkpoints")
    logs_dir: Path = Path("/home/acernitro/Desktop/X3D/ACDMICS/DL/proj/outputs/logs")
    visualizations_dir: Path = Path("/home/acernitro/Desktop/X3D/ACDMICS/DL/proj/outputs/visualizations")
    evaluation_dir: Path = Path("/home/acernitro/Desktop/X3D/ACDMICS/DL/proj/outputs/evaluation")

    # --- Dataset & Labels ---
    label_map: Dict[int, str] = field(default_factory=lambda: {
        0: "BL1",  # Baseline (no pain)
        1: "PA1",  # Lowest pain intensity
        2: "PA2",  # Pain level 2
        3: "PA3",  # Pain level 3
        4: "PA4",  # Highest pain intensity
    })
    num_classes: int = 5
    raw_fps: float = 25.0
    raw_frame_count: int = 138
    raw_width: int = 1388
    raw_height: int = 1038

    # --- Temporal & Spatial Processing ---
    num_frames: int = 16           # T: sampled frames per video clip
    image_size: int = 224          # H = W = 224
    crop_mode: str = "center_square" # 1038x1038 center crop then resize to 224x224
    sampling_strategy: str = "uniform" # Uniformly spaced indices across 138 frames

    # --- ViViT Architecture Parameters ---
    patch_size: int = 16           # Spatial patch dimension P
    tubelet_t: int = 1             # Temporal tubelet size t
    embed_dim: int = 256           # Token embedding dimension D
    spatial_layers: int = 4        # Number of Spatial Transformer blocks
    temporal_layers: int = 4       # Number of Temporal Transformer blocks
    num_heads: int = 4             # Attention heads
    mlp_dim: int = 512             # Feedforward hidden dimension
    dropout: float = 0.1           # Dropout rate

    # --- Training Hyperparameters ---
    batch_size: int = 8            # Batch size B (chosen for RTX 4050 6GB VRAM safety)
    learning_rate: float = 1e-4    # AdamW initial learning rate
    weight_decay: float = 1e-2     # Weight decay
    epochs: int = 5                # Number of epochs for demonstration/training
    num_workers: int = 2           # DataLoader worker threads
    use_amp: bool = True           # Mixed precision (torch.cuda.amp)
    gradient_clip_norm: float = 1.0# Max gradient norm
    seed: int = 42                 # Random seed for full reproducibility

    # --- Subject Splitting ---
    train_ratio: float = 0.70      # ~61 subjects
    val_ratio: float = 0.15        # ~13 subjects
    test_ratio: float = 0.15       # ~13 subjects

    def __post_init__(self):
        for d in [
            self.output_dir,
            self.manifest_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.visualizations_dir,
            self.evaluation_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


config = ProjectConfig()
