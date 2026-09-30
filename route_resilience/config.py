"""
Central configuration for the Route Resilience pipeline.

Keep every tunable in one place so a training run, an ablation, and a
graph/criticality run all read from the same source of truth. This is
also what makes the "fair comparison" requirement from Phase 0 actually
enforceable in code: if two models are compared, they should be built
from the same Config object with only ONE field changed (e.g. `arch`),
not two independently-hand-edited notebook cells.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    dataset_dir: str = "/content/RouteResilience"
    image_subdir: str = "images_png"
    mask_subdir: str = "masks_png"
    split_csv: str = "data_split.csv"
    image_size: int = 256          # verify this matches your actual tiles
    batch_size: int = 16
    num_workers: int = 2
    # If True, train_transform includes a synthetic occlusion (coarse
    # dropout) augmentation. This is CORE, not optional -- without it,
    # nothing in training actually engages the "occlusion-robust" claim.
    use_occlusion_augmentation: bool = True
    occlusion_max_holes: int = 6
    occlusion_max_frac: float = 0.15  # max fraction of tile edge length per hole


@dataclass
class ModelConfig:
    arch: str = "unet"             # "unet" | "unet_smp" | "deeplabv3plus"
    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"  # never None for a fair run
    in_channels: int = 3
    classes: int = 1
    dropout_p: float = 0.2         # used by the custom MC-Dropout U-Net


@dataclass
class TrainConfig:
    epochs: int = 50
    patience: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    loss: str = "focal_dice"       # "bce_dice" | "focal_dice"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    threshold: float = 0.5         # only a *starting* value -- always re-tune on val
    seed: int = 42


@dataclass
class UncertaintyConfig:
    mc_samples: int = 20
    ece_bins: int = 15


@dataclass
class GraphConfig:
    # Gap-healing is confidence-gated per Phase 0 -- never pure geometry.
    max_gap_px: float = 15.0
    min_gap_confidence: float = 0.15   # evidence threshold, deliberately
                                        # lower than the segmentation
                                        # decision threshold -- this is
                                        # "is there any support at all
                                        # for a road here", not "is this
                                        # confidently a road"
    min_edge_length_px: float = 3.0    # drop skeleton spurs shorter than this


@dataclass
class Phase3Config:
    """CPU-only pixel-space graph-resilience experiment settings."""
    weight: str = "length_px"
    od_pair_limit: Optional[int] = 200
    od_seed: int = 42
    top_n: int = 10
    failure_count: int = 1
    failure_seed: int = 42
    recovery_budget: Optional[int] = None  # None restores all failed baseline assets
    recovery_ordering: str = "explicit_failure_order"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    device: str = "cuda"           # falls back to cpu automatically in code
