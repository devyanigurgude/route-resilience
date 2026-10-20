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
    occlusion_aug_type: str = "coarse_dropout"  # "coarse_dropout" | "multi_type"


@dataclass
class ModelConfig:
    arch: str = "unet"             # "unet" | "unet_smp" | "deeplabv3plus" | "segformer"
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
    tangent_dot_threshold: float = 0.7  # cos(~45.6°) — minimum dot product
                                        # between each endpoint tangent and the
                                        # gap direction for a candidate to be
                                        # accepted. Increase toward 1.0 for
                                        # stricter angle alignment (fewer but
                                        # more precise healed edges).
                                        # cos(30°)≈0.866, cos(25°)≈0.906,
                                        # cos(20°)≈0.940. Default 0.7 preserves
                                        # the Phase 1A production behaviour.
    continuity_penalty_weight: float = 0.0  # lambda >= 0: penalizes candidates
                                            # whose tangents diverge from collinear
                                            # alignment when sorting candidate pairs.
    min_component_len: float = 0.0          # min individual component length for endpoint branches
    min_total_component_len: float = 0.0    # min combined component length for candidate pair
    max_sinuosity: float = 1.5              # max allowed path length / Euclidean distance ratio
    min_flank_contrast: Optional[float] = None # min centerline vs perpendicular flank contrast
    filter_tile_borders: bool = False       # prune candidates with both endpoints on tile border
    # Production continuity recovery remains evidence-gated: it requires
    # facing endpoint directions, limited separation, a viable probability
    # corridor, and rejects already-connected/unaligned segments.
    enable_occlusion_recovery: bool = True
    max_occlusion_gap_px: float = 30.0      # max gap distance considered for occlusion candidates
    min_occlusion_confidence: float = 0.05  # threshold for occluded corridors with high geometric alignment
    occlusion_heading_threshold: float = 0.75  # strict tangent alignment required for low-confidence spans
    max_lateral_offset_px: float = 4.0      # max lateral displacement between road continuation axes
    # A large bright cloud can suppress every road pixel between two otherwise
    # substantial, facing road branches.  This separate path is deliberately
    # stricter than ordinary occlusion recovery: it requires a cloud mask,
    # long visible trunks, mutual heading, and a smooth tangent-continuous arc.
    enable_cloud_occlusion_recovery: bool = True
    max_cloud_occlusion_gap_px: float = 240.0
    cloud_heading_threshold: float = 0.78
    cloud_min_component_len: float = 20.0
    cloud_min_coverage: float = 0.55
    cloud_max_sinuosity: float = 1.35
    cloud_min_mutual_heading: float = 0.60    # floor for distance-adaptive mutual heading;
                                               # short cloud gaps use 0.85, long gaps relax
                                               # linearly down to this value



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
