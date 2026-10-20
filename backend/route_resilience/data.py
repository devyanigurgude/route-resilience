"""
Dataset + augmentation.

Fixes applied relative to the reviewed notebook:
  - occlusion-simulating augmentation is included and used for EVERY
    model, not bolted onto one run only (fairness + it's the one
    augmentation actually tied to the research question)
  - a single augmentation pipeline builder so train/val/test never
    silently diverge between experiments
  - a split-file sanity check that warns (does not silently pass) if
    the split looks like a random tile shuffle rather than a
    geography-disjoint split, per the Phase 0 data-leakage concern
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:  # pragma: no cover
    A = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_rgb_image(image: np.ndarray | Image.Image) -> torch.Tensor:
    """Validation/test preprocessing shared by dataset loading and inference."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    elif not isinstance(image, np.ndarray):
        image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {image.shape}")
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32)


class MultiTypeOcclusionAugmentation(A.ImageOnlyTransform):
    """
    Multi-type synthetic occlusion augmentation for training occlusion-robust models:
      1. 'rectangular_dropout': Black rectangular patches (simulating sensor dropouts & buildings).
      2. 'canopy_cluster': Textured foliage-green organic clusters (simulating natural tree canopy).
      3. 'procedural_cloud': Bright, low-saturation organic patches with soft Gaussian alpha edges
         and variable opacity (simulating clouds & haze).

    IMPORTANT: Ground-truth road masks are NEVER modified because they represent
    the true physical underlying road network beneath the occlusion.
    """

    def __init__(
        self,
        num_holes_range=(1, 4),
        hole_size_range=(16, 36),
        occlusion_types=("rectangular_dropout", "canopy_cluster", "procedural_cloud"),
        type_weights=(0.34, 0.33, 0.33),
        seed=None,
        p=0.5,
    ):
        super().__init__(p=p)
        self.num_holes_range = num_holes_range
        self.hole_size_range = hole_size_range
        self.occlusion_types = occlusion_types
        self.type_weights = type_weights
        self.rng = np.random.default_rng(seed)

    def apply(self, img, **params):
        import cv2

        H, W, _ = img.shape
        img_out = img.copy()

        n_holes = int(self.rng.integers(self.num_holes_range[0], self.num_holes_range[1] + 1))
        if n_holes <= 0:
            return img_out

        chosen_type = str(self.rng.choice(self.occlusion_types, p=self.type_weights))

        for _ in range(n_holes):
            sz = int(self.rng.integers(self.hole_size_range[0], self.hole_size_range[1] + 1))
            cr = int(self.rng.integers(sz // 2, max(sz // 2 + 1, H - sz // 2)))
            cc = int(self.rng.integers(sz // 2, max(sz // 2 + 1, W - sz // 2)))

            if chosen_type == "rectangular_dropout":
                half_h = sz // 2
                half_w = int(self.rng.integers(self.hole_size_range[0], self.hole_size_range[1] + 1)) // 2
                r0, r1 = max(0, cr - half_h), min(H, cr + half_h + 1)
                c0, c1 = max(0, cc - half_w), min(W, cc + half_w + 1)
                img_out[r0:r1, c0:c1] = 0

            elif chosen_type == "canopy_cluster":
                patch = np.zeros((H, W), dtype=bool)
                num_sub = int(self.rng.integers(3, 6))
                rad = max(3, sz // 3)
                for _ in range(num_sub):
                    dr = int(self.rng.integers(-rad, rad + 1))
                    dc = int(self.rng.integers(-rad, rad + 1))
                    pr = int(np.clip(cr + dr, 0, H - 1))
                    pc = int(np.clip(cc + dc, 0, W - 1))
                    patch[pr, pc] = True
                dilated = cv2.dilate(patch.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1, rad * 2 + 1))) > 0

                base_color = np.array([45, 75, 38], dtype=np.float32)
                texture_noise = self.rng.normal(0, 10.0, (H, W, 3))
                foliage = np.clip(base_color + texture_noise, 0, 255).astype(np.uint8)
                img_out[dilated] = foliage[dilated]

            elif chosen_type == "procedural_cloud":
                patch = np.zeros((H, W), dtype=np.float32)
                num_sub = int(self.rng.integers(4, 7))
                rad = max(4, sz // 2)
                for _ in range(num_sub):
                    dr = int(self.rng.integers(-rad, rad + 1))
                    dc = int(self.rng.integers(-rad, rad + 1))
                    pr = int(np.clip(cr + dr, 0, H - 1))
                    pc = int(np.clip(cc + dc, 0, W - 1))
                    patch[pr, pc] = 1.0

                ksize = max(5, rad * 2 + 1) | 1
                soft_mask = cv2.GaussianBlur(patch, (ksize, ksize), sigmaX=rad * 0.8)
                if soft_mask.max() > 0:
                    soft_mask = soft_mask / soft_mask.max()
                opacity = float(self.rng.uniform(0.60, 0.95))
                alpha = np.clip(soft_mask * opacity, 0.0, 1.0)[:, :, np.newaxis]

                cloud_base = np.array([235, 238, 242], dtype=np.float32)
                cloud_noise = self.rng.normal(0, 4.0, (H, W, 3))
                cloud_rgb = np.clip(cloud_base + cloud_noise, 0, 255).astype(np.float32)

                blended = alpha * cloud_rgb + (1.0 - alpha) * img_out.astype(np.float32)
                img_out = np.clip(blended, 0, 255).astype(np.uint8)

        return img_out


def build_transforms(cfg):
    """
    One augmentation pipeline, shared by every model trained in a
    comparison run. `use_occlusion_augmentation` controls whether
    occlusion augmentation is included. `occlusion_aug_type` chooses
    between baseline 'coarse_dropout' and 'multi_type'.
    """
    if A is None:
        raise ImportError("pip install albumentations --break-system-packages")

    ops = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ]

    if cfg.use_occlusion_augmentation:
        aug_type = getattr(cfg, "occlusion_aug_type", "coarse_dropout")
        if aug_type == "multi_type":
            max_sz = max(16, int(cfg.image_size * cfg.occlusion_max_frac))
            ops.append(
                MultiTypeOcclusionAugmentation(
                    num_holes_range=(1, cfg.occlusion_max_holes),
                    hole_size_range=(16, max_sz),
                    p=0.5,
                )
            )
        else:
            max_h = max(1, int(cfg.image_size * cfg.occlusion_max_frac))
            max_w = max(1, int(cfg.image_size * cfg.occlusion_max_frac))
            ops.append(
                A.CoarseDropout(
                    num_holes_range=(1, cfg.occlusion_max_holes),
                    hole_height_range=(1, max_h),
                    hole_width_range=(1, max_w),
                    fill=0,
                    fill_mask=None,
                    p=0.5,
                )
            )

    return A.Compose(ops)


def check_split_leakage_risk(df: pd.DataFrame, filename_col: str = "filename"):
    """
    Heuristic-only check: if filenames look like they encode a tile
    grid from a single contiguous scene (e.g. sequential indices or a
    row_col pattern) and are shuffled across splits without any city/
    region prefix, warn loudly. This cannot prove leakage, only flag
    the pattern that most commonly causes it -- confirm manually
    against how data_split.csv was actually generated.
    """
    names = df[filename_col].astype(str)
    has_region_prefix = names.str.contains(r"^[a-zA-Z]+_", regex=True).mean()
    if has_region_prefix < 0.5:
        warnings.warn(
            "Most filenames don't carry an obvious region/city prefix. "
            "If train/val/test were split by random shuffling of tiles "
            "from the SAME scene(s), you likely have spatial-autocorrelation "
            "leakage between splits (Phase 0, Part 7). Verify the split was "
            "made by city/district, not by tile index, before trusting "
            "these metrics as a generalization estimate.",
            stacklevel=2,
        )


class RoadDataset(Dataset):
    def __init__(self, dataframe, image_dir, mask_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]["filename"]

        image = np.array(
            Image.open(os.path.join(self.image_dir, filename)).convert("RGB"),
            dtype=np.uint8,
        )
        mask = np.array(
            Image.open(os.path.join(self.mask_dir, filename)).convert("L"),
            dtype=np.uint8,
        )

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        mask = (mask > 0).astype(np.float32)

        image = preprocess_rgb_image(image)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask


def load_splits(cfg):
    """Returns (train_df, val_df, test_df) and runs the leakage-risk check."""
    split_path = os.path.join(cfg.dataset_dir, cfg.split_csv)
    df = pd.read_csv(split_path)
    check_split_leakage_risk(df)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]
    return train_df, val_df, test_df


def build_datasets(cfg):
    image_dir = os.path.join(cfg.dataset_dir, cfg.image_subdir)
    mask_dir = os.path.join(cfg.dataset_dir, cfg.mask_subdir)
    train_df, val_df, test_df = load_splits(cfg)

    train_tf = build_transforms(cfg)  # augmentation ON for train
    train_ds = RoadDataset(train_df, image_dir, mask_dir, transform=train_tf)
    val_ds = RoadDataset(val_df, image_dir, mask_dir, transform=None)
    test_ds = RoadDataset(test_df, image_dir, mask_dir, transform=None)
    return train_ds, val_ds, test_ds
