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


def build_transforms(cfg):
    """
    One augmentation pipeline, shared by every model trained in a
    comparison run. `use_occlusion_augmentation` controls whether
    CoarseDropout (a stand-in for cloud/canopy/shadow occlusion) is
    included -- keep this ON for every model in an ablation unless the
    experiment is specifically "with vs without occlusion aug".
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
        max_h = max(1, int(cfg.image_size * cfg.occlusion_max_frac))
        max_w = max(1, int(cfg.image_size * cfg.occlusion_max_frac))
        ops.append(
            A.CoarseDropout(
                # Albumentations >= 2.0 API. Integer ranges are pixel sizes.
                num_holes_range=(1, cfg.occlusion_max_holes),
                hole_height_range=(1, max_h),
                hole_width_range=(1, max_w),
                fill=0,   # blacks out the patch, standing in for cloud/shadow
                fill_mask=None,  # IMPORTANT: leave the mask untouched --
                # we want the model to learn to recover the *hidden* road,
                # not to learn that occluded pixels mean "not a road"
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
