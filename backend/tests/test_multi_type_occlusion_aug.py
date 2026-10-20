"""Unit tests for MultiTypeOcclusionAugmentation in Route Resilience.

Verifies:
  1. MultiTypeOcclusionAugmentation strictly preserves the ground-truth mask untouched.
  2. MultiTypeOcclusionAugmentation outputs valid uint8 RGB images matching input dimensions.
  3. Deterministic seeding produces identical augmentations.
  4. Build transforms properly selects coarse_dropout vs multi_type.
"""

import numpy as np
import pytest

from route_resilience.config import Config
from route_resilience.data import MultiTypeOcclusionAugmentation, build_transforms


def test_multi_type_augmentation_preserves_mask():
    H, W = 128, 128
    image = np.full((H, W, 3), 100, dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[50:60, :] = 255  # horizontal road
    mask_copy = mask.copy()

    for occ_type in ["rectangular_dropout", "canopy_cluster", "procedural_cloud"]:
        aug = MultiTypeOcclusionAugmentation(
            num_holes_range=(2, 3),
            hole_size_range=(16, 24),
            occlusion_types=(occ_type,),
            type_weights=(1.0,),
            seed=42,
            p=1.0,
        )
        res = aug(image=image, mask=mask)
        # 1. Mask must be 100% identical and unmutated
        assert np.array_equal(res["mask"], mask_copy), f"Mask was mutated by {occ_type}!"
        # 2. Image must be altered
        assert not np.array_equal(res["image"], image), f"Image was not modified by {occ_type}!"
        # 3. Output shape and type valid
        assert res["image"].shape == (H, W, 3)
        assert res["image"].dtype == np.uint8


def test_deterministic_seeding():
    image = np.full((128, 128, 3), 120, dtype=np.uint8)
    aug1 = MultiTypeOcclusionAugmentation(seed=1234, p=1.0)
    aug2 = MultiTypeOcclusionAugmentation(seed=1234, p=1.0)

    res1 = aug1(image=image)
    res2 = aug2(image=image)

    assert np.array_equal(res1["image"], res2["image"]), "Identical seeds produced different augmentations!"


def test_build_transforms_wiring():
    cfg = Config()
    cfg.data.use_occlusion_augmentation = True
    cfg.data.occlusion_aug_type = "coarse_dropout"
    tf_coarse = build_transforms(cfg.data)
    has_coarse = any("CoarseDropout" in t.__class__.__name__ for t in tf_coarse.transforms)
    assert has_coarse, "CoarseDropout not present when occlusion_aug_type='coarse_dropout'"

    cfg.data.occlusion_aug_type = "multi_type"
    tf_multi = build_transforms(cfg.data)
    has_multi = any("MultiTypeOcclusionAugmentation" in t.__class__.__name__ for t in tf_multi.transforms)
    assert has_multi, "MultiTypeOcclusionAugmentation not present when occlusion_aug_type='multi_type'"

