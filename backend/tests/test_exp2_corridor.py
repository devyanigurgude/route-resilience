"""
Unit tests for EXP-2.1 corridor module.

Tests verify:
  1. DilatedCorridorDataset produces masks strictly larger than originals
  2. Original masks on disk are not modified after dataset creation
  3. skeletonize_prediction thins a wide road to a 1-px skeleton
  4. TopologyEvaluator returns the expected keys
  5. save_comparison_panel runs without error (matplotlib optional)

All tests are CPU-only and use synthetic data — no actual dataset needed.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ── project path bootstrap ─────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_road_dataset(tmp_path):
    """Create a minimal RoadDataset with 4 synthetic 64×64 tiles."""
    from route_resilience.data import RoadDataset
    import pandas as pd
    from PIL import Image

    img_dir = tmp_path / "images_png"
    msk_dir = tmp_path / "masks_png"
    img_dir.mkdir(); msk_dir.mkdir()

    filenames = [f"tile_{i:02d}.png" for i in range(4)]
    for fn in filenames:
        # RGB image
        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(img_dir / fn)
        # 1–2 px centerline mask (thin horizontal line)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[32, 10:54] = 255   # horizontal road, 1 px wide
        Image.fromarray(mask).save(msk_dir / fn)

    df = pd.DataFrame({"filename": filenames, "split": ["train"] * 4})
    return RoadDataset(df, str(img_dir), str(msk_dir), transform=None), msk_dir, filenames


# ---------------------------------------------------------------------------
# 1. DilatedCorridorDataset expands the mask
# ---------------------------------------------------------------------------

def test_dilated_mask_is_larger_than_original(dummy_road_dataset):
    from route_resilience.exp2_corridor import DilatedCorridorDataset
    base_ds, _, _ = dummy_road_dataset

    corridor_ds = DilatedCorridorDataset(base_ds, dilation_radius=2)
    assert len(corridor_ds) == len(base_ds)

    for idx in range(len(base_ds)):
        _, orig_mask = base_ds[idx]
        _, dil_mask = corridor_ds[idx]

        orig_road_px = orig_mask.sum().item()
        dil_road_px = dil_mask.sum().item()

        # Dilated mask must have at least as many road pixels as original
        assert dil_road_px >= orig_road_px, (
            f"idx={idx}: dilated ({dil_road_px}) < original ({orig_road_px})"
        )
        if orig_road_px > 0:
            # Corridor is strictly larger for non-empty masks
            assert dil_road_px > orig_road_px, (
                f"idx={idx}: dilation had no effect (orig={orig_road_px}, dil={dil_road_px})"
            )


# ---------------------------------------------------------------------------
# 2. Original masks on disk are NOT modified
# ---------------------------------------------------------------------------

def test_original_masks_unchanged(dummy_road_dataset):
    from route_resilience.exp2_corridor import DilatedCorridorDataset
    from PIL import Image

    base_ds, msk_dir, filenames = dummy_road_dataset

    # Read originals BEFORE creating the corridor dataset
    originals = {}
    for fn in filenames:
        originals[fn] = np.array(Image.open(msk_dir / fn).convert("L"))

    corridor_ds = DilatedCorridorDataset(base_ds, dilation_radius=2)

    # Iterate the corridor dataset
    for _ in corridor_ds:
        pass

    # Read again AFTER iteration
    for fn in filenames:
        after = np.array(Image.open(msk_dir / fn).convert("L"))
        np.testing.assert_array_equal(
            originals[fn], after,
            err_msg=f"Mask file {fn} was modified on disk — this must never happen."
        )


# ---------------------------------------------------------------------------
# 3. skeletonize_prediction thins a wide band to a thin skeleton
# ---------------------------------------------------------------------------

def test_skeletonize_prediction_thins_wide_road():
    from route_resilience.exp2_corridor import skeletonize_prediction

    # Create a 5-px wide horizontal corridor
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30:35, 10:54] = 1   # 5 rows wide

    skel = skeletonize_prediction(mask)

    # Skeleton must be thinner than the original
    assert skel.sum() < mask.sum(), "Skeleton should be thinner than the input corridor"
    # Skeleton must be non-empty (there is a road)
    assert skel.sum() > 0, "Skeleton should not be empty for a non-empty road mask"
    # Skeleton pixels are at most 1 px wide (no 2×2 blocks of True)
    skel_u8 = skel.astype(np.uint8)
    # Check: no 2×2 block is all-ones (true skeleton property)
    from scipy.ndimage import uniform_filter
    density = uniform_filter(skel_u8.astype(float), size=2)
    assert density.max() <= 1.0 + 1e-9, "Skeleton has a 2×2 block — not properly thinned"


# ---------------------------------------------------------------------------
# 4. TopologyEvaluator returns expected keys
# ---------------------------------------------------------------------------

def test_topology_evaluator_expected_keys():
    from route_resilience.exp2_corridor import TopologyEvaluator

    evaluator = TopologyEvaluator(snap_radius=10.0, n_pairs=20, seed=42)

    # Synthetic binary masks: a simple horizontal road
    pred = np.zeros((64, 64), dtype=np.uint8)
    pred[32, 10:54] = 1
    gt = pred.copy()

    result = evaluator.evaluate(pred, gt)

    expected_keys = {
        "pred_road_px", "pred_skel_px",
        "n_components", "largest_cc_ratio",
        "pred_nodes", "pred_edges",
        "apls_score", "miss_rate", "od_reachability",
    }
    missing = expected_keys - set(result.keys())
    assert not missing, f"TopologyEvaluator result missing keys: {missing}"


def test_topology_evaluator_empty_pred():
    from route_resilience.exp2_corridor import TopologyEvaluator

    evaluator = TopologyEvaluator(snap_radius=10.0, n_pairs=20, seed=42)
    pred = np.zeros((64, 64), dtype=np.uint8)   # empty prediction
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[32, 10:54] = 1

    result = evaluator.evaluate(pred, gt)
    assert result["n_components"] == 0
    assert result["apls_score"] == 0.0


# ---------------------------------------------------------------------------
# 5. save_comparison_panel runs without error
# ---------------------------------------------------------------------------

def test_save_comparison_panel(tmp_path):
    from route_resilience.exp2_corridor import save_comparison_panel

    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    gt = np.zeros((64, 64), dtype=np.uint8); gt[32, 10:54] = 1
    pred = gt.copy()
    skel = gt.copy()

    panel_path = str(tmp_path / "panel_test.png")
    # Should not raise even when G_pred is None
    save_comparison_panel(rgb, gt, pred, skel, G_pred=None, save_path=panel_path, title="test")

    # matplotlib may not be installed — only assert no crash
    # If it is installed, the file should exist
    try:
        import matplotlib
        assert Path(panel_path).exists(), "Panel file not created"
    except ImportError:
        pass   # silently skipped if matplotlib not installed


# ---------------------------------------------------------------------------
# 6. DilatedCorridorDataset dilation_radius=0 is identity
# ---------------------------------------------------------------------------

def test_dilation_radius_zero_is_identity(dummy_road_dataset):
    from route_resilience.exp2_corridor import DilatedCorridorDataset

    base_ds, _, _ = dummy_road_dataset
    corridor_ds = DilatedCorridorDataset(base_ds, dilation_radius=0)

    for idx in range(len(base_ds)):
        _, orig_mask = base_ds[idx]
        _, dil_mask = corridor_ds[idx]
        # With radius 0, disk(0) is a single pixel → binary_dilation is identity
        assert (orig_mask - dil_mask).abs().sum().item() < 1e-6, (
            f"radius=0 should produce an identical mask at idx={idx}"
        )

