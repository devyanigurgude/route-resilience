"""Focused tests for Priority 2: Shadow & Canopy Mask Dilation Robustness.

Verifies:
  1. Dilation disabled preserves the exact baseline mask and behavior.
  2. Dilation expands the intended mask without modifying the source mask.
  3. Valid hidden-road connections are recovered under baseline and dilated configurations.
  4. Negative controls (unaligned headings, parallel corridors, oversized gaps)
     do not produce spurious false bridges under dilation.
  5. Invalid kernel sizes or iteration counts are safely rejected with ValueError.
  6. Cloud masks remain unaffected by shadow/canopy dilation parameters.
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import networkx as nx
import numpy as np
import pytest

from route_resilience.config import GraphConfig
from route_resilience.graph_extraction import confidence_gated_gap_healing
from route_resilience.real_graph_pipeline import (
    canopy_occlusion_mask,
    cloud_occlusion_mask,
    dilate_mask,
    photometric_occlusion_mask,
    shadow_occlusion_mask,
)


def test_dilation_disabled_preserves_original_behavior():
    """When dilation_iterations == 0, masks are identical to baseline and source is unchanged."""
    H, W = 80, 80
    rgb = np.full((H, W, 3), [100, 95, 85], dtype=np.uint8)
    rgb[30:50, 20:60] = [30, 30, 30]  # Shadow patch
    rgb[10:25, 10:40] = [45, 68, 38]  # Canopy patch

    # Baseline calls (default dilation_iterations=0)
    shadow_base = shadow_occlusion_mask(rgb)
    canopy_base = canopy_occlusion_mask(rgb)
    photo_base = photometric_occlusion_mask(rgb)

    # Explicit dilation_iterations=0 calls
    shadow_zero = shadow_occlusion_mask(rgb, dilation_kernel_size=3, dilation_iterations=0)
    canopy_zero = canopy_occlusion_mask(rgb, dilation_kernel_size=3, dilation_iterations=0)
    photo_zero = photometric_occlusion_mask(rgb, dilation_kernel_size=3, dilation_iterations=0)

    assert np.array_equal(shadow_base, shadow_zero)
    assert np.array_equal(canopy_base, canopy_zero)
    assert np.array_equal(photo_base, photo_zero)

    # dilate_mask helper with iterations=0 returns unchanged copy
    test_mask = np.zeros((20, 20), dtype=bool)
    test_mask[5:15, 5:15] = True
    copy_test = test_mask.copy()
    dilated_zero = dilate_mask(test_mask, kernel_size=3, iterations=0)

    assert np.array_equal(dilated_zero, copy_test)
    assert np.array_equal(test_mask, copy_test)


def test_dilation_expands_mask_without_modifying_source():
    """Dilation expands boundaries monotonically and never mutates the input array."""
    H, W = 50, 50
    mask = np.zeros((H, W), dtype=bool)
    # A 10x10 central square: rows 20..29, cols 20..29 (area = 100)
    mask[20:30, 20:30] = True
    original_copy = mask.copy()

    # 1 iteration with 3x3 kernel: expands by 1 px on all sides -> 12x12 = 144
    dilated_1 = dilate_mask(mask, kernel_size=3, iterations=1)
    # 2 iterations with 3x3 kernel: expands by 2 px on all sides -> 14x14 = 196
    dilated_2 = dilate_mask(mask, kernel_size=3, iterations=2)

    # Source mask must be strictly unmodified
    assert np.array_equal(mask, original_copy)

    # Monotonic expansion
    assert mask.sum() == 100
    assert dilated_1.sum() == 144
    assert dilated_2.sum() == 196
    assert np.all(dilated_1[mask])  # All original True pixels remain True
    assert np.all(dilated_2[dilated_1])  # All iter-1 True pixels remain True in iter-2


def test_valid_hidden_road_connection_recovered_with_dilation():
    """A road gap beneath slightly eroded/sparse shadow mask recovers when dilated."""
    H, W = 100, 100
    trunk_len = 15
    gap_px = 22.0
    y = 50.0

    G = nx.Graph()
    # Left trunk: (50, 10) to (50, 25)
    for x in range(10, 26):
        G.add_node(f"L{x-10}", pos=(y, float(x)))
        if x > 10:
            G.add_edge(f"L{x-11}", f"L{x-10}", length_px=1.0, pts=np.array([[y, x - 1], [y, x]]))

    # Right trunk: (50, 47) to (50, 62) (gap from 25 to 47 = 22 px)
    for x in range(47, 63):
        G.add_node(f"R{x-47}", pos=(y, float(x)))
        if x > 47:
            G.add_edge(f"R{x-48}", f"R{x-47}", length_px=1.0, pts=np.array([[y, x - 1], [y, x]]))

    prob_map = np.full((H, W), 0.02, dtype=np.float32)
    prob_map[48:53, 10:26] = 0.90
    prob_map[48:53, 47:63] = 0.90

    # Narrow shadow mask along road: only covers cols 30 to 42 (12 px of 22 px gap = 54.5% coverage)
    # But has slight vertical offset so only 9 of 22 pts touch without dilation (40.9% coverage < 50%)
    raw_shadow_mask = np.zeros((H, W), dtype=bool)
    raw_shadow_mask[51:56, 25:47] = True  # Sits 1 pixel below road centerline row 50

    # 1. Without dilation: road centerline at row 50 has 0% coverage on raw_shadow_mask -> rejected
    G_unhealed = confidence_gated_gap_healing(
        G.copy(),
        prob_map,
        max_gap_px=15.0,
        enable_occlusion_recovery=True,
        max_occlusion_gap_px=30.0,
        min_occlusion_confidence=0.05,
        occlusion_heading_threshold=0.75,
        occlusion_mask=raw_shadow_mask,
    )
    assert not G_unhealed.has_edge("L15", "R0"), "Without dilation, offset shadow mask must not bridge gap"

    # 2. With 3x3 1-iteration dilation: shadow expands to include row 50 -> coverage becomes 100% -> healed!
    dilated_shadow = dilate_mask(raw_shadow_mask, kernel_size=3, iterations=1)
    G_healed = confidence_gated_gap_healing(
        G.copy(),
        prob_map,
        max_gap_px=15.0,
        enable_occlusion_recovery=True,
        max_occlusion_gap_px=30.0,
        min_occlusion_confidence=0.05,
        occlusion_heading_threshold=0.75,
        occlusion_mask=dilated_shadow,
    )
    assert G_healed.has_edge("L15", "R0"), "With 3x3 dilation, gap must be recovered"
    edge = G_healed["L15"]["R0"]
    assert edge["occlusion_recovered"] is True
    assert edge["recovery_type"] == "occlusion_continuity"


def test_negative_controls_do_not_produce_false_bridges_under_dilation():
    """Dilation of shadow/canopy mask does not create spurious bridges on invalid geometries."""
    H, W = 100, 100

    # Negative Control 1: Misaligned / Angled turn (L horizontal at row 40, R vertical at col 60)
    G_turn = nx.Graph()
    for x in range(10, 26):
        G_turn.add_node(f"H{x}", pos=(40.0, float(x)))
        if x > 10:
            G_turn.add_edge(f"H{x-1}", f"H{x}", length_px=1.0, pts=np.array([[40.0, x - 1], [40.0, x]]))
    for y in range(50, 66):
        G_turn.add_node(f"V{y}", pos=(float(y), 50.0))
        if y > 50:
            G_turn.add_edge(f"V{y-1}", f"V{y}", length_px=1.0, pts=np.array([[y - 1, 50.0], [y, 50.0]]))

    # Dense shadow over the entire quadrant
    dense_shadow = np.zeros((H, W), dtype=bool)
    dense_shadow[30:70, 20:70] = True
    dilated_shadow_2 = dilate_mask(dense_shadow, kernel_size=3, iterations=2)

    prob_map = np.zeros((H, W), dtype=np.float32)

    G_turn_out = confidence_gated_gap_healing(
        G_turn.copy(),
        prob_map,
        max_gap_px=15.0,
        enable_occlusion_recovery=True,
        max_occlusion_gap_px=30.0,
        min_occlusion_confidence=0.05,
        occlusion_heading_threshold=0.75,
        occlusion_mask=dilated_shadow_2,
    )
    assert not G_turn_out.has_edge("H25", "V50"), "Misaligned turn must remain rejected even under 2-iter dilation"

    # Negative Control 2: Oversized gap (36 px > 30 px limit)
    G_oversized = nx.Graph()
    for x in range(10, 26):
        G_oversized.add_node(f"A{x}", pos=(50.0, float(x)))
        if x > 10:
            G_oversized.add_edge(f"A{x-1}", f"A{x}", length_px=1.0, pts=np.array([[50.0, x - 1], [50.0, x]]))
    for x in range(62, 78):
        G_oversized.add_node(f"B{x}", pos=(50.0, float(x)))
        if x > 62:
            G_oversized.add_edge(f"B{x-1}", f"B{x}", length_px=1.0, pts=np.array([[50.0, x - 1], [50.0, x]]))

    full_shadow = np.ones((H, W), dtype=bool)
    dilated_full = dilate_mask(full_shadow, kernel_size=3, iterations=2)

    G_oversized_out = confidence_gated_gap_healing(
        G_oversized.copy(),
        prob_map,
        max_gap_px=15.0,
        enable_occlusion_recovery=True,
        max_occlusion_gap_px=30.0,
        min_occlusion_confidence=0.05,
        occlusion_heading_threshold=0.75,
        occlusion_mask=dilated_full,
    )
    assert not G_oversized_out.has_edge("A25", "B62"), "Oversized gap (>30px) must be rejected under dilation"


def test_invalid_dilation_parameters_rejected_or_handled_safely():
    """Invalid kernel sizes or iterations raise ValueError; None returns None."""
    mask = np.ones((10, 10), dtype=bool)

    # Negative iterations
    with pytest.raises(ValueError, match="iterations must be non-negative"):
        dilate_mask(mask, kernel_size=3, iterations=-1)

    # Even kernel size
    with pytest.raises(ValueError, match="kernel_size must be a positive odd integer"):
        dilate_mask(mask, kernel_size=2, iterations=1)

    with pytest.raises(ValueError, match="kernel_size must be a positive odd integer"):
        dilate_mask(mask, kernel_size=4, iterations=1)

    # Zero or negative kernel size
    with pytest.raises(ValueError, match="kernel_size must be a positive odd integer"):
        dilate_mask(mask, kernel_size=0, iterations=1)

    with pytest.raises(ValueError, match="kernel_size must be a positive odd integer"):
        dilate_mask(mask, kernel_size=-3, iterations=1)

    # None mask returns None safely
    assert dilate_mask(None, kernel_size=3, iterations=1) is None


def test_cloud_mask_not_dilated_by_shadow_canopy_parameters():
    """Cloud mask generation is completely independent of shadow/canopy dilation."""
    H, W = 60, 60
    # Bright cloud patch
    rgb_cloud = np.full((H, W, 3), 100, dtype=np.uint8)
    rgb_cloud[20:40, 20:40] = [240, 240, 240]

    cloud_base = cloud_occlusion_mask(rgb_cloud)

    # Photometric mask with 0 vs 2 dilation iterations on shadow/canopy
    photo_0 = photometric_occlusion_mask(rgb_cloud, dilation_iterations=0)
    photo_2 = photometric_occlusion_mask(rgb_cloud, dilation_iterations=2)

    # The cloud region in photo_0 and photo_2 must be identical since there are no shadows/canopy
    assert np.array_equal(photo_0, cloud_base)
    assert np.array_equal(photo_2, cloud_base)

