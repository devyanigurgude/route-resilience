"""Focused tests for Phase 3A Controlled Occlusion Simulation & Topology Matching.

Verifies:
  1. Synthetic occlusion patterns (rectangular, irregular, transverse) never mutate
     the ground-truth road masks.
  2. Ground-truth matching rules accurately distinguish true positive road connections
     from false/hallucinated connections.
  3. Connection precision, recall, and component deltas are correctly calculated.
"""

import networkx as nx
import numpy as np
import pytest

from route_resilience.occlusion_simulation import generate_road_centered_occlusions
from route_resilience.topology_matching import evaluate_connection_recovery


def test_controlled_occlusion_patterns_preserve_ground_truth():
    H, W = 100, 100
    image = np.full((H, W, 3), 120, dtype=np.uint8)
    gt_mask = np.zeros((H, W), dtype=np.uint8)
    # Draw horizontal road across the middle
    gt_mask[48:52, 10:90] = 255
    gt_copy = gt_mask.copy()

    for pattern in ["rectangular_block", "irregular_patch", "linear_transverse"]:
        occluded, occ_mask, meta = generate_road_centered_occlusions(
            image,
            gt_mask,
            pattern=pattern,
            num_occlusions=2,
            size_range=(14, 20),
            seed=123,
        )
        # 1. Ground truth mask MUST NOT be mutated
        assert np.array_equal(gt_mask, gt_copy), f"Pattern {pattern} mutated gt_mask!"

        # 2. Occlusion was placed on known road pixels
        assert meta["affected_road_pixels"] > 0
        assert occ_mask.sum() > 0

        # 3. Only occluded pixels in image were altered
        assert np.all(image[~occ_mask] == occluded[~occ_mask])
        assert not np.all(image[occ_mask] == occluded[occ_mask])


def test_topology_matching_distinguishes_tp_from_fp():
    H, W = 80, 80
    gt_mask = np.zeros((H, W), dtype=np.uint8)
    # A single continuous horizontal road at row 40
    gt_mask[39:42, 10:70] = 255

    G_gt = nx.Graph()
    G_gt.add_node("gt_a", pos=(40.0, 20.0))
    G_gt.add_node("gt_b", pos=(40.0, 50.0))
    G_gt.add_edge("gt_a", "gt_b", pts=np.array([[40.0, c] for c in range(20, 51)]))

    # G_unhealed has a gap between cols 30 and 40
    G_unhealed = nx.Graph()
    G_unhealed.add_node("u", pos=(40.0, 30.0))
    G_unhealed.add_node("u_inner", pos=(40.0, 20.0))
    G_unhealed.add_node("v", pos=(40.0, 40.0))
    G_unhealed.add_node("v_inner", pos=(40.0, 50.0))
    G_unhealed.add_edge("u", "u_inner", pts=np.array([[40.0, 20.0], [40.0, 30.0]]))
    G_unhealed.add_edge("v", "v_inner", pts=np.array([[40.0, 40.0], [40.0, 50.0]]))

    # Case A: Legitimate connection along row 40 (follows true road)
    G_healed_tp = G_unhealed.copy()
    tp_line = np.array([[40.0, float(c)] for c in range(30, 41)])
    G_healed_tp.add_edge("u", "v", healed=True, occlusion_recovered=True, pts=tp_line, length_px=10.0)

    eval_tp = evaluate_connection_recovery(
        G_unhealed=G_unhealed,
        G_healed=G_healed_tp,
        G_gt=G_gt,
        gt_mask=gt_mask,
        tolerance_radius_px=3.0,
        snap_radius_px=12.0,
    )
    assert eval_tp["total_recovered_edges"] == 1
    assert eval_tp["true_recovered_connections"] == 1
    assert eval_tp["false_recovered_connections"] == 0
    assert eval_tp["connection_recovery_precision"] == 1.0

    # Case B: Spurious / False connection cutting across empty background
    G_healed_fp = G_unhealed.copy()
    # Path bows up to row 10 (30 pixels away from ground truth road!)
    fp_line = np.array([[10.0, float(c)] for c in range(30, 41)])
    G_healed_fp.add_edge("u", "v", healed=True, occlusion_recovered=True, pts=fp_line, length_px=20.0)

    eval_fp = evaluate_connection_recovery(
        G_unhealed=G_unhealed,
        G_healed=G_healed_fp,
        G_gt=G_gt,
        gt_mask=gt_mask,
        tolerance_radius_px=3.0,
        snap_radius_px=12.0,
    )
    assert eval_fp["total_recovered_edges"] == 1
    assert eval_fp["true_recovered_connections"] == 0
    assert eval_fp["false_recovered_connections"] == 1
    assert eval_fp["connection_recovery_precision"] == 0.0


def test_topology_matching_reports_component_reduction():
    G_unhealed = nx.Graph()
    G_unhealed.add_edge(1, 2)
    G_unhealed.add_edge(3, 4)
    # 2 components

    G_healed = G_unhealed.copy()
    G_healed.add_edge(2, 3, healed=True, pts=np.array([[10, 10], [10, 20]]))
    # 1 component

    gt_mask = np.zeros((30, 30), dtype=np.uint8)
    gt_mask[10, 5:25] = 255

    res = evaluate_connection_recovery(
        G_unhealed=G_unhealed,
        G_healed=G_healed,
        G_gt=None,
        gt_mask=gt_mask,
    )
    assert res["unhealed_component_count"] == 2
    assert res["healed_component_count"] == 1
    assert res["component_reduction"] == 1
