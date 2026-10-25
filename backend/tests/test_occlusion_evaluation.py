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


def test_graph_from_probability_map_with_occluded_rgb_exercises_recovery():
    """Verify passing occluded RGB image enables photometric detection and gap recovery."""
    from route_resilience.config import GraphConfig
    from route_resilience.real_graph_pipeline import (
        attach_width,
        clean_road_mask,
        cloud_occlusion_mask,
        mask_to_skeleton,
        photometric_occlusion_mask,
        skeleton_to_graph,
    )
    from route_resilience.graph_extraction import confidence_gated_gap_healing

    H, W = 100, 100
    prob_map = np.full((H, W), 0.02, dtype=np.float32)
    # Horizontal road: row 50, left branch cols 10..35, right branch cols 55..80 (gap 20 px)
    prob_map[50, 10:36] = 0.90
    prob_map[50, 55:81] = 0.90

    # Build an RGB image with dark shadow [30, 30, 30] over the gap corridor
    rgb_image = np.full((H, W, 3), [100, 95, 85], dtype=np.uint8)
    rgb_image[45:55, 30:60] = [30, 30, 30]

    assert rgb_image.shape == (H, W, 3)
    assert rgb_image.dtype == np.uint8

    graph_cfg = GraphConfig(
        max_gap_px=15.0,
        min_gap_confidence=0.15,
        enable_occlusion_recovery=True,
        max_occlusion_gap_px=30.0,
        min_occlusion_confidence=0.05,
        occlusion_heading_threshold=0.75,
    )

    bin_mask = clean_road_mask(prob_map >= 0.50, min_component_pixels=3)
    skel = mask_to_skeleton(bin_mask)
    G_before = skeleton_to_graph(skel, prob_map=prob_map, min_edge_length_px=graph_cfg.min_edge_length_px)
    G_before = attach_width(G_before, bin_mask)

    p_mask = photometric_occlusion_mask(rgb_image)
    c_mask = cloud_occlusion_mask(rgb_image)

    # 1. Without occlusion mask (old defect in run_occlusion_experiment): 20px zero-evidence gap is rejected
    G_without_mask = confidence_gated_gap_healing(
        G_before.copy(),
        prob_map,
        max_gap_px=graph_cfg.max_gap_px,
        min_gap_confidence=graph_cfg.min_gap_confidence,
        enable_occlusion_recovery=True,
        max_occlusion_gap_px=30.0,
        min_occlusion_confidence=0.05,
        occlusion_heading_threshold=0.75,
        occlusion_mask=None,
        cloud_mask=None,
    )
    unhealed_edges = [d for _, _, d in G_without_mask.edges(data=True) if d.get("healed")]
    assert len(unhealed_edges) == 0, "Without photometric occlusion mask, gap must not be healed"

    # 2. With occlusion mask (fixed run_occlusion_experiment path): photometric mask enables occlusion continuity
    G_with_mask = confidence_gated_gap_healing(
        G_before.copy(),
        prob_map,
        max_gap_px=graph_cfg.max_gap_px,
        min_gap_confidence=graph_cfg.min_gap_confidence,
        enable_occlusion_recovery=True,
        max_occlusion_gap_px=30.0,
        min_occlusion_confidence=0.05,
        occlusion_heading_threshold=0.75,
        occlusion_mask=p_mask,
        cloud_mask=c_mask,
    )
    healed_edges = [d for _, _, d in G_with_mask.edges(data=True) if d.get("healed")]
    assert len(healed_edges) == 1, "With photometric occlusion mask, gap must be recovered"
    assert healed_edges[0]["occlusion_recovered"] is True
    assert healed_edges[0]["recovery_type"] == "occlusion_continuity"


def test_phase2_benchmark_safe_output_dir_resolution(tmp_path):
    """Verify run_phase2_benchmark output directory handling prevents historical overwrite."""
    import sys
    from pathlib import Path
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import run_phase2_benchmark

    dummy_dir = tmp_path / "test_benchmark_dir"
    dummy_dir.mkdir(parents=True)
    summary_file = dummy_dir / "benchmark_summary.json"
    summary_file.write_text("{\"historical\": true}", encoding="utf-8")

    # Calling run_benchmark targeting an existing summary file with overwrite=False must raise FileExistsError
    with pytest.raises(FileExistsError, match="Historical benchmark results already exist"):
        run_phase2_benchmark.run_benchmark(
            dataset_dir=Path("dummy"),
            split_csv=Path("dummy.csv"),
            unet_checkpoint=Path("dummy.pth"),
            segformer_checkpoint=None,
            sample_size=1,
            output_dir=dummy_dir,
            overwrite=False,
        )