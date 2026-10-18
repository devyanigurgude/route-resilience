"""
Regression tests for the no-GPU-required half of the pipeline. Run with:
    pytest tests/test_graph_pipeline.py -v

These codify the adversarial checks done manually while building this
(most importantly: the confidence gate must refuse an unsupported gap).
If you change graph_extraction.py, run these before trusting output
again -- a passing "it ran" is not the same as "it's still correct".
"""

import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from route_resilience.criticality import add_capacity_weight, compute_centrality, detour_cost
from route_resilience.disruption import flood_conditioned_removal, simulate_removal
from route_resilience.graph_extraction import (
    confidence_gated_gap_healing,
    connectivity_ratio,
    mask_to_skeleton,
    skeleton_to_graph,
)


def test_connectivity_ratio_detects_fragmentation():
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 10:90] = True
    skel_whole = mask_to_skeleton(mask)
    assert connectivity_ratio(skel_whole) == 1.0

    mask[50, 45:55] = False  # punch a gap
    skel_gapped = mask_to_skeleton(mask)
    assert connectivity_ratio(skel_gapped) < 1.0


def test_gap_healing_accepts_real_evidence():
    mask = np.zeros((120, 120), dtype=bool)
    mask[60, 8:112] = True
    mask[55:65, 60:68] = False  # occlusion gap
    prob_map = np.where(mask, 0.9, 0.05).astype(float)
    prob_map[55:65, 60:68] = 0.25  # real, weak evidence in the gap

    skel = mask_to_skeleton(mask)
    G = skeleton_to_graph(skel, prob_map=prob_map, min_edge_length_px=3.0)
    assert nx.number_connected_components(G) > 1

    G = confidence_gated_gap_healing(G, prob_map, max_gap_px=20, min_gap_confidence=0.15)
    assert nx.number_connected_components(G) == 1


def test_gap_healing_refuses_unsupported_gap():
    """The critical regression test: no evidence -> no healing, ever."""
    mask = np.zeros((120, 120), dtype=bool)
    mask[30, 10:40] = True
    mask[30, 45:80] = True  # disjoint, no bridging evidence
    prob_map = np.where(mask, 0.9, 0.02).astype(float)

    skel = mask_to_skeleton(mask)
    G = skeleton_to_graph(skel, prob_map=prob_map, min_edge_length_px=3.0)
    n_components_before = nx.number_connected_components(G)
    assert n_components_before > 1

    G = confidence_gated_gap_healing(G, prob_map, max_gap_px=20, min_gap_confidence=0.15)
    assert nx.number_connected_components(G) == n_components_before, (
        "Gate healed a gap with no supporting evidence -- this must never happen."
    )


def test_gap_healing_refuses_strong_probability_without_aligned_road_geometry():
    G = nx.Graph()
    G.add_node("a", pos=(20.0, 20.0))
    G.add_node("a_inner", pos=(20.0, 5.0))
    G.add_node("b", pos=(26.0, 25.0))
    G.add_node("b_inner", pos=(26.0, 40.0))
    G.add_edge("a", "a_inner", pts=np.array([[20, 5], [20, 20]]), length_px=15.0)
    G.add_edge("b", "b_inner", pts=np.array([[26, 25], [26, 40]]), length_px=15.0)
    probability = np.full((64, 64), 0.9, dtype=np.float32)

    healed = confidence_gated_gap_healing(
        G, probability, max_gap_px=12.0, min_gap_confidence=0.15
    )

    assert nx.number_connected_components(healed) == 2
    assert not healed.has_edge("a", "b")


def test_gap_healing_requires_support_across_most_of_the_gap():
    G = nx.Graph()
    G.add_node("left", pos=(20.0, 10.0))
    G.add_node("left_inner", pos=(20.0, 5.0))
    G.add_node("right", pos=(20.0, 25.0))
    G.add_node("right_inner", pos=(20.0, 30.0))
    G.add_edge("left", "left_inner", pts=np.array([[20, 5], [20, 10]]), length_px=5.0)
    G.add_edge("right", "right_inner", pts=np.array([[20, 30], [20, 25]]), length_px=5.0)
    probability = np.full((48, 48), 0.02, dtype=np.float32)
    probability[20, 16:19] = 0.9

    healed = confidence_gated_gap_healing(
        G, probability, max_gap_px=20.0, min_gap_confidence=0.15
    )

    assert nx.number_connected_components(healed) == 2
    assert not healed.has_edge("left", "right")


def test_gap_healing_does_not_add_shortcuts_inside_an_existing_component():
    G = nx.Graph()
    positions = {
        "left": (20.0, 20.0),
        "left_inner": (20.0, 5.0),
        "left_turn": (50.0, 5.0),
        "right_turn": (50.0, 40.0),
        "right_inner": (20.0, 40.0),
        "right": (20.0, 25.0),
    }
    for node, pos in positions.items():
        G.add_node(node, pos=pos)
    for source, target in [
        ("left", "left_inner"),
        ("left_inner", "left_turn"),
        ("left_turn", "right_turn"),
        ("right_turn", "right_inner"),
        ("right_inner", "right"),
    ]:
        G.add_edge(source, target, length_px=10.0)
    probability = np.full((64, 64), 0.9, dtype=np.float32)

    healed = confidence_gated_gap_healing(
        G, probability, max_gap_px=8.0, min_gap_confidence=0.15
    )

    assert nx.number_connected_components(healed) == 1
    assert not healed.has_edge("left", "right")


def test_bottleneck_edge_has_highest_betweenness_and_infinite_detour():
    G = nx.Graph()
    G.add_edge("A", "B", length_px=10, width_px=6)
    G.add_edge("B", "C", length_px=10, width_px=6)
    G.add_edge("C", "D", length_px=25, width_px=3)  # sole bridge
    G.add_edge("D", "E", length_px=10, width_px=6)
    G.add_edge("E", "F", length_px=10, width_px=6)

    cen = compute_centrality(G)
    bottleneck_score = cen["betweenness"]["C"]
    assert bottleneck_score == max(cen["betweenness"].values())
    assert detour_cost(G, "C", "D") == float("inf")


def test_disruption_removal_reports_disconnection():
    G = nx.Graph()
    G.add_edge("A", "B", length_px=10)
    G.add_edge("B", "C", length_px=10)  # sole connector

    result = simulate_removal(G, edges_to_remove=[("B", "C")])
    assert result["after_components"] > result["before_components"]


def test_flood_conditioned_removal_only_affects_low_elevation():
    G = nx.Graph()
    for u, v in [("A", "B"), ("B", "C"), ("C", "D")]:
        G.add_edge(u, v, length_px=10)

    elevations = {"A": 50, "B": 48, "C": 5, "D": 3}  # C-D low-lying
    result = flood_conditioned_removal(G, elevations, flood_level=10)
    assert result["affected_edge_count"] >= 1
    assert result["after_components"] >= result["before_components"]


def test_occlusion_recovery_reconnects_curved_gap():
    """Fallback probability-guided search must recover curved paths across occluded gaps."""
    G = nx.Graph()
    G.add_node("left", pos=(40.0, 20.0))
    G.add_node("left_inner", pos=(40.0, 10.0))
    G.add_node("right", pos=(40.0, 36.0))
    G.add_node("right_inner", pos=(40.0, 46.0))
    G.add_edge("left", "left_inner", pts=np.array([[40.0, 10.0], [40.0, 20.0]]), length_px=10.0)
    G.add_edge("right", "right_inner", pts=np.array([[40.0, 46.0], [40.0, 36.0]]), length_px=10.0)

    # Straight line chord between (40, 20) and (40, 36) is 16 px.
    # Background and straight-line chord have low probability (0.02),
    # but a curved road arc bows upward between cols 20 and 36.
    prob_map = np.full((80, 80), 0.02, dtype=np.float32)
    for col in range(20, 37):
        row = int(round(40 + 6 * np.sin((col - 20) / 16.0 * np.pi)))
        prob_map[row - 1:row + 2, col] = 0.85

    assert nx.number_connected_components(G) == 2

    healed = confidence_gated_gap_healing(G, prob_map, max_gap_px=20.0, min_gap_confidence=0.15)

    assert nx.number_connected_components(healed) == 1
    assert healed.has_edge("left", "right")

    edge_data = healed.edges["left", "right"]
    assert edge_data["healed"] is True
    assert edge_data["occlusion_recovered"] is True
    assert edge_data["confidence"] >= 0.15
    assert edge_data["length_px"] > 16.0
    assert len(edge_data["pts"]) > 2


def test_occlusion_recovery_rejects_unsupported_gap():
    """Fallback search must still refuse gaps that have no probability support."""
    G = nx.Graph()
    G.add_node("left", pos=(40.0, 20.0))
    G.add_node("left_inner", pos=(40.0, 10.0))
    G.add_node("right", pos=(40.0, 36.0))
    G.add_node("right_inner", pos=(40.0, 46.0))
    G.add_edge("left", "left_inner", pts=np.array([[40.0, 10.0], [40.0, 20.0]]), length_px=10.0)
    G.add_edge("right", "right_inner", pts=np.array([[40.0, 46.0], [40.0, 36.0]]), length_px=10.0)

    prob_map = np.full((80, 80), 0.02, dtype=np.float32)

    healed = confidence_gated_gap_healing(G, prob_map, max_gap_px=20.0, min_gap_confidence=0.15)

    assert nx.number_connected_components(healed) == 2
    assert not healed.has_edge("left", "right")


def test_occlusion_recovery_boundary_handling():
    """Path search must cleanly handle endpoints adjacent to image boundaries without IndexError."""
    G = nx.Graph()
    G.add_node("b_left", pos=(0.0, 5.0))
    G.add_node("b_left_inner", pos=(0.0, 1.0))
    G.add_node("b_right", pos=(0.0, 15.0))
    G.add_node("b_right_inner", pos=(0.0, 19.0))
    G.add_edge("b_left", "b_left_inner", pts=np.array([[0.0, 1.0], [0.0, 5.0]]), length_px=4.0)
    G.add_edge("b_right", "b_right_inner", pts=np.array([[0.0, 19.0], [0.0, 15.0]]), length_px=4.0)

    prob_map = np.full((32, 32), 0.02, dtype=np.float32)
    prob_map[0:2, 5:16] = 0.85

    healed = confidence_gated_gap_healing(G, prob_map, max_gap_px=15.0, min_gap_confidence=0.15)
    assert nx.number_connected_components(healed) == 1
    assert healed.has_edge("b_left", "b_right")


def test_occlusion_recovery_preserves_shortcut_refusal():
    """Curved or occluded paths must never bridge nodes that already have connectivity."""
    G = nx.Graph()
    positions = {
        "left": (20.0, 20.0),
        "left_inner": (20.0, 5.0),
        "loop_top": (60.0, 5.0),
        "loop_bottom": (60.0, 35.0),
        "right_inner": (20.0, 35.0),
        "right": (20.0, 28.0),
    }
    for n, pos in positions.items():
        G.add_node(n, pos=pos)
    for u, v in [
        ("left", "left_inner"),
        ("left_inner", "loop_top"),
        ("loop_top", "loop_bottom"),
        ("loop_bottom", "right_inner"),
        ("right_inner", "right"),
    ]:
        G.add_edge(u, v, length_px=15.0)

    prob_map = np.full((70, 70), 0.02, dtype=np.float32)
    for col in range(20, 29):
        prob_map[22:25, col] = 0.95

    assert nx.has_path(G, "left", "right")
    healed = confidence_gated_gap_healing(G, prob_map, max_gap_px=12.0, min_gap_confidence=0.15)

    assert not healed.has_edge("left", "right")


if __name__ == "__main__":
    # allow `python tests/test_graph_pipeline.py` without pytest installed
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
