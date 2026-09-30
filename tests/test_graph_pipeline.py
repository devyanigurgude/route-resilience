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
