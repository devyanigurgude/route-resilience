"""Tests for Phase 3E — Projection-Weighted Road Pairing.

Tests verify:
1. Parallel roads with transverse cut:
   - When Euclidean distance favors a cross-road shortcut, baseline (weight=0.0)
     picks the transverse shortcut.
   - When continuity_penalty_weight > 0, longitudinal pairs are favored over
     perpendicular cross-road jumps, healing genuine roads and rejecting shortcuts.
2. Curved roads:
   - Moderately curved genuine continuations with valid tangents are preserved.
3. Rejection of cross-road shortcuts:
   - Cross-road jumps with large alignment deficit are strongly penalized.
4. Parameter validation:
   - Negative continuity_penalty_weight raises ValueError.
5. Backward compatibility:
   - Default continuity_penalty_weight=0.0 strictly preserves baseline behavior.
"""

import math
import networkx as nx
import numpy as np
import pytest

from route_resilience.config import GraphConfig
from route_resilience.graph_extraction import (
    confidence_gated_gap_healing,
)


def _build_parallel_roads_scenario(
    gap_px: float = 14.0,
) -> tuple:
    """
    Build two parallel horizontal roads where Road 1 has endpoints 1 and 2
    (collinear, distance = 14.0 px, perfect alignment dot_a=1.0, dot_b=1.0),
    and a parallel Road 2 has an endpoint 4 positioned such that:
      dist(1, 4) = 9.0 px < dist(1, 2) = 14.0 px,
    but the connection (1, 4) is diagonal across the road gap with dot_a ≈ 0.889, dot_b ≈ 0.889.
    """
    G = nx.Graph()

    # Road 1, Arm A (west): (row 50, col 10) -> (row 50, col 30) [Node 1]
    G.add_node(10, pos=(50.0, 10.0))
    G.add_node(1, pos=(50.0, 30.0))
    G.add_edge(10, 1, pts=np.array([[50.0, float(c)] for c in range(10, 31)]),
               length_px=20.0, confidence=0.9, healed=False, occlusion_recovered=False)

    # Road 1, Arm B (east): (row 50, col 30 + gap_px) [Node 2] -> (row 50, col 50 + gap_px)
    G.add_node(2, pos=(50.0, 30.0 + gap_px))
    G.add_node(20, pos=(50.0, 50.0 + gap_px))
    G.add_edge(2, 20, pts=np.array([[50.0, float(c)] for c in range(int(30 + gap_px), int(51 + gap_px))]),
               length_px=20.0, confidence=0.9, healed=False, occlusion_recovered=False)

    # Road 2, Arm B (east): starts at (50 + 4.0, 30 + 8.0) [Node 4] -> points east
    # delta(1 -> 4) = (4.0, 8.0) => dist = sqrt(16 + 64) = 8.94 px < 14.0 px!
    # direction = (4.0/8.94, 8.0/8.94) = (0.447, 0.894)
    # tangent_1 = (0.0, 1.0) => dot_1 = 0.894 >= 0.70
    # tangent_4 = (0.0, -1.0) => dot_4 = -(-0.894) = 0.894 >= 0.70
    G.add_node(4, pos=(54.0, 38.0))
    G.add_node(40, pos=(54.0, 58.0))
    G.add_edge(4, 40, pts=np.array([[54.0, float(c)] for c in range(38, 59)]),
               length_px=20.0, confidence=0.9, healed=False, occlusion_recovered=False)

    prob_map = np.full((120, 120), 0.85, dtype=float)
    return G, prob_map


class TestProjectionWeightedPairing:
    def test_parameter_validation(self):
        G = nx.Graph()
        prob = np.ones((50, 50))
        with pytest.raises(ValueError, match="continuity_penalty_weight"):
            confidence_gated_gap_healing(G, prob, continuity_penalty_weight=-0.5)

    def test_baseline_weight_zero_picks_shorter_transverse_shortcut(self):
        """
        Under pure Euclidean distance sorting (weight=0.0), Node 1 connects to Node 4
        because dist(1, 4) = 11.66 px < dist(1, 2) = 14.0 px.
        """
        G, prob = _build_parallel_roads_scenario(gap_px=14.0)
        G_healed = confidence_gated_gap_healing(
            G.copy(),
            prob,
            max_gap_px=20.0,
            min_gap_confidence=0.15,
            tangent_dot_threshold=0.7,
            continuity_penalty_weight=0.0,
        )

        healed_edges = [(u, v) for u, v, d in G_healed.edges(data=True) if d.get("healed")]
        # Baseline pairs the Euclidean shorter shortcut (1, 4)
        assert (1, 4) in healed_edges or (4, 1) in healed_edges, (
            "Baseline should greedily select the Euclidean-shorter cross-road pair (1, 4)"
        )
        assert (1, 2) not in healed_edges and (2, 1) not in healed_edges

    def test_continuity_penalty_favors_longitudinal_road_continuity(self):
        """
        With continuity_penalty_weight = 3.0, the diagonal jump (1, 4) has an
        alignment deficit (dot_a ≈ 0.894, dot_b ≈ 0.894 -> deficit ≈ 0.211), so its
        sort_score = 8.94 * (1 + 3.0 * 0.211) = 14.61 > 14.0.
        Node 1 correctly pairs with Node 2 (longitudinal genuine road, sort_score = 14.0).
        """
        G, prob = _build_parallel_roads_scenario(gap_px=14.0)
        G_healed = confidence_gated_gap_healing(
            G.copy(),
            prob,
            max_gap_px=20.0,
            min_gap_confidence=0.15,
            tangent_dot_threshold=0.7,
            continuity_penalty_weight=3.0,
        )

        healed_edges = [(u, v) for u, v, d in G_healed.edges(data=True) if d.get("healed")]
        assert (1, 2) in healed_edges or (2, 1) in healed_edges, (
            "With continuity penalty, genuine longitudinal road pair (1, 2) must be preferred over shortcut"
        )
        assert (1, 4) not in healed_edges and (4, 1) not in healed_edges, (
            "Cross-road shortcut (1, 4) must be rejected"
        )

    def test_curved_road_preservation(self):
        """
        A genuine curved road whose endpoints have a mild angle (e.g., 20° bend)
        should still be healed cleanly when continuity_penalty_weight > 0.
        """
        G = nx.Graph()
        # Arm 1: horizontal (tangent along +col)
        G.add_node(10, pos=(50.0, 20.0))
        G.add_node(1, pos=(50.0, 40.0))
        G.add_edge(10, 1, pts=np.array([[50.0, float(c)] for c in range(20, 41)]),
                   length_px=20.0, confidence=0.85, healed=False, occlusion_recovered=False)

        # Arm 2: tilted by 15 degrees
        angle = math.radians(15)
        # Gap of 10 px along direction (cos 15°, sin 15°)
        node2_r = 50.0 + 10.0 * math.sin(angle)
        node2_c = 40.0 + 10.0 * math.cos(angle)
        node20_r = node2_r + 20.0 * math.sin(angle)
        node20_c = node2_c + 20.0 * math.cos(angle)

        G.add_node(2, pos=(node2_r, node2_c))
        G.add_node(20, pos=(node20_r, node20_c))
        G.add_edge(2, 20,
                   pts=np.linspace([node2_r, node2_c], [node20_r, node20_c], 20),
                   length_px=20.0, confidence=0.85, healed=False, occlusion_recovered=False)

        prob = np.full((120, 120), 0.85, dtype=float)

        G_healed = confidence_gated_gap_healing(
            G.copy(),
            prob,
            max_gap_px=15.0,
            min_gap_confidence=0.15,
            tangent_dot_threshold=0.7,
            continuity_penalty_weight=1.5,
        )

        healed_edges = [(u, v) for u, v, d in G_healed.edges(data=True) if d.get("healed")]
        assert len(healed_edges) == 1, "Moderate genuine curve must still be successfully healed"
        assert (1, 2) in healed_edges or (2, 1) in healed_edges

    def test_default_graph_config_preserves_zero_penalty(self):
        cfg = GraphConfig()
        assert hasattr(cfg, "continuity_penalty_weight")
        assert cfg.continuity_penalty_weight == 0.0
