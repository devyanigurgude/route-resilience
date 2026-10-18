"""Unit and regression tests for Occlusion-Aware Road Detection & Continuity Recovery.

Verifies:
1. Candidate occlusion detection identifying facing degree-1 endpoints across gaps.
2. Positive recovery of collinear roads across low-confidence canopy/shadow gaps (P < 0.15, gap > 15px).
3. Explicit baseline safety: when enable_occlusion_recovery=False, large gaps are not healed.
4. Rejection of false parallel bridges, perpendicular shortcuts, floating noise specks, and already-connected paths.

Real function signatures (verified from graph_extraction.py):
- detect_candidate_occlusions(G, max_gap_px=30.0, heading_threshold=0.70, max_lateral_offset_px=4.5, min_trunk_len=6.0)
  - Reads G.nodes[n]["pos"] and edge "pts" / "length_px" attributes
  - Returns list[dict] with keys: source, target, pos_source, pos_target, gap_px, heading_alignment, mutual_heading, lateral_offset_px, status, recovered
- confidence_gated_gap_healing(G, prob_map, max_gap_px=15.0, min_gap_confidence=0.15, ..., enable_occlusion_recovery=True, max_occlusion_gap_px=30.0, ...)
  - Flat keyword args — no GraphConfig wrapper
  - Reads G.nodes[n]["pos"] and edge "pts" (ndarray) / "length_px" attributes
  - Writes G.graph["candidate_occlusions"] and edge healed=True, occlusion_recovered=True attrs
"""

import networkx as nx
import numpy as np
import pytest

from route_resilience.config import GraphConfig
from route_resilience.graph_extraction import (
    confidence_gated_gap_healing,
    detect_candidate_occlusions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_horizontal_segment(G, prefix: str, y: float, x_start: float, n_nodes: int, step: float = 1.0):
    """Add a horizontal road segment to G with correct node (pos=) and edge (pts, length_px) attributes."""
    pts_list = []
    for i in range(n_nodes):
        x = x_start + i * step
        G.add_node(f"{prefix}{i}", pos=(y, x))
        pts_list.append((y, x))
        if i > 0:
            prev_x = x - step
            G.add_edge(
                f"{prefix}{i-1}",
                f"{prefix}{i}",
                length_px=step,
                pts=np.array([(y, prev_x), (y, x)]),
            )
    return f"{prefix}0", f"{prefix}{n_nodes - 1}"


def _build_straight_road_pair(gap_px: float = 20.0, trunk_len: int = 15):
    """Two horizontal road segments facing each other across a gap.

    Left segment: y=50, x from 10 to 10+trunk_len
    Right segment: y=50, x from 10+trunk_len+gap_px to end
    Endpoints facing each other: L{trunk_len} and R0
    """
    G = nx.Graph()
    y = 50.0
    left_start, left_end = _add_horizontal_segment(G, "L", y, 10.0, trunk_len + 1)
    right_x0 = 10.0 + trunk_len + gap_px
    right_start, right_end = _add_horizontal_segment(G, "R", y, right_x0, trunk_len + 1)
    return G, left_end, right_start  # facing tips


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectCandidateOcclusions:
    def test_finds_facing_collinear_endpoints(self):
        """Detects collinear, facing degree-1 endpoints across a 20px gap."""
        G, left_tip, right_tip = _build_straight_road_pair(gap_px=20.0, trunk_len=15)

        candidates = detect_candidate_occlusions(
            G,
            max_gap_px=30.0,
            heading_threshold=0.70,
            max_lateral_offset_px=4.5,
            min_trunk_len=6.0,
        )

        assert len(candidates) == 1, f"Expected 1 candidate, got {len(candidates)}"
        cand = candidates[0]
        assert cand["status"] == "candidate"
        assert {cand["source"], cand["target"]} == {left_tip, right_tip}
        assert cand["gap_px"] == pytest.approx(20.0, abs=0.5)
        assert cand["heading_alignment"] > 0.85
        assert cand["lateral_offset_px"] < 1.0

    def test_already_connected_shortcut_rejected(self):
        """Endpoints already reachable from each other must not form candidates."""
        G, left_tip, right_tip = _build_straight_road_pair(gap_px=20.0, trunk_len=15)
        # Connect via a back-route
        G.add_node("BACK0", pos=(50.0, 10.0))
        G.add_node("BACK1", pos=(50.0, 60.0))
        G.add_edge(left_tip, "BACK0", length_px=5.0, pts=np.array([(50.0, 24.0), (50.0, 10.0)]))
        G.add_edge("BACK0", "BACK1", length_px=50.0, pts=np.array([(50.0, 10.0), (50.0, 60.0)]))
        G.add_edge("BACK1", right_tip, length_px=5.0, pts=np.array([(50.0, 60.0), (50.0, 50.0)]))
        assert nx.has_path(G, left_tip, right_tip)

        candidates = detect_candidate_occlusions(G, max_gap_px=30.0)
        # Must reject because path already exists
        assert len(candidates) == 0

    def test_floating_noise_specks_rejected(self):
        """Specks shorter than min_trunk_len (6.0px) must be rejected."""
        G = nx.Graph()
        # Main road: y=50, length 20px
        _add_horizontal_segment(G, "M", 50.0, 10.0, 6, step=4.0)  # 20px total

        # Tiny speck: 2px long at y=50, 15px gap from main road
        G.add_node("S0", pos=(50.0, 45.0))
        G.add_node("S1", pos=(50.0, 47.0))
        G.add_edge("S0", "S1", length_px=2.0, pts=np.array([(50.0, 45.0), (50.0, 47.0)]))

        candidates = detect_candidate_occlusions(G, max_gap_px=30.0, min_trunk_len=6.0)
        # Speck trunk length 2.0px < 6.0px min → rejected
        assert len(candidates) == 0

    def test_parallel_roads_rejected(self):
        """Parallel road endpoints across a lateral gap must NOT be candidates (tangent ⊥ gap vec)."""
        G = nx.Graph()
        # Road 1: y=30, x from 10 to 40 (horizontal)
        _add_horizontal_segment(G, "A", 30.0, 10.0, 11, step=3.0)
        # Road 2: y=45 (15px lateral gap), x from 10 to 40 (parallel horizontal)
        _add_horizontal_segment(G, "B", 45.0, 10.0, 11, step=3.0)

        candidates = detect_candidate_occlusions(G, max_gap_px=30.0, heading_threshold=0.70)
        # Gap vector is along y-axis; road tangents are along x-axis: dot ≈ 0 < 0.70 → rejected
        assert len(candidates) == 0

    def test_perpendicular_crossroads_rejected(self):
        """Perpendicular road endpoints must not form candidates (mutual heading ≈ 0 < 0.50)."""
        G = nx.Graph()
        # Horizontal road: y=50, x from 10 to 30 (right-facing tip at (50,30))
        _add_horizontal_segment(G, "H", 50.0, 10.0, 11, step=2.0)
        # Vertical road: x=45 (15px gap), y from 50 to 80 (upper tip at (50,45))
        for i in range(11):
            y = 50.0 + i * 3.0
            G.add_node(f"V{i}", pos=(y, 45.0))
            if i > 0:
                G.add_edge(
                    f"V{i-1}", f"V{i}",
                    length_px=3.0,
                    pts=np.array([(y - 3.0, 45.0), (y, 45.0)]),
                )

        candidates = detect_candidate_occlusions(G, max_gap_px=30.0)
        # H's right tangent ≈ (+1,0); V's up tangent ≈ (-1,0); mutual_heading ≈ -1 < 0.50 → rejected
        assert len(candidates) == 0


class TestOcclusionAwareRecovery:
    def test_occlusion_recovery_is_enabled_by_default(self):
        """Production extraction enables the evidence-gated continuity pass."""
        assert GraphConfig().enable_occlusion_recovery is True

    def test_reconnects_low_confidence_canopy_gap(self):
        """Occlusion recovery reconnects facing collinear roads across a 20px gap with P=0.08."""
        G, left_tip, right_tip = _build_straight_road_pair(gap_px=20.0, trunk_len=15)
        assert nx.number_connected_components(G) == 2

        # Low-confidence probability map in the gap (typical for dense canopy / cloud shadow)
        H, W = 100, 100
        prob_map = np.full((H, W), 0.02, dtype=np.float32)
        prob_map[48:53, 10:26] = 0.90   # left trunk
        prob_map[48:53, 26:46] = 0.08   # 20px gap (below standard P=0.15 threshold)
        prob_map[48:53, 46:70] = 0.90   # right trunk

        G_healed = confidence_gated_gap_healing(
            G,
            prob_map,
            max_gap_px=15.0,               # Standard threshold: 15px → gap of 20px fails
            min_gap_confidence=0.15,       # Standard P threshold: 15% → gap P=0.08 fails
            enable_occlusion_recovery=True,
            max_occlusion_gap_px=30.0,     # Occlusion allows up to 30px
            min_occlusion_confidence=0.05, # Occlusion allows as low as P=0.05
            occlusion_heading_threshold=0.75,
            max_lateral_offset_px=4.0,
        )

        # Must be reconnected into 1 component
        assert nx.number_connected_components(G_healed) == 1
        assert G_healed.has_edge(left_tip, right_tip)

        edge_data = G_healed[left_tip][right_tip]
        assert edge_data.get("healed") is True
        assert edge_data.get("occlusion_recovered") is True
        assert edge_data.get("inferred") is True
        assert edge_data.get("occlusion_suspected") is True
        assert edge_data.get("recovery_type") == "occlusion_continuity"

    def test_candidate_occlusions_stored_in_graph_attr(self):
        """After healing, G.graph['candidate_occlusions'] lists the recovered candidates."""
        G, left_tip, right_tip = _build_straight_road_pair(gap_px=20.0, trunk_len=15)
        prob_map = np.full((100, 100), 0.06, dtype=np.float32)
        prob_map[48:53, 10:70] = 0.08  # low confidence corridor throughout

        G_healed = confidence_gated_gap_healing(
            G, prob_map,
            max_gap_px=15.0,
            enable_occlusion_recovery=True,
            max_occlusion_gap_px=30.0,
            min_occlusion_confidence=0.05,
            occlusion_heading_threshold=0.75,
            max_lateral_offset_px=4.0,
        )

        cands = G_healed.graph.get("candidate_occlusions", [])
        assert len(cands) >= 1
        # At least one candidate should be recovered if healing succeeded
        statuses = {c["status"] for c in cands}
        assert "recovered" in statuses or nx.number_connected_components(G_healed) == 1
        assert all(c.get("suspected_occlusion") is True for c in cands)

    def test_baseline_does_not_heal_large_gap(self):
        """When enable_occlusion_recovery=False, 20px gaps remain unhealed."""
        G, left_tip, right_tip = _build_straight_road_pair(gap_px=20.0, trunk_len=15)
        prob_map = np.full((100, 100), 0.02, dtype=np.float32)
        prob_map[48:53, 10:70] = 0.90  # high confidence but gap is 20px > max_gap_px=15

        G_healed = confidence_gated_gap_healing(
            G,
            prob_map,
            max_gap_px=15.0,           # 20px gap exceeds this
            min_gap_confidence=0.15,
            enable_occlusion_recovery=False,  # Baseline mode
        )

        # Must remain disconnected: standard healing cannot bridge 20px gap
        assert nx.number_connected_components(G_healed) == 2
        assert not G_healed.has_edge(left_tip, right_tip)

    def test_baseline_heals_small_high_confidence_gap(self):
        """Standard healing still works for a 10px gap with P >= 0.15."""
        G, left_tip, right_tip = _build_straight_road_pair(gap_px=10.0, trunk_len=15)
        prob_map = np.full((100, 100), 0.5, dtype=np.float32)

        G_healed = confidence_gated_gap_healing(
            G, prob_map,
            max_gap_px=15.0,
            min_gap_confidence=0.15,
            enable_occlusion_recovery=False,
        )

        assert nx.number_connected_components(G_healed) == 1
        assert G_healed.has_edge(left_tip, right_tip)
        # Standard healed edges should NOT be marked as occlusion_recovered
        assert not G_healed[left_tip][right_tip].get("occlusion_recovered", False)

    def test_cloud_gap_recovers_smooth_long_continuation_and_preserves_coordinates(self):
        """A cloud-covered bend reconnects only through a tangent-continuous cloud corridor."""
        graph = nx.Graph()
        # The two exposed road tips model the attached image: a long diagonal
        # interruption with the road entering and leaving on a gentle bend.
        start = np.array([110.0, 70.0])
        end = np.array([70.0, 190.0])
        tangent_start = np.array([-0.5, 0.8660254])
        tangent_end = np.array([0.5, -0.8660254])
        graph.add_node("left_tip", pos=start)
        graph.add_node("left_trunk", pos=start - 32.0 * tangent_start)
        graph.add_edge(
            "left_trunk", "left_tip", length_px=32.0,
            pts=np.array([start - 32.0 * tangent_start, start]),
        )
        graph.add_node("right_tip", pos=end)
        graph.add_node("right_trunk", pos=end - 32.0 * tangent_end)
        graph.add_edge(
            "right_trunk", "right_tip", length_px=32.0,
            pts=np.array([end - 32.0 * tangent_end, end]),
        )

        probability = np.zeros((180, 280), dtype=np.float32)
        cloud = np.zeros_like(probability, dtype=bool)
        cloud[45:135, 55:215] = True

        healed = confidence_gated_gap_healing(
            graph,
            probability,
            max_gap_px=30.0,
            enable_occlusion_recovery=True,
            max_occlusion_gap_px=30.0,
            occlusion_mask=cloud,
            enable_cloud_occlusion_recovery=True,
            max_cloud_occlusion_gap_px=180.0,
            cloud_heading_threshold=0.78,
            cloud_min_component_len=20.0,
            cloud_min_coverage=0.70,
        )

        assert nx.number_connected_components(healed) == 1
        assert healed.has_edge("left_tip", "right_tip")
        edge = healed["left_tip"]["right_tip"]
        assert edge["recovery_type"] == "cloud_continuity"
        assert edge["inferred"] is True
        assert edge["occlusion_suspected"] is True
        assert edge["occlusion_kind"] == "cloud"
        points = np.asarray(edge["pts"])
        assert np.allclose(points[0], start)
        assert np.allclose(points[-1], end)
        assert edge["cloud_coverage"] >= 0.70
        assert np.count_nonzero(cloud[np.rint(points[:, 0]).astype(int), np.rint(points[:, 1]).astype(int)]) > len(points) // 2

    def test_long_aligned_gap_without_cloud_mask_is_rejected(self):
        """Long geometry alone is not enough to invent a road connection."""
        graph, left_tip, right_tip = _build_straight_road_pair(gap_px=100.0, trunk_len=32)
        healed = confidence_gated_gap_healing(
            graph,
            np.zeros((100, 220), dtype=np.float32),
            max_gap_px=30.0,
            enable_cloud_occlusion_recovery=True,
            max_cloud_occlusion_gap_px=180.0,
            cloud_min_component_len=20.0,
        )
        assert nx.number_connected_components(healed) == 2
        assert not healed.has_edge(left_tip, right_tip)
