"""Tests for photometric occlusion detection (cloud, shadow, canopy) and safe gap-healing recovery.

Verifies:
1. Backward compatibility of cloud_occlusion_mask.
2. Conservative shadow and tree canopy mask detection.
3. Safe gap-healing across zero-evidence spans covered by verified occlusion masks.
4. Rejection of zero-evidence gaps without occlusion support (prevents hallucinating roads).
5. Rejection of implausible bridges (perpendicular directions, large gaps exceeding limits).
"""

import networkx as nx
import numpy as np
import pytest

from route_resilience.graph_extraction import (
    confidence_gated_gap_healing,
    detect_candidate_occlusions,
)
from route_resilience.real_graph_pipeline import (
    canopy_occlusion_mask,
    cloud_occlusion_mask,
    photometric_occlusion_mask,
    shadow_occlusion_mask,
)


def _build_horizontal_road_pair(gap_px: float = 20.0, trunk_len: int = 15):
    """Builds two horizontal road segments facing each other across a gap."""
    G = nx.Graph()
    y = 50.0
    for i in range(trunk_len + 1):
        x = 10.0 + i
        G.add_node(f"L{i}", pos=(y, x))
        if i > 0:
            G.add_edge(f"L{i-1}", f"L{i}", length_px=1.0, pts=np.array([(y, x - 1), (y, x)]))

    right_x0 = 10.0 + trunk_len + gap_px
    for i in range(trunk_len + 1):
        x = right_x0 + i
        G.add_node(f"R{i}", pos=(y, x))
        if i > 0:
            G.add_edge(f"R{i-1}", f"R{i}", length_px=1.0, pts=np.array([(y, x - 1), (y, x)]))

    return G, f"L{trunk_len}", "R0"


class TestPhotometricOcclusionDetection:
    def test_cloud_mask_compatibility(self):
        """Bright, low-saturation pixels are flagged as clouds."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)  # neutral grey
        img[20:40, 20:40] = [240, 240, 245]  # bright cloud
        c_mask = cloud_occlusion_mask(img)
        p_mask = photometric_occlusion_mask(img)

        assert c_mask[30, 30]
        assert p_mask[30, 30]
        assert not c_mask[10, 10]
        assert not p_mask[10, 10]

    def test_shadow_and_canopy_detection(self):
        """Shadows and dense vegetation canopy are correctly flagged by candidate masks."""
        img = np.full((100, 100, 3), 120, dtype=np.uint8)  # neutral grey asphalt
        img[10:30, 10:30] = [25, 25, 25]     # deep dark shadow
        img[50:70, 50:70] = [35, 95, 30]     # dense green foliage (ExG > 0.08, G dominant)

        shadow_m = shadow_occlusion_mask(img)
        canopy_m = canopy_occlusion_mask(img)
        photo_m = photometric_occlusion_mask(img)

        assert shadow_m[20, 20]
        assert not shadow_m[60, 60]
        assert not shadow_m[90, 90]  # asphalt is not shadow

        assert canopy_m[60, 60]
        assert not canopy_m[20, 20]
        assert not canopy_m[90, 90]  # asphalt is not canopy

        # Combined photometric mask covers both
        assert photo_m[20, 20]
        assert photo_m[60, 60]
        assert not photo_m[90, 90]


class TestOcclusionRecoverySafeBridging:
    def test_recovers_zero_evidence_span_under_verified_occlusion_mask(self):
        """Collinear road endpoints with zero probability in the gap ARE bridged when covered by occlusion mask."""
        G, left_tip, right_tip = _build_horizontal_road_pair(gap_px=20.0, trunk_len=15)
        prob_map = np.zeros((100, 100), dtype=np.float32)

        # Occlusion mask covers the gap region [40:60, 20:50]
        occ_mask = np.zeros((100, 100), dtype=bool)
        occ_mask[45:55, 20:50] = True

        healed = confidence_gated_gap_healing(
            G,
            prob_map,
            max_gap_px=15.0,
            enable_occlusion_recovery=True,
            max_occlusion_gap_px=30.0,
            occlusion_mask=occ_mask,
        )

        assert healed.has_edge(left_tip, right_tip)
        edge = healed[left_tip][right_tip]
        assert edge["healed"] is True
        assert edge["occlusion_recovered"] is True
        assert edge["recovery_type"] == "occlusion_continuity"
        assert edge["occlusion_coverage"] >= 0.50

    def test_rejects_zero_evidence_span_without_occlusion_mask(self):
        """Collinear road endpoints with zero probability ARE NOT bridged if there is no occlusion mask."""
        G, left_tip, right_tip = _build_horizontal_road_pair(gap_px=20.0, trunk_len=15)
        prob_map = np.zeros((100, 100), dtype=np.float32)
        # Empty occlusion mask (bare ground / open missing road)
        occ_mask = np.zeros((100, 100), dtype=bool)

        healed = confidence_gated_gap_healing(
            G,
            prob_map,
            max_gap_px=15.0,
            enable_occlusion_recovery=True,
            max_occlusion_gap_px=30.0,
            occlusion_mask=occ_mask,
        )

        # Must remain disconnected to prevent hallucinating non-existent roads
        assert not healed.has_edge(left_tip, right_tip)
        assert nx.number_connected_components(healed) == 2

    def test_rejects_implausible_bridges_even_under_occlusion(self):
        """Perpendicular endpoints or gaps exceeding max_occlusion_gap_px are strictly rejected."""
        # Case A: Gap exceeds max_occlusion_gap_px (e.g. 50px gap vs limit 30px)
        G_large, left_tip, right_tip = _build_horizontal_road_pair(gap_px=50.0, trunk_len=15)
        occ_mask = np.ones((120, 120), dtype=bool)
        prob_map = np.zeros((120, 120), dtype=np.float32)

        healed_large = confidence_gated_gap_healing(
            G_large,
            prob_map,
            max_gap_px=15.0,
            enable_occlusion_recovery=True,
            max_occlusion_gap_px=30.0,
            occlusion_mask=occ_mask,
        )
        assert not healed_large.has_edge(left_tip, right_tip)

        # Case B: Perpendicular road segments
        G_perp = nx.Graph()
        # Horizontal segment: (50, 10) to (50, 30)
        for x in range(10, 31):
            G_perp.add_node(f"H{x}", pos=(50.0, float(x)))
            if x > 10:
                G_perp.add_edge(f"H{x-1}", f"H{x}", length_px=1.0, pts=np.array([(50.0, x - 1), (50.0, x)]))
        # Vertical segment: (60, 35) to (80, 35)
        for y in range(60, 81):
            G_perp.add_node(f"V{y}", pos=(float(y), 35.0))
            if y > 60:
                G_perp.add_edge(f"V{y-1}", f"V{y}", length_px=1.0, pts=np.array([(y - 1, 35.0), (y, 35.0)]))

        healed_perp = confidence_gated_gap_healing(
            G_perp,
            np.zeros((100, 100), dtype=np.float32),
            max_gap_px=15.0,
            enable_occlusion_recovery=True,
            max_occlusion_gap_px=30.0,
            occlusion_mask=occ_mask[:100, :100],
        )
        assert not healed_perp.has_edge("H30", "V60")

