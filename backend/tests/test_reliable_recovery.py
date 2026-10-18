"""Focused unit tests for Phase 6 Reliable Occlusion Recovery candidate validation."""

import networkx as nx
import numpy as np
import pytest

from route_resilience.config import GraphConfig
from route_resilience.graph_extraction import (
    confidence_gated_gap_healing,
)


def _create_test_graph_with_spur():
    """Builds a synthetic graph with a genuine road corridor and an isolated noise spur."""
    G = nx.Graph()
    # Main road segment: 40px long
    G.add_node(1, pos=(100.0, 50.0))
    G.add_node(2, pos=(100.0, 90.0))
    G.add_edge(1, 2, length_px=40.0)

    # Isolated floating noise spur: 3px long, 10px away from node 2
    G.add_node(3, pos=(100.0, 100.0))
    G.add_node(4, pos=(100.0, 103.0))
    G.add_edge(3, 4, length_px=3.0)

    return G


def test_floating_noise_spur_rejection():
    """Verify that isolated short spurs (<4px) are rejected when min_component_len is active."""
    G = _create_test_graph_with_spur()
    prob = np.full((200, 200), 0.5, dtype=np.float32)

    # 1. Baseline: heals the spur into the main road
    G_base = confidence_gated_gap_healing(
        G.copy(), prob, max_gap_px=15.0, min_gap_confidence=0.15, min_component_len=0.0
    )
    assert G_base.has_edge(2, 3), "Baseline should have healed edge (2, 3)"

    # 2. Robust validation: rejects the spur because component length of [3, 4] is only 3px
    G_rob = confidence_gated_gap_healing(
        G.copy(), prob, max_gap_px=15.0, min_gap_confidence=0.15, min_component_len=4.0
    )
    assert not G_rob.has_edge(2, 3), "Robust validation must reject connection to 3px noise spur"


def test_tile_border_artifact_rejection():
    """Verify that border stumps along image margins are rejected by filter_tile_borders."""
    G = nx.Graph()
    # Two stumps near the left image border (c=2 and c=3)
    G.add_node(1, pos=(150.0, 2.0))
    G.add_node(2, pos=(160.0, 2.0))
    G.add_edge(1, 2, length_px=10.0)

    G.add_node(3, pos=(170.0, 3.0))
    G.add_node(4, pos=(180.0, 3.0))
    G.add_edge(3, 4, length_px=10.0)

    prob = np.full((256, 256), 0.5, dtype=np.float32)

    # Baseline connects them
    G_base = confidence_gated_gap_healing(
        G.copy(), prob, max_gap_px=15.0, min_gap_confidence=0.15, filter_tile_borders=False
    )
    assert G_base.has_edge(2, 3)

    # Robust validation filters them
    G_rob = confidence_gated_gap_healing(
        G.copy(), prob, max_gap_px=15.0, min_gap_confidence=0.15, filter_tile_borders=True
    )
    assert not G_rob.has_edge(2, 3), "Border stumps within 4px of tile margin must be filtered"


def test_flank_contrast_rejection_of_diffuse_blobs():
    """Verify that connections across diffuse probability blobs (no corridor contrast) are rejected."""
    G = nx.Graph()
    G.add_node(1, pos=(50.0, 50.0))
    G.add_node(2, pos=(50.0, 80.0))
    G.add_edge(1, 2, length_px=30.0)

    G.add_node(3, pos=(50.0, 90.0))
    G.add_node(4, pos=(50.0, 120.0))
    G.add_edge(3, 4, length_px=30.0)

    # Inverted probability trough: center has lower probability than flanks
    prob = np.zeros((150, 150), dtype=np.float32)
    prob[46:55, 75:95] = 0.50  # flanks are high
    prob[50, 75:95] = 0.20     # centerline is low

    # Baseline with min_gap_confidence=0.15 accepts it because 0.20 >= 0.15
    G_base = confidence_gated_gap_healing(
        G.copy(), prob, max_gap_px=15.0, min_gap_confidence=0.15, min_flank_contrast=None
    )
    assert G_base.has_edge(2, 3)

    # Robust validation detects negative contrast (flanks > center) and rejects it
    G_rob = confidence_gated_gap_healing(
        G.copy(), prob, max_gap_px=15.0, min_gap_confidence=0.15, min_flank_contrast=-0.02
    )
    assert not G_rob.has_edge(2, 3), "Inverted or diffuse background trough must be rejected"


def test_genuine_road_connection_preservation():
    """Verify that a genuine collinear road with positive ridge contrast is preserved and healed."""
    G = nx.Graph()
    G.add_node(1, pos=(50.0, 20.0))
    G.add_node(2, pos=(50.0, 60.0))
    G.add_edge(1, 2, length_px=40.0)

    G.add_node(3, pos=(50.0, 70.0))
    G.add_node(4, pos=(50.0, 110.0))
    G.add_edge(3, 4, length_px=40.0)

    # Genuine road: high along centerline, zero on flanks
    prob = np.zeros((150, 150), dtype=np.float32)
    prob[50, :] = 0.85
    prob[49:52, 60:70] = 0.40  # residual corridor evidence

    G_rob = confidence_gated_gap_healing(
        G.copy(),
        prob,
        max_gap_px=15.0,
        min_gap_confidence=0.15,
        min_component_len=4.0,
        min_total_component_len=12.0,
        min_flank_contrast=-0.02,
        filter_tile_borders=True,
    )
    assert G_rob.has_edge(2, 3), "Genuine road corridor with ridge contrast must be successfully healed"
    assert G_rob.edges[2, 3]["healed"] is True


def test_sinuosity_guard_rejects_wandering_fallback_path():
    """Verify that max_sinuosity rejects serpentine wandering paths."""
    G = nx.Graph()
    G.add_node(1, pos=(20.0, 20.0))
    G.add_node(2, pos=(40.0, 20.0))
    G.add_edge(1, 2, length_px=20.0)

    G.add_node(3, pos=(50.0, 20.0))
    G.add_node(4, pos=(70.0, 20.0))
    G.add_edge(3, 4, length_px=20.0)

    prob = np.zeros((100, 100), dtype=np.float32)
    # Block straight line between (40, 20) and (50, 20)
    # Force an unnatural zigzag path of length ~25px across a 10px gap (sinuosity = 2.5)
    prob[40:51, 20] = 0.0
    for r in range(40, 46):
        prob[r, 20 + (r - 40)] = 0.8
    for r in range(45, 51):
        prob[r, 25 - (r - 45)] = 0.8

    G_rob = confidence_gated_gap_healing(
        G.copy(), prob, max_gap_px=15.0, min_gap_confidence=0.15, max_sinuosity=1.35
    )
    # Sinuosity of zigzag exceeds 1.35 -> must not be added
    assert not G_rob.has_edge(2, 3), "Serpentine path with sinuosity > 1.35 must be rejected"


def test_default_config_backward_compatibility():
    """Verify that GraphConfig default values preserve exact baseline behavior."""
    cfg = GraphConfig()
    assert cfg.min_component_len == 0.0
    assert cfg.min_total_component_len == 0.0
    assert cfg.max_sinuosity == 1.5
    assert cfg.min_flank_contrast is None
    assert cfg.filter_tile_borders is False
