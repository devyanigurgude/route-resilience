"""Focused unit tests for Phase 7 Real-World Occlusion Validation."""

from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from route_resilience.config import GraphConfig
from route_resilience.graph_extraction import (
    confidence_gated_gap_healing,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "RouteResilience").exists() and (PROJECT_ROOT.parent / "RouteResilience").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent


def test_phase7_selected_tiles_provenance_and_integrity():
    """Verify that the 60 selected real tiles have verified test provenance and expected subset counts."""
    csv_path = PROJECT_ROOT / "RouteResilience" / "phase7_selected_tiles.csv"
    assert csv_path.exists(), "phase7_selected_tiles.csv must exist"

    df = pd.read_csv(csv_path)
    assert len(df) == 60, f"Expected 60 selected tiles, got {len(df)}"

    counts = df["subset"].value_counts().to_dict()
    assert counts.get("natural_canopy") == 20, "Expected 20 natural canopy tiles"
    assert counts.get("natural_shadow") == 20, "Expected 20 natural shadow tiles"
    assert counts.get("open_terrain_control") == 20, "Expected 20 open terrain control tiles"

    # Verify tile files actually exist on disk
    for _, row in df.iterrows():
        img = PROJECT_ROOT / "RouteResilience" / "images_png" / row["filename"]
        mask = PROJECT_ROOT / "RouteResilience" / "masks_png" / row["filename"]
        assert img.exists(), f"Image {img} missing"
        assert mask.exists(), f"Mask {mask} missing"


def test_natural_canopy_and_shadow_distributions():
    """Verify that natural canopy tiles have high greenness and shadow tiles have low luminance."""
    csv_path = PROJECT_ROOT / "RouteResilience" / "phase7_selected_tiles.csv"
    df = pd.read_csv(csv_path)

    canopy_df = df[df["subset"] == "natural_canopy"]
    shadow_df = df[df["subset"] == "natural_shadow"]
    control_df = df[df["subset"] == "open_terrain_control"]

    assert canopy_df["green_frac"].mean() > 0.95, "Canopy subset must exhibit >95% vegetation along roads"
    assert shadow_df["shadow_frac"].mean() > 0.95, "Shadow subset must exhibit >95% low luminance along roads"
    assert control_df["green_frac"].mean() < 0.25, "Control subset must exhibit <25% vegetation along roads"


def test_candidate_border_and_flank_guards_on_real_geometry():
    """Verify that candidate parameters filter edge padding stumps while connecting real collinear road segments."""
    # Synthetic case representing real satellite border artifact
    G = nx.Graph()
    # Real road in center: (100, 50) to (100, 75) and (100, 85) to (100, 110) -> 10px gap
    G.add_node(1, pos=(100.0, 50.0))
    G.add_node(2, pos=(100.0, 75.0))
    G.add_edge(1, 2, length_px=25.0)

    G.add_node(3, pos=(100.0, 85.0))
    G.add_node(4, pos=(100.0, 110.0))
    G.add_edge(3, 4, length_px=25.0)

    # False border artifact stumps along top edge (row=2 and row=3)
    G.add_node(5, pos=(2.0, 150.0))
    G.add_node(6, pos=(2.0, 160.0))
    G.add_edge(5, 6, length_px=10.0)

    G.add_node(7, pos=(3.0, 170.0))
    G.add_node(8, pos=(3.0, 180.0))
    G.add_edge(7, 8, length_px=10.0)

    prob = np.full((256, 256), 0.10, dtype=np.float32)
    # High confidence on real road
    prob[100, 45:115] = 0.85
    # Border noise
    prob[1:5, 145:185] = 0.40

    # Baseline connects both the real road AND the border stumps
    G_base = confidence_gated_gap_healing(
        G.copy(), prob, max_gap_px=15.0, min_gap_confidence=0.15,
        filter_tile_borders=False, min_flank_contrast=None
    )
    assert G_base.has_edge(2, 3), "Baseline must heal the genuine road"
    assert G_base.has_edge(6, 7), "Baseline erroneously heals border stumps"

    # Phase 6 Candidate connects the real road and rejects the border stumps
    G_cand = confidence_gated_gap_healing(
        G.copy(), prob, max_gap_px=15.0, min_gap_confidence=0.15,
        filter_tile_borders=True, min_flank_contrast=-0.02
    )
    assert G_cand.has_edge(2, 3), "Candidate must preserve the genuine road"
    assert not G_cand.has_edge(6, 7), "Candidate must reject false border stumps"
