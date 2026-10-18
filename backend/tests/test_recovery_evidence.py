"""Tests for Explainable Recovery Evidence in occlusion recovery pipeline and API.

Verifies:
1. Accepted cloud-continuity connections contain structured evidence, checks, and acceptance reason.
2. Accepted evidence-guided/direct connections contain probability support and continuity checks.
3. Rejected candidate occlusions retain exact rejection reasons and evaluated check statuses.
4. GET /api/analysis/{analysis_id}/recovery-evidence endpoint returns structured recovery evidence.
"""

import json
import os
import networkx as nx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_server
from route_resilience.graph_extraction import (
    confidence_gated_gap_healing,
    detect_candidate_occlusions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_horizontal_segment(G, prefix: str, y: float, x_start: float, n_nodes: int, step: float = 1.0):
    for i in range(n_nodes):
        x = x_start + i * step
        G.add_node(f"{prefix}{i}", pos=(y, x))
        if i > 0:
            prev_x = x - step
            G.add_edge(
                f"{prefix}{i-1}",
                f"{prefix}{i}",
                length_px=step,
                pts=np.array([(y, prev_x), (y, x)]),
            )
    return f"{prefix}0", f"{prefix}{n_nodes - 1}"


def _build_facing_segments(gap_px: float = 20.0, trunk_len: int = 25, y: float = 50.0):
    G = nx.Graph()
    left_start, left_end = _add_horizontal_segment(G, "L", y, 10.0, trunk_len + 1)
    right_x0 = 10.0 + trunk_len + gap_px
    right_start, right_end = _add_horizontal_segment(G, "R", y, right_x0, trunk_len + 1)
    return G, left_end, right_start


# ---------------------------------------------------------------------------
# Unit Tests for Recovery Evidence Recording
# ---------------------------------------------------------------------------

def test_cloud_continuity_recovery_evidence():
    """Verify accepted cloud continuity edge carries complete evidence metadata."""
    G, left_end, right_end = _build_facing_segments(gap_px=22.0, trunk_len=25, y=50.0)
    prob_map = np.full((100, 100), 0.05, dtype=np.float32)
    # Cloud mask covering the gap
    cloud_mask = np.zeros((100, 100), dtype=bool)
    cloud_mask[45:56, 20:65] = True

    G_healed = confidence_gated_gap_healing(
        G,
        prob_map,
        max_gap_px=10.0,
        enable_occlusion_recovery=True,
        max_occlusion_gap_px=30.0,
        enable_cloud_occlusion_recovery=True,
        max_cloud_occlusion_gap_px=50.0,
        cloud_min_component_len=15.0,
        cloud_min_coverage=0.50,
        occlusion_mask=cloud_mask,
    )

    assert G_healed.has_edge(left_end, right_end), "Cloud continuity should heal the facing gap"
    edge_data = G_healed.get_edge_data(left_end, right_end)

    assert edge_data.get("occlusion_recovered") is True
    assert edge_data.get("recovery_type") == "cloud_continuity"
    assert "acceptance_reason" in edge_data
    assert "Cloud Occlusion Continuity" in edge_data["acceptance_reason"]

    evidence = edge_data.get("evidence")
    assert isinstance(evidence, dict), "Evidence dict must be attached to the edge"
    assert "gap_px" in evidence
    assert 21.0 <= evidence["gap_px"] <= 23.0
    assert "heading_alignment" in evidence
    assert evidence["heading_alignment"] > 0.9
    assert "mutual_heading" in evidence
    assert evidence["mutual_heading"] > 0.9
    assert evidence["cloud_coverage"] is not None
    assert evidence["cloud_coverage"] >= 0.5
    assert "sinuosity" in evidence
    assert evidence["sinuosity"] == 1.0

    checks = edge_data.get("checks", [])
    assert len(checks) >= 6, "All key continuity checks should be recorded"
    check_names = {c["name"] for c in checks}
    assert "Gap Distance" in check_names
    assert "Cloud Mask Coverage" in check_names
    assert "Endpoint Heading Alignment" in check_names
    assert "Acyclicity & Degree" in check_names
    assert all(c["passed"] for c in checks), "All recorded checks for accepted edge should have passed"


def test_evidence_guided_recovery_evidence():
    """Verify accepted evidence-guided edge records probability support and contrast."""
    G, left_end, right_end = _build_facing_segments(gap_px=18.0, trunk_len=15, y=50.0)
    # Provide strong probability corridor through the gap
    prob_map = np.full((100, 100), 0.05, dtype=np.float32)
    prob_map[49:52, 24:45] = 0.85

    G_healed = confidence_gated_gap_healing(
        G,
        prob_map,
        max_gap_px=10.0,
        enable_occlusion_recovery=True,
        max_occlusion_gap_px=25.0,
        min_flank_contrast=0.10,
    )

    assert G_healed.has_edge(left_end, right_end), "Evidence corridor should heal the gap"
    edge_data = G_healed.get_edge_data(left_end, right_end)

    assert edge_data.get("occlusion_recovered") is True
    evidence = edge_data.get("evidence")
    assert isinstance(evidence, dict)
    assert evidence.get("probability_evidence") is not None
    assert evidence["probability_evidence"] > 0.5
    assert evidence.get("flank_contrast") is not None
    assert len(edge_data.get("checks", [])) >= 5


def test_rejected_candidate_evidence_and_reason():
    """Verify rejected candidates retain exact rejection reasons and check statuses."""
    G, left_end, right_end = _build_facing_segments(gap_px=22.0, trunk_len=15, y=50.0)
    # Pure zero probability across gap, no cloud mask -> must fail continuity qualification
    prob_map = np.full((100, 100), 0.0, dtype=np.float32)

    G_healed = confidence_gated_gap_healing(
        G,
        prob_map,
        max_gap_px=10.0,
        enable_occlusion_recovery=True,
        max_occlusion_gap_px=30.0,
    )

    assert not G_healed.has_edge(left_end, right_end), "Should not heal unsupported gap"
    candidates = G_healed.graph.get("candidate_occlusions", [])
    assert len(candidates) > 0, "Candidates should be recorded in graph metadata"

    cand = next((c for c in candidates if {c["source"], c["target"]} == {left_end, right_end}), None)
    assert cand is not None, "Candidate between left and right endpoints must exist"
    assert cand["status"] == "rejected"
    assert cand.get("rejection_reason") is not None
    assert len(cand["rejection_reason"]) > 0
    assert "checks" in cand
    assert any(c["passed"] is False for c in cand["checks"]), "At least one check must be marked failed"


# ---------------------------------------------------------------------------
# API Endpoint Integration Tests
# ---------------------------------------------------------------------------

def test_api_recovery_evidence_endpoint(tmp_path):
    """Verify GET /api/analysis/{analysis_id}/recovery-evidence returns proper payload."""
    analysis_root = tmp_path / "analysis"
    analysis_id = "evidence-test-01"
    analysis_dir = analysis_root / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    healed_payload = {
        "coordinate_system": "pixel-space",
        "nodes": [
            {"id": "L15", "attributes": {"pos": [50.0, 25.0]}},
            {"id": "R0", "attributes": {"pos": [50.0, 47.0]}},
        ],
        "edges": [
            {
                "source": "L15",
                "target": "R0",
                "attributes": {
                    "length_px": 22.0,
                    "confidence": 0.85,
                    "healed": True,
                    "inferred": True,
                    "occlusion_recovered": True,
                    "recovery_type": "cloud_continuity",
                    "acceptance_reason": "Cloud continuity accepted: 22.0px gap aligned with 85% cloud coverage",
                    "evidence": {
                        "gap_px": 22.0,
                        "heading_alignment": 0.98,
                        "mutual_heading": 0.99,
                        "dot_a": 0.99,
                        "dot_b": 0.99,
                        "lateral_offset_px": 0.0,
                        "cloud_coverage": 0.85,
                        "sinuosity": 1.0,
                    },
                    "checks": [
                        {"name": "distance_within_limit", "passed": True, "criteria": "dist (22.0px) <= max (30.0px)"},
                        {"name": "cloud_mask_coverage", "passed": True, "criteria": "coverage (85%) >= 50%"},
                    ],
                },
            }
        ],
        "candidate_occlusions": [
            {
                "source": "A",
                "target": "B",
                "pos_source": [10.0, 10.0],
                "pos_target": [10.0, 35.0],
                "gap_px": 25.0,
                "status": "rejected",
                "rejection_reason": "Flank contrast (0.02) below threshold (0.10)",
                "checks": [
                    {"name": "distance_within_limit", "passed": True, "criteria": "dist <= 30.0px"},
                    {"name": "flank_contrast", "passed": False, "criteria": "contrast (0.02) >= 0.10"},
                ],
            }
        ],
    }

    (analysis_dir / "healed_graph.json").write_text(json.dumps(healed_payload), encoding="utf-8")
    os.environ["ROUTE_RESILIENCE_ANALYSIS_DIR"] = str(analysis_root)

    client = TestClient(api_server.app)
    response = client.get(f"/api/analysis/{analysis_id}/recovery-evidence")

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_id"] == analysis_id
    assert data["status"] == "completed"
    assert data["inferred_count"] == 1
    assert data["candidates_count"] == 1

    inferred = data["inferred_connections"]
    assert len(inferred) == 1
    assert inferred[0]["source"] == "L15"
    assert inferred[0]["target"] == "R0"
    assert inferred[0]["recovery_type"] == "cloud_continuity"
    assert inferred[0]["evidence"]["cloud_coverage"] == 0.85
    assert len(inferred[0]["checks"]) == 2

    candidates = data["candidate_occlusions"]
    assert len(candidates) == 1
    assert candidates[0]["status"] == "rejected"
    assert "Flank contrast" in candidates[0]["rejection_reason"]


def test_api_recovery_evidence_404(tmp_path):
    """Verify 404 response for nonexistent analysis."""
    os.environ["ROUTE_RESILIENCE_ANALYSIS_DIR"] = str(tmp_path / "empty_analysis")
    client = TestClient(api_server.app)
    response = client.get("/api/analysis/nonexistent-id/recovery-evidence")
    assert response.status_code == 404
