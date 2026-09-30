"""
Real-image disruption smoke test.
Uploads RouteResilience/images_png/1.png, then runs three disruption
strategies against the produced analysis. Prints a structured report.

Run: python scripts/test_disrupt_real_image.py
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

# ── add project root to path so api_server is importable ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

os.environ.setdefault(
    "ROUTE_RESILIENCE_ANALYSIS_DIR",
    str(PROJECT_ROOT / "outputs" / "analysis"),
)

import api_server  # noqa: E402 – must come after sys.path & env setup

IMAGE_PATH = PROJECT_ROOT / "RouteResilience" / "images_png" / "1.png"


def _fmt(d, indent=2):
    return json.dumps(d, indent=indent, default=str)


def main():
    print("=" * 70)
    print("REAL-IMAGE DISRUPTION TEST  -  images_png/1.png")
    print("=" * 70)

    assert IMAGE_PATH.exists(), f"Image not found: {IMAGE_PATH}"

    app = api_server.create_app()
    client = TestClient(app, raise_server_exceptions=True)

    # Step 1: Upload image and run analysis
    print("\n[1/4] Uploading 1.png and running road analysis ...")
    with open(IMAGE_PATH, "rb") as fh:
        image_bytes = fh.read()

    resp = client.post(
        "/api/road-analysis",
        files={"file": ("1.png", image_bytes, "image/png")},
    )
    assert resp.status_code == 200, f"Analysis failed: {resp.text}"
    analysis = resp.json()
    analysis_id = analysis["analysis_id"]

    print(f"  analysis_id = {analysis_id}")
    summary = analysis.get("summary", {})
    print(f"  baseline nodes   = {summary.get('n_nodes', '?')}")
    print(f"  baseline edges   = {summary.get('n_edges', '?')}")
    print(f"  connected        = {summary.get('is_connected', '?')}")
    print(f"  threshold used   = {analysis.get('threshold_used', '?')}")

    base_url = f"/api/analysis/{analysis_id}/disrupt"

    # Step 2: Random failure
    print("\n[2/4] Strategy: random (1 edge, seed=42) ...")
    resp2 = client.post(base_url, json={
        "strategy": "random",
        "target_type": "edge",
        "count": 1,
        "seed": 42,
    })
    assert resp2.status_code == 200, f"Random disruption failed: {resp2.text}"
    r2 = resp2.json()
    print(f"  failed edges = {r2['scenario']['failed_edges']}")
    m2 = r2["impact_metrics"]
    print(f"  efficiency_drop_frac        = {m2['efficiency_drop_frac']:.6f}")
    print(f"  newly_disconnected_components = {m2['newly_disconnected_components']}")
    if r2["affected_od_service"]:
        od2 = r2["affected_od_service"]
        print(f"  od pairs evaluated          = {od2['total_od_pairs_evaluated']}")
        print(f"  disrupted_service_perf      = {od2['disrupted_service_performance']:.6f}")
        print(f"  severed pairs               = {od2['severed_pairs_count']}")

    # Step 3: Targeted betweenness
    print("\n[3/4] Strategy: targeted_betweenness (1 edge) ...")
    resp3 = client.post(base_url, json={
        "strategy": "targeted_betweenness",
        "target_type": "edge",
        "count": 1,
    })
    assert resp3.status_code == 200, f"Betweenness disruption failed: {resp3.text}"
    r3 = resp3.json()
    print(f"  failed edges = {r3['scenario']['failed_edges']}")
    m3 = r3["impact_metrics"]
    print(f"  efficiency_drop_frac        = {m3['efficiency_drop_frac']:.6f}")
    print(f"  newly_disconnected_components = {m3['newly_disconnected_components']}")
    net3 = r3["network_state"]
    print(f"  baseline  nodes/edges = {net3['baseline']['n_nodes']} / {net3['baseline']['n_edges']}")
    print(f"  disrupted nodes/edges = {net3['disrupted']['n_nodes']} / {net3['disrupted']['n_edges']}")

    # Step 4: Explicit (re-remove the top-betweenness edge)
    top_edge = r3["scenario"]["failed_edges"][0]
    print(f"\n[4/4] Strategy: explicit (re-remove top-betweenness edge {top_edge}) ...")
    resp4 = client.post(base_url, json={
        "strategy": "explicit",
        "target_type": "edge",
        "failed_edges": [top_edge],
    })
    assert resp4.status_code == 200, f"Explicit disruption failed: {resp4.text}"
    r4 = resp4.json()
    m4 = r4["impact_metrics"]
    print(f"  efficiency_drop_frac        = {m4['efficiency_drop_frac']:.6f}")
    assert abs(m4["efficiency_drop_frac"] - m3["efficiency_drop_frac"]) < 1e-9, (
        "Explicit re-removal must yield same efficiency drop as targeted run"
    )
    print("  Matches targeted_betweenness result (deterministic)")

    # Error cases
    print("\n[Error cases]")
    r_404 = client.post("/api/analysis/bad-id-xyz/disrupt", json={"strategy": "random", "target_type": "edge"})
    assert r_404.status_code == 404
    print(f"  404 for unknown analysis_id: {r_404.json()['detail'][:60]}")

    r_bad_node = client.post(base_url, json={
        "strategy": "explicit", "target_type": "node", "failed_nodes": [999999]
    })
    assert r_bad_node.status_code == 400
    print(f"  400 for bad node id: {r_bad_node.json()['detail'][:60]}")

    r_bad_edge = client.post(base_url, json={
        "strategy": "explicit", "target_type": "edge", "failed_edges": [[0, 999999]]
    })
    assert r_bad_edge.status_code == 400
    print(f"  400 for bad edge id: {r_bad_edge.json()['detail'][:60]}")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)

    print("\nSUMMARY JSON:")
    summary_out = {
        "analysis_id": analysis_id,
        "baseline": {
            "n_nodes": summary.get("n_nodes"),
            "n_edges": summary.get("n_edges"),
            "is_connected": summary.get("is_connected"),
        },
        "random_disruption": {
            "failed_edges": r2["scenario"]["failed_edges"],
            "efficiency_drop_frac": m2["efficiency_drop_frac"],
            "newly_disconnected_components": m2["newly_disconnected_components"],
            "disrupted_service_performance": r2["affected_od_service"]["disrupted_service_performance"] if r2["affected_od_service"] else None,
        },
        "targeted_betweenness_disruption": {
            "failed_edges": r3["scenario"]["failed_edges"],
            "efficiency_drop_frac": m3["efficiency_drop_frac"],
            "newly_disconnected_components": m3["newly_disconnected_components"],
            "n_disrupted_nodes": net3["disrupted"]["n_nodes"],
            "n_disrupted_edges": net3["disrupted"]["n_edges"],
        },
    }
    print(_fmt(summary_out))


if __name__ == "__main__":
    main()
