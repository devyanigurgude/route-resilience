"""Targeted End-to-End Workflow Verification Script.

Tests the full user workflow across the real backend:
  1. Upload satellite image (1.png) -> POST /api/road-analysis
  2. Inspect road graph -> GET /api/analysis/{analysis_id}
  3. View critical-road rankings/heatmap -> GET /api/analysis/{analysis_id}/criticality
  4. Disable a critical road -> POST /api/analysis/{analysis_id}/disrupt
  5. Verify persisted disruption scenario -> GET /api/analysis/{analysis_id}/scenario
  6. Request a reroute -> POST /api/analysis/{analysis_id}/route
  7. Test recovery/restoration -> POST /api/analysis/{analysis_id}/recovery
"""

import sys
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from PIL import Image

import api_server


def run_e2e_verification():
    print("[1/7] Initializing FastAPI TestClient...")
    app = api_server.create_app()
    client = TestClient(app)

    # 1. Upload satellite image
    img_path = Path("RouteResilience/images_png/1.png")
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at {img_path}")

    print(f"[2/7] Uploading real satellite image: {img_path}...")
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    upload_resp = client.post(
        "/api/road-analysis",
        files={"file": ("1.png", img_bytes, "image/png")},
    )
    assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
    analysis = upload_resp.json()
    analysis_id = analysis["analysis_id"]
    print(f"      -> Analysis created: ID={analysis_id}, status={analysis['status']}")
    print(f"      -> Healed edges: {analysis['summary']['healed_edge_count']}")

    # 2. Inspect road graph & artifacts
    print(f"[3/7] Inspecting analysis artifacts for {analysis_id}...")
    get_resp = client.get(f"/api/analysis/{analysis_id}")
    assert get_resp.status_code == 200, f"Get analysis failed: {get_resp.text}"
    analysis_data = get_resp.json()
    assert analysis_data["artifacts"]["healed_graph_json"].endswith("/healed_graph.json")
    assert analysis_data["artifacts"]["original"].endswith("/original.png")

    # 3. View critical-road rankings/heatmap
    print("[4/7] Querying edge criticality report...")
    crit_resp = client.get(f"/api/analysis/{analysis_id}/criticality")
    assert crit_resp.status_code == 200, f"Criticality query failed: {crit_resp.text}"
    crit_data = crit_resp.json()
    top_vulnerable = crit_data["criticality"]["top_vulnerable_edges_by_betweenness"]
    assert len(top_vulnerable) > 0, "No vulnerable edges found!"
    target_edge = top_vulnerable[0]
    print(f"      -> Identified top vulnerable edge: {target_edge}")

    # 4. Disable a road in Disruption
    print(f"[5/7] Simulating failure of critical edge {target_edge}...")
    disrupt_resp = client.post(
        f"/api/analysis/{analysis_id}/disrupt",
        json={
            "strategy": "explicit",
            "target_type": "edge",
            "failed_edges": [target_edge],
        },
    )
    assert disrupt_resp.status_code == 200, f"Disruption failed: {disrupt_resp.text}"
    disrupt_data = disrupt_resp.json()
    metrics = disrupt_data["impact_metrics"]
    print(f"      -> Efficiency drop: {metrics['efficiency_drop_frac'] * 100:.2f}%")
    print(f"      -> Newly disconnected components: {metrics['newly_disconnected_components']}")
    assert metrics["efficiency_drop_frac"] >= 0.0, "Efficiency drop should be positive!"
    assert disrupt_data["network_state"]["disrupted"]["n_edges"] < disrupt_data["network_state"]["baseline"]["n_edges"]

    # Check scenario persistence
    scen_resp = client.get(f"/api/analysis/{analysis_id}/scenario")
    assert scen_resp.status_code == 200
    assert scen_resp.json()["disruption"] is not None

    # 5. Request a reroute
    print(f"[6/7] Querying reroute around blocked edge {target_edge}...")
    route_resp = client.post(
        f"/api/analysis/{analysis_id}/route",
        json={
            "origin": target_edge[0],
            "destination": target_edge[1],
            "current_state": "disrupted",
            "failed_edges": [target_edge],
        },
    )
    assert route_resp.status_code == 200, f"Routing failed: {route_resp.text}"
    route_data = route_resp.json()
    print(f"      -> Route status: {route_data['route_status']}")
    if route_data["route_exists"]:
        print(f"      -> Alternative route: {route_data['alternative_route']}")
        assert target_edge not in zip(route_data["alternative_route"][:-1], route_data["alternative_route"][1:])
    else:
        print("      -> Destination severed as expected (no alternative cycle)")

    # 6. Test recovery/restoration
    print(f"[7/7] Testing recovery of {target_edge} under budget=1...")
    rec_resp = client.post(
        f"/api/analysis/{analysis_id}/recovery",
        json={
            "failed_edges": [target_edge],
            "budget": 1,
            "ordering": "criticality_greedy",
        },
    )
    assert rec_resp.status_code == 200, f"Recovery failed: {rec_resp.text}"
    rec_data = rec_resp.json()
    resilience = rec_data["resilience_metrics"]
    print(f"      -> P_D: {resilience['P_D']:.3f}, P_R: {resilience['P_R']:.3f}")
    print(f"      -> RecoveryGain: {resilience['RecoveryGain']:.3f}, RI: {resilience['RI']:.3f}")
    assert resilience["P_R"] >= resilience["P_D"]
    assert len(rec_data["recovery_summary"]["recovered_edges"]) == 1

    print("\nSUCCESS: All 7 end-to-end workflow stages verified on live backend!")


if __name__ == "__main__":
    run_e2e_verification()
