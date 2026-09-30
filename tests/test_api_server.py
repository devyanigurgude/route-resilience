import io
import json
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import networkx as nx

import api_server


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_healed_graph_json(graph: nx.Graph) -> dict:
    """Convert a networkx graph to the healed_graph.json wire format."""
    return {
        "coordinate_system": "pixel-space",
        "nodes": [{"id": n, "attributes": data} for n, data in graph.nodes(data=True)],
        "edges": [
            {"source": u, "target": v, "attributes": data}
            for u, v, data in graph.edges(data=True)
        ],
    }


def _setup_analysis_dir(tmp_path, graph: nx.Graph, analysis_id: str = "test-analysis") -> tuple:
    """Create a minimal analysis directory with a healed_graph.json and original.png."""
    analysis_root = tmp_path / "analysis"
    analysis_dir = analysis_root / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (32, 32), color=(20, 30, 40)).save(analysis_dir / "original.png")
    (analysis_dir / "healed_graph.json").write_text(
        json.dumps(_make_healed_graph_json(graph)), encoding="utf-8"
    )
    os.environ["ROUTE_RESILIENCE_ANALYSIS_DIR"] = str(analysis_root)
    return analysis_root, analysis_dir


def _triangle_graph() -> nx.Graph:
    """A 3-node triangle: enough connectivity to test re-routing."""
    G = nx.Graph()
    G.add_node(1, pos=(0, 0))
    G.add_node(2, pos=(1, 0))
    G.add_node(3, pos=(0, 1))
    G.add_edge(1, 2, length_px=10.0, confidence=0.9, healed=False, width_px=3.0)
    G.add_edge(2, 3, length_px=10.0, confidence=0.8, healed=False, width_px=3.0)
    G.add_edge(1, 3, length_px=14.0, confidence=0.7, healed=False, width_px=2.0)
    return G


# ---------------------------------------------------------------------------
# Tests for POST /api/analysis/{analysis_id}/disrupt
# ---------------------------------------------------------------------------

class TestDisruptEndpoint:

    def _client(self, tmp_path, graph=None):
        if graph is None:
            graph = _triangle_graph()
        _setup_analysis_dir(tmp_path, graph)
        app = api_server.create_app()
        return TestClient(app)

    def test_explicit_edge_failure_returns_correct_shape(self, tmp_path):
        client = self._client(tmp_path)
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={
                "strategy": "explicit",
                "target_type": "edge",
                "failed_edges": [[1, 2]],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["analysis_id"] == "test-analysis"
        assert body["status"] == "completed"

        # Scenario fields
        scenario = body["scenario"]
        assert scenario["strategy"] == "explicit"
        assert scenario["target_type"] == "edge"
        assert [1, 2] in scenario["failed_edges"] or [2, 1] in scenario["failed_edges"]

        # Impact metrics present and numerically valid
        metrics = body["impact_metrics"]
        assert "efficiency_drop_frac" in metrics
        assert "newly_disconnected_components" in metrics
        assert 0.0 <= metrics["efficiency_drop_frac"] <= 1.0

        # Network state: disrupted graph has fewer edges
        net = body["network_state"]
        assert net["baseline"]["n_edges"] == 3
        assert net["disrupted"]["n_edges"] == 2

        # OD service block present
        assert body["affected_od_service"] is not None
        od = body["affected_od_service"]
        assert "total_od_pairs_evaluated" in od
        assert "disrupted_service_performance" in od

        # Disrupted graph included
        assert "disrupted_graph" in body
        assert len(body["disrupted_graph"]["nodes"]) > 0

    def test_explicit_node_failure(self, tmp_path):
        client = self._client(tmp_path)
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={
                "strategy": "explicit",
                "target_type": "node",
                "failed_nodes": [2],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        net = body["network_state"]
        # Node 2 removed → only nodes 1 and 3 remain
        assert net["disrupted"]["n_nodes"] == 2
        assert 2 in body["scenario"]["failed_nodes"]

    def test_targeted_betweenness_edge_strategy(self, tmp_path):
        client = self._client(tmp_path)
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={
                "strategy": "targeted_betweenness",
                "target_type": "edge",
                "count": 1,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "completed"
        assert body["scenario"]["strategy"] == "targeted_betweenness"
        assert len(body["scenario"]["failed_edges"]) == 1

    def test_targeted_betweenness_node_strategy(self, tmp_path):
        client = self._client(tmp_path)
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={
                "strategy": "targeted_betweenness",
                "target_type": "node",
                "count": 1,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["scenario"]["failed_nodes"]) == 1

    def test_random_failure_strategy(self, tmp_path):
        client = self._client(tmp_path)
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={
                "strategy": "random",
                "target_type": "edge",
                "count": 1,
                "seed": 99,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["scenario"]["strategy"] == "random"
        assert len(body["scenario"]["failed_edges"]) == 1

    def test_random_failure_is_reproducible(self, tmp_path):
        """Same seed must yield the same failed edges."""
        _setup_analysis_dir(tmp_path, _triangle_graph())
        client = TestClient(api_server.create_app())

        def _run():
            return client.post(
                "/api/analysis/test-analysis/disrupt",
                json={"strategy": "random", "target_type": "edge", "count": 1, "seed": 7},
            ).json()["scenario"]["failed_edges"]

        assert _run() == _run()

    def test_404_for_missing_analysis_id(self, tmp_path):
        _setup_analysis_dir(tmp_path, _triangle_graph())
        client = TestClient(api_server.create_app())
        resp = client.post(
            "/api/analysis/does-not-exist/disrupt",
            json={"strategy": "random", "target_type": "edge"},
        )
        assert resp.status_code == 404
        assert "does-not-exist" in resp.json()["detail"]

    def test_400_for_nonexistent_node_id(self, tmp_path):
        client = self._client(tmp_path)
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={
                "strategy": "explicit",
                "target_type": "node",
                "failed_nodes": [999],
            },
        )
        assert resp.status_code == 400
        assert "999" in resp.json()["detail"]

    def test_400_for_nonexistent_edge_id(self, tmp_path):
        client = self._client(tmp_path)
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={
                "strategy": "explicit",
                "target_type": "edge",
                "failed_edges": [[1, 99]],
            },
        )
        assert resp.status_code == 400
        assert "99" in resp.json()["detail"] or "1" in resp.json()["detail"]

    def test_400_for_unsupported_strategy(self, tmp_path):
        client = self._client(tmp_path)
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={"strategy": "flood", "target_type": "edge"},
        )
        assert resp.status_code == 400
        assert "flood" in resp.json()["detail"]

    def test_400_for_explicit_with_no_targets(self, tmp_path):
        client = self._client(tmp_path)
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={"strategy": "explicit", "target_type": "edge"},
        )
        assert resp.status_code == 400

    def test_400_for_invalid_target_type(self, tmp_path):
        client = self._client(tmp_path)
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={"strategy": "random", "target_type": "link"},
        )
        assert resp.status_code == 400
        assert "link" in resp.json()["detail"]

    def test_400_on_empty_graph(self, tmp_path):
        """Disruption on a graph with no nodes must return a useful 400."""
        empty = nx.Graph()
        _setup_analysis_dir(tmp_path, empty)
        client = TestClient(api_server.create_app())
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={"strategy": "random", "target_type": "edge"},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_disrupted_graph_has_correct_node_count(self, tmp_path):
        """After removing 1 of 3 nodes, disrupted graph must have exactly 2 nodes."""
        client = self._client(tmp_path)
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={"strategy": "explicit", "target_type": "node", "failed_nodes": [3]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["network_state"]["disrupted"]["n_nodes"] == 2
        dg_nodes = {n["id"] for n in body["disrupted_graph"]["nodes"]}
        assert 3 not in dg_nodes

    def test_efficiency_drop_is_nonnegative(self, tmp_path):
        """Removing an edge cannot increase network efficiency."""
        client = self._client(tmp_path)
        resp = client.post(
            "/api/analysis/test-analysis/disrupt",
            json={"strategy": "explicit", "target_type": "edge", "failed_edges": [[1, 3]]},
        )
        assert resp.status_code == 200
        assert resp.json()["impact_metrics"]["efficiency_drop_frac"] >= 0.0


# ---------------------------------------------------------------------------
# Route & recovery API contract tests
# ---------------------------------------------------------------------------

class TestRoutingAndRecoveryEndpoints:

    def _client(self, tmp_path, graph=None):
        if graph is None:
            graph = _triangle_graph()
        _setup_analysis_dir(tmp_path, graph)
        return TestClient(api_server.create_app())

    def test_route_endpoint_valid_and_invalid_requests(self, tmp_path):
        client = self._client(tmp_path)

        valid = client.post(
            "/api/analysis/test-analysis/route",
            json={"origin": 1, "destination": 3, "current_state": "baseline"},
        )
        assert valid.status_code == 200, valid.text
        body = valid.json()
        assert body["route_exists"] is True
        assert body["route"] == [1, 3] or body["route"][0] == 1
        assert body["route_length"] > 0

        invalid = client.post(
            "/api/analysis/test-analysis/route",
            json={"origin": 999, "destination": 3, "current_state": "baseline"},
        )
        assert invalid.status_code == 400
        assert "999" in invalid.json()["detail"]

    def test_recovery_endpoint_valid_and_invalid_requests(self, tmp_path):
        client = self._client(tmp_path)

        valid = client.post(
            "/api/analysis/test-analysis/recovery",
            json={
                "failed_edges": [[1, 2]],
                "budget": 1,
                "ordering": "explicit_failure_order",
                "weight": "length_px",
            },
        )
        assert valid.status_code == 200, valid.text
        body = valid.json()
        assert "disrupted_state" in body
        assert "restoration_sequence" in body
        assert "resilience_metrics" in body
        assert "final_recovered_state" in body

        invalid = client.post(
            "/api/analysis/test-analysis/recovery",
            json={"ordering": "bad_policy"},
        )
        assert invalid.status_code == 400
        assert "ordering" in invalid.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------

def test_post_and_get_analysis_workflow(tmp_path):
    original_dir = tmp_path / "analysis"
    original_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROUTE_RESILIENCE_ANALYSIS_DIR"] = str(original_dir)

    app = api_server.create_app()
    client = TestClient(app)

    image = Image.new("RGB", (320, 220), color=(20, 30, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/api/road-analysis",
        files={"file": ("sample.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["analysis_id"]
    assert payload["threshold_used"]
    assert payload["input_image_ref"].endswith("/original.png")
    assert payload["artifacts"]["summary_json"].endswith("/summary.json")
    assert payload["artifacts"]["critical_roads"].endswith("/critical_roads.png")

    # Also test /analyze alias
    buffer.seek(0)
    alias_response = client.post(
        "/analyze",
        files={"file": ("sample.png", buffer.getvalue(), "image/png")},
    )
    assert alias_response.status_code == 200, alias_response.text
    alias_payload = alias_response.json()
    assert alias_payload["status"] == "completed"
    assert alias_payload["analysis_id"]

    analysis_id = payload["analysis_id"]
    get_response = client.get(f"/api/analysis/{analysis_id}")
    assert get_response.status_code == 200, get_response.text
    get_payload = get_response.json()
    assert get_payload["analysis_id"] == analysis_id
    assert get_payload["artifacts"]["summary_json"].endswith("/summary.json")

    artifact_dir = original_dir / analysis_id
    assert (artifact_dir / "original.png").exists()
    assert (artifact_dir / "summary.json").exists()
    assert (artifact_dir / "graph.json").exists()
    assert (artifact_dir / "healed_graph.json").exists()
    assert (artifact_dir / "visualization.png").exists()
    assert (artifact_dir / "critical_roads.png").exists()


def test_criticality_endpoint_persists_real_report(tmp_path):
    analysis_dir = tmp_path / "analysis" / "analysis-1"
    analysis_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(20, 30, 40)).save(analysis_dir / "original.png")
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_node(2, pos=(0, 1))
    graph.add_edge(1, 2, length_px=4.0, confidence=0.9, healed=False, width_px=2.0, pts=[[10, 5], [10, 25]])
    (analysis_dir / "healed_graph.json").write_text(json.dumps({
        "coordinate_system": "pixel-space",
        "nodes": [{"id": node, "attributes": data} for node, data in graph.nodes(data=True)],
        "edges": [{"source": u, "target": v, "attributes": data} for u, v, data in graph.edges(data=True)],
    }), encoding="utf-8")

    os.environ["ROUTE_RESILIENCE_ANALYSIS_DIR"] = str(tmp_path / "analysis")
    client = TestClient(api_server.create_app())
    response = client.get("/api/analysis/analysis-1/criticality")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["analysis_id"] == "analysis-1"
    assert payload["criticality"]["graph_state"] == "G_healed"
    assert len(payload["criticality"]["edge_criticality"]) == 1
    assert (analysis_dir / "criticality.json").exists()
    assert payload["artifacts"]["critical_roads"].endswith("/critical_roads.png")
    assert (analysis_dir / "critical_roads.png").exists()
    with Image.open(analysis_dir / "critical_roads.png") as critical_roads:
        assert critical_roads.getpixel((5, 10)) == (255, 64, 64)
