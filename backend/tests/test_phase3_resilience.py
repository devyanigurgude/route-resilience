"""CPU-only regression tests for the fixed-OD Phase III graph experiment."""

import json

import networkx as nx
import pytest

from route_resilience.criticality import articulation_points, bridge_edges, criticality_report
from route_resilience.disruption import create_failure_scenario, simulate_removal
from route_resilience.phase3_pipeline import (
    graph_from_jsonable,
    graph_to_jsonable,
    load_graph_json,
    run_phase3_pipeline,
)
from route_resilience.recovery import _select_greedy_recovery_assets, reinstate_baseline_assets
from route_resilience.resilience import evaluate_resilience, service_score
from route_resilience.routing import route_result, select_baseline_od_pairs


def _graph():
    G = nx.Graph()
    G.add_edge("A", "B", length_px=1.0)
    G.add_edge("B", "C", length_px=2.0)
    G.add_edge("B", "E", length_px=2.0)
    G.add_edge("E", "C", length_px=2.0)
    G.add_edge("C", "D", length_px=1.0)
    return G


def _chain():
    G = nx.path_graph(["A", "B", "C"])
    nx.set_edge_attributes(G, 1.0, "length_px")
    return G


def test_articulation_points_and_bridges_are_detected():
    G = _chain()
    assert articulation_points(G) == ["B"]
    assert {frozenset(edge) for edge in bridge_edges(G)} == {frozenset(("A", "B")), frozenset(("B", "C"))}


def test_criticality_report_keeps_node_and_edge_metrics_separate():
    report = criticality_report(_chain(), top_n=2)
    node = next(record for record in report["node_criticality"] if record["node_id"] == "B")
    assert node["is_articulation_point"] is True
    assert {"degree_centrality", "betweenness_centrality", "is_articulation_point"} <= set(node)
    assert report["edge_criticality"][0]["is_bridge"] is True
    assert "standalone_efficiency_drop" in report["edge_criticality"][0]
    assert "occlusion_recovered" in report["edge_criticality"][0]


def test_criticality_report_preserves_occlusion_recovered_flag():
    G = nx.path_graph(["A", "B", "C"])
    nx.set_edge_attributes(G, 1.0, "length_px")
    G.edges["A", "B"]["healed"] = True
    G.edges["A", "B"]["occlusion_recovered"] = True
    G.edges["B", "C"]["healed"] = True
    G.edges["B", "C"]["occlusion_recovered"] = False
    report = criticality_report(G)
    records = {frozenset(r["edge_id"]): r for r in report["edge_criticality"]}
    assert records[frozenset(("A", "B"))]["occlusion_recovered"] is True
    assert records[frozenset(("A", "B"))]["healed"] is True
    assert records[frozenset(("B", "C"))]["occlusion_recovered"] is False
    assert records[frozenset(("B", "C"))]["healed"] is True


def test_single_edge_failure_and_rerouting():
    G = _graph()
    baseline = route_result(G, "B", "C")
    disrupted = simulate_removal(G, edges_to_remove=[("B", "C")])["surviving_graph"]
    rerouted = route_result(disrupted, "B", "C")
    assert baseline["cost_px"] == 2.0
    assert rerouted["reachable"] is True
    assert rerouted["cost_px"] == 4.0
    assert rerouted["path"] == ["B", "E", "C"]


def test_node_failure_marks_removed_endpoint_pairs_unreachable():
    G = _chain()
    pairs = select_baseline_od_pairs(G, max_pairs=None)
    disrupted = simulate_removal(G, nodes_to_remove=["B"])["surviving_graph"]
    result = evaluate_resilience(G, disrupted, disrupted, pairs)
    route = next(record for record in result["od_routes"] if record["origin"] == "A" and record["destination"] == "B")
    assert route["disrupted"]["reachable"] is False
    assert route["disrupted_service"] == 0.0


def test_multiple_targeted_failure_scenario_is_deterministic():
    G = _graph()
    scenario = create_failure_scenario(G, "multi", "edge", "top_edge_betweenness", count=2)
    assert scenario["requested_failure_count"] == 2
    assert scenario["selection_method"] == "top_edge_betweenness"
    assert len(scenario["failed_edges"]) == 2


def test_seeded_random_failure_scenarios_match():
    G = _graph()
    first = create_failure_scenario(G, "one", "node", "random_uniform", count=2, seed=17)
    second = create_failure_scenario(G, "two", "node", "random_uniform", count=2, seed=17)
    assert first["failed_nodes"] == second["failed_nodes"]
    assert first["random_seed"] == 17


def test_baseline_asset_reinstatement_restores_only_baseline_edge():
    G = _chain()
    disrupted = simulate_removal(G, edges_to_remove=[("A", "B")])["surviving_graph"]
    recovery = reinstate_baseline_assets(G, disrupted, failed_edges=[("A", "B")], budget=1)
    assert recovery["recovered_graph"].has_edge("A", "B")
    assert recovery["policy"] == "restore_failed_baseline_assets"


def test_greedy_recovery_selects_using_weighted_efficiency():
    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 3, 20.0), (0, 4, 20.0), (1, 2, 1.0), (1, 3, 10.0),
        (2, 3, 20.0), (2, 4, 20.0), (3, 4, 10.0),
    ], weight="length_px")
    failed_edges = [(0, 3), (1, 2)]
    disrupted = G.copy()
    disrupted.remove_edges_from(failed_edges)

    selected = _select_greedy_recovery_assets(
        G, disrupted, [("edge", edge) for edge in failed_edges], 1, "length_px"
    )

    assert selected == [("edge", (1, 2))]


def test_node_reinstatement_restores_baseline_node_and_available_incident_edges():
    G = _chain()
    disrupted = simulate_removal(G, nodes_to_remove=["B"])["surviving_graph"]
    recovery = reinstate_baseline_assets(G, disrupted, failed_nodes=["B"], budget=1)
    restored = recovery["recovered_graph"]
    assert restored.has_node("B")
    assert restored.has_edge("A", "B")
    assert restored.has_edge("B", "C")


def test_fixed_od_pairs_keep_denominator_after_node_failure():
    G = _chain()
    pairs = select_baseline_od_pairs(G, max_pairs=None)
    disrupted = simulate_removal(G, nodes_to_remove=["B"])["surviving_graph"]
    result = evaluate_resilience(G, disrupted, disrupted, pairs)
    assert len(result["od_routes"]) == len(pairs) == 3
    assert result["P_D"] == 0.0


def test_graph_json_round_trip_preserves_edge_attributes(tmp_path):
    G = nx.Graph()
    G.add_edge(
        "A", "B", length_px=3.5, width_px=2.0, confidence=0.75,
        healed=True, pts=[(1, 2), (3, 4)],
    )
    path = tmp_path / "graph_healed.json"
    path.write_text(json.dumps(graph_to_jsonable(G)), encoding="utf-8")

    loaded = load_graph_json(path)
    assert loaded.edges["A", "B"] == {
        "length_px": 3.5, "width_px": 2.0, "confidence": 0.75,
        "healed": True, "pts": [[1, 2], [3, 4]],
    }


def test_resilience_formula_for_no_loss_full_loss_and_full_recovery():
    G = _chain()
    pairs = select_baseline_od_pairs(G, max_pairs=None)
    no_loss = evaluate_resilience(G, G, G, pairs)
    broken = simulate_removal(G, edges_to_remove=[("A", "B"), ("B", "C")])["surviving_graph"]
    loss = evaluate_resilience(G, broken, broken, pairs)
    recovered = reinstate_baseline_assets(G, broken, failed_edges=[("A", "B"), ("B", "C")])["recovered_graph"]
    full_recovery = evaluate_resilience(G, broken, recovered, pairs)
    assert no_loss["RI"] == 1.0
    assert loss["RI"] == 0.0
    assert full_recovery["P_D"] == 0.0
    assert full_recovery["P_R"] == 1.0
    assert full_recovery["RI"] == 0.5
    assert service_score(10.0, {"reachable": True, "cost_px": 20.0}) == 0.5


def test_end_to_end_synthetic_phase3_artifacts(tmp_path):
    G = _chain()
    result = run_phase3_pipeline(
        G, G, tmp_path, [{
            "scenario_id": "edge_01", "failure_type": "edge", "selection_method": "top_edge_betweenness",
            "count": 1, "recovery_budget": 1,
        }], od_pair_limit=None,
    )
    assert result["scenarios"][0]["P_R"] == 1.0
    assert result["scenarios"][0]["RI"] == pytest.approx(2.0 / 3.0)
    expected = [
        "graph/graph_extracted.json", "graph/graph_healed.json", "graph/graph_summary.json",
        "criticality/criticality.json", "scenarios/od_pairs.json", "scenarios/edge_01.json",
        "scenarios/edge_01_disrupted_graph.json", "scenarios/edge_01_disruption_metrics.json",
        "recovery/edge_01_recovery_policy.json", "recovery/edge_01_recovered_graph.json",
        "recovery/edge_01_recovery_metrics.json", "resilience/edge_01_od_routes.json",
        "resilience/edge_01_resilience.json", "resilience/resilience_summary.json",
    ]
    assert all((tmp_path / item).exists() for item in expected)
    payload = json.loads((tmp_path / "graph" / "graph_healed.json").read_text(encoding="utf-8"))
    assert graph_from_jsonable(payload).number_of_edges() == G.number_of_edges()
    assert graph_to_jsonable(G)["coordinate_system"] == "pixel-space"
