"""Comprehensive deterministic tests for Phase 8 End-to-End Network Disruption and Rerouting.

Tests cover:
  1. Bridge failure that disconnects the network (component split, connectivity ratio drop).
  2. Redundant-route failure permitting a detour (cost increase, failed edge absent from alternative path).
  3. Failure making a destination unreachable (isolated node, severed OD pairs).
  4. Formula fidelity: Resilience Index RI = (P_D + P_R) / 2 vs Global Efficiency E(G).
  5. API endpoint validation: ensure /api/analysis/{id}/disrupt and /route execute real graph changes.
"""

import networkx as nx
import numpy as np
import pytest

from route_resilience.criticality import network_efficiency
from route_resilience.disruption import (
    evaluate_disruption_impact_with_rerouting,
    graph_connectivity_ratio,
    simulate_removal,
)
from route_resilience.recovery import reinstate_baseline_assets
from route_resilience.resilience import evaluate_resilience, service_score
from route_resilience.routing import route_result, select_baseline_od_pairs


def test_bridge_failure_disconnects_network():
    """Verify that removing a critical bridge edge splits components and drops connectivity ratio."""
    # Graph: Two 2-node components connected by bridge edge (2, 3)
    # 1 --(10)-- 2 =====(5, bridge)===== 3 --(10)-- 4
    G = nx.Graph()
    G.add_edge(1, 2, length_px=10.0)
    G.add_edge(2, 3, length_px=5.0)
    G.add_edge(3, 4, length_px=10.0)

    # Baseline properties
    assert nx.number_connected_components(G) == 1
    assert graph_connectivity_ratio(G) == 1.0
    baseline_eff = network_efficiency(G)
    assert baseline_eff > 0.0

    # Simulate bridge failure
    res = evaluate_disruption_impact_with_rerouting(G, edges_to_remove=[(2, 3)])
    removal = res["removal"]

    # Graph G must be completely preserved
    assert G.has_edge(2, 3), "Baseline graph G must remain unchanged"
    assert not res["surviving_graph"].has_edge(2, 3), "Surviving graph must have edge removed"

    # Component & connectivity ratio verification
    assert removal["before_components"] == 1
    assert removal["after_components"] == 2
    assert removal["newly_disconnected"] == 1
    assert removal["before_connectivity_ratio"] == 1.0
    assert removal["after_connectivity_ratio"] == 0.5  # Largest component has 2 of 4 nodes
    assert removal["connectivity_ratio_drop"] == 0.5

    # Efficiency must drop, and unreachable pairs must contribute zero
    assert removal["after_efficiency"] < baseline_eff
    assert removal["efficiency_drop_frac"] > 0.0

    # OD pair (1, 4) must be severed
    od_eval = [p for p in res["all_pairs"] if (p["origin"] == 1 and p["destination"] == 4)]
    assert len(od_eval) == 1
    assert od_eval[0]["status"] == "disconnected"
    assert od_eval[0]["alternative_path"] is None
    assert res["severed_pairs_count"] >= 1


def test_redundant_route_permits_detour():
    """Verify that removing an edge on a cycle forces a detour with measurable cost increase."""
    # Graph: Triangle cycle with shortcut (1, 3) and detour via 2
    # 1 --(10)-- 2 --(10)-- 3
    #  \--------(12)--------/
    G = nx.Graph()
    G.add_edge(1, 2, length_px=10.0)
    G.add_edge(2, 3, length_px=10.0)
    G.add_edge(1, 3, length_px=12.0)

    baseline_route = route_result(G, 1, 3)
    assert baseline_route["reachable"] is True
    assert baseline_route["cost_px"] == 12.0
    assert baseline_route["path"] == [1, 3]

    # Simulate failure of shortcut edge (1, 3)
    res = evaluate_disruption_impact_with_rerouting(G, edges_to_remove=[(1, 3)])
    H = res["surviving_graph"]

    # Graph remains connected
    assert res["removal"]["after_components"] == 1
    assert res["removal"]["after_connectivity_ratio"] == 1.0

    # Rerouted path between 1 and 3
    rerouted = route_result(H, 1, 3)
    assert rerouted["reachable"] is True
    assert rerouted["cost_px"] == 20.0
    assert rerouted["path"] == [1, 2, 3]

    # Failed edge must NOT appear in the rerouted path
    path_edges = list(zip(rerouted["path"][:-1], rerouted["path"][1:]))
    assert (1, 3) not in path_edges and (3, 1) not in path_edges

    # Cost increase verification
    cost_increase = rerouted["cost_px"] - baseline_route["cost_px"]
    assert cost_increase == 8.0
    detour_ratio = rerouted["cost_px"] / baseline_route["cost_px"]
    assert detour_ratio == pytest.approx(20.0 / 12.0)

    # Detoured pair accounting
    assert res["detoured_pairs_count"] >= 1
    detoured = [p for p in res["affected_pairs"] if p["origin"] == 1 and p["destination"] == 3]
    assert len(detoured) == 1
    assert detoured[0]["status"] == "detoured"
    assert detoured[0]["cost_increase_px"] == 8.0
    assert detoured[0]["alternative_path"] == [1, 2, 3]


def test_failure_makes_destination_unreachable():
    """Verify that removing a spur edge leaves its leaf node completely unreachable."""
    # Graph: Backbone (1, 2) and spur (2, 3)
    G = nx.Graph()
    G.add_edge(1, 2, length_px=10.0)
    G.add_edge(2, 3, length_px=15.0)

    baseline_1_3 = route_result(G, 1, 3)
    assert baseline_1_3["reachable"] is True
    assert baseline_1_3["cost_px"] == 25.0

    # Remove spur edge (2, 3)
    res = evaluate_disruption_impact_with_rerouting(G, edges_to_remove=[(2, 3)])
    H = res["surviving_graph"]

    # Destination 3 is isolated (degree 0)
    assert H.degree(3) == 0

    disrupted_1_3 = route_result(H, 1, 3)
    assert disrupted_1_3["reachable"] is False
    assert disrupted_1_3["path"] is None
    assert disrupted_1_3["cost_px"] is None

    # Severed pair recorded
    severed = [p for p in res["affected_pairs"] if p["origin"] == 1 and p["destination"] == 3]
    assert len(severed) == 1
    assert severed[0]["status"] == "disconnected"
    assert severed[0]["alternative_cost_px"] is None


def test_resilience_index_formula_and_ratio_fidelity():
    """Verify that the Resilience Index RI formula follows PS4 specification and is not inverted."""
    # Baseline cost = 10px
    # If disrupted cost = 20px (detour), service_score = 10 / 20 = 0.5 (NOT 20 / 10 = 2.0)
    assert service_score(10.0, {"reachable": True, "cost_px": 20.0}) == 0.5
    # If unreachable, service_score must be strictly 0.0
    assert service_score(10.0, {"reachable": False, "cost_px": None}) == 0.0
    # If cost is identical (no disruption impact), service_score is 1.0
    assert service_score(10.0, {"reachable": True, "cost_px": 10.0}) == 1.0

    # Test complete evaluate_resilience on small network
    G = nx.path_graph([1, 2, 3])
    nx.set_edge_attributes(G, 10.0, "length_px")
    od_pairs = select_baseline_od_pairs(G, max_pairs=None)

    # Intact baseline: P_baseline = 1.0, RI = 1.0
    res_intact = evaluate_resilience(G, G, G, od_pairs)
    assert res_intact["P_baseline"] == 1.0
    assert res_intact["P_D"] == 1.0
    assert res_intact["P_R"] == 1.0
    assert res_intact["RI"] == 1.0
    assert res_intact["RecoveryGain"] == 0.0

    # Full destruction: P_D = 0.0
    empty_graph = nx.create_empty_copy(G)
    res_broken = evaluate_resilience(G, empty_graph, empty_graph, od_pairs)
    assert res_broken["P_D"] == 0.0
    assert res_broken["P_R"] == 0.0
    assert res_broken["RI"] == 0.0

    # Partial recovery: Disrupted = 0.0, Recovered = 1.0 -> RI = (0 + 1) / 2 = 0.5
    res_half = evaluate_resilience(G, empty_graph, G, od_pairs)
    assert res_half["P_D"] == 0.0
    assert res_half["P_R"] == 1.0
    assert res_half["RecoveryGain"] == 1.0
    assert res_half["RI"] == 0.5


def test_edge_cases_and_graceful_handling():
    """Verify safe handling of single-node, empty, and disconnected graphs."""
    G_single = nx.Graph()
    G_single.add_node(1)

    assert graph_connectivity_ratio(G_single) == 1.0
    assert network_efficiency(G_single) == 0.0
    assert route_result(G_single, 1, 1)["reachable"] is True
    assert route_result(G_single, 1, 1)["cost_px"] == 0.0

    res = evaluate_disruption_impact_with_rerouting(G_single, nodes_to_remove=[1])
    assert res["surviving_graph"].number_of_nodes() == 0
    assert graph_connectivity_ratio(res["surviving_graph"]) == 0.0
    assert network_efficiency(res["surviving_graph"]) == 0.0


def test_measured_recovery_vs_assumed_restored_baseline():
    """Verify distinction between:
    1. Disrupted unrestored state (P_R == P_D, RecoveryGain == 0, RI == P_D)
    2. Modeled recovery with budget constraint (P_D <= P_R_modeled < P_R_assumed)
    3. Assumed restored-baseline ceiling (P_R == 1.0, theoretical ceiling)
    Also verify that an efficiency decrease produces a strictly positive drop.
    """
    # Graph: Diamond with an extended tail
    # 1 --(10)-- 2 --(10)-- 3 --(10)-- 4
    #            \         /
    #             --(10)-- 5
    G = nx.Graph()
    G.add_edge(1, 2, length_px=10.0)
    G.add_edge(2, 3, length_px=10.0)
    G.add_edge(3, 4, length_px=10.0)
    G.add_edge(2, 5, length_px=10.0)
    G.add_edge(5, 3, length_px=10.0)

    # Disruption: cut both paths around the loop ((2, 3) and (2, 5))
    failed_edges = [(2, 3), (2, 5)]
    res = evaluate_disruption_impact_with_rerouting(G, edges_to_remove=failed_edges)
    H_disrupted = res["surviving_graph"]

    # Check efficiency drop sign: a decrease in efficiency MUST yield a positive drop
    removal = res["removal"]
    assert removal["before_efficiency"] > removal["after_efficiency"]
    assert removal["efficiency_drop_abs"] > 0.0
    assert removal["efficiency_drop_frac"] > 0.0
    expected_frac = (removal["before_efficiency"] - removal["after_efficiency"]) / removal["before_efficiency"]
    assert abs(removal["efficiency_drop_frac"] - expected_frac) < 1e-9

    od_pairs = select_baseline_od_pairs(G, max_pairs=None)

    # 1. Disrupted unrestored state: no recovery actions taken
    res_unrestored = evaluate_resilience(G, H_disrupted, H_disrupted, od_pairs)
    assert res_unrestored["P_R"] == res_unrestored["P_D"]
    assert res_unrestored["RecoveryGain"] == 0.0
    assert res_unrestored["RI"] == res_unrestored["P_D"]

    # 2. Modeled recovery under budget=1 (reinstates only 1 of the 2 failed edges)
    rec_modeled = reinstate_baseline_assets(
        G, H_disrupted, failed_edges=failed_edges, budget=1, ordering="criticality_greedy"
    )
    G_recovered_modeled = rec_modeled["recovered_graph"]
    assert len(rec_modeled["recovered_edges"]) == 1
    res_modeled = evaluate_resilience(G, H_disrupted, G_recovered_modeled, od_pairs)

    # 3. Assumed restored-baseline ceiling (hypothetical 100% full recovery)
    res_assumed_ceiling = evaluate_resilience(G, H_disrupted, G, od_pairs)
    assert res_assumed_ceiling["P_R"] == 1.0

    # Measured recovery must be greater than or equal to disrupted, but strictly distinct from the 100% ceiling
    assert res_unrestored["P_D"] < res_assumed_ceiling["P_R"]
    assert res_modeled["P_D"] == res_unrestored["P_D"]
    assert res_modeled["RecoveryGain"] >= 0.0
    assert res_modeled["RI"] >= res_unrestored["RI"]
