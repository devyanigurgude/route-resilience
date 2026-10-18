"""
Disruption simulation.

Two removal strategies are implemented, matching the Phase 0 critique
that random/degree-based removal doesn't reflect how real disruptions
happen:
  - `simulate_removal`: generic, given an explicit edge/node list (use
    for topological what-if analysis, or as the mechanism underneath
    the flood-conditioned version below)
  - `flood_conditioned_removal`: removes edges/nodes whose elevation
    (from a per-node DEM sample, attached by the caller) falls below a
    flood level -- physically grounded, not random/degree-based
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import networkx as nx

from .criticality import network_efficiency


def graph_connectivity_ratio(G: nx.Graph) -> float:
    """Fraction of graph nodes belonging to the single largest connected component."""
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    largest_cc = max(nx.connected_components(G), key=len)
    return float(len(largest_cc) / n)


def simulate_removal(
    G: nx.Graph,
    edges_to_remove: Optional[Iterable[Tuple]] = None,
    nodes_to_remove: Optional[Iterable] = None,
    weight: str = "length_px",
) -> Dict:
    """
    Removes the given edges/nodes on a copy of G and reports before/after impact:
    network efficiency drop, connected components, and connectivity ratio change.
    The baseline graph G remains completely unchanged.
    """
    before_efficiency = network_efficiency(G, weight=weight)
    before_components = nx.number_connected_components(G)
    before_cr = graph_connectivity_ratio(G)

    H = G.copy()
    if edges_to_remove:
        H.remove_edges_from([e for e in edges_to_remove if H.has_edge(*e)])
    if nodes_to_remove:
        H.remove_nodes_from([n for n in nodes_to_remove if H.has_node(n)])

    after_efficiency = network_efficiency(H, weight=weight)
    after_components = nx.number_connected_components(H)
    after_cr = graph_connectivity_ratio(H)

    return {
        "before_efficiency": before_efficiency,
        "after_efficiency": after_efficiency,
        "efficiency_drop_abs": float(before_efficiency - after_efficiency),
        "efficiency_drop_frac": (
            (before_efficiency - after_efficiency) / before_efficiency
            if before_efficiency > 0 else 0.0
        ),
        "before_components": before_components,
        "after_components": after_components,
        "newly_disconnected": after_components - before_components,
        "before_connectivity_ratio": before_cr,
        "after_connectivity_ratio": after_cr,
        "connectivity_ratio_drop": float(before_cr - after_cr),
        "surviving_graph": H,
    }


def evaluate_disruption_impact_with_rerouting(
    G: nx.Graph,
    edges_to_remove: Optional[Iterable[Tuple]] = None,
    nodes_to_remove: Optional[Iterable] = None,
    od_pairs: Optional[Iterable[Dict]] = None,
    weight: str = "length_px",
) -> Dict:
    """Simulates edge/node failure on G.copy() and recalculates full network resilience metrics:
    - Connected components and connectivity ratio before/after
    - Global network efficiency before/after (unreachable pairs contribute zero)
    - Origin-Destination rerouting: detours, cost increases, and severed/disconnected pairs
    - Guarantees failed edges/nodes cannot appear in any alternative route
    """
    removal_result = simulate_removal(
        G, edges_to_remove=edges_to_remove, nodes_to_remove=nodes_to_remove, weight=weight
    )
    H = removal_result["surviving_graph"]

    from .routing import select_baseline_od_pairs
    from .resilience import evaluate_state_service

    if od_pairs is None:
        od_pairs = select_baseline_od_pairs(G, max_pairs=100, weight=weight)
    else:
        od_pairs = list(od_pairs)

    od_eval = evaluate_state_service(H, od_pairs, weight=weight) if od_pairs else {"performance": 1.0, "pair_records": []}
    pair_records = od_eval.get("pair_records", [])

    severed_pairs = []
    detoured_pairs = []
    intact_pairs = []

    for r in pair_records:
        orig = r["origin"]
        dest = r["destination"]
        base_route = r["baseline"]
        state_route = r["state"]

        if not state_route["reachable"]:
            severed_pairs.append({
                "origin": orig,
                "destination": dest,
                "status": "disconnected",
                "baseline_cost_px": base_route["cost_px"],
                "alternative_cost_px": None,
                "cost_increase_px": None,
                "detour_ratio": None,
                "alternative_path": None,
            })
        elif r["service_score"] < 1.0:
            cost_increase = float(state_route["cost_px"] - base_route["cost_px"])
            detour_ratio = float(state_route["cost_px"] / base_route["cost_px"]) if base_route["cost_px"] else 1.0
            detoured_pairs.append({
                "origin": orig,
                "destination": dest,
                "status": "detoured",
                "baseline_cost_px": base_route["cost_px"],
                "alternative_cost_px": state_route["cost_px"],
                "cost_increase_px": cost_increase,
                "detour_ratio": detour_ratio,
                "alternative_path": state_route["path"],
            })
        else:
            intact_pairs.append({
                "origin": orig,
                "destination": dest,
                "status": "intact",
                "baseline_cost_px": base_route["cost_px"],
                "alternative_cost_px": state_route["cost_px"],
                "cost_increase_px": 0.0,
                "detour_ratio": 1.0,
                "alternative_path": state_route["path"],
            })

    cost_increases = [p["cost_increase_px"] for p in detoured_pairs if p["cost_increase_px"] is not None]
    detour_ratios = [p["detour_ratio"] for p in detoured_pairs if p["detour_ratio"] is not None]

    return {
        "removal": removal_result,
        "surviving_graph": H,
        "total_od_pairs_evaluated": len(pair_records),
        "disrupted_service_performance": float(od_eval["performance"]),
        "performance_drop": float(1.0 - od_eval["performance"]),
        "severed_pairs_count": len(severed_pairs),
        "detoured_pairs_count": len(detoured_pairs),
        "intact_pairs_count": len(intact_pairs),
        "mean_cost_increase_px": float(np.mean(cost_increases)) if cost_increases else 0.0,
        "max_cost_increase_px": float(np.max(cost_increases)) if cost_increases else 0.0,
        "mean_detour_ratio": float(np.mean(detour_ratios)) if detour_ratios else 1.0,
        "max_detour_ratio": float(np.max(detour_ratios)) if detour_ratios else 1.0,
        "affected_pairs": detoured_pairs + severed_pairs,
        "all_pairs": detoured_pairs + severed_pairs + intact_pairs,
    }


def flood_conditioned_removal(
    G: nx.Graph,
    node_elevation: Dict,
    flood_level: float,
    elevation_attr: str = "elevation",
) -> Dict:
    """
    node_elevation: {node_id: elevation_value} -- caller is responsible
    for sampling this from a real DEM raster at each node's
    georeferenced position (out of scope for this module, which only
    knows about the graph, not the raster/CRS).

    An edge is treated as flood-affected if EITHER endpoint's elevation
    is at or below flood_level -- a conservative, simple rule. A more
    refined version would sample elevation along the edge path, not
    just at endpoints, and incorporate flow direction/drainage rather
    than a flat "water level" -- flagged as a real simplification here,
    not hidden.
    """
    for n, elev in node_elevation.items():
        if G.has_node(n):
            G.nodes[n][elevation_attr] = elev

    affected_nodes = [
        n for n, elev in node_elevation.items()
        if G.has_node(n) and elev <= flood_level
    ]
    affected_edges = [
        (u, v) for u, v in G.edges()
        if node_elevation.get(u, float("inf")) <= flood_level
        or node_elevation.get(v, float("inf")) <= flood_level
    ]

    result = simulate_removal(G, edges_to_remove=affected_edges)
    result["flood_level"] = flood_level
    result["affected_node_count"] = len(affected_nodes)
    result["affected_edge_count"] = len(affected_edges)
    return result


def rank_edges_by_disruption_impact(
    G: nx.Graph, candidate_edges: Optional[List[Tuple]] = None, weight: str = "length_px"
) -> List[Dict]:
    """
    For each candidate edge (defaults to all edges -- expensive on
    large graphs, see Phase 0 §D's scalability warning), removes it
    alone and records the efficiency drop it alone causes. This is the
    brute-force, exact version; for city-scale graphs replace with a
    sampled/approximate variant (e.g. only test top-K edges by
    betweenness as candidates) rather than testing every edge.
    """
    if candidate_edges is None:
        candidate_edges = list(G.edges())

    results = []
    for u, v in candidate_edges:
        impact = simulate_removal(G, edges_to_remove=[(u, v)], weight=weight)
        results.append({
            "edge": (u, v),
            "efficiency_drop_frac": impact["efficiency_drop_frac"],
            "disconnects": impact["newly_disconnected"] > 0,
            "newly_disconnected": impact["newly_disconnected"],
        })

    results.sort(key=lambda r: r["efficiency_drop_frac"], reverse=True)
    return results


def create_failure_scenario(
    G: nx.Graph,
    scenario_id: str,
    failure_type: str,
    selection_method: str,
    count: int = 1,
    seed: Optional[int] = None,
    weight: str = "length_px",
) -> Dict:
    """Create a reproducible node or edge failure scenario from ``G``.

    Supported selection methods are ``random_uniform``,
    ``top_node_betweenness``, ``articulation_points``,
    ``top_edge_betweenness``, and ``top_edge_disruption_impact``.
    """
    if failure_type not in {"node", "edge"}:
        raise ValueError("failure_type must be 'node' or 'edge'")
    if count < 1:
        raise ValueError("count must be at least 1")

    if failure_type == "node":
        candidates = list(G.nodes())
        if selection_method == "random_uniform":
            chosen = _seeded_choice(candidates, count, seed)
        elif selection_method == "top_node_betweenness":
            scores = nx.betweenness_centrality(G, weight=weight, normalized=True)
            chosen = _top_items(scores, count)
        elif selection_method == "articulation_points":
            chosen = sorted(nx.articulation_points(G), key=repr)[:count]
        else:
            raise ValueError(f"Unsupported node selection method: {selection_method}")
        return _scenario_record(scenario_id, failure_type, selection_method, count, seed, chosen, [], weight)

    candidates = list(G.edges())
    if selection_method == "random_uniform":
        chosen = _seeded_choice(candidates, count, seed)
    elif selection_method == "top_edge_betweenness":
        scores = nx.edge_betweenness_centrality(G, weight=weight, normalized=True)
        chosen = _top_items(scores, count)
    elif selection_method == "top_edge_disruption_impact":
        chosen = [r["edge"] for r in rank_edges_by_disruption_impact(G, weight=weight)[:count]]
    else:
        raise ValueError(f"Unsupported edge selection method: {selection_method}")
    return _scenario_record(scenario_id, failure_type, selection_method, count, seed, [], chosen, weight)


def _seeded_choice(candidates, count: int, seed: Optional[int]):
    if count > len(candidates):
        raise ValueError("failure count exceeds available graph assets")
    ordered = sorted(candidates, key=repr)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ordered), size=count, replace=False)
    return [ordered[int(index)] for index in indices]


def _top_items(scores: Dict, count: int):
    return [item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], repr(kv[0])))[:count]]


def _scenario_record(scenario_id, failure_type, selection_method, count, seed, nodes, edges, weight):
    return {
        "scenario_id": scenario_id,
        "baseline_graph_state": "G_healed",
        "failure_type": failure_type,
        "selection_method": selection_method,
        "selection_metric": None if selection_method == "random_uniform" else selection_method.removeprefix("top_"),
        "random_seed": seed if selection_method == "random_uniform" else None,
        "requested_failure_count": count,
        "failed_nodes": list(nodes),
        "failed_edges": [tuple(edge) for edge in edges],
        "weight": weight,
    }
