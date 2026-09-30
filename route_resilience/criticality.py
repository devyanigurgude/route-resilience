"""
Graph-theoretic criticality analysis.

Implements, in order of how load-bearing they are to the project's
actual novelty claim (Phase 0, Part 2/3):
  - degree / betweenness centrality (standard baseline, NOT the novelty)
  - width-weighted centrality (capacity proxy, cheap and defensible)
  - detour cost (the planner-relevant output -- how much worse does a
    route get, not just "is it disconnected")
  - confidence-aware vs confidence-blind ranking stability under
    bootstrap perturbation -- this is the H4 experiment, the actual
    headline result the project needs
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


def compute_centrality(G: nx.Graph, weight: str = None) -> Dict:
    """
    weight=None -> unweighted (pure topology).
    weight="length_px" -> distance-weighted (standard betweenness).
    weight="inv_width" -> capacity-weighted, see `add_capacity_weight`.
    """
    return {
        "degree": dict(nx.degree_centrality(G)),
        "betweenness": dict(nx.betweenness_centrality(G, weight=weight, normalized=True)),
    }


def add_capacity_weight(G: nx.Graph, width_attr: str = "width_px") -> nx.Graph:
    """
    Adds an 'inv_width' edge attribute = 1 / width, so that shortest-
    path / betweenness algorithms (which minimize weight) naturally
    prefer wider, higher-capacity roads -- turning "shortest path" into
    something closer to "fastest/most-capable path" without needing a
    real traffic model. This directly operationalizes the "unweighted
    centrality is a weak proxy" critique from Phase 0.
    """
    for u, v, data in G.edges(data=True):
        width = max(data.get(width_attr, 1.0), 1e-3)
        data["inv_width"] = 1.0 / width
    return G


def add_confidence_weight(G: nx.Graph, confidence_attr: str = "confidence") -> nx.Graph:
    """
    Adds an 'inv_confidence' edge attribute = 1 / confidence, so that
    shortest-path/betweenness algorithms treat low-confidence
    (possibly-fabricated) edges as "expensive" and route around them
    when possible -- i.e. a road the model is unsure about contributes
    less to computed criticality than one it's confident about.

    NOTE on the H4 experiment (Phase 0, Part 3): a complete version of
    H4 compares TWO things, not one:
      1. rankings computed with weight="inv_confidence" (confidence
         baked into the criticality computation itself) vs weight=None
         (confidence-blind ranking) -- this function enables that.
      2. the STABILITY of each ranking under `criticality_ranking_stability`
         (confidence-aware perturbation, already implemented below).
    Doing only #2 (perturbation) without #1 (confidence-weighted
    ranking) is a simplified, partial operationalization -- it tests
    whether the *perturbation* found low-confidence edges, not whether
    a confidence-aware *ranking method* is more trustworthy. Use both
    together for the real result, not perturbation stability alone.
    """
    for u, v, data in G.edges(data=True):
        conf = max(data.get(confidence_attr, 1.0), 1e-3)
        data["inv_confidence"] = 1.0 / conf
    return G


def detour_cost(G: nx.Graph, u, v, weight: str = "length_px") -> float:
    """
    Increase in shortest-path distance between u and v's OTHER
    neighbours if edge (u, v) is removed, vs. the direct edge length.
    Returns np.inf if removal disconnects them entirely (a real
    binary-criticality event), otherwise the extra distance a detour
    costs -- the more planner-legible output (Phase 0, Part 4).
    """
    if not G.has_edge(u, v):
        raise ValueError(f"No edge ({u}, {v}) in graph")

    direct_length = G.edges[u, v].get(weight, 1.0)
    H = G.copy()
    H.remove_edge(u, v)

    try:
        alt_length = nx.shortest_path_length(H, u, v, weight=weight)
        return float(alt_length - direct_length)
    except nx.NetworkXNoPath:
        return float("inf")


def top_n_edges_by_betweenness(G: nx.Graph, n: int = 10, weight: str = None) -> List[Tuple]:
    eb = nx.edge_betweenness_centrality(G, weight=weight, normalized=True)
    ranked = sorted(eb.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:n]


def _perturb_graph(G: nx.Graph, rng: np.random.Generator, confidence_aware: bool) -> nx.Graph:
    """
    Bootstrap-resample the graph by randomly dropping edges, with the
    drop probability driven by (1 - confidence) when confidence_aware,
    or uniform when not. This simulates "what if the extraction model's
    own uncertainty corresponds to real errors" -- the direct
    operationalization of H4 (Phase 0, Part 3/6, experiment E5).
    """
    H = G.copy()
    edges = list(H.edges(data=True))
    for u, v, data in edges:
        if confidence_aware:
            drop_prob = 1.0 - data.get("confidence", 1.0)
        else:
            drop_prob = 0.1  # uniform baseline noise, same expected magnitude order
        if rng.random() < drop_prob * 0.3:  # scale down so we don't shred the graph
            H.remove_edge(u, v)
    return H


def criticality_ranking_stability(
    G: nx.Graph,
    n_bootstrap: int = 50,
    top_n: int = 10,
    weight: str = None,
    confidence_aware: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Runs the H4 experiment: how stable is the top-N edge-criticality
    ranking under repeated perturbation of the graph?

    Stability is measured as mean Jaccard overlap between the
    unperturbed top-N ranked edges and each bootstrap sample's top-N.
    Compare the output of this function with confidence_aware=True vs
    confidence_aware=False on the SAME graph to test H4 directly --
    that comparison, not either number alone, is the actual result.
    """
    rng = np.random.default_rng(seed)

    base_top = set(e for e, _ in top_n_edges_by_betweenness(G, n=top_n, weight=weight))

    overlaps = []
    for _ in range(n_bootstrap):
        H = _perturb_graph(G, rng, confidence_aware=confidence_aware)
        if H.number_of_edges() == 0:
            overlaps.append(0.0)
            continue
        boot_top = set(e for e, _ in top_n_edges_by_betweenness(H, n=top_n, weight=weight))
        # edges may not exist in H  if dropped -- Jaccard over what's comparable
        if not base_top and not boot_top:
            overlaps.append(1.0)
            continue
        intersection = len(base_top & boot_top)
        union = len(base_top | boot_top)
        overlaps.append(intersection / union if union else 0.0)

    return {
        "mean_top_n_overlap": float(np.mean(overlaps)),
        "std_top_n_overlap": float(np.std(overlaps)),
        "n_bootstrap": n_bootstrap,
        "confidence_aware": confidence_aware,
    }


def network_efficiency(G: nx.Graph, weight: str = "length_px") -> float:
    """
    Global efficiency = average of 1/shortest_path_length over all
    node pairs (0 for disconnected pairs). Standard aggregate
    resilience metric -- report alongside per-edge criticality, not
    instead of it (Phase 0, §D).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n < 2:
        return 0.0

    total = 0.0
    for i, u in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, u, weight=weight)
        for v in nodes[i + 1:]:
            d = lengths.get(v)
            if d and d > 0:
                total += 1.0 / d

    n_pairs = n * (n - 1) / 2
    return float(total / n_pairs) if n_pairs else 0.0


def articulation_points(G: nx.Graph) -> List:
    """Return graph cut vertices; each removal increases component count."""
    return list(nx.articulation_points(G))


def bridge_edges(G: nx.Graph) -> List[Tuple]:
    """Return graph bridges; each removal increases component count."""
    return list(nx.bridges(G))


def criticality_report(G: nx.Graph, weight: str = "length_px", top_n: int = 10) -> Dict:
    """Keep node and edge criticality measures separate in one JSON-ready report."""
    from .disruption import rank_edges_by_disruption_impact

    centrality = compute_centrality(G, weight=weight)
    articulation = set(articulation_points(G))
    bridges = {frozenset(edge) for edge in bridge_edges(G)}
    impact_by_edge = {
        frozenset(record["edge"]): record for record in rank_edges_by_disruption_impact(G, weight=weight)
    }

    node_records = []
    for node in G.nodes():
        record = {
            "node_id": node,
            "degree_centrality": float(centrality["degree"][node]),
            "betweenness_centrality": float(centrality["betweenness"][node]),
            "is_articulation_point": node in articulation,
        }
        if "pos" in G.nodes[node]:
            record["pos"] = G.nodes[node]["pos"]
        node_records.append(record)
    node_records.sort(key=lambda r: (-r["betweenness_centrality"], repr(r["node_id"])))

    edge_records = []
    edge_betweenness = dict(nx.edge_betweenness_centrality(G, weight=weight, normalized=True))
    for u, v, data in G.edges(data=True):
        detour = detour_cost(G, u, v, weight=weight)
        impact = impact_by_edge[frozenset((u, v))]
        betweenness = edge_betweenness.get((u, v))
        if betweenness is None:
            betweenness = edge_betweenness[(v, u)]
        edge_records.append({
            "edge_id": (u, v),
            "length_px": float(data.get("length_px", 1.0)),
            "confidence": data.get("confidence"),
            "healed": bool(data.get("healed", False)),
            "width_px": data.get("width_px"),
            "edge_betweenness": float(betweenness),
            "detour_cost_px": None if np.isinf(detour) else float(detour),
            "detour_disconnected": bool(np.isinf(detour)),
            "is_bridge": frozenset((u, v)) in bridges,
            "standalone_efficiency_drop": float(impact["efficiency_drop_frac"]),
            "standalone_component_delta": int(impact["newly_disconnected"]),
        })
    edge_records.sort(key=lambda r: (-r["edge_betweenness"], repr(r["edge_id"])))
    by_impact = sorted(edge_records, key=lambda r: (-r["standalone_efficiency_drop"], repr(r["edge_id"])))

    return {
        "graph_state": "G_healed",
        "weight": weight,
        "node_criticality": node_records,
        "edge_criticality": edge_records,
        "articulation_points": sorted(articulation, key=repr),
        "bridge_edges": [record["edge_id"] for record in edge_records if record["is_bridge"]],
        "top_vulnerable_nodes_by_betweenness": [r["node_id"] for r in node_records[:top_n]],
        "top_vulnerable_edges_by_betweenness": [r["edge_id"] for r in edge_records[:top_n]],
        "top_vulnerable_edges_by_disruption_impact": [r["edge_id"] for r in by_impact[:top_n]],
    }
