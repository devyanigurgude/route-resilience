"""Fixed baseline origin-destination routing for pixel-space graphs."""

from itertools import combinations
from typing import Dict, Iterable, List, Optional

import networkx as nx
import numpy as np


def route_result(G: nx.Graph, origin, destination, weight: str = "length_px") -> Dict:
    """Return the shortest available pixel-space path and cost, if one exists."""
    result = {"origin": origin, "destination": destination, "reachable": False, "path": None, "cost_px": None}
    if not G.has_node(origin) or not G.has_node(destination):
        return result
    try:
        result["path"] = list(nx.shortest_path(G, origin, destination, weight=weight))
        result["cost_px"] = float(nx.shortest_path_length(G, origin, destination, weight=weight))
        result["reachable"] = True
    except nx.NetworkXNoPath:
        pass
    return result


def select_automatic_route_pair(G: nx.Graph, weight: str = "length_px"):
    """Choose a deterministic, well-separated pair from the largest routable component.

    A two-sweep weighted search avoids an all-pairs search while ensuring the
    selected endpoints are connected by actual graph edges.
    """
    components = [
        component
        for component in nx.connected_components(G)
        if len(component) > 1
    ]
    if not components:
        return None

    component = min(
        components,
        key=lambda nodes: (-len(nodes), tuple(sorted(map(repr, nodes)))),
    )
    subgraph = G.subgraph(component)
    start = min(component, key=repr)
    first_distances = nx.single_source_dijkstra_path_length(subgraph, start, weight=weight)
    first = max(first_distances, key=lambda node: (first_distances[node], repr(node)))
    second_distances = nx.single_source_dijkstra_path_length(subgraph, first, weight=weight)
    second = max(
        (node for node in second_distances if node != first),
        key=lambda node: (second_distances[node], repr(node)),
    )
    return first, second


def select_reachable_node(G: nx.Graph, origin, weight: str = "length_px"):
    """Choose a deterministic destination reachable from a supplied origin."""
    if not G.has_node(origin):
        return None
    distances = nx.single_source_dijkstra_path_length(G, origin, weight=weight)
    candidates = (node for node in distances if node != origin)
    return max(candidates, key=lambda node: (distances[node], repr(node)), default=None)


def select_baseline_od_pairs(
    G: nx.Graph, max_pairs: Optional[int] = 200, seed: int = 42, weight: str = "length_px"
) -> List[Dict]:
    """Build a fixed set of baseline-connected OD pairs with baseline routes."""
    candidates = []
    nodes = sorted(G.nodes(), key=repr)
    for origin, destination in combinations(nodes, 2):
        route = route_result(G, origin, destination, weight=weight)
        if route["reachable"]:
            candidates.append(route)
    if max_pairs is None or max_pairs >= len(candidates):
        return candidates
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive or None")
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(candidates), size=max_pairs, replace=False).tolist())
    return [candidates[index] for index in indices]


def evaluate_od_pairs(G: nx.Graph, od_pairs: Iterable[Dict], weight: str = "length_px") -> List[Dict]:
    """Route an unchanged baseline OD set on another graph state."""
    return [route_result(G, pair["origin"], pair["destination"], weight=weight) for pair in od_pairs]
