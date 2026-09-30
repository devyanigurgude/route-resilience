"""Recovery policies for pixel-space road graph experiments."""

from typing import Dict, Iterable, Optional, Tuple

import networkx as nx

from .criticality import network_efficiency


def reinstate_baseline_assets(
    baseline_graph: nx.Graph,
    disrupted_graph: nx.Graph,
    failed_nodes: Optional[Iterable] = None,
    failed_edges: Optional[Iterable[Tuple]] = None,
    budget: Optional[int] = None,
    ordering: str = "explicit_failure_order",
    weight: str = "length_px",
) -> Dict:
    """
    Recover failed baseline assets using a selected recovery policy.

    Policies:
      - explicit_failure_order:
          Restore failed assets in the order supplied.
      - criticality_greedy:
          Iteratively restore the failed asset that gives the largest
          improvement in network efficiency.
    """

    nodes = list(failed_nodes or [])
    edges = [tuple(edge) for edge in (failed_edges or [])]

    assets = [("node", node) for node in nodes] + [
        ("edge", edge) for edge in edges
    ]

    if budget is None:
        budget = len(assets)

    if budget < 0:
        raise ValueError("budget must be non-negative or None")

    H = disrupted_graph.copy()

    recovered_nodes = []
    recovered_edges = []

    if ordering == "explicit_failure_order":
        selected_assets = assets[:budget]

    elif ordering == "criticality_greedy":
        selected_assets = _select_greedy_recovery_assets(
            baseline_graph,
            H,
            assets,
            budget,
            weight,
        )

    else:
        raise ValueError(
            "ordering must be 'explicit_failure_order' "
            "or 'criticality_greedy'"
        )

    for asset_type, asset in selected_assets:

        if asset_type == "node":

            node = asset

            if not baseline_graph.has_node(node):
                continue

            H.add_node(
                node,
                **baseline_graph.nodes[node],
            )

            recovered_nodes.append(node)

            for neighbor, data in baseline_graph[node].items():

                if not H.has_node(neighbor):
                    continue

                H.add_edge(
                    node,
                    neighbor,
                    **data,
                )

                _append_edge_once(
                    recovered_edges,
                    (node, neighbor),
                )

        else:

            u, v = asset

            if not baseline_graph.has_edge(u, v):
                continue

            if not H.has_node(u) or not H.has_node(v):
                continue

            H.add_edge(
                u,
                v,
                **baseline_graph.edges[u, v],
            )

            _append_edge_once(
                recovered_edges,
                (u, v),
            )

    return {
        "recovered_graph": H,
        "policy": "restore_failed_baseline_assets",
        "ordering": ordering,
        "budget": budget,
        "recovered_nodes": recovered_nodes,
        "recovered_edges": recovered_edges,
    }


def _select_greedy_recovery_assets(
    baseline_graph: nx.Graph,
    disrupted_graph: nx.Graph,
    assets,
    budget,
    weight,
):
    """
    Select recovery assets greedily.

    At every step, temporarily restores each remaining candidate and
    measures the resulting network efficiency. The candidate producing
    the largest efficiency improvement is selected.
    """

    selected = []
    remaining = list(assets)

    current_graph = disrupted_graph.copy()

    for _ in range(min(budget, len(remaining))):

        best_asset = None
        best_score = float("-inf")

        current_efficiency = _network_efficiency(
            current_graph,
            weight,
        )

        for asset in remaining:

            candidate_graph = current_graph.copy()

            _restore_asset(
                baseline_graph,
                candidate_graph,
                asset,
            )

            candidate_efficiency = _network_efficiency(
                candidate_graph,
                weight,
            )

            gain = candidate_efficiency - current_efficiency

            if gain > best_score:
                best_score = gain
                best_asset = asset

        if best_asset is None:
            break

        _restore_asset(
            baseline_graph,
            current_graph,
            best_asset,
        )

        selected.append(best_asset)
        remaining.remove(best_asset)

    return selected


def _restore_asset(
    baseline_graph,
    graph,
    asset,
):

    asset_type, value = asset

    if asset_type == "node":

        node = value

        if not baseline_graph.has_node(node):
            return

        graph.add_node(
            node,
            **baseline_graph.nodes[node],
        )

        for neighbor, data in baseline_graph[node].items():

            if graph.has_node(neighbor):
                graph.add_edge(
                    node,
                    neighbor,
                    **data,
                )

    else:

        u, v = value

        if (
            baseline_graph.has_edge(u, v)
            and graph.has_node(u)
            and graph.has_node(v)
        ):
            graph.add_edge(
                u,
                v,
                **baseline_graph.edges[u, v],
            )


def _network_efficiency(
    graph: nx.Graph,
    weight: str,
) -> float:

    if graph.number_of_nodes() < 2:
        return 0.0

    try:
        return network_efficiency(graph, weight=weight)
    except Exception:
        return 0.0


def _append_edge_once(edges, edge):

    key = frozenset(edge)

    if not any(
        frozenset(existing) == key
        for existing in edges
    ):
        edges.append(edge)