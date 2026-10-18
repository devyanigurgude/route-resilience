"""
Topology-level validation (Phase 0, Part 9, tier 2): pixel IoU alone
cannot tell you whether the extracted graph is USABLE, because it's
blind to connectivity. This module gives a simplified, honestly-labeled
approximation of APLS, plus a graph-vs-graph edge-count/length sanity
comparison. It is NOT a reimplementation of the official SpaceNet APLS
metric (which uses proper spatial indexing and edge-snapping at fixed
intervals) -- treat this as a fast proxy for development iteration, and
switch to the official `apls` package for anything you intend to report
as a benchmark-comparable number.
"""

from typing import Dict

import networkx as nx
import numpy as np


def simplified_apls(
    G_pred: nx.Graph,
    G_gt: nx.Graph,
    pos_attr: str = "pos",
    weight: str = "length_px",
    snap_radius: float = 10.0,
    n_pairs: int = 200,
    seed: int = 0,
) -> Dict:
    """
    Simplified path-length-similarity metric:
      1. Sample random node pairs from G_gt.
      2. For each GT node, find its nearest node in G_pred within
         `snap_radius` pixels (unmatched -> pair is skipped, counted as
         a miss).
      3. Compare shortest-path length between the matched pair in
         G_pred against the GT path length; score =
         1 - min(1, |len_pred - len_gt| / len_gt).
      4. Average the scores, including 0 for skipped/no-path pairs.

    Returns a dict with the averaged score and the miss rate, so you
    can see whether a low score is coming from routing errors or from
    node-matching misses (fabricated/missing topology) -- these mean
    different things and shouldn't be collapsed into one number without
    also reporting which is driving it.
    """
    rng = np.random.default_rng(seed)
    gt_nodes = list(G_gt.nodes())
    if len(gt_nodes) < 2:
        return {"score": 0.0, "miss_rate": 1.0, "n_pairs_evaluated": 0}

    pred_positions = {n: np.array(G_pred.nodes[n][pos_attr]) for n in G_pred.nodes()}

    def nearest_pred_node(gt_pos):
        if not pred_positions:
            return None
        dists = {n: np.linalg.norm(np.array(gt_pos) - p) for n, p in pred_positions.items()}
        nearest = min(dists, key=dists.get)
        return nearest if dists[nearest] <= snap_radius else None

    pairs = rng.choice(len(gt_nodes), size=(min(n_pairs, len(gt_nodes) ** 1), 2), replace=True)
    scores = []
    misses = 0
    evaluated = 0

    for i, j in pairs:
        if i == j:
            continue
        a, b = gt_nodes[i], gt_nodes[j]
        try:
            gt_len = nx.shortest_path_length(G_gt, a, b, weight=weight)
        except nx.NetworkXNoPath:
            continue
        evaluated += 1

        pa = nearest_pred_node(G_gt.nodes[a][pos_attr])
        pb = nearest_pred_node(G_gt.nodes[b][pos_attr])
        if pa is None or pb is None:
            misses += 1
            scores.append(0.0)
            continue

        try:
            pred_len = nx.shortest_path_length(G_pred, pa, pb, weight=weight)
        except nx.NetworkXNoPath:
            misses += 1
            scores.append(0.0)
            continue

        score = 1.0 - min(1.0, abs(pred_len - gt_len) / gt_len) if gt_len > 0 else 1.0
        scores.append(score)

    if evaluated == 0:
        return {"score": 0.0, "miss_rate": 1.0, "n_pairs_evaluated": 0}

    return {
        "score": float(np.mean(scores)),
        "miss_rate": float(misses / evaluated),
        "n_pairs_evaluated": evaluated,
    }


def graph_summary(G: nx.Graph) -> Dict:
    """Quick sanity-check numbers to compare predicted vs. GT graphs at a glance."""
    total_length = sum(d.get("length_px", 0) for _, _, d in G.edges(data=True))
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_components": nx.number_connected_components(G),
        "total_length_px": float(total_length),
    }
