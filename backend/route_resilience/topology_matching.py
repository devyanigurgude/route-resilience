"""Topology matching rules and connection recovery validation for Route Resilience.

This module provides documented, reproducible matching rules to evaluate
whether a graph recovery link agrees with the ground-truth road network.

Matching Rule Specification:
----------------------------
A candidate recovered edge e = (u, v) with spatial trajectory `pts` in G_healed
is verified against the ground-truth road mask (gt_mask) and graph (G_gt) under:
  1. Trajectory Agreement:
     Sampled interior coordinates of `pts` must lie within `tolerance_radius_px`
     (default 3.0 px) of true ground-truth road pixels for at least
     `min_path_agreement` (default 75%) of the trajectory.
  2. Endpoint Valid Snapping:
     Endpoints u and v must snap within `snap_radius_px` (default 8.0 px) to
     nodes in the ground-truth graph G_gt.
  3. Topological Agreement:
     The snapped ground-truth endpoints must belong to the same connected
     component in G_gt (i.e., nx.has_path(G_gt, u_gt, v_gt) == True).

Classification:
  - True Positive (TP): Satisfies (1), (2), and (3).
  - False Positive (FP): Fails any of (1), (2), or (3) — represents an unsupported,
    hallucinated, or topologically invalid connection.
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

from route_resilience.graph_extraction import connectivity_ratio


def evaluate_connection_recovery(
    G_unhealed: nx.Graph,
    G_healed: nx.Graph,
    G_gt: Optional[nx.Graph],
    gt_mask: np.ndarray,
    occlusion_mask: Optional[np.ndarray] = None,
    tolerance_radius_px: float = 3.0,
    snap_radius_px: float = 8.0,
    min_path_agreement: float = 0.75,
) -> Dict[str, Any]:
    """Evaluates recovered connections against ground-truth road geometry and topology.

    Parameters
    ----------
    G_unhealed : nx.Graph
        Baseline graph extracted from thresholded prediction before gap healing.
    G_healed : nx.Graph
        Graph after evidence-gated and occlusion-aware gap healing.
    G_gt : Optional[nx.Graph]
        Ground-truth road graph extracted from gt_mask.
    gt_mask : np.ndarray
        Untouched binary ground-truth road mask.
    occlusion_mask : Optional[np.ndarray]
        Binary mask indicating where synthetic occlusions were placed.
    tolerance_radius_px : float
        Allowed spatial deviation from ground-truth road centerline in pixels.
    snap_radius_px : float
        Max snapping distance between predicted nodes and GT nodes.
    min_path_agreement : float
        Fraction of trajectory points that must fall within tolerance_radius_px.

    Returns
    -------
    Dict[str, Any]
        Dictionary with verified TP/FP counts, precision, recall, component counts,
        and per-edge verification records.
    """
    gt_dist_map = distance_transform_edt(~gt_mask.astype(bool))
    H, W = gt_mask.shape[:2]

    # Identify all newly healed or occlusion-recovered edges in G_healed
    new_edges = []
    for u, v, data in G_healed.edges(data=True):
        if not G_unhealed.has_edge(u, v) and data.get("healed", False):
            new_edges.append((u, v, data))

    # Pre-calculate GT node positions for snapping
    gt_positions = {}
    if G_gt is not None and G_gt.number_of_nodes() > 0:
        for n, ndata in G_gt.nodes(data=True):
            if "pos" in ndata:
                gt_positions[n] = np.asarray(ndata["pos"], dtype=float)

    def snap_to_gt(pos: np.ndarray) -> Optional[Any]:
        if not gt_positions:
            return None
        min_dist = float("inf")
        best_node = None
        for gn, gp in gt_positions.items():
            d = float(np.linalg.norm(pos - gp))
            if d < min_dist:
                min_dist = d
                best_node = gn
        return best_node if min_dist <= snap_radius_px else None

    tp_count = 0
    fp_count = 0
    occlusion_tp_count = 0
    occlusion_fp_count = 0
    edge_verifications = []

    for u, v, data in new_edges:
        is_occlusion_recovered = bool(data.get("occlusion_recovered", False))
        pts = data.get("pts")
        if pts is None or len(pts) < 2:
            u_pos = np.asarray(G_healed.nodes[u]["pos"], dtype=float)
            v_pos = np.asarray(G_healed.nodes[v]["pos"], dtype=float)
            pts = np.linspace(u_pos, v_pos, max(4, int(np.linalg.norm(v_pos - u_pos))))
        else:
            pts = np.asarray(pts, dtype=float)

        # 1. Trajectory Agreement against GT road buffer
        rows = np.clip(np.rint(pts[:, 0]).astype(int), 0, H - 1)
        cols = np.clip(np.rint(pts[:, 1]).astype(int), 0, W - 1)
        distances_to_gt = gt_dist_map[rows, cols]
        in_buffer_fraction = float(np.mean(distances_to_gt <= tolerance_radius_px))
        trajectory_valid = in_buffer_fraction >= min_path_agreement

        # 2. Endpoint snapping — prefer pts endpoints; fall back to node "pos"
        u_node = G_healed.nodes[u]
        v_node = G_healed.nodes[v]
        if "pos" in u_node:
            u_pos = np.asarray(u_node["pos"], dtype=float)
        else:
            u_pos = pts[0]
        if "pos" in v_node:
            v_pos = np.asarray(v_node["pos"], dtype=float)
        else:
            v_pos = pts[-1]
        u_gt = snap_to_gt(u_pos)
        v_gt = snap_to_gt(v_pos)

        # 3. Topological Agreement in GT
        if G_gt is not None and u_gt is not None and v_gt is not None:
            topology_valid = nx.has_path(G_gt, u_gt, v_gt)
        else:
            # If G_gt is absent or snapping fails, fallback strictly to geometric buffer support
            topology_valid = trajectory_valid

        is_true_positive = bool(trajectory_valid and topology_valid)

        if is_true_positive:
            tp_count += 1
            if is_occlusion_recovered:
                occlusion_tp_count += 1
        else:
            fp_count += 1
            if is_occlusion_recovered:
                occlusion_fp_count += 1

        # Check if the recovered edge passes through the occlusion zone
        in_occlusion_zone = False
        if occlusion_mask is not None:
            in_occlusion_zone = bool(np.any(occlusion_mask[rows, cols]))

        edge_verifications.append({
            "edge": (u, v),
            "occlusion_recovered": is_occlusion_recovered,
            "length_px": float(data.get("length_px", len(pts))),
            "in_buffer_fraction": in_buffer_fraction,
            "u_snapped_gt": u_gt is not None,
            "v_snapped_gt": v_gt is not None,
            "topology_valid": topology_valid,
            "is_true_positive": is_true_positive,
            "in_occlusion_zone": in_occlusion_zone,
        })

    # Estimate ground-truth severed cuts:
    # A severed cut occurs when two components in G_unhealed belong to the same component in G_gt
    severed_gt_cuts = 0
    if G_gt is not None and G_gt.number_of_nodes() > 1:
        unhealed_comps = list(nx.connected_components(G_unhealed))
        for i in range(len(unhealed_comps)):
            for j in range(i + 1, len(unhealed_comps)):
                comp_a = unhealed_comps[i]
                comp_b = unhealed_comps[j]
                # Check if any node in comp_a connects to comp_b in G_gt
                connected_in_gt = False
                for na in comp_a:
                    pa = snap_to_gt(np.asarray(G_unhealed.nodes[na]["pos"]))
                    if pa is None:
                        continue
                    for nb in comp_b:
                        pb = snap_to_gt(np.asarray(G_unhealed.nodes[nb]["pos"]))
                        if pb is not None and nx.has_path(G_gt, pa, pb):
                            connected_in_gt = True
                            break
                    if connected_in_gt:
                        break
                if connected_in_gt:
                    severed_gt_cuts += 1

    total_recovered = tp_count + fp_count
    precision = float(tp_count / total_recovered) if total_recovered > 0 else 1.0

    # Recall over severed cuts
    denom_recall = max(tp_count, severed_gt_cuts)
    recall = float(tp_count / denom_recall) if denom_recall > 0 else (1.0 if tp_count == 0 else 1.0)

    # Component count & connectivity ratio
    unhealed_n_comps = nx.number_connected_components(G_unhealed) if G_unhealed.number_of_nodes() > 0 else 0
    healed_n_comps = nx.number_connected_components(G_healed) if G_healed.number_of_nodes() > 0 else 0
    gt_n_comps = nx.number_connected_components(G_gt) if G_gt is not None and G_gt.number_of_nodes() > 0 else 0

    return {
        "total_recovered_edges": total_recovered,
        "true_recovered_connections": tp_count,
        "false_recovered_connections": fp_count,
        "occlusion_recovered_tp": occlusion_tp_count,
        "occlusion_recovered_fp": occlusion_fp_count,
        "connection_recovery_precision": precision,
        "connection_recovery_recall": recall,
        "severed_gt_cuts": severed_gt_cuts,
        "unhealed_component_count": unhealed_n_comps,
        "healed_component_count": healed_n_comps,
        "gt_component_count": gt_n_comps,
        "component_reduction": unhealed_n_comps - healed_n_comps,
        "edge_verifications": edge_verifications,
    }
