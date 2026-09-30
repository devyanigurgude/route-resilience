"""
Runnable, self-contained demo of the graph reconstruction -> criticality
-> disruption simulation pipeline (everything AFTER segmentation).

No GPU, no dataset, no torch required -- this uses a synthetic
probability map standing in for a trained model's output, so you can
verify the whole downstream pipeline works before your segmentation
model is even trained, and reuse this exact wiring once it is (just
replace `make_synthetic_prediction()` with your model's real output).

Run: python scripts/demo_graph_pipeline.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from route_resilience.criticality import (
    add_capacity_weight,
    add_confidence_weight,
    compute_centrality,
    criticality_ranking_stability,
    network_efficiency,
    top_n_edges_by_betweenness,
)
from route_resilience.disruption import flood_conditioned_removal, rank_edges_by_disruption_impact
from route_resilience.evaluate_topology import graph_summary, simplified_apls
from route_resilience.graph_extraction import (
    attach_width,
    confidence_gated_gap_healing,
    connectivity_ratio,
    mask_to_skeleton,
    skeleton_to_graph,
)


def make_synthetic_scene():
    """
    A small road network with:
      - a main east-west arterial (wide, high confidence)
      - a north-south residential branch (narrower)
      - a deliberate occlusion gap in the arterial (simulating a cloud)
      - a low-lying spur (for the flood scenario)
    Stands in for what a trained segmentation model would actually output.
    """
    H, W = 200, 260
    gt_mask = np.zeros((H, W), dtype=bool)

    for r in range(98, 103):
        gt_mask[r, 10:250] = True         # arterial
    for c in range(48, 53):
        gt_mask[10:100, c] = True         # north branch
    for c in range(198, 203):
        gt_mask[100:190, c] = True        # south spur (low-lying)

    pred_mask = gt_mask.copy()
    pred_mask[95:106, 130:145] = False    # occlusion gap in the arterial

    prob_map = np.where(pred_mask, 0.92, 0.05).astype(float)
    prob_map[95:106, 130:145] = 0.24      # weak residual evidence under the "cloud"

    return gt_mask, pred_mask, prob_map


def main():
    print("=" * 70)
    print("ROUTE RESILIENCE -- graph/criticality/disruption pipeline demo")
    print("=" * 70)

    gt_mask, pred_mask, prob_map = make_synthetic_scene()

    # ---- Stage 1: skeleton + connectivity diagnostic -------------------
    skel_gt = mask_to_skeleton(gt_mask)
    skel_pred = mask_to_skeleton(pred_mask)

    print(f"\n[1] Connectivity Ratio (GT):   {connectivity_ratio(skel_gt):.3f}")
    print(f"[1] Connectivity Ratio (pred, unhealed): {connectivity_ratio(skel_pred):.3f}  "
          f"<- occlusion gap fragments the network, this is the diagnostic that catches it")

    # ---- Stage 2: graph reconstruction ----------------------------------
    G_gt = skeleton_to_graph(skel_gt, prob_map=np.where(gt_mask, 0.95, 0.05), min_edge_length_px=3.0)
    G_gt = attach_width(G_gt, gt_mask)

    G_pred = skeleton_to_graph(skel_pred, prob_map=prob_map, min_edge_length_px=3.0)
    G_pred = attach_width(G_pred, pred_mask)

    print(f"\n[2] GT graph:            {graph_summary(G_gt)}")
    print(f"[2] Pred graph (unhealed): {graph_summary(G_pred)}")

    apls_unhealed = simplified_apls(G_pred, G_gt, snap_radius=8, n_pairs=150, seed=1)
    print(f"[2] Simplified APLS (unhealed): {apls_unhealed}")

    # ---- Stage 3: confidence-gated gap healing --------------------------
    G_pred = confidence_gated_gap_healing(G_pred, prob_map, max_gap_px=20, min_gap_confidence=0.15)
    print(f"\n[3] Pred graph (healed): {graph_summary(G_pred)}")

    apls_healed = simplified_apls(G_pred, G_gt, snap_radius=8, n_pairs=150, seed=1)
    print(f"[3] Simplified APLS (healed):   {apls_healed}")
    print("    -> this before/after comparison IS the concrete evidence that "
          "confidence-gated healing is doing useful work, not just adding edges")

    # ---- Stage 4: criticality analysis -----------------------------------
    G_pred = add_capacity_weight(G_pred)
    G_pred = add_confidence_weight(G_pred)

    cen_unweighted = compute_centrality(G_pred)
    cen_weighted = compute_centrality(G_pred, weight="inv_width")

    print("\n[4] Top-5 edges by UNWEIGHTED betweenness:")
    for edge, score in top_n_edges_by_betweenness(G_pred, n=5):
        print(f"      {edge}: {score:.3f}")

    print("[4] Top-5 edges by WIDTH-WEIGHTED betweenness:")
    for edge, score in top_n_edges_by_betweenness(G_pred, n=5, weight="inv_width"):
        print(f"      {edge}: {score:.3f}")

    # ---- Stage 5: H4 -- confidence-aware vs confidence-blind stability ---
    stab_aware = criticality_ranking_stability(G_pred, n_bootstrap=300, top_n=3,
                                                confidence_aware=True, seed=7)
    stab_blind = criticality_ranking_stability(G_pred, n_bootstrap=300, top_n=3,
                                                confidence_aware=False, seed=7)
    print(f"\n[5] H4 check -- top-3 ranking stability under noise:")
    print(f"      confidence-aware perturbation: {stab_aware['mean_top_n_overlap']:.3f} "
          f"(+/- {stab_aware['std_top_n_overlap']:.3f})")
    print(f"      confidence-blind perturbation: {stab_blind['mean_top_n_overlap']:.3f} "
          f"(+/- {stab_blind['std_top_n_overlap']:.3f})")
    print("      NOTE: this is a simplified, partial operationalization of H4 -- "
          "see criticality.add_confidence_weight() docstring for what a complete "
          "version additionally requires.")

    # ---- Stage 6: disruption simulation -----------------------------------
    ranked = rank_edges_by_disruption_impact(G_pred)
    print("\n[6] Edges ranked by standalone disruption impact (top 3):")
    for r in ranked[:3]:
        print(f"      {r['edge']}: efficiency_drop={r['efficiency_drop_frac']:.3f} "
              f"disconnects={r['disconnects']}")

    # crude synthetic elevations: south spur is low-lying -> flood scenario
    node_elevation = {}
    for n, data in G_pred.nodes(data=True):
        row, col = data["pos"]
        node_elevation[n] = 5.0 if col > 190 else 40.0  # south spur is "low"

    flood = flood_conditioned_removal(G_pred, node_elevation, flood_level=10.0)
    print(f"\n[6] Flood-conditioned removal (level=10): "
          f"affected_edges={flood['affected_edge_count']} "
          f"efficiency_drop={flood['efficiency_drop_frac']:.3f} "
          f"components {flood['before_components']} -> {flood['after_components']}")

    print("\n" + "=" * 70)
    print("Pipeline ran end-to-end successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
