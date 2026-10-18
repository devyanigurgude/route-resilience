"""Phase 8 — End-to-End Network Disruption and Rerouting Experiment.

Executes real-world network disruption, connectivity ratio calculation,
global efficiency drop, and origin-destination rerouting on real satellite
imagery road networks.

Outputs:
  - outputs/phase8_disruption/phase8_disruption_metrics.json
  - outputs/phase8_disruption/Phase8_Report.md
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.config import Config, GraphConfig
from route_resilience.criticality import network_efficiency, top_n_edges_by_betweenness
from route_resilience.disruption import (
    evaluate_disruption_impact_with_rerouting,
    graph_connectivity_ratio,
    simulate_removal,
)
from route_resilience.graph_extraction import (
    attach_width,
    clean_road_mask,
    confidence_gated_gap_healing,
    mask_to_skeleton,
    skeleton_to_graph,
)
from route_resilience.real_graph_pipeline import (
    load_model_for_inference,
    load_rgb_image,
    predict_probability_map,
)
from route_resilience.resilience import evaluate_resilience
from route_resilience.routing import route_result, select_baseline_od_pairs
from route_resilience.train import get_device


def run_phase8_experiment(output_dir: Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device("cpu")
    cfg = Config()
    cfg.model.arch = "unet_smp"
    cfg.model.encoder_name = "resnet34"
    cfg.model.encoder_weights = "imagenet"
    cfg.train.loss = "focal_dice"

    ckpt_path = PROJECT_ROOT / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth"
    model, _ = load_model_for_inference(str(ckpt_path), cfg.model, cfg.train.loss, device)

    # Use real satellite image 1.png and 1301.png
    tiles_to_evaluate = ["1.png", "1301.png"]
    graph_cfg = GraphConfig(max_gap_px=22.0, min_gap_confidence=0.15, tangent_dot_threshold=0.7)

    all_experiments = []

    for fn in tiles_to_evaluate:
        img_path = PROJECT_ROOT / "RouteResilience" / "images_png" / fn
        orig_rgb = load_rgb_image(img_path)

        # 1. Real Image Extraction
        prob_map = predict_probability_map(model, orig_rgb, device)
        bin_clean = clean_road_mask(prob_map >= 0.5, 3)
        skel = mask_to_skeleton(bin_clean)
        G_raw = skeleton_to_graph(skel, prob_map=prob_map, min_edge_length_px=graph_cfg.min_edge_length_px)
        G_raw = attach_width(G_raw, bin_clean)

        # Healed Baseline Graph
        G_baseline = confidence_gated_gap_healing(
            G_raw, prob_map, max_gap_px=graph_cfg.max_gap_px, min_gap_confidence=graph_cfg.min_gap_confidence
        )

        n_nodes = G_baseline.number_of_nodes()
        n_edges = G_baseline.number_of_edges()
        if n_nodes < 4 or n_edges < 3:
            continue

        base_eff = network_efficiency(G_baseline)
        base_cr = graph_connectivity_ratio(G_baseline)
        base_components = nx.number_connected_components(G_baseline)

        # Find top vulnerable edges by betweenness
        ranked_edges = top_n_edges_by_betweenness(G_baseline, n=5, weight="length_px")
        top_edge = ranked_edges[0][0] if ranked_edges else list(G_baseline.edges())[0]

        # Select fixed baseline OD pairs
        baseline_od = select_baseline_od_pairs(G_baseline, max_pairs=50, seed=42)

        # Test Strategy A: Single Top-Critical Edge Failure
        res_crit = evaluate_disruption_impact_with_rerouting(
            G_baseline, edges_to_remove=[top_edge], od_pairs=baseline_od
        )

        # Test Strategy B: Multi-Edge Targeted Disruption (Top 2 edges)
        top_2_edges = [e for e, _ in ranked_edges[:2]] if len(ranked_edges) >= 2 else [top_edge]
        res_multi = evaluate_disruption_impact_with_rerouting(
            G_baseline, edges_to_remove=top_2_edges, od_pairs=baseline_od
        )

        # Verify failed edges NEVER appear in any alternative path
        for exp_res, failed_set in [(res_crit, [top_edge]), (res_multi, top_2_edges)]:
            for p in exp_res["affected_pairs"]:
                if p["status"] == "detoured" and p["alternative_path"]:
                    path_edges = list(zip(p["alternative_path"][:-1], p["alternative_path"][1:]))
                    for fe in failed_set:
                        assert fe not in path_edges and (fe[1], fe[0]) not in path_edges, (
                            f"Failed edge {fe} found in alternative path {p['alternative_path']}!"
                        )

        # Compute resilience metrics across 3 explicit scenarios:
        # Scenario 1: Disrupted unrestored (zero recovery)
        H_disrupted = res_crit["surviving_graph"]
        resilience_unrestored = evaluate_resilience(G_baseline, H_disrupted, H_disrupted, baseline_od)

        # Scenario 2: Modeled recovery using greedy budget reinstatement (budget=1)
        from route_resilience.recovery import reinstate_baseline_assets
        recovery_crit = reinstate_baseline_assets(
            G_baseline, H_disrupted, failed_edges=[top_edge], budget=1, ordering="criticality_greedy"
        )
        G_recovered_crit = recovery_crit["recovered_graph"]
        resilience_modeled_crit = evaluate_resilience(G_baseline, H_disrupted, G_recovered_crit, baseline_od)

        # Multi-edge modeled partial recovery (budget=1 out of 2 failed edges)
        H_multi_disrupted = res_multi["surviving_graph"]
        recovery_multi_partial = reinstate_baseline_assets(
            G_baseline, H_multi_disrupted, failed_edges=top_2_edges, budget=1, ordering="criticality_greedy"
        )
        G_recovered_multi_partial = recovery_multi_partial["recovered_graph"]
        resilience_modeled_multi = evaluate_resilience(
            G_baseline, H_multi_disrupted, G_recovered_multi_partial, baseline_od
        )

        # Scenario 3: Assumed 100% restored-baseline ceiling (idealized reference)
        resilience_assumed = evaluate_resilience(G_baseline, H_disrupted, G_baseline, baseline_od)

        record = {
            "tile": fn,
            "baseline": {
                "nodes": n_nodes,
                "edges": n_edges,
                "components": base_components,
                "connectivity_ratio": float(base_cr),
                "network_efficiency": float(base_eff),
                "od_pairs_evaluated": len(baseline_od),
            },
            "critical_edge_failure": {
                "failed_edge": [list(top_edge)],
                "disrupted_components": res_crit["removal"]["after_components"],
                "newly_disconnected_components": res_crit["removal"]["newly_disconnected"],
                "disrupted_connectivity_ratio": float(res_crit["removal"]["after_connectivity_ratio"]),
                "connectivity_ratio_drop": float(res_crit["removal"]["connectivity_ratio_drop"]),
                "disrupted_efficiency": float(res_crit["removal"]["after_efficiency"]),
                "efficiency_drop_frac": float(res_crit["removal"]["efficiency_drop_frac"]),
                "severed_od_pairs": res_crit["severed_pairs_count"],
                "detoured_od_pairs": res_crit["detoured_pairs_count"],
                "intact_od_pairs": res_crit["intact_pairs_count"],
                "mean_cost_increase_px": float(res_crit["mean_cost_increase_px"]),
                "max_detour_ratio": float(res_crit["max_detour_ratio"]),
                "disrupted_service_performance_PD": float(res_crit["disrupted_service_performance"]),
                "resilience_unrestored": {
                    "P_D": float(resilience_unrestored["P_D"]),
                    "P_R": float(resilience_unrestored["P_R"]),
                    "RecoveryGain": float(resilience_unrestored["RecoveryGain"]),
                    "RI": float(resilience_unrestored["RI"]),
                    "note": "Evaluated on unrestored disrupted graph (zero recovery)",
                },
                "modeled_recovery": {
                    "budget": 1,
                    "restored_edges": [list(e) for e in recovery_crit["recovered_edges"]],
                    "P_D": float(resilience_modeled_crit["P_D"]),
                    "P_R": float(resilience_modeled_crit["P_R"]),
                    "RecoveryGain": float(resilience_modeled_crit["RecoveryGain"]),
                    "RI": float(resilience_modeled_crit["RI"]),
                    "note": "Evaluated on modeled restored graph under budget=1",
                },
                "assumed_restored_ceiling": {
                    "P_D": float(resilience_assumed["P_D"]),
                    "P_R": float(resilience_assumed["P_R"]),
                    "RecoveryGain": float(resilience_assumed["RecoveryGain"]),
                    "RI_ceiling": float(resilience_assumed["RI"]),
                    "note": "Hypothetical ceiling assuming 100% full recovery without resource constraints",
                },
                "sample_detour": next((p for p in res_crit["affected_pairs"] if p["status"] == "detoured"), None),
                "sample_severed": next((p for p in res_crit["affected_pairs"] if p["status"] == "disconnected"), None),
            },
            "multi_edge_failure": {
                "failed_edges": [list(e) for e in top_2_edges],
                "disrupted_components": res_multi["removal"]["after_components"],
                "disrupted_connectivity_ratio": float(res_multi["removal"]["after_connectivity_ratio"]),
                "efficiency_drop_frac": float(res_multi["removal"]["efficiency_drop_frac"]),
                "severed_od_pairs": res_multi["severed_pairs_count"],
                "detoured_od_pairs": res_multi["detoured_pairs_count"],
                "disrupted_service_performance_PD": float(res_multi["disrupted_service_performance"]),
                "modeled_partial_recovery": {
                    "budget": 1,
                    "total_failed": len(top_2_edges),
                    "restored_edges": [list(e) for e in recovery_multi_partial["recovered_edges"]],
                    "P_D": float(resilience_modeled_multi["P_D"]),
                    "P_R": float(resilience_modeled_multi["P_R"]),
                    "RecoveryGain": float(resilience_modeled_multi["RecoveryGain"]),
                    "RI": float(resilience_modeled_multi["RI"]),
                    "note": "Budget=1 restoration restoring only the top-impact asset",
                },
            },
        }
        all_experiments.append(record)

    output_json = output_dir / "phase8_disruption_metrics.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"experiments": all_experiments}, f, indent=2)
    print(f"Saved disruption metrics to {output_json}")

    return {"experiments": all_experiments}


if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "outputs" / "phase8_disruption"
    run_phase8_experiment(out_dir)
