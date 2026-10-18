"""Phase 3E — Projection-Weighted Road Pairing Comparative Evaluation.

Evaluates the projection/continuity penalty weight on the exact same 32 unique tiles
and 3 occlusion patterns used in Phase 3D:

  Variants:
  1. baseline:                   continuity_penalty_weight = 0.0 (exact production baseline)
  2. continuity_weight_1_0:      continuity_penalty_weight = 1.0
  3. continuity_weight_2_0:      continuity_penalty_weight = 2.0
  4. continuity_weight_3_0:      continuity_penalty_weight = 3.0

Outputs are saved to outputs/projection_pairing_ablation/ (does NOT overwrite Phase 3C or 3D).
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, jaccard_score
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.config import Config, GraphConfig
from route_resilience.graph_extraction import (
    attach_width,
    clean_road_mask,
    confidence_gated_gap_healing,
    connectivity_ratio,
    mask_to_skeleton,
    skeleton_to_graph,
)
from route_resilience.occlusion_simulation import generate_road_centered_occlusions
from route_resilience.real_graph_pipeline import (
    load_model_for_inference,
    load_rgb_image,
    predict_probability_map,
)
from route_resilience.topology_matching import evaluate_connection_recovery
from route_resilience.train import get_device

WEIGHT_VARIANTS = [
    {"name": "baseline_w0", "weight": 0.0, "desc": "Baseline (weight=0.0, pure Euclidean distance)"},
    {"name": "penalty_w1",   "weight": 1.0, "desc": "Continuity penalty weight = 1.0"},
    {"name": "penalty_w2",   "weight": 2.0, "desc": "Continuity penalty weight = 2.0"},
    {"name": "penalty_w3",   "weight": 3.0, "desc": "Continuity penalty weight = 3.0"},
]


def _extract_gt_graph(gt_mask: np.ndarray, min_edge_length_px: float = 3.0) -> Optional[nx.Graph]:
    skel = mask_to_skeleton(gt_mask > 0)
    if skel.sum() == 0:
        return None
    return skeleton_to_graph(skel, prob_map=None, min_edge_length_px=min_edge_length_px)


def run_phase_3e_ablation(
    dataset_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    seed: int = 42,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identical 32 tiles from Phase 3D (20 top-FP + 12 val)
    top_20_fp_tiles = [213, 4526, 6314, 7074, 5061, 7624, 7728, 1930, 5621, 4547,
                       2719, 7546, 4119, 4782, 6231, 8030, 8952, 8996, 7881, 2904]
    val_12_tiles = [4428, 9671, 76, 5146, 6448, 8172, 1319, 4407, 1494, 7195, 9541, 2444]

    tile_groups = {}
    for tid in top_20_fp_tiles:
        tile_groups[tid] = "top_20_fp"
    for tid in val_12_tiles:
        tile_groups[tid] = "val_12"

    unique_tiles = list(tile_groups.keys())
    print(f"Total unique tiles to evaluate: {len(unique_tiles)} (Top-20 FP: {len(top_20_fp_tiles)}, Val-12: {len(val_12_tiles)})")

    device = get_device(device_name)
    cfg = Config()
    cfg.model.arch = "unet_smp"
    cfg.model.encoder_name = "resnet34"
    cfg.model.encoder_weights = "imagenet"
    cfg.train.loss = "focal_dice"

    print(f"Loading trained model from {checkpoint_path}...")
    model, _ = load_model_for_inference(checkpoint_path, cfg.model, cfg.train.loss, device)
    threshold = 0.5
    patterns = ["rectangular_block", "irregular_patch", "linear_transverse"]
    base_graph_cfg = GraphConfig(max_gap_px=22.0, min_gap_confidence=0.15, tangent_dot_threshold=0.7)

    all_records = []
    print("\nExecuting Phase 3E Projection-Weighted Road Pairing Evaluation across 4 weights...")
    print("-" * 80)

    for t_idx, tile_id in enumerate(unique_tiles, start=1):
        filename = f"{tile_id}.png"
        image_path = dataset_dir / "images_png" / filename
        mask_path = dataset_dir / "masks_png" / filename
        group_tag = tile_groups[tile_id]

        if not image_path.exists() or not mask_path.exists():
            continue

        orig_rgb = load_rgb_image(image_path)
        gt_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        G_gt = _extract_gt_graph(gt_mask)

        clean_prob = predict_probability_map(model, orig_rgb, device)
        clean_bin = clean_prob >= threshold
        clean_iou = float(jaccard_score(gt_mask.ravel(), clean_bin.ravel(), zero_division=1.0))

        for p_idx, pattern in enumerate(patterns):
            case_seed = seed + t_idx * 100 + p_idx * 17
            occluded_rgb, occ_mask, occ_meta = generate_road_centered_occlusions(
                orig_rgb,
                gt_mask,
                pattern=pattern,
                num_occlusions=2 if pattern != "irregular_patch" else 3,
                size_range=(16, 24),
                fill_mode="shadow" if pattern != "irregular_patch" else "canopy",
                seed=case_seed,
            )

            occ_prob = predict_probability_map(model, occluded_rgb, device)
            occ_bin = occ_prob >= threshold
            occ_iou = float(jaccard_score(gt_mask.ravel(), occ_bin.ravel(), zero_division=1.0))
            occ_dice = float(f1_score(gt_mask.ravel(), occ_bin.ravel(), zero_division=1.0))

            binary_mask = clean_road_mask(
                occ_bin,
                min_component_pixels=max(1, int(np.ceil(base_graph_cfg.min_edge_length_px))),
            )
            skeleton = mask_to_skeleton(binary_mask)
            graph_before = skeleton_to_graph(
                skeleton,
                prob_map=occ_prob,
                min_edge_length_px=base_graph_cfg.min_edge_length_px,
            )
            graph_before = attach_width(graph_before, binary_mask)
            cr_unhealed = connectivity_ratio(skeleton)

            for vw in WEIGHT_VARIANTS:
                G_healed = confidence_gated_gap_healing(
                    G=graph_before.copy(),
                    prob_map=occ_prob,
                    max_gap_px=base_graph_cfg.max_gap_px,
                    min_gap_confidence=base_graph_cfg.min_gap_confidence,
                    tangent_dot_threshold=base_graph_cfg.tangent_dot_threshold,
                    continuity_penalty_weight=vw["weight"],
                )

                matching_eval = evaluate_connection_recovery(
                    G_unhealed=graph_before,
                    G_healed=G_healed,
                    G_gt=G_gt,
                    gt_mask=gt_mask,
                    occlusion_mask=occ_mask,
                    tolerance_radius_px=3.0,
                    snap_radius_px=8.0,
                )

                rec = {
                    "tile_id": tile_id,
                    "tile_group": group_tag,
                    "pattern": pattern,
                    "variant": vw["name"],
                    "continuity_penalty_weight": vw["weight"],
                    "clean_iou": clean_iou,
                    "occluded_iou": occ_iou,
                    "occluded_dice": occ_dice,
                    "unhealed_components": matching_eval["unhealed_component_count"],
                    "healed_components": matching_eval["healed_component_count"],
                    "component_reduction": matching_eval["component_reduction"],
                    "total_recovered_edges": matching_eval["total_recovered_edges"],
                    "true_recovered_connections": matching_eval["true_recovered_connections"],
                    "false_recovered_connections": matching_eval["false_recovered_connections"],
                    "occlusion_recovered_tp": matching_eval["occlusion_recovered_tp"],
                    "occlusion_recovered_fp": matching_eval["occlusion_recovered_fp"],
                    "connection_recovery_precision": matching_eval["connection_recovery_precision"],
                    "connection_recovery_recall": matching_eval["connection_recovery_recall"],
                    "connectivity_ratio_unhealed": cr_unhealed,
                }
                all_records.append(rec)

        print(f"[{t_idx}/{len(unique_tiles)}] Tile {tile_id} ({group_tag}) — all patterns & 4 weights evaluated.")

    results_df = pd.DataFrame(all_records)
    csv_path = output_dir / "ablation_3e_metrics.csv"
    results_df.to_csv(csv_path, index=False)

    summary_by_variant = {}
    for vw in WEIGHT_VARIANTS:
        v_name = vw["name"]
        v_df = results_df[results_df["variant"] == v_name]
        v_rec = int(v_df["total_recovered_edges"].sum())
        v_tp = int(v_df["true_recovered_connections"].sum())
        v_fp = int(v_df["false_recovered_connections"].sum())
        v_occ_tp = int(v_df["occlusion_recovered_tp"].sum())
        v_occ_fp = int(v_df["occlusion_recovered_fp"].sum())

        group_summary = {}
        for grp in ["top_20_fp", "val_12"]:
            grp_df = v_df[v_df["tile_group"] == grp]
            g_rec = int(grp_df["total_recovered_edges"].sum())
            g_tp = int(grp_df["true_recovered_connections"].sum())
            g_fp = int(grp_df["false_recovered_connections"].sum())
            group_summary[grp] = {
                "scenarios": len(grp_df),
                "total_recovered_edges": g_rec,
                "true_positive_connections": g_tp,
                "false_positive_connections": g_fp,
                "precision": float(g_tp / g_rec) if g_rec > 0 else 1.0,
                "mean_recall": float(grp_df["connection_recovery_recall"].mean()),
                "mean_component_reduction": float(grp_df["component_reduction"].mean()),
            }

        pat_summary = {}
        for p in patterns:
            p_df = v_df[v_df["pattern"] == p]
            p_rec = int(p_df["total_recovered_edges"].sum())
            p_tp = int(p_df["true_recovered_connections"].sum())
            p_fp = int(p_df["false_recovered_connections"].sum())
            pat_summary[p] = {
                "total_recovered_edges": p_rec,
                "true_positive_connections": p_tp,
                "false_positive_connections": p_fp,
                "precision": float(p_tp / p_rec) if p_rec > 0 else 1.0,
                "mean_recall": float(p_df["connection_recovery_recall"].mean()),
                "mean_component_reduction": float(p_df["component_reduction"].mean()),
            }

        summary_by_variant[v_name] = {
            "description": vw["desc"],
            "continuity_penalty_weight": vw["weight"],
            "total_evaluations": len(v_df),
            "total_recovered_edges": v_rec,
            "true_positive_connections": v_tp,
            "false_positive_connections": v_fp,
            "overall_precision": float(v_tp / v_rec) if v_rec > 0 else 1.0,
            "mean_recall": float(v_df["connection_recovery_recall"].mean()),
            "occlusion_specific_tp": v_occ_tp,
            "occlusion_specific_fp": v_occ_fp,
            "occlusion_recovery_precision": float(v_occ_tp / (v_occ_tp + v_occ_fp)) if (v_occ_tp + v_occ_fp) > 0 else 1.0,
            "mean_component_reduction": float(v_df["component_reduction"].mean()),
            "by_tile_group": group_summary,
            "by_pattern": pat_summary,
        }

    final_report = {
        "experiment": "Phase 3E — Projection-Weighted Road Pairing Evaluation",
        "num_tiles_evaluated": len(unique_tiles),
        "total_scenarios_per_variant": len(unique_tiles) * len(patterns),
        "variants": summary_by_variant,
        "artifacts_saved": {
            "per_evaluation_csv": str(csv_path),
            "summary_json": str(output_dir / "ablation_3e_summary.json"),
        },
    }

    with open(output_dir / "ablation_3e_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 3E ABLATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    header = f"{'Variant':<22} {'Weight':>7} {'Recov':>6} {'TP':>5} {'FP':>5} {'Prec':>7} {'Recall':>7} {'CompRed':>8}"
    print(header)
    print("-" * 80)
    for v_name, s in summary_by_variant.items():
        print(
            f"{v_name:<22} {s['continuity_penalty_weight']:>7.1f} {s['total_recovered_edges']:>6} {s['true_positive_connections']:>5} "
            f"{s['false_positive_connections']:>5} {s['overall_precision']:>7.1%} "
            f"{s['mean_recall']:>7.1%} {s['mean_component_reduction']:>8.2f}"
        )
    print("=" * 80)

    return final_report


def main():
    parser = argparse.ArgumentParser(description="Phase 3E Projection-Weighted Pairing Evaluation")
    parser.add_argument("--dataset_dir", type=Path, default=PROJECT_ROOT / "RouteResilience")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "projection_pairing_ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_phase_3e_ablation(
        dataset_dir=args.dataset_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
