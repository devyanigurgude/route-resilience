"""Phase 6 — Reliable Occlusion Recovery Evaluation.

Evaluates robust candidate validation on the exact same 32 unique tiles
(20 top-FP tiles + 12 standard validation tiles) across all 3 occlusion patterns
under identical random seeds and conditions:

  1. baseline:
     min_component_len       = 0.0
     min_total_component_len = 0.0
     min_flank_contrast      = None
     filter_tile_borders     = False
     max_sinuosity           = 1.5

  2. ablation_scale_guard:
     min_component_len       = 4.0
     min_total_component_len = 12.0
     min_flank_contrast      = None
     filter_tile_borders     = False
     max_sinuosity           = 1.5

  3. ablation_border_guard:
     min_component_len       = 0.0
     min_total_component_len = 0.0
     min_flank_contrast      = None
     filter_tile_borders     = True
     max_sinuosity           = 1.5

  4. ablation_contrast_guard:
     min_component_len       = 0.0
     min_total_component_len = 0.0
     min_flank_contrast      = -0.02
     filter_tile_borders     = False
     max_sinuosity           = 1.5

  5. candidate_robust_full:
     min_component_len       = 4.0
     min_total_component_len = 12.0
     min_flank_contrast      = -0.02
     filter_tile_borders     = True
     max_sinuosity           = 1.35

Outputs are saved to outputs/phase6_occlusion_recovery/ (does NOT overwrite Phase 3C/3D/3E).
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

CONFIG_VARIANTS = [
    {
        "name": "baseline",
        "min_component_len": 0.0,
        "min_total_component_len": 0.0,
        "min_flank_contrast": None,
        "filter_tile_borders": False,
        "max_sinuosity": 1.5,
        "desc": "Baseline (production defaults)",
    },
    {
        "name": "ablation_scale_guard",
        "min_component_len": 4.0,
        "min_total_component_len": 12.0,
        "min_flank_contrast": None,
        "filter_tile_borders": False,
        "max_sinuosity": 1.5,
        "desc": "Component scale guard (min_comp=4px, min_total=12px)",
    },
    {
        "name": "ablation_border_guard",
        "min_component_len": 0.0,
        "min_total_component_len": 0.0,
        "min_flank_contrast": None,
        "filter_tile_borders": True,
        "max_sinuosity": 1.5,
        "desc": "Tile border artifact guard (margin=4px)",
    },
    {
        "name": "ablation_contrast_guard",
        "min_component_len": 0.0,
        "min_total_component_len": 0.0,
        "min_flank_contrast": -0.02,
        "filter_tile_borders": False,
        "max_sinuosity": 1.5,
        "desc": "Flank contrast guard (min_contrast=-0.02)",
    },
    {
        "name": "candidate_robust_full",
        "min_component_len": 4.0,
        "min_total_component_len": 12.0,
        "min_flank_contrast": -0.02,
        "filter_tile_borders": True,
        "max_sinuosity": 1.35,
        "desc": "Candidate Robust Full (Scale + Border + Flank + Sinuosity)",
    },
]


def _extract_gt_graph(gt_mask: np.ndarray, min_edge_length_px: float = 3.0) -> Optional[nx.Graph]:
    skel = mask_to_skeleton(gt_mask > 0)
    if skel.sum() == 0:
        return None
    return skeleton_to_graph(skel, prob_map=None, min_edge_length_px=min_edge_length_px)


def run_phase6_evaluation(
    dataset_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    seed: int = 42,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identical 32 tiles from Phase 3D & 3E
    top_20_fp_tiles = [
        213, 4526, 6314, 7074, 5061, 7624, 7728, 1930, 5621, 4547,
        2719, 7546, 4119, 4782, 6231, 8030, 8952, 8996, 7881, 2904
    ]
    val_12_tiles = [
        4428, 9671, 76, 5146, 6448, 8172, 1319, 4407, 1494, 7195, 9541, 2444
    ]

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
    print("\nExecuting Phase 6 Reliable Occlusion Recovery Evaluation across 5 configurations...")
    print("-" * 80)

    for t_idx, tile_id in enumerate(unique_tiles, start=1):
        filename = f"{tile_id}.png"
        image_path = dataset_dir / "images_png" / filename
        mask_path = dataset_dir / "masks_png" / filename
        group_tag = tile_groups[tile_id]

        if not image_path.exists():
            raise FileNotFoundError(f"Image tile not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask tile not found: {mask_path}")

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

            for cv in CONFIG_VARIANTS:
                G_healed = confidence_gated_gap_healing(
                    G=graph_before.copy(),
                    prob_map=occ_prob,
                    max_gap_px=base_graph_cfg.max_gap_px,
                    min_gap_confidence=base_graph_cfg.min_gap_confidence,
                    tangent_dot_threshold=base_graph_cfg.tangent_dot_threshold,
                    min_component_len=cv["min_component_len"],
                    min_total_component_len=cv["min_total_component_len"],
                    min_flank_contrast=cv["min_flank_contrast"],
                    filter_tile_borders=cv["filter_tile_borders"],
                    max_sinuosity=cv["max_sinuosity"],
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
                    "variant": cv["name"],
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

        print(f"[{t_idx}/{len(unique_tiles)}] Tile {tile_id} ({group_tag}) — all patterns & 5 variants evaluated.")

    results_df = pd.DataFrame(all_records)
    csv_path = output_dir / "phase6_comparison_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved per-scenario metrics to {csv_path}")

    summary_by_variant = {}
    for cv in CONFIG_VARIANTS:
        v_name = cv["name"]
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
            "description": cv["desc"],
            "total_recovered_edges": v_rec,
            "true_positive_connections": v_tp,
            "false_positive_connections": v_fp,
            "overall_precision": float(v_tp / v_rec) if v_rec > 0 else 1.0,
            "macro_mean_recall": float(v_df["connection_recovery_recall"].mean()),
            "mean_component_reduction": float(v_df["component_reduction"].mean()),
            "occlusion_recovered_tp": v_occ_tp,
            "occlusion_recovered_fp": v_occ_fp,
            "groups": group_summary,
            "patterns": pat_summary,
        }

    summary_path = output_dir / "phase6_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_by_variant, f, indent=2)
    print(f"Saved aggregate summary to {summary_path}")

    print("\n" + "=" * 90)
    print(f"{'Variant':<25} | {'Rec':>5} | {'TP':>5} | {'FP':>5} | {'Precision':>9} | {'Mean Recall':>11} | {'Comp Red':>8}")
    print("-" * 90)
    for v_name, s in summary_by_variant.items():
        print(
            f"{v_name:<25} | {s['total_recovered_edges']:>5} | {s['true_positive_connections']:>5} | "
            f"{s['false_positive_connections']:>5} | {s['overall_precision'] * 100:>8.1f}% | "
            f"{s['macro_mean_recall'] * 100:>10.1f}% | {s['mean_component_reduction']:>8.2f}"
        )
    print("=" * 90)

    return summary_by_variant


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 6 Reliable Occlusion Recovery Evaluation.")
    parser.add_argument("--dataset_dir", type=str, default=str(PROJECT_ROOT / "RouteResilience"), help="Dataset directory containing images_png and masks_png")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_unet_smp_resnet34_focaldice.pth", help="Model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/phase6_occlusion_recovery", help="Directory to save Phase 6 outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (cpu or cuda)")
    args = parser.parse_args()

    run_phase6_evaluation(
        dataset_dir=Path(args.dataset_dir),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        device_name=args.device,
    )
