"""Phase 7 — Real-World Occlusion Validation Evaluation.

Evaluates the Route Resilience pipeline on real satellite imagery containing
natural tree canopy and building/topographical shadows, comparing the production
baseline against the Phase 6 candidate (filter_tile_borders=True, min_flank_contrast=-0.02).

Evaluation Scope:
  - 60 held-out test tiles from RouteResilience/data_split_grouped.csv:
      * 20 Natural Tree Canopy tiles (mean road vegetation coverage: 98.7%)
      * 20 Natural Shadow tiles (mean road shadow coverage: 99.0%)
      * 20 Open-Terrain Control tiles (mean vegetation: 16.7%, shadow: 14.9%)
  - Zero synthetic occlusions injected: evaluates authentic model fragmentation
    and recovery under natural aerial conditions.
  - Metrics: IoU, Dice, unhealed components, healed components, component reduction,
    total recovered connections, true positives (TP), false positives (FP),
    precision, recall, connectivity ratio.

Outputs:
  - outputs/phase7_real_occlusion/phase7_metrics.csv
  - outputs/phase7_real_occlusion/phase7_summary.json
  - outputs/phase7_real_occlusion/visuals/*.png
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
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
        "filter_tile_borders": False,
        "min_flank_contrast": None,
        "desc": "Baseline production defaults (border_filter=False, flank_contrast=None)",
    },
    {
        "name": "phase6_candidate",
        "filter_tile_borders": True,
        "min_flank_contrast": -0.02,
        "desc": "Phase 6 Candidate (border_filter=True, min_flank_contrast=-0.02)",
    },
]


def _extract_gt_graph(gt_mask: np.ndarray, min_edge_length_px: float = 3.0) -> Optional[nx.Graph]:
    skel = mask_to_skeleton(gt_mask > 0)
    if skel.sum() == 0:
        return None
    return skeleton_to_graph(skel, prob_map=None, min_edge_length_px=min_edge_length_px)


def _save_visual_comparison(
    out_path: Path,
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    prob_map: np.ndarray,
    G_raw: nx.Graph,
    G_base: nx.Graph,
    G_cand: nx.Graph,
    tile_id: int,
    subset: str,
    iou: float,
    eval_base: Dict[str, Any],
    eval_cand: Dict[str, Any],
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Original RGB Image
    axes[0].imshow(rgb)
    axes[0].set_title(f"Tile {tile_id} ({subset})\nNatural Aerial Imagery", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Ground-Truth Reference Road Network
    axes[1].imshow(rgb)
    gt_y, gt_x = np.where(gt_mask)
    axes[1].scatter(gt_x, gt_y, s=1, c="lime", alpha=0.6, label="GT Road Centerline")
    axes[1].set_title(f"Reference Road Network\nIoU: {iou:.3f}", fontsize=11)
    axes[1].axis("off")

    # Panel 3: Probability Map & Unhealed Graph (Natural Gaps)
    axes[2].imshow(prob_map, cmap="magma", vmin=0.0, vmax=1.0)
    for u, v in G_raw.edges():
        p1 = G_raw.nodes[u]["pos"]
        p2 = G_raw.nodes[v]["pos"]
        axes[2].plot([p1[1], p2[1]], [p1[0], p2[0]], c="white", linewidth=1.5)
    axes[2].set_title(
        f"Model Confidence & Raw Graph\nUnhealed Components: {eval_base['unhealed_component_count']}",
        fontsize=11,
    )
    axes[2].axis("off")

    # Panel 4: Healed Comparison (Baseline vs Candidate)
    axes[3].imshow(rgb)
    # Plot raw edges in white
    for u, v in G_raw.edges():
        p1 = G_raw.nodes[u]["pos"]
        p2 = G_raw.nodes[v]["pos"]
        axes[3].plot([p1[1], p2[1]], [p1[0], p2[0]], c="gray", linewidth=1.5, alpha=0.7)

    # Plot baseline healed edges in orange
    for u, v, data in G_base.edges(data=True):
        if data.get("healed"):
            p1 = G_base.nodes[u]["pos"]
            p2 = G_base.nodes[v]["pos"]
            axes[3].plot([p1[1], p2[1]], [p1[0], p2[0]], c="orange", linewidth=2.5, linestyle="--", label="Baseline Healed" if "Baseline Healed" not in [l.get_label() for l in axes[3].lines] else "")

    # Plot candidate healed edges in cyan
    for u, v, data in G_cand.edges(data=True):
        if data.get("healed"):
            p1 = G_cand.nodes[u]["pos"]
            p2 = G_cand.nodes[v]["pos"]
            axes[3].plot([p1[1], p2[1]], [p1[0], p2[0]], c="cyan", linewidth=2.0, label="Candidate Healed" if "Candidate Healed" not in [l.get_label() for l in axes[3].lines] else "")

    axes[3].set_title(
        f"Healing Comparison\nBase: TP={eval_base['true_recovered_connections']}, FP={eval_base['false_recovered_connections']} | Cand: TP={eval_cand['true_recovered_connections']}, FP={eval_cand['false_recovered_connections']}",
        fontsize=10,
    )
    axes[3].legend(loc="lower right", fontsize=8)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_phase7_evaluation(
    dataset_dir: Path,
    selected_csv: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    if not selected_csv.exists():
        raise FileNotFoundError(f"Selected tiles CSV not found: {selected_csv}")

    tiles_df = pd.read_csv(selected_csv)
    print(f"Loaded {len(tiles_df)} selected tiles across subsets: {tiles_df['subset'].value_counts().to_dict()}")

    device = get_device(device_name)
    cfg = Config()
    cfg.model.arch = "unet_smp"
    cfg.model.encoder_name = "resnet34"
    cfg.model.encoder_weights = "imagenet"
    cfg.train.loss = "focal_dice"

    print(f"Loading trained model from {checkpoint_path}...")
    model, _ = load_model_for_inference(checkpoint_path, cfg.model, cfg.train.loss, device)
    threshold = 0.5
    base_graph_cfg = GraphConfig(max_gap_px=22.0, min_gap_confidence=0.15, tangent_dot_threshold=0.7)

    all_records = []
    print("\nExecuting Phase 7 Real-World Occlusion Validation...")
    print("-" * 80)

    # Keep track of visuals saved per subset
    visuals_saved = {"natural_canopy": 0, "natural_shadow": 0, "open_terrain_control": 0}

    for t_idx, row in tiles_df.iterrows():
        tile_id = int(row["tile_id"])
        filename = str(row["filename"])
        subset = str(row["subset"])
        image_path = dataset_dir / "images_png" / filename
        mask_path = dataset_dir / "masks_png" / filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        orig_rgb = load_rgb_image(image_path)
        gt_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        G_gt = _extract_gt_graph(gt_mask)

        # Real imagery inference (NO synthetic masks applied)
        prob_map = predict_probability_map(model, orig_rgb, device)
        bin_mask_raw = prob_map >= threshold
        iou = float(jaccard_score(gt_mask.ravel(), bin_mask_raw.ravel(), zero_division=1.0))
        dice = float(f1_score(gt_mask.ravel(), bin_mask_raw.ravel(), zero_division=1.0))

        # Extract unhealed graph from real prediction
        bin_clean = clean_road_mask(
            bin_mask_raw,
            min_component_pixels=max(1, int(np.ceil(base_graph_cfg.min_edge_length_px))),
        )
        skel = mask_to_skeleton(bin_clean)
        G_raw = skeleton_to_graph(skel, prob_map=prob_map, min_edge_length_px=base_graph_cfg.min_edge_length_px)
        G_raw = attach_width(G_raw, bin_clean)
        cr_unhealed = connectivity_ratio(skel)

        variant_evals = {}
        for cv in CONFIG_VARIANTS:
            v_name = cv["name"]
            G_healed = confidence_gated_gap_healing(
                G=G_raw.copy(),
                prob_map=prob_map,
                max_gap_px=base_graph_cfg.max_gap_px,
                min_gap_confidence=base_graph_cfg.min_gap_confidence,
                tangent_dot_threshold=base_graph_cfg.tangent_dot_threshold,
                filter_tile_borders=cv["filter_tile_borders"],
                min_flank_contrast=cv["min_flank_contrast"],
            )

            matching_eval = evaluate_connection_recovery(
                G_unhealed=G_raw,
                G_healed=G_healed,
                G_gt=G_gt,
                gt_mask=gt_mask,
                occlusion_mask=None,  # Real-world: evaluate across entire tile
                tolerance_radius_px=3.0,
                snap_radius_px=8.0,
            )
            variant_evals[v_name] = (G_healed, matching_eval)

            rec = {
                "tile_id": tile_id,
                "filename": filename,
                "subset": subset,
                "variant": v_name,
                "green_frac": row["green_frac"],
                "shadow_frac": row["shadow_frac"],
                "road_percentage": row["road_percentage"],
                "real_image_iou": iou,
                "real_image_dice": dice,
                "unhealed_components": matching_eval["unhealed_component_count"],
                "healed_components": matching_eval["healed_component_count"],
                "component_reduction": matching_eval["component_reduction"],
                "total_recovered_edges": matching_eval["total_recovered_edges"],
                "true_recovered_connections": matching_eval["true_recovered_connections"],
                "false_recovered_connections": matching_eval["false_recovered_connections"],
                "connection_recovery_precision": matching_eval["connection_recovery_precision"],
                "connection_recovery_recall": matching_eval["connection_recovery_recall"],
                "connectivity_ratio_unhealed": cr_unhealed,
            }
            all_records.append(rec)

        # Save representative visual comparisons (first 3 per subset or notable difference)
        G_base, eval_base = variant_evals["baseline"]
        G_cand, eval_cand = variant_evals["phase6_candidate"]
        diff_exists = (eval_base["false_recovered_connections"] != eval_cand["false_recovered_connections"] or
                       eval_base["true_recovered_connections"] != eval_cand["true_recovered_connections"])

        if visuals_saved[subset] < 3 or (diff_exists and visuals_saved[subset] < 5):
            vis_file = visuals_dir / f"{subset}_{tile_id}.png"
            _save_visual_comparison(
                out_path=vis_file,
                rgb=orig_rgb,
                gt_mask=gt_mask,
                prob_map=prob_map,
                G_raw=G_raw,
                G_base=G_base,
                G_cand=G_cand,
                tile_id=tile_id,
                subset=subset,
                iou=iou,
                eval_base=eval_base,
                eval_cand=eval_cand,
            )
            visuals_saved[subset] += 1

        print(
            f"[{t_idx + 1:2d}/60] Tile {tile_id} ({subset:<20}) | IoU: {iou:.3f} | "
            f"Base: TP={eval_base['true_recovered_connections']} FP={eval_base['false_recovered_connections']} | "
            f"Cand: TP={eval_cand['true_recovered_connections']} FP={eval_cand['false_recovered_connections']}"
        )

    results_df = pd.DataFrame(all_records)
    csv_path = output_dir / "phase7_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved per-tile metrics to {csv_path}")

    # Compute summary breakdown by variant and subset
    summary = {}
    for cv in CONFIG_VARIANTS:
        v_name = cv["name"]
        v_df = results_df[results_df["variant"] == v_name]
        v_rec = int(v_df["total_recovered_edges"].sum())
        v_tp = int(v_df["true_recovered_connections"].sum())
        v_fp = int(v_df["false_recovered_connections"].sum())

        subset_summary = {}
        for sub in ["natural_canopy", "natural_shadow", "open_terrain_control"]:
            s_df = v_df[v_df["subset"] == sub]
            s_rec = int(s_df["total_recovered_edges"].sum())
            s_tp = int(s_df["true_recovered_connections"].sum())
            s_fp = int(s_df["false_recovered_connections"].sum())
            subset_summary[sub] = {
                "tile_count": len(s_df),
                "mean_iou": float(s_df["real_image_iou"].mean()),
                "mean_dice": float(s_df["real_image_dice"].mean()),
                "total_recovered_edges": s_rec,
                "true_positive_connections": s_tp,
                "false_positive_connections": s_fp,
                "precision": float(s_tp / s_rec) if s_rec > 0 else 1.0,
                "mean_recall": float(s_df["connection_recovery_recall"].mean()),
                "mean_component_reduction": float(s_df["component_reduction"].mean()),
            }

        summary[v_name] = {
            "description": cv["desc"],
            "total_tiles": len(v_df),
            "mean_iou": float(v_df["real_image_iou"].mean()),
            "mean_dice": float(v_df["real_image_dice"].mean()),
            "total_recovered_edges": v_rec,
            "true_positive_connections": v_tp,
            "false_positive_connections": v_fp,
            "overall_precision": float(v_tp / v_rec) if v_rec > 0 else 1.0,
            "macro_mean_recall": float(v_df["connection_recovery_recall"].mean()),
            "mean_component_reduction": float(v_df["component_reduction"].mean()),
            "subsets": subset_summary,
        }

    summary_path = output_dir / "phase7_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved aggregate summary to {summary_path}")

    print("\n" + "=" * 90)
    print(f"{'Variant / Subset':<30} | {'Rec':>5} | {'TP':>5} | {'FP':>5} | {'Precision':>9} | {'Recall':>8} | {'IoU':>6}")
    print("-" * 90)
    for v_name, s in summary.items():
        print(f"[{v_name}]")
        print(
            f"  {'Overall':<28} | {s['total_recovered_edges']:>5} | {s['true_positive_connections']:>5} | "
            f"{s['false_positive_connections']:>5} | {s['overall_precision'] * 100:>8.1f}% | "
            f"{s['macro_mean_recall'] * 100:>7.1f}% | {s['mean_iou']:>6.3f}"
        )
        for sub_name, sub in s["subsets"].items():
            print(
                f"  - {sub_name:<26} | {sub['total_recovered_edges']:>5} | {sub['true_positive_connections']:>5} | "
                f"{sub['false_positive_connections']:>5} | {sub['precision'] * 100:>8.1f}% | "
                f"{sub['mean_recall'] * 100:>7.1f}% | {sub['mean_iou']:>6.3f}"
            )
    print("=" * 90)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 7 Real-World Occlusion Validation.")
    parser.add_argument("--dataset_dir", type=str, default=str(PROJECT_ROOT / "RouteResilience"))
    parser.add_argument("--selected_csv", type=str, default=str(PROJECT_ROOT / "RouteResilience" / "phase7_selected_tiles.csv"))
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_unet_smp_resnet34_focaldice.pth")
    parser.add_argument("--output_dir", type=str, default="outputs/phase7_real_occlusion")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_phase7_evaluation(
        dataset_dir=Path(args.dataset_dir),
        selected_csv=Path(args.selected_csv),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        device_name=args.device,
    )
