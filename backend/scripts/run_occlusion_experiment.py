"""Phase 3A — Controlled Occlusion Evaluation Experiment.

Evaluates road segmentation model and occlusion-aware graph continuity
recovery under controlled, synthetic occlusions (rectangular blocks,
irregular patches, and transverse cuts) placed over known road corridors.

Usage:
    python scripts/run_occlusion_experiment.py --num_tiles 12
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import f1_score, jaccard_score
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR
sys.path.insert(0, str(BACKEND_DIR))

from route_resilience.config import Config, GraphConfig
from route_resilience.graph_extraction import (
    connectivity_ratio,
    mask_to_skeleton,
    skeleton_to_graph,
)
from route_resilience.occlusion_simulation import generate_road_centered_occlusions
from route_resilience.real_graph_pipeline import (
    graph_from_probability_map,
    load_model_for_inference,
    load_rgb_image,
    predict_probability_map,
)
from route_resilience.topology_matching import evaluate_connection_recovery
from route_resilience.train import get_device


def _extract_gt_graph(gt_mask: np.ndarray, min_edge_length_px: float = 3.0) -> Optional[nx.Graph]:
    skel = mask_to_skeleton(gt_mask > 0)
    if skel.sum() == 0:
        return None
    return skeleton_to_graph(skel, prob_map=None, min_edge_length_px=min_edge_length_px)


def _render_graph_overlay(
    rgb_image: np.ndarray,
    graph: nx.Graph,
    highlight_healed: bool = True,
    verifications: Optional[List[Dict]] = None,
) -> Image.Image:
    canvas = Image.fromarray(rgb_image).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    tp_edges = set()
    fp_edges = set()
    if verifications:
        for v in verifications:
            edge_pair = frozenset(v["edge"])
            if v["is_true_positive"]:
                tp_edges.add(edge_pair)
            else:
                fp_edges.add(edge_pair)

    for u, v, data in graph.edges(data=True):
        pts = data.get("pts")
        if pts is None or len(pts) < 2:
            continue
        line = [(int(col), int(row)) for row, col in np.asarray(pts)]
        if len(line) < 2:
            continue

        edge_pair = frozenset((u, v))
        is_healed = data.get("healed", False)
        is_occlusion_recovered = data.get("occlusion_recovered", False)

        if highlight_healed and is_healed:
            if edge_pair in tp_edges:
                color = (0, 230, 118)  # Emerald green = Verified True Positive
                width = 3
            elif edge_pair in fp_edges:
                color = (255, 61, 0)   # Vivid red = False Positive / Spurious
                width = 3
            elif is_occlusion_recovered:
                color = (255, 171, 0)  # Amber gold = Occlusion Recovered
                width = 3
            else:
                color = (255, 215, 0)  # Yellow = Healed
                width = 2
        else:
            color = (0, 150, 255)      # Cyan blue = directly observed road
            width = 2

        draw.line(line, fill=color, width=width)

    # Draw nodes
    for n, data in graph.nodes(data=True):
        if "pos" in data:
            row, col = data["pos"]
            deg = graph.degree(n)
            fill = (255, 0, 80) if deg == 1 else (0, 255, 255)
            draw.ellipse((int(col) - 2, int(row) - 2, int(col) + 2, int(row) + 2), fill=fill)

    return canvas


def _create_comparison_panel(
    original_rgb: np.ndarray,
    occluded_rgb: np.ndarray,
    gt_mask: np.ndarray,
    prob_map: np.ndarray,
    G_unhealed: nx.Graph,
    G_healed: nx.Graph,
    verifications: List[Dict],
    title: str,
) -> Image.Image:
    H, W = original_rgb.shape[:2]

    # Panel 1: Original image
    p1 = Image.fromarray(original_rgb).convert("RGB")
    # Panel 2: Occluded image
    p2 = Image.fromarray(occluded_rgb).convert("RGB")
    # Panel 3: GT Mask
    p3 = Image.fromarray(np.where(gt_mask > 0, 255, 0).astype(np.uint8)).convert("RGB")
    # Panel 4: Probability Map
    p4_arr = np.clip(prob_map * 255, 0, 255).astype(np.uint8)
    p4 = Image.fromarray(p4_arr).convert("RGB")
    # Panel 5: Unhealed Graph (baseline with gaps)
    p5 = _render_graph_overlay(occluded_rgb, G_unhealed, highlight_healed=False)
    # Panel 6: Healed Recovered Graph (with TP/FP colors)
    p6 = _render_graph_overlay(occluded_rgb, G_healed, highlight_healed=True, verifications=verifications)

    panels = [p1, p2, p3, p4, p5, p6]
    labels = [
        "1. Original RGB",
        "2. Synthetic Occluded RGB",
        "3. Ground-Truth Road Mask",
        "4. Model Probability Map",
        "5. Unhealed Graph (Severed)",
        "6. Recovered Graph (Green=TP, Gold=Recov)",
    ]

    cols = 3
    rows = 2
    banner_height = 24
    cell_w = W
    cell_h = H + banner_height
    grid_img = Image.new("RGB", (cell_w * cols, cell_h * rows), (20, 24, 30))

    for idx, (panel, label) in enumerate(zip(panels, labels)):
        r = idx // cols
        c = idx % cols
        x = c * cell_w
        y = r * cell_h

        # Paste panel
        grid_img.paste(panel, (x, y + banner_height))

        # Draw banner
        draw = ImageDraw.Draw(grid_img)
        draw.rectangle((x, y, x + cell_w, y + banner_height), fill=(15, 20, 28))
        draw.text((x + 6, y + 4), label, fill=(240, 240, 240))

    return grid_img


def run_experiment(
    dataset_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    num_tiles: int = 12,
    seed: int = 42,
    device_name: str = "cpu",
    overwrite: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    if (output_dir / "results_summary.json").exists() and not overwrite:
        raise FileExistsError(
            f"Historical benchmark results already exist in {output_dir}. "
            "To prevent accidental data loss, provide a different --output_dir or pass --overwrite."
        )
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    split_csv = dataset_dir / "data_split.csv"
    if not split_csv.exists():
        split_csv = dataset_dir / "data_split_grouped.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Split file not found in {dataset_dir}")

    df = pd.read_csv(split_csv)
    test_df = df[df["split"] == "test"].copy()
    # Select representative tiles with meaningful road coverage (0.5% - 3.5%)
    candidates = test_df[(test_df["road_percentage"] >= 0.5) & (test_df["road_percentage"] <= 3.5)]
    if len(candidates) < num_tiles:
        candidates = test_df[test_df["road_percentage"] > 0.2]

    # Deterministic selection
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(candidates), size=min(num_tiles, len(candidates)), replace=False)
    selected_tiles = candidates.iloc[chosen_indices].copy()

    device = get_device(device_name)
    cfg = Config()
    cfg.model.arch = "unet_smp"
    cfg.model.encoder_name = "resnet34"
    cfg.model.encoder_weights = "imagenet"
    cfg.train.loss = "focal_dice"

    print(f"Loading trained model from {checkpoint_path}...")
    model, checkpoint = load_model_for_inference(checkpoint_path, cfg.model, cfg.train.loss, device)
    threshold = 0.5
    graph_cfg = GraphConfig(max_gap_px=22.0, min_gap_confidence=0.15)

    patterns = ["rectangular_block", "irregular_patch", "linear_transverse"]
    per_image_records = []

    print(f"\nRunning Controlled Occlusion Evaluation on {len(selected_tiles)} tiles...")
    print(f"Patterns: {patterns}")
    print("-" * 70)

    for t_idx, row in enumerate(selected_tiles.itertuples(index=False), start=1):
        filename = row.filename
        image_path = dataset_dir / "images_png" / filename
        mask_path = dataset_dir / "masks_png" / filename

        if not image_path.exists() or not mask_path.exists():
            continue

        orig_rgb = load_rgb_image(image_path)
        gt_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        G_gt = _extract_gt_graph(gt_mask)

        # Baseline Clean Inference
        clean_prob = predict_probability_map(model, orig_rgb, device)
        clean_bin = clean_prob >= threshold
        clean_iou = float(jaccard_score(gt_mask.ravel(), clean_bin.ravel(), zero_division=1.0))
        clean_skel = mask_to_skeleton(clean_bin)
        clean_cr = connectivity_ratio(clean_skel)

        # Run through each occlusion pattern
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

            # Model prediction on occluded image (NEVER sees gt_mask)
            occ_prob = predict_probability_map(model, occluded_rgb, device)
            occ_bin = occ_prob >= threshold
            occ_iou = float(jaccard_score(gt_mask.ravel(), occ_bin.ravel(), zero_division=1.0))
            occ_dice = float(f1_score(gt_mask.ravel(), occ_bin.ravel(), zero_division=1.0))

            # Run graph extraction and recovery with occluded RGB image for photometric detection
            extraction_result = graph_from_probability_map(
                occ_prob,
                threshold=threshold,
                graph_cfg=graph_cfg,
                rgb_image=occluded_rgb,
            )
            G_unhealed = extraction_result["graph_before"]
            G_healed = extraction_result["graph_after"]

            # Ground truth verification of recovered connections
            matching_eval = evaluate_connection_recovery(
                G_unhealed=G_unhealed,
                G_healed=G_healed,
                G_gt=G_gt,
                gt_mask=gt_mask,
                occlusion_mask=occ_mask,
                tolerance_radius_px=3.0,
                snap_radius_px=8.0,
            )

            skel_unhealed = mask_to_skeleton(extraction_result["binary_mask"])
            cr_unhealed = connectivity_ratio(skel_unhealed)
            cr_healed = connectivity_ratio(mask_to_skeleton(clean_bin)) # proxy after recovery

            rec = {
                "tile_id": Path(filename).stem,
                "pattern": pattern,
                "road_percentage": float(row.road_percentage),
                "affected_road_pixels": occ_meta["affected_road_pixels"],
                "clean_iou": clean_iou,
                "occluded_iou": occ_iou,
                "occluded_dice": occ_dice,
                "iou_drop": clean_iou - occ_iou,
                "unhealed_components": matching_eval["unhealed_component_count"],
                "healed_components": matching_eval["healed_component_count"],
                "gt_components": matching_eval["gt_component_count"],
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
            per_image_records.append(rec)

            # Generate visual comparison for representative cases (first 10 tiles, plus prominent TP/FP cases)
            should_save_visual = (t_idx <= 10 and p_idx == 0) or (matching_eval["false_recovered_connections"] > 0 and len(list(visuals_dir.glob("fp_*.png"))) < 5)
            if should_save_visual:
                prefix = "fp_" if matching_eval["false_recovered_connections"] > 0 else ""
                visual_path = visuals_dir / f"{prefix}{Path(filename).stem}_{pattern}.png"
                if not visual_path.exists():
                    panel = _create_comparison_panel(
                        original_rgb=orig_rgb,
                        occluded_rgb=occluded_rgb,
                        gt_mask=gt_mask,
                        prob_map=occ_prob,
                        G_unhealed=G_unhealed,
                        G_healed=G_healed,
                        verifications=matching_eval["edge_verifications"],
                        title=f"Tile {Path(filename).stem} — {pattern}",
                    )
                    panel.save(visual_path)

        print(f"[{t_idx}/{len(selected_tiles)}] Tile {filename} evaluated across 3 patterns.")

    results_df = pd.DataFrame(per_image_records)
    results_df.to_csv(output_dir / "per_tile_metrics.csv", index=False)

    # Compute Aggregate Summary
    total_rec_edges = int(results_df["total_recovered_edges"].sum())
    total_tp = int(results_df["true_recovered_connections"].sum())
    total_fp = int(results_df["false_recovered_connections"].sum())
    total_occ_tp = int(results_df["occlusion_recovered_tp"].sum())
    total_occ_fp = int(results_df["occlusion_recovered_fp"].sum())

    aggregate_precision = float(total_tp / total_rec_edges) if total_rec_edges > 0 else 1.0
    occ_precision = float(total_occ_tp / (total_occ_tp + total_occ_fp)) if (total_occ_tp + total_occ_fp) > 0 else 1.0

    # Per-pattern breakdown
    per_pattern_summary = {}
    for p in patterns:
        p_df = results_df[results_df["pattern"] == p]
        p_rec = int(p_df["total_recovered_edges"].sum())
        p_tp = int(p_df["true_recovered_connections"].sum())
        p_fp = int(p_df["false_recovered_connections"].sum())
        p_occ_tp = int(p_df["occlusion_recovered_tp"].sum())
        p_occ_fp = int(p_df["occlusion_recovered_fp"].sum())
        per_pattern_summary[p] = {
            "evaluations": len(p_df),
            "mean_clean_iou": float(p_df["clean_iou"].mean()),
            "mean_occluded_iou": float(p_df["occluded_iou"].mean()),
            "mean_occluded_dice": float(p_df["occluded_dice"].mean()),
            "mean_iou_degradation": float(p_df["iou_drop"].mean()),
            "total_recovered_edges": p_rec,
            "true_positive_connections": p_tp,
            "false_positive_connections": p_fp,
            "overall_recovery_precision": float(p_tp / p_rec) if p_rec > 0 else 1.0,
            "mean_recovery_recall": float(p_df["connection_recovery_recall"].mean()),
            "occlusion_specific_tp": p_occ_tp,
            "occlusion_specific_fp": p_occ_fp,
            "occlusion_recovery_precision": float(p_occ_tp / (p_occ_tp + p_occ_fp)) if (p_occ_tp + p_occ_fp) > 0 else 1.0,
            "mean_unhealed_components": float(p_df["unhealed_components"].mean()),
            "mean_healed_components": float(p_df["healed_components"].mean()),
            "mean_component_reduction": float(p_df["component_reduction"].mean()),
            "mean_connectivity_ratio_unhealed": float(p_df["connectivity_ratio_unhealed"].mean()),
        }

    summary = {
        "experiment": "Phase 3C — Full-Scale Occlusion-Recovery Evaluation",
        "num_tiles_evaluated": len(selected_tiles),
        "total_test_evaluations": len(results_df),
        "occlusion_patterns_evaluated": patterns,
        "segmentation_metrics": {
            "mean_clean_iou": float(results_df["clean_iou"].mean()),
            "mean_occluded_iou": float(results_df["occluded_iou"].mean()),
            "mean_occluded_dice": float(results_df["occluded_dice"].mean()),
            "mean_iou_degradation": float(results_df["iou_drop"].mean()),
        },
        "recovery_connection_metrics": {
            "total_recovered_edges": total_rec_edges,
            "true_positive_connections": total_tp,
            "false_positive_connections": total_fp,
            "overall_recovery_precision": aggregate_precision,
            "mean_recovery_recall": float(results_df["connection_recovery_recall"].mean()),
            "occlusion_specific_tp": total_occ_tp,
            "occlusion_specific_fp": total_occ_fp,
            "occlusion_recovery_precision": occ_precision,
        },
        "topological_impact_metrics": {
            "mean_unhealed_components": float(results_df["unhealed_components"].mean()),
            "mean_healed_components": float(results_df["healed_components"].mean()),
            "mean_component_reduction": float(results_df["component_reduction"].mean()),
            "mean_connectivity_ratio_unhealed": float(results_df["connectivity_ratio_unhealed"].mean()),
        },
        "per_pattern_metrics": per_pattern_summary,
        "artifacts_saved": {
            "per_tile_csv": str(output_dir / "per_tile_metrics.csv"),
            "summary_json": str(output_dir / "results_summary.json"),
            "visuals_directory": str(visuals_dir),
        },
    }

    with open(output_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 3C EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Total test evaluations: {len(results_df)}")
    print(f"Mean Clean IoU: {summary['segmentation_metrics']['mean_clean_iou']:.4f}")
    print(f"Mean Occluded IoU: {summary['segmentation_metrics']['mean_occluded_iou']:.4f}")
    print(f"Total Recovered Connections: {total_rec_edges}")
    print(f"  - Verified True Positives: {total_tp}")
    print(f"  - False Positives (Spurious): {total_fp}")
    print(f"Connection Recovery Precision: {aggregate_precision * 100:.2f}%")
    print(f"Mean Connection Recovery Recall: {summary['recovery_connection_metrics']['mean_recovery_recall'] * 100:.2f}%")
    print(f"Mean Component Reduction: {summary['topological_impact_metrics']['mean_component_reduction']:.2f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 3A Controlled Occlusion Evaluation")
    default_dataset = WORKSPACE_ROOT / "RouteResilience"
    if not default_dataset.exists():
        default_dataset = BACKEND_DIR / "RouteResilience"
    parser.add_argument("--dataset_dir", type=Path, default=default_dataset)
    parser.add_argument("--checkpoint", type=Path, default=BACKEND_DIR / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth")
    parser.add_argument("--output_dir", type=Path, default=WORKSPACE_ROOT / "outputs" / "occlusion_experiment_eval")
    parser.add_argument("--num_tiles", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs in output_dir")
    args = parser.parse_args()

    run_experiment(
        dataset_dir=args.dataset_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_tiles=args.num_tiles,
        seed=args.seed,
        device_name=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
