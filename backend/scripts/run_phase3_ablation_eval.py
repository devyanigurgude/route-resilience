"""Phase 3 — 3-Way Controlled Ablation Evaluation Script.

Evaluates and compares:
  - Model A: Baseline U-Net (trained on random split with coarse dropout)
  - Model B: Grouped U-Net (trained on grouped split with coarse dropout)
  - Model C: Multi-Type U-Net (trained on grouped split with multi-type occlusion augmentation)

Evaluates on the identical held-out spatial test partition across:
  1. clean_baseline
  2. rect_dropout_mild & rect_dropout_severe
  3. canopy_mild & canopy_severe
  4. cloud_mild & cloud_severe

Computes:
  - Pixel metrics: IoU, Dice, Precision, Recall, Inside-Occlusion Recall
  - Topology metrics: Unhealed/Healed Connected Components, Component Reduction, LCR
  - Metric proxies: Unhealed APLS, Healed APLS, APLS Gain
  - Gap Recovery: TP, FP, Total Candidates, Micro-Pooled Precision, Macro Active Precision, Legacy Macro Precision

Outputs:
  - backend/outputs/phase3_ablation/phase3_ablation_summary.json
  - backend/outputs/phase3_ablation/phase3_ablation_per_tile.csv
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from route_resilience.config import GraphConfig, ModelConfig
from route_resilience.evaluate_topology import simplified_apls
from route_resilience.graph_extraction import mask_to_skeleton, skeleton_to_graph
from route_resilience.occlusion_simulation import generate_road_centered_occlusions
from route_resilience.real_graph_pipeline import (
    graph_from_probability_map,
    load_model_for_inference,
    load_rgb_image,
    predict_probability_map,
)
from route_resilience.topology_matching import evaluate_connection_recovery
from route_resilience.train import get_device


def calculate_pixel_metrics(pred_bin: np.ndarray, target_bin: np.ndarray) -> Dict[str, float]:
    pred = pred_bin.astype(bool)
    target = target_bin.astype(bool)

    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())

    union = tp + fp + fn
    iou = tp / union if union > 0 else (1.0 if target.sum() == 0 else 0.0)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else (1.0 if target.sum() == 0 else 0.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def extract_gt_graph(gt_mask: np.ndarray, min_edge_length_px: float = 3.0) -> Optional[nx.Graph]:
    skel = mask_to_skeleton(gt_mask > 0)
    if skel.sum() == 0:
        return None
    return skeleton_to_graph(skel, prob_map=None, min_edge_length_px=min_edge_length_px)


def compute_graph_topology_stats(graph: Optional[nx.Graph]) -> Dict[str, Any]:
    if graph is None or graph.number_of_nodes() == 0:
        return {
            "node_count": 0,
            "edge_count": 0,
            "num_components": 0,
            "largest_component_ratio": 0.0,
        }

    components = list(nx.connected_components(graph))
    num_nodes = graph.number_of_nodes()
    largest_cc_nodes = max(len(c) for c in components) if components else 0
    lcr = largest_cc_nodes / num_nodes if num_nodes > 0 else 0.0

    return {
        "node_count": num_nodes,
        "edge_count": graph.number_of_edges(),
        "num_components": len(components),
        "largest_component_ratio": float(lcr),
    }


def get_test_tiles(dataset_dir: Path, split_csv: Path, sample_size: int = 30, seed: int = 42) -> pd.DataFrame:
    df_grp = pd.read_csv(split_csv)
    test_tiles_all = df_grp[df_grp["split"] == "test"].copy()
    valid_candidates = test_tiles_all[test_tiles_all["road_percentage"] >= 0.2].copy()

    if len(valid_candidates) <= sample_size:
        return valid_candidates.copy()

    valid_candidates["bin"] = pd.qcut(
        valid_candidates["road_percentage"],
        q=min(5, sample_size),
        labels=False,
        duplicates="drop",
    )
    sampled_list = []
    for _, group in valid_candidates.groupby("bin"):
        n_b = max(1, int(round(sample_size * len(group) / len(valid_candidates))))
        sampled_list.append(group.sample(n=min(len(group), n_b), random_state=seed))
    selected = pd.concat(sampled_list).head(sample_size).copy()
    return selected


def run_ablation(
    sample_size: int = 30,
    seed: int = 42,
    device_name: str = "cpu",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = BACKEND_DIR / "outputs" / "phase3_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = WORKSPACE_ROOT / "RouteResilience"
    split_csv = dataset_dir / "data_split_grouped.csv"
    device = get_device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    print(f"Inference Device: {device} (threads={torch.get_num_threads()})")

    print("=" * 75)
    print("PHASE 3 — 3-WAY CONTROLLED ABLATION STUDY")
    print("=" * 75)

    test_tiles = get_test_tiles(dataset_dir, split_csv, sample_size=sample_size, seed=seed)
    print(f"Selected {len(test_tiles)} held-out spatial test tiles.")

    models_info = [
        {
            "id": "model_a_baseline",
            "name": "Model A (Baseline Random-Split)",
            "checkpoint": BACKEND_DIR / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth",
            "threshold_file": BACKEND_DIR / "checkpoints" / "threshold_unet_smp_resnet34_focaldice.json",
            "default_thresh": 0.50,
        },
        {
            "id": "model_b_grouped_coarse",
            "name": "Model B (Grouped Split + Coarse Dropout)",
            "checkpoint": BACKEND_DIR / "checkpoints" / "best_unet_smp_resnet34_grouped_coarse_dropout.pth",
            "threshold_file": BACKEND_DIR / "checkpoints" / "threshold_unet_smp_resnet34_grouped_coarse_dropout.json",
            "default_thresh": 0.50,
        },
        {
            "id": "model_c_grouped_multi",
            "name": "Model C (Grouped Split + Multi-Type Occlusion)",
            "checkpoint": BACKEND_DIR / "checkpoints" / "best_unet_smp_resnet34_grouped_multi_type.pth",
            "threshold_file": BACKEND_DIR / "checkpoints" / "threshold_unet_smp_resnet34_grouped_multi_type.json",
            "default_thresh": 0.50,
        },
    ]

    loaded_models = {}
    for m in models_info:
        if not m["checkpoint"].exists():
            raise FileNotFoundError(f"Missing required checkpoint: {m['checkpoint']}")

        thresh = m["default_thresh"]
        if m["threshold_file"].exists():
            with open(m["threshold_file"], "r", encoding="utf-8") as tf:
                t_data = json.load(tf)
                thresh = float(t_data.get("threshold", thresh))

        m_cfg = ModelConfig(arch="unet_smp", encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, dropout_p=0.2)
        model_obj, meta = load_model_for_inference(m["checkpoint"], m_cfg, "focal_dice", device)
        loaded_models[m["id"]] = {
            "model": model_obj,
            "threshold": thresh,
            "info": m,
            "epoch": meta.get("epoch", "unknown"),
        }
        print(f"Loaded {m['name']} from {m['checkpoint'].name} (Threshold: {thresh:.2f})")

    occlusion_scenarios = [
        {"name": "clean_baseline", "pattern": None, "severity": "none", "num": 0, "size": (0, 0), "fill": None},
        {"name": "rect_dropout_mild", "pattern": "rectangular_block", "severity": "mild", "num": 1, "size": (16, 22), "fill": "black"},
        {"name": "rect_dropout_severe", "pattern": "rectangular_block", "severity": "severe", "num": 2, "size": (26, 36), "fill": "black"},
        {"name": "canopy_mild", "pattern": "canopy_cluster", "severity": "mild", "num": 1, "size": (16, 22), "fill": "canopy"},
        {"name": "canopy_severe", "pattern": "canopy_cluster", "severity": "severe", "num": 2, "size": (26, 36), "fill": "canopy"},
        {"name": "cloud_mild", "pattern": "procedural_cloud", "severity": "mild", "num": 1, "size": (16, 22), "fill": "cloud"},
        {"name": "cloud_severe", "pattern": "procedural_cloud", "severity": "severe", "num": 2, "size": (26, 36), "fill": "cloud"},
    ]

    graph_cfg = GraphConfig(
        max_gap_px=15.0,
        min_gap_confidence=0.15,
        enable_occlusion_recovery=True,
        max_occlusion_gap_px=30.0,
        min_occlusion_confidence=0.05,
        occlusion_heading_threshold=0.60,
        max_lateral_offset_px=4.0,
        enable_cloud_occlusion_recovery=True,
        max_cloud_occlusion_gap_px=240.0,
        cloud_heading_threshold=0.78,
    )

    per_tile_rows = []
    matrix_summary = {}

    for m_entry in models_info:
        m_id = m_entry["id"]
        m_data = loaded_models[m_id]
        model = m_data["model"]
        threshold = m_data["threshold"]
        print(f"\nEvaluating Model: {m_entry['name']}...")

        matrix_summary[m_id] = {
            "model_id": m_id,
            "model_name": m_entry["name"],
            "threshold": threshold,
            "checkpoint": m_entry["checkpoint"].name,
            "scenarios": {},
        }

        for sc_idx, sc in enumerate(occlusion_scenarios):
            sc_name = sc["name"]
            metrics = {
                "iou": [], "dice": [], "precision": [], "recall": [],
                "occ_road_recall": [],
                "unhealed_comp": [], "healed_comp": [], "comp_reduction": [],
                "unhealed_lcr": [], "healed_lcr": [],
                "unhealed_apls": [], "healed_apls": [],
                "tp": [], "fp": [], "total_cand": [],
                "macro_prec_active": [], "legacy_macro_prec": [],
            }

            for t_idx, row in enumerate(test_tiles.itertuples(index=False)):
                filename = row.filename
                img_path = dataset_dir / "images_png" / filename
                mask_path = dataset_dir / "masks_png" / filename

                orig_rgb = load_rgb_image(img_path)
                gt_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
                G_gt = extract_gt_graph(gt_mask)

                if sc["pattern"] is not None:
                    tile_seed = seed + t_idx * 137 + sc_idx * 19
                    eval_rgb, occ_mask, occ_meta = generate_road_centered_occlusions(
                        orig_rgb,
                        gt_mask,
                        pattern=sc["pattern"],
                        num_occlusions=sc["num"],
                        size_range=sc["size"],
                        fill_mode=sc["fill"],
                        seed=tile_seed,
                    )
                else:
                    eval_rgb = orig_rgb
                    occ_mask = np.zeros(gt_mask.shape, dtype=bool)

                prob_map = predict_probability_map(model, eval_rgb, device)
                pred_bin = prob_map >= threshold

                pm = calculate_pixel_metrics(pred_bin, gt_mask)
                metrics["iou"].append(pm["iou"])
                metrics["dice"].append(pm["dice"])
                metrics["precision"].append(pm["precision"])
                metrics["recall"].append(pm["recall"])

                occ_road_total = int(np.logical_and(gt_mask, occ_mask).sum())
                if occ_road_total > 0:
                    occ_road_pred = int(np.logical_and(np.logical_and(pred_bin, gt_mask), occ_mask).sum())
                    occ_recall = occ_road_pred / occ_road_total
                    metrics["occ_road_recall"].append(occ_recall)
                else:
                    metrics["occ_road_recall"].append(1.0 if sc["pattern"] is None else 0.0)

                pipe_res = graph_from_probability_map(
                    prob_map,
                    rgb_image=eval_rgb,
                    threshold=threshold,
                    graph_cfg=graph_cfg,
                )
                G_unhealed = pipe_res["graph_before"]
                G_healed = pipe_res["graph_after"]

                stat_unhealed = compute_graph_topology_stats(G_unhealed)
                stat_healed = compute_graph_topology_stats(G_healed)

                metrics["unhealed_comp"].append(stat_unhealed["num_components"])
                metrics["healed_comp"].append(stat_healed["num_components"])
                metrics["comp_reduction"].append(stat_unhealed["num_components"] - stat_healed["num_components"])
                metrics["unhealed_lcr"].append(stat_unhealed["largest_component_ratio"])
                metrics["healed_lcr"].append(stat_healed["largest_component_ratio"])

                if G_gt is not None and G_gt.number_of_nodes() >= 2:
                    apls_unhealed = simplified_apls(G_unhealed, G_gt)["score"]
                    apls_healed = simplified_apls(G_healed, G_gt)["score"]
                else:
                    apls_unhealed = 0.0
                    apls_healed = 0.0
                metrics["unhealed_apls"].append(apls_unhealed)
                metrics["healed_apls"].append(apls_healed)

                rec_eval = evaluate_connection_recovery(
                    G_unhealed,
                    G_healed,
                    G_gt,
                    gt_mask,
                    occlusion_mask=occ_mask if sc["pattern"] is not None else None,
                )
                tp = rec_eval["true_recovered_connections"]
                fp = rec_eval["false_recovered_connections"]
                tot = tp + fp
                metrics["tp"].append(tp)
                metrics["fp"].append(fp)
                metrics["total_cand"].append(tot)

                # Active vs Legacy macro precision
                if tot > 0:
                    metrics["macro_prec_active"].append(tp / tot)
                    metrics["legacy_macro_prec"].append(tp / tot)
                else:
                    metrics["legacy_macro_prec"].append(1.0)

                per_tile_rows.append({
                    "model_id": m_id,
                    "scenario": sc_name,
                    "pattern": sc["pattern"] or "none",
                    "severity": sc["severity"],
                    "filename": filename,
                    "iou": pm["iou"],
                    "dice": pm["dice"],
                    "precision": pm["precision"],
                    "recall": pm["recall"],
                    "occ_road_recall": metrics["occ_road_recall"][-1],
                    "unhealed_comp": stat_unhealed["num_components"],
                    "healed_comp": stat_healed["num_components"],
                    "unhealed_lcr": stat_unhealed["largest_component_ratio"],
                    "healed_lcr": stat_healed["largest_component_ratio"],
                    "unhealed_apls": apls_unhealed,
                    "healed_apls": apls_healed,
                    "tp": tp,
                    "fp": fp,
                    "tot_cand": tot,
                })

            total_tp = int(np.sum(metrics["tp"]))
            total_fp = int(np.sum(metrics["fp"]))
            total_cand = total_tp + total_fp
            micro_prec = (total_tp / total_cand) if total_cand > 0 else 1.0
            macro_prec_active = float(np.mean(metrics["macro_prec_active"])) if metrics["macro_prec_active"] else 1.0
            legacy_macro_prec = float(np.mean(metrics["legacy_macro_prec"]))

            matrix_summary[m_id]["scenarios"][sc_name] = {
                "mean_iou": float(np.mean(metrics["iou"])),
                "mean_dice": float(np.mean(metrics["dice"])),
                "mean_precision": float(np.mean(metrics["precision"])),
                "mean_recall": float(np.mean(metrics["recall"])),
                "mean_occ_road_recall": float(np.mean(metrics["occ_road_recall"])),
                "mean_unhealed_comp": float(np.mean(metrics["unhealed_comp"])),
                "mean_healed_comp": float(np.mean(metrics["healed_comp"])),
                "mean_comp_reduction": float(np.mean(metrics["comp_reduction"])),
                "mean_unhealed_lcr": float(np.mean(metrics["unhealed_lcr"])),
                "mean_healed_lcr": float(np.mean(metrics["healed_lcr"])),
                "mean_unhealed_apls": float(np.mean(metrics["unhealed_apls"])),
                "mean_healed_apls": float(np.mean(metrics["healed_apls"])),
                "apls_gain": float(np.mean(metrics["healed_apls"]) - np.mean(metrics["unhealed_apls"])),
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_candidates": total_cand,
                "micro_recovery_precision": float(micro_prec),
                "macro_active_recovery_precision": float(macro_prec_active),
                "legacy_macro_recovery_precision": float(legacy_macro_prec),
                "active_tiles_count": len(metrics["macro_prec_active"]),
            }

    # Save outputs
    summary_path = output_dir / "phase3_ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(matrix_summary, f, indent=2)

    df_per_tile = pd.DataFrame(per_tile_rows)
    csv_path = output_dir / "phase3_ablation_per_tile.csv"
    df_per_tile.to_csv(csv_path, index=False)

    print("\n" + "=" * 75)
    print(f"Ablation evaluation complete! Results saved to:")
    print(f"  Summary JSON: {summary_path}")
    print(f"  Per-tile CSV: {csv_path}")
    print("=" * 75)

    return matrix_summary


def main():
    parser = argparse.ArgumentParser(description="Phase 3 3-Way Ablation Evaluation.")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_ablation(sample_size=args.sample_size, seed=args.seed, device_name=args.device)


if __name__ == "__main__":
    main()
