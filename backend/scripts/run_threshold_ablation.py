"""Phase 3B — Tangent-Alignment Threshold Ablation.

Evaluates three tangent-alignment dot-product thresholds on the same 12
tiles and 3 occlusion patterns used in Phase 3A.

  Threshold  Angle (approx)  dot-product
  ---------  --------------  -----------
  baseline         ~45.6°       0.700
  30°               30.0°       0.866
  25°               25.0°       0.906
  20°               20.0°       0.940

All other parameters, random seeds, tile selection, and GT matching rules
are identical to Phase 3A.  Model inference is performed ONCE per
(tile, pattern) and cached; only gap-healing is re-run per threshold.

Usage:
    python scripts/run_threshold_ablation.py --num_tiles 12 --device cpu
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

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

# ---------------------------------------------------------------------------
# Threshold definitions  (dot-product = cos(angle_in_degrees))
# ---------------------------------------------------------------------------
THRESHOLDS: List[Dict[str, Any]] = [
    {"label": "baseline_0.700", "angle_deg": 45.6, "dot": 0.700},
    {"label": "30deg_0.866",    "angle_deg": 30.0, "dot": round(math.cos(math.radians(30)), 6)},
    {"label": "25deg_0.906",    "angle_deg": 25.0, "dot": round(math.cos(math.radians(25)), 6)},
    {"label": "20deg_0.940",    "angle_deg": 20.0, "dot": round(math.cos(math.radians(20)), 6)},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_gt_graph(gt_mask: np.ndarray, min_edge_length_px: float = 3.0) -> Optional[nx.Graph]:
    skel = mask_to_skeleton(gt_mask > 0)
    if skel.sum() == 0:
        return None
    return skeleton_to_graph(skel, prob_map=None, min_edge_length_px=min_edge_length_px)


def _run_gap_healing(
    graph_before: nx.Graph,
    prob_map: np.ndarray,
    graph_cfg: GraphConfig,
    dot_threshold: float,
) -> nx.Graph:
    """Re-run only gap healing on an already-extracted graph_before."""
    cfg = GraphConfig(
        max_gap_px=graph_cfg.max_gap_px,
        min_gap_confidence=graph_cfg.min_gap_confidence,
        min_edge_length_px=graph_cfg.min_edge_length_px,
        tangent_dot_threshold=dot_threshold,
    )
    return confidence_gated_gap_healing(
        graph_before.copy(),
        prob_map,
        max_gap_px=cfg.max_gap_px,
        min_gap_confidence=cfg.min_gap_confidence,
        tangent_dot_threshold=cfg.tangent_dot_threshold,
    )


def _aggregate_records(records: List[Dict]) -> Dict[str, Any]:
    """Compute summary stats from a list of per-evaluation dicts."""
    df = pd.DataFrame(records)
    total_rec = int(df["total_recovered_edges"].sum())
    total_tp  = int(df["true_recovered_connections"].sum())
    total_fp  = int(df["false_recovered_connections"].sum())
    precision = total_tp / total_rec if total_rec > 0 else 1.0
    fp_rate   = total_fp / total_rec if total_rec > 0 else 0.0
    mean_recall = float(df["connection_recovery_recall"].mean())
    mean_comp_reduc = float(df["component_reduction"].mean())
    mean_healed_comp = float(df["healed_components"].mean())
    return {
        "total_recovered_edges": total_rec,
        "true_positive_connections": total_tp,
        "false_positive_connections": total_fp,
        "precision": round(precision, 6),
        "fp_rate": round(fp_rate, 6),
        "mean_recall": round(mean_recall, 6),
        "mean_component_reduction": round(mean_comp_reduc, 6),
        "mean_healed_components": round(mean_healed_comp, 6),
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_ablation(
    dataset_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    num_tiles: int = 12,
    seed: int = 42,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Tile selection: identical to Phase 3A (same seed, same filters) ---
    split_csv = dataset_dir / "data_split.csv"
    if not split_csv.exists():
        split_csv = dataset_dir / "data_split_grouped.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Split file not found in {dataset_dir}")

    df_split = pd.read_csv(split_csv)
    test_df = df_split[df_split["split"] == "test"].copy()
    candidates = test_df[
        (test_df["road_percentage"] >= 0.5) & (test_df["road_percentage"] <= 3.5)
    ]
    if len(candidates) < num_tiles:
        candidates = test_df[test_df["road_percentage"] > 0.2]

    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(candidates), size=min(num_tiles, len(candidates)), replace=False)
    selected_tiles = candidates.iloc[chosen_indices].copy()

    # --- Model setup ---
    device = get_device(device_name)
    cfg = Config()
    cfg.model.arch = "unet_smp"
    cfg.model.encoder_name = "resnet34"
    cfg.model.encoder_weights = "imagenet"
    cfg.train.loss = "focal_dice"

    print(f"Loading trained model from {checkpoint_path}...")
    model, _ = load_model_for_inference(checkpoint_path, cfg.model, cfg.train.loss, device)
    threshold = 0.5
    base_graph_cfg = GraphConfig(max_gap_px=22.0, min_gap_confidence=0.15)

    patterns = ["rectangular_block", "irregular_patch", "linear_transverse"]

    # per_records[threshold_label] = list of row-dicts
    per_records: Dict[str, List[Dict]] = {t["label"]: [] for t in THRESHOLDS}

    print(f"\nRunning Threshold Ablation on {len(selected_tiles)} tiles × "
          f"{len(patterns)} patterns × {len(THRESHOLDS)} thresholds...")
    print(f"Thresholds: {[t['label'] for t in THRESHOLDS]}")
    print("-" * 70)

    for t_idx, row in enumerate(selected_tiles.itertuples(index=False), start=1):
        filename = row.filename
        image_path = dataset_dir / "images_png" / filename
        mask_path  = dataset_dir / "masks_png"  / filename

        if not image_path.exists() or not mask_path.exists():
            print(f"  [SKIP] {filename}: missing image or mask.")
            continue

        orig_rgb = load_rgb_image(image_path)
        gt_mask  = np.asarray(Image.open(mask_path).convert("L")) > 0
        G_gt     = _extract_gt_graph(gt_mask)

        # Baseline clean IoU (once per tile, threshold-independent)
        clean_prob = predict_probability_map(model, orig_rgb, device)
        clean_bin  = clean_prob >= threshold
        clean_iou  = float(jaccard_score(gt_mask.ravel(), clean_bin.ravel(), zero_division=1.0))

        for p_idx, pattern in enumerate(patterns):
            # --- Occlusion (deterministic, identical seed to Phase 3A) ---
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

            # --- Model inference: done ONCE, cached for all thresholds ---
            occ_prob = predict_probability_map(model, occluded_rgb, device)
            occ_bin  = occ_prob >= threshold
            occ_iou  = float(jaccard_score(gt_mask.ravel(), occ_bin.ravel(), zero_division=1.0))
            occ_dice = float(f1_score(gt_mask.ravel(), occ_bin.ravel(), zero_division=1.0))

            # --- Graph extraction (before healing): also done ONCE per (tile, pattern) ---
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
            cr_unhealed  = connectivity_ratio(skeleton)

            # --- Re-run gap healing for each threshold ---
            for tdef in THRESHOLDS:
                G_healed = _run_gap_healing(
                    graph_before, occ_prob, base_graph_cfg, tdef["dot"]
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

                per_records[tdef["label"]].append({
                    "tile_id":                    Path(filename).stem,
                    "pattern":                    pattern,
                    "threshold_label":            tdef["label"],
                    "threshold_dot":              tdef["dot"],
                    "threshold_angle_deg":        tdef["angle_deg"],
                    "road_percentage":            float(row.road_percentage),
                    "affected_road_pixels":       occ_meta["affected_road_pixels"],
                    "clean_iou":                  clean_iou,
                    "occluded_iou":               occ_iou,
                    "occluded_dice":              occ_dice,
                    "iou_drop":                   clean_iou - occ_iou,
                    "unhealed_components":        matching_eval["unhealed_component_count"],
                    "healed_components":          matching_eval["healed_component_count"],
                    "gt_components":              matching_eval["gt_component_count"],
                    "component_reduction":        matching_eval["component_reduction"],
                    "total_recovered_edges":      matching_eval["total_recovered_edges"],
                    "true_recovered_connections": matching_eval["true_recovered_connections"],
                    "false_recovered_connections":matching_eval["false_recovered_connections"],
                    "occlusion_recovered_tp":     matching_eval["occlusion_recovered_tp"],
                    "occlusion_recovered_fp":     matching_eval["occlusion_recovered_fp"],
                    "connection_recovery_precision": matching_eval["connection_recovery_precision"],
                    "connection_recovery_recall": matching_eval["connection_recovery_recall"],
                    "connectivity_ratio_unhealed": cr_unhealed,
                })

        print(f"[{t_idx}/{len(selected_tiles)}] Tile {filename} — all patterns & thresholds done.")

    # --- Flatten to single CSV ---
    all_rows = []
    for rows in per_records.values():
        all_rows.extend(rows)
    results_df = pd.DataFrame(all_rows)
    csv_path = output_dir / "ablation_per_tile_metrics.csv"
    results_df.to_csv(csv_path, index=False)

    # --- Build comparison summary ---
    threshold_summaries = {}
    for tdef in THRESHOLDS:
        agg = _aggregate_records(per_records[tdef["label"]])
        threshold_summaries[tdef["label"]] = {
            "angle_deg": tdef["angle_deg"],
            "dot_product": tdef["dot"],
            **agg,
        }

    # --- Per-pattern breakdown per threshold ---
    per_pattern_breakdown: Dict[str, Dict] = {}
    for tdef in THRESHOLDS:
        label = tdef["label"]
        pat_breakdown = {}
        label_rows = [r for r in per_records[label]]
        for pattern in patterns:
            pat_rows = [r for r in label_rows if r["pattern"] == pattern]
            pat_breakdown[pattern] = _aggregate_records(pat_rows)
        per_pattern_breakdown[label] = pat_breakdown

    summary = {
        "experiment": "Phase 3B — Tangent-Alignment Threshold Ablation",
        "num_tiles_evaluated": len(selected_tiles),
        "total_test_evaluations": len(results_df),
        "occlusion_patterns": patterns,
        "thresholds_evaluated": [
            {"label": t["label"], "angle_deg": t["angle_deg"], "dot": t["dot"]}
            for t in THRESHOLDS
        ],
        "threshold_comparison": threshold_summaries,
        "per_pattern_breakdown": per_pattern_breakdown,
        "artifacts": {
            "per_tile_csv": str(csv_path),
            "summary_json": str(output_dir / "ablation_summary.json"),
        },
        "metric_limitations": [
            "Recall denominator: tiles with 0 broken connections receive recall=1.0 by convention, "
            "inflating mean recall — inherited from Phase 3A matching rules.",
            "fp_rate = FP/(TP+FP), not FP/(FP+TN). True-negative connections are not enumerated.",
            "n=12 tiles x 3 patterns = 36 evaluations per threshold. Results are directional only.",
            "Inference is cached: each (tile, pattern) prob_map is computed once and shared across "
            "thresholds for a fair comparison; only confidence_gated_gap_healing is re-run.",
        ],
    }

    json_path = output_dir / "ablation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Console report ---
    print("\n" + "=" * 70)
    print("PHASE 3B ABLATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    header = f"{'Threshold':<22} {'Angle':>6} {'Recov':>6} {'TP':>5} {'FP':>5} {'Prec':>7} {'Recall':>7} {'CompRed':>8}"
    print(header)
    print("-" * 70)
    baseline_prec = None
    for tdef in THRESHOLDS:
        s = threshold_summaries[tdef["label"]]
        flag = ""
        if baseline_prec is None:
            baseline_prec = s["precision"]
        else:
            delta = s["precision"] - baseline_prec
            flag = f"  dPrec={delta:+.3f}"
        print(
            f"{tdef['label']:<22} {s['angle_deg']:>5.1f}deg "
            f"{s['total_recovered_edges']:>6} {s['true_positive_connections']:>5} "
            f"{s['false_positive_connections']:>5} {s['precision']:>7.1%} "
            f"{s['mean_recall']:>7.1%} {s['mean_component_reduction']:>8.2f}"
            f"{flag}"
        )

    print("-" * 70)
    print("\nPer-pattern breakdown (Precision):")
    pat_header = f"  {'Threshold':<22}" + "".join(f"  {p[:12]:>13}" for p in patterns)
    print(pat_header)
    for tdef in THRESHOLDS:
        label = tdef["label"]
        row_str = f"  {label:<22}"
        for pat in patterns:
            p_val = per_pattern_breakdown[label][pat]["precision"]
            row_str += f"  {p_val:>12.1%}"
        print(row_str)

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 3B Threshold Ablation")
    parser.add_argument("--dataset_dir",  type=Path, default=PROJECT_ROOT / "RouteResilience")
    parser.add_argument("--checkpoint",   type=Path, default=PROJECT_ROOT / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth")
    parser.add_argument("--output_dir",   type=Path, default=PROJECT_ROOT / "outputs" / "threshold_ablation")
    parser.add_argument("--num_tiles",    type=int,  default=12)
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--device",       type=str,  default="cpu")
    args = parser.parse_args()

    run_ablation(
        dataset_dir=args.dataset_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_tiles=args.num_tiles,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
