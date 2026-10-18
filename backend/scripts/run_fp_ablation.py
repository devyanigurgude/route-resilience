"""Phase 3D — Controlled False-Positive Ablation Experiment.

Evaluates 4 configurations on 32 unique tiles (20 top-FP tiles + 12 standard validation tiles)
across all 3 occlusion patterns under identical random seeds and conditions:

  1. Baseline:
     min_endpoint_confidence = 0.00
     min_path_evidence       = 0.15 (standard production default)
  2. Variant A (Endpoint Gate Only):
     min_endpoint_confidence = 0.35
     min_path_evidence       = 0.15
  3. Variant B (Path Evidence Gate Only):
     min_endpoint_confidence = 0.00
     min_path_evidence       = 0.25
  4. Variant C (Combined):
     min_endpoint_confidence = 0.35
     min_path_evidence       = 0.25

Outputs are saved to outputs/false_positive_ablation/ (does NOT overwrite Phase 3C).
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
    _find_probability_guided_path,
    _outward_endpoint_tangent,
)
from route_resilience.occlusion_simulation import generate_road_centered_occlusions
from route_resilience.real_graph_pipeline import (
    load_model_for_inference,
    load_rgb_image,
    predict_probability_map,
)
from route_resilience.topology_matching import evaluate_connection_recovery
from route_resilience.train import get_device


VARIANTS = [
    {
        "name": "baseline",
        "min_endpoint_confidence": 0.0,
        "min_path_evidence": 0.15,
        "desc": "Production baseline (0.15 min_gap_confidence, no endpoint gate)",
    },
    {
        "name": "variant_a_endpoint_gate",
        "min_endpoint_confidence": 0.35,
        "min_path_evidence": 0.15,
        "desc": "Endpoint confidence gate at 0.35, baseline path evidence 0.15",
    },
    {
        "name": "variant_b_path_evidence",
        "min_endpoint_confidence": 0.0,
        "min_path_evidence": 0.25,
        "desc": "Baseline endpoints, path evidence threshold boosted to 0.25",
    },
    {
        "name": "variant_c_combined",
        "min_endpoint_confidence": 0.35,
        "min_path_evidence": 0.25,
        "desc": "Combined: endpoint confidence >= 0.35 AND path evidence >= 0.25",
    },
]


def _extract_gt_graph(gt_mask: np.ndarray, min_edge_length_px: float = 3.0) -> Optional[nx.Graph]:
    skel = mask_to_skeleton(gt_mask > 0)
    if skel.sum() == 0:
        return None
    return skeleton_to_graph(skel, prob_map=None, min_edge_length_px=min_edge_length_px)


def _custom_gap_healing(
    G: nx.Graph,
    prob_map: np.ndarray,
    max_gap_px: float = 22.0,
    min_gap_confidence: float = 0.15,
    tangent_dot_threshold: float = 0.7,
    min_endpoint_confidence: float = 0.0,
    min_path_evidence: float = 0.15,
) -> nx.Graph:
    """Gap healing with controllable endpoint confidence gate and path evidence gate."""
    G = G.copy()
    all_endpoints = [n for n in G.nodes() if G.degree(n) == 1]

    # Filter endpoints by incident edge confidence if min_endpoint_confidence > 0
    if min_endpoint_confidence > 0.0:
        valid_endpoints = []
        for n in all_endpoints:
            (nbr,) = G.neighbors(n)
            edge_conf = float(G.edges[n, nbr].get("confidence", 1.0))
            if edge_conf >= min_endpoint_confidence:
                valid_endpoints.append(n)
        endpoints = valid_endpoints
    else:
        endpoints = all_endpoints

    positions = {n: np.array(G.nodes[n]["pos"]) for n in endpoints}
    tangents = {n: _outward_endpoint_tangent(G, n) for n in endpoints}

    candidates = []
    for i, a in enumerate(endpoints):
        for b in endpoints[i + 1:]:
            if G.has_edge(a, b):
                continue
            delta = positions[b] - positions[a]
            dist = np.linalg.norm(delta)
            if dist <= max_gap_px:
                if dist == 0:
                    continue
                direction = delta / dist
                if (
                    np.dot(tangents[a], direction) >= tangent_dot_threshold
                    and np.dot(tangents[b], -direction) >= tangent_dot_threshold
                ):
                    candidates.append((dist, a, b))

    candidates.sort(key=lambda c: c[0])

    for dist, a, b in candidates:
        if a not in G or b not in G:
            continue
        if G.degree(a) != 1 or G.degree(b) != 1:
            continue
        if nx.has_path(G, a, b):
            continue

        n_samples = max(4, int(dist))
        line = np.linspace(positions[a], positions[b], n_samples)
        margin = max(1, n_samples // 5)
        interior = line[margin:-margin] if n_samples > 2 * margin else line
        rows = np.clip(np.rint(interior[:, 0]).astype(int), 0, prob_map.shape[0] - 1)
        cols = np.clip(np.rint(interior[:, 1]).astype(int), 0, prob_map.shape[1] - 1)
        support = prob_map[rows, cols]
        evidence = float(support.mean())
        supported_fraction = float(np.mean(support >= min_gap_confidence))

        # Straight-line recovery gate
        if evidence >= min_gap_confidence and supported_fraction >= 0.7:
            G.add_edge(
                a, b,
                length_px=float(dist),
                confidence=float(evidence),
                healed=True,
                occlusion_recovered=False,
                pts=line,
            )
        else:
            # Fallback path search: uses min_path_evidence
            recovered_path = _find_probability_guided_path(
                prob_map=prob_map,
                start_pos=positions[a],
                end_pos=positions[b],
                max_path_length=float(max_gap_px * 1.5),
            )
            if recovered_path is not None and len(recovered_path) >= 2:
                diffs = np.diff(recovered_path, axis=0)
                path_length = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))

                n_pts = len(recovered_path)
                p_margin = max(1, n_pts // 5)
                p_interior = (
                    recovered_path[p_margin:-p_margin]
                    if n_pts > 2 * p_margin
                    else recovered_path
                )
                p_rows = np.clip(np.rint(p_interior[:, 0]).astype(int), 0, prob_map.shape[0] - 1)
                p_cols = np.clip(np.rint(p_interior[:, 1]).astype(int), 0, prob_map.shape[1] - 1)
                p_support = prob_map[p_rows, p_cols]
                p_evidence = float(p_support.mean())
                p_supported_fraction = float(np.mean(p_support >= min_path_evidence))

                if (
                    p_evidence >= min_path_evidence
                    and p_supported_fraction >= 0.7
                    and path_length <= max_gap_px * 1.5
                ):
                    G.add_edge(
                        a, b,
                        length_px=path_length,
                        confidence=p_evidence,
                        healed=True,
                        occlusion_recovered=True,
                        pts=recovered_path,
                    )

    return G


def run_phase_3d_ablation(
    dataset_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    seed: int = 42,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Define tile sets without duplicates
    top_20_fp_tiles = [213, 4526, 6314, 7074, 5061, 7624, 7728, 1930, 5621, 4547,
                       2719, 7546, 4119, 4782, 6231, 8030, 8952, 8996, 7881, 2904]
    val_12_tiles = [4428, 9671, 76, 5146, 6448, 8172, 1319, 4407, 1494, 7195, 9541, 2444]

    # Tag each tile by group (unique list of 32 tiles)
    tile_groups = {}
    for tid in top_20_fp_tiles:
        tile_groups[tid] = "top_20_fp"
    for tid in val_12_tiles:
        tile_groups[tid] = "val_12"

    unique_tiles = list(tile_groups.keys())
    print(f"Total unique tiles to evaluate: {len(unique_tiles)} (Top-20 FP: {len(top_20_fp_tiles)}, Val-12: {len(val_12_tiles)})")

    # 2. Setup model
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
    base_graph_cfg = GraphConfig(max_gap_px=22.0, min_gap_confidence=0.15)

    all_records = []
    print("\nExecuting Phase 3D False-Positive Ablation across 4 variants...")
    print("-" * 75)

    for t_idx, tile_id in enumerate(unique_tiles, start=1):
        filename = f"{tile_id}.png"
        image_path = dataset_dir / "images_png" / filename
        mask_path = dataset_dir / "masks_png" / filename
        group_tag = tile_groups[tile_id]

        if not image_path.exists() or not mask_path.exists():
            print(f"  [SKIP] {filename} missing on disk.")
            continue

        orig_rgb = load_rgb_image(image_path)
        gt_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        G_gt = _extract_gt_graph(gt_mask)

        # Baseline Clean Inference
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

            # Model inference done once per scenario
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

            # Evaluate each variant on the exact same graph_before and occ_prob
            for var in VARIANTS:
                G_healed = _custom_gap_healing(
                    G=graph_before,
                    prob_map=occ_prob,
                    max_gap_px=base_graph_cfg.max_gap_px,
                    min_gap_confidence=base_graph_cfg.min_gap_confidence,
                    tangent_dot_threshold=base_graph_cfg.tangent_dot_threshold,
                    min_endpoint_confidence=var["min_endpoint_confidence"],
                    min_path_evidence=var["min_path_evidence"],
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
                    "variant": var["name"],
                    "min_endpoint_confidence": var["min_endpoint_confidence"],
                    "min_path_evidence": var["min_path_evidence"],
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

        print(f"[{t_idx}/{len(unique_tiles)}] Tile {tile_id} ({group_tag}) — all patterns & 4 variants evaluated.")

    results_df = pd.DataFrame(all_records)
    csv_path = output_dir / "ablation_3d_metrics.csv"
    results_df.to_csv(csv_path, index=False)

    # 3. Compute summaries
    summary_by_variant = {}
    for var in VARIANTS:
        v_name = var["name"]
        v_df = results_df[results_df["variant"] == v_name]
        v_rec = int(v_df["total_recovered_edges"].sum())
        v_tp = int(v_df["true_recovered_connections"].sum())
        v_fp = int(v_df["false_recovered_connections"].sum())
        v_occ_tp = int(v_df["occlusion_recovered_tp"].sum())
        v_occ_fp = int(v_df["occlusion_recovered_fp"].sum())

        # Split summaries by tile group
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

        # Pattern summaries
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
            "description": var["desc"],
            "min_endpoint_confidence": var["min_endpoint_confidence"],
            "min_path_evidence": var["min_path_evidence"],
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
        "experiment": "Phase 3D — Controlled False-Positive Ablation",
        "num_tiles_evaluated": len(unique_tiles),
        "total_scenarios_per_variant": len(unique_tiles) * len(patterns),
        "variants": summary_by_variant,
        "artifacts_saved": {
            "per_evaluation_csv": str(csv_path),
            "summary_json": str(output_dir / "ablation_3d_summary.json"),
        },
    }

    with open(output_dir / "ablation_3d_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 75)
    print("PHASE 3D ABLATION COMPLETED SUCCESSFULLY")
    print("=" * 75)
    header = f"{'Variant':<26} {'Recov':>6} {'TP':>5} {'FP':>5} {'Prec':>7} {'Recall':>7} {'CompRed':>8}"
    print(header)
    print("-" * 75)
    for v_name, s in summary_by_variant.items():
        print(
            f"{v_name:<26} {s['total_recovered_edges']:>6} {s['true_positive_connections']:>5} "
            f"{s['false_positive_connections']:>5} {s['overall_precision']:>7.1%} "
            f"{s['mean_recall']:>7.1%} {s['mean_component_reduction']:>8.2f}"
        )
    print("=" * 75)

    return final_report


def main():
    parser = argparse.ArgumentParser(description="Phase 3D Controlled False-Positive Ablation")
    parser.add_argument("--dataset_dir", type=Path, default=PROJECT_ROOT / "RouteResilience")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "false_positive_ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_phase_3d_ablation(
        dataset_dir=args.dataset_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
