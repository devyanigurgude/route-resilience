"""Priority 2: Shadow & Canopy Mask Dilation Benchmark.

Evaluates baseline (no dilation, iterations=0) vs 3x3 1-iteration vs 3x3 2-iterations
morphological dilation across controlled synthetic shadow, canopy, and negative-control scenarios.

Reports:
  - Mask area before and after dilation (px and % expansion)
  - True Positives (TP), False Positives (FP), False Negatives (FN)
  - Recovery Precision and Recall
  - False-Bridge Rate (FP / (TP + FP))
  - Connected component counts and connectivity improvement
  - Breakdown by category: Shadow, Canopy, Negative Controls, and Total Aggregate
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import networkx as nx
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from route_resilience.graph_extraction import confidence_gated_gap_healing
from route_resilience.real_graph_pipeline import (
    canopy_occlusion_mask,
    dilate_mask,
    photometric_occlusion_mask,
    shadow_occlusion_mask,
)


def _add_oriented_segment(
    G: nx.Graph,
    prefix: str,
    start_pos: Tuple[float, float],
    angle_deg: float,
    n_nodes: int,
    step: float = 1.0,
) -> Tuple[str, str]:
    rad = np.radians(angle_deg)
    dr, dc = float(np.sin(rad)), float(np.cos(rad))
    y0, x0 = start_pos
    for i in range(n_nodes):
        pos = (y0 + i * step * dr, x0 + i * step * dc)
        G.add_node(f"{prefix}{i}", pos=pos)
        if i > 0:
            prev = (y0 + (i - 1) * step * dr, x0 + (i - 1) * step * dc)
            G.add_edge(
                f"{prefix}{i-1}", f"{prefix}{i}",
                length_px=step,
                pts=np.array([prev, pos]),
            )
    return f"{prefix}0", f"{prefix}{n_nodes - 1}"


def build_scenario_data(sc_def: Dict[str, Any], image_size: Tuple[int, int] = (160, 220)) -> Dict[str, Any]:
    """Constructs the road graph, probability map, and synthetic RGB image for a scenario."""
    H, W = image_size
    G = nx.Graph()
    gap_px = sc_def["gap_px"]
    trunk_len = sc_def.get("trunk_len", 15)
    angle_deg = sc_def.get("angle_deg", 0.0)
    turn_deg = sc_def.get("turn_deg", 0.0)
    lat_shift = sc_def.get("lateral_shift_px", 0.0)
    category = sc_def["category"]
    pattern = sc_def.get("pattern", "full")

    rad = np.radians(angle_deg)
    dr, dc = float(np.sin(rad)), float(np.cos(rad))
    norm_r, norm_c = -dc, dr

    total_len = 2 * trunk_len + gap_px
    y0 = max(25.0, (H - total_len * dr) / 2.0)
    x0 = max(25.0, (W - total_len * dc) / 2.0)

    # Left trunk
    left_start, left_end = _add_oriented_segment(G, "L", (y0, x0), angle_deg, trunk_len + 1)

    # Right trunk start
    r_start_y = y0 + trunk_len * dr + gap_px * dr + lat_shift * norm_r
    r_start_x = x0 + trunk_len * dc + gap_px * dc + lat_shift * norm_c
    right_angle = angle_deg + turn_deg
    right_start, right_end = _add_oriented_segment(G, "R", (r_start_y, r_start_x), right_angle, trunk_len + 1)

    # Prob map: high prob on trunks, zero in gap
    prob_map = np.full((H, W), 0.02, dtype=np.float32)
    for i in range(trunk_len + 1):
        lr, lc = int(round(y0 + i * dr)), int(round(x0 + i * dc))
        prob_map[max(0, lr - 2):min(H, lr + 3), max(0, lc - 2):min(W, lc + 3)] = 0.90

    r_rad = np.radians(right_angle)
    r_dr, r_dc = float(np.sin(r_rad)), float(np.cos(r_rad))
    for i in range(trunk_len + 1):
        rr, rc = int(round(r_start_y + i * r_dr)), int(round(r_start_x + i * r_dc))
        prob_map[max(0, rr - 2):min(H, rr + 3), max(0, rc - 2):min(W, rc + 3)] = 0.90

    # Default RGB: neutral background (soil/gravel: [110, 105, 95])
    rgb = np.full((H, W, 3), [110, 105, 95], dtype=np.uint8)

    # Add shadow/canopy occlusion in RGB
    if category == "shadow":
        # Shadow color: dark [30, 30, 30]
        if pattern == "full":
            for t in np.linspace(-0.1, 1.1, int(gap_px * 3)):
                gr = y0 + trunk_len * dr + t * gap_px * dr
                gc = x0 + trunk_len * dc + t * gap_px * dc
                r_int, c_int = int(round(gr)), int(round(gc))
                rgb[max(0, r_int - 8):min(H, r_int + 9), max(0, c_int - 8):min(W, c_int + 9)] = [30, 30, 30]
        elif pattern == "sparse_offset":
            # Shadow swath is slightly offset laterally (+1px normal) on 70% of gap
            # Raw centerline coverage is ~41.2% (< 50% threshold), so baseline fails
            # Dilation by 1 pixel restores centerline coverage to 100%
            for t in np.linspace(0.0, 1.0, int(gap_px * 3)):
                gr = y0 + trunk_len * dr + t * gap_px * dr
                gc = x0 + trunk_len * dc + t * gap_px * dc
                r_int, c_int = int(round(gr)), int(round(gc))
                if t <= 0.30:
                    rgb[max(0, r_int - 2):min(H, r_int + 3), max(0, c_int - 2):min(W, c_int + 3)] = [30, 30, 30]
                else:
                    rgb[max(0, r_int + 1):min(H, r_int + 5), max(0, c_int - 2):min(W, c_int + 3)] = [30, 30, 30]

    elif category == "canopy":
        # Canopy color: foliage green [45, 75, 40]
        if pattern == "full":
            for t in np.linspace(-0.1, 1.1, int(gap_px * 3)):
                gr = y0 + trunk_len * dr + t * gap_px * dr
                gc = x0 + trunk_len * dc + t * gap_px * dc
                r_int, c_int = int(round(gr)), int(round(gc))
                rgb[max(0, r_int - 8):min(H, r_int + 9), max(0, c_int - 8):min(W, c_int + 9)] = [45, 75, 40]
        elif pattern == "sparse_boundary":
            # Foliage with a slight lateral boundary offset (+1px normal) on 70% of gap
            for t in np.linspace(0.0, 1.0, int(gap_px * 3)):
                gr = y0 + trunk_len * dr + t * gap_px * dr
                gc = x0 + trunk_len * dc + t * gap_px * dc
                r_int, c_int = int(round(gr)), int(round(gc))
                if t <= 0.30:
                    rgb[max(0, r_int - 2):min(H, r_int + 3), max(0, c_int - 2):min(W, c_int + 3)] = [45, 75, 40]
                else:
                    rgb[max(0, r_int + 1):min(H, r_int + 5), max(0, c_int - 2):min(W, c_int + 3)] = [45, 75, 40]

    elif category == "negative_control":
        # Negative control scenarios
        if sc_def.get("has_shadow", False):
            for t in np.linspace(-0.2, 1.2, int(gap_px * 3)):
                gr = y0 + trunk_len * dr + t * gap_px * dr
                gc = x0 + trunk_len * dc + t * gap_px * dc
                r_int, c_int = int(round(gr)), int(round(gc))
                rgb[max(0, r_int - 15):min(H, r_int + 16), max(0, c_int - 15):min(W, c_int + 16)] = [30, 30, 30]
        elif sc_def.get("has_canopy", False):
            for t in np.linspace(-0.2, 1.2, int(gap_px * 3)):
                gr = y0 + trunk_len * dr + t * gap_px * dr
                gc = x0 + trunk_len * dc + t * gap_px * dc
                r_int, c_int = int(round(gr)), int(round(gc))
                rgb[max(0, r_int - 15):min(H, r_int + 16), max(0, c_int - 15):min(W, c_int + 16)] = [45, 75, 40]
        # clear sky has no occlusion stamps

    gt_edge = (left_end, right_start)
    return {
        "G": G,
        "prob_map": prob_map,
        "rgb": rgb,
        "gt_edge": gt_edge,
        "is_true_connection": sc_def["is_true_connection"],
        "category": category,
        "name": sc_def["name"],
        "gap_px": gap_px,
        "angle_deg": angle_deg,
    }


def define_scenarios() -> List[Dict[str, Any]]:
    return [
        # --- Shadow Scenarios ---
        {
            "name": "shadow_collinear_horiz_22px",
            "category": "shadow",
            "gap_px": 22.0,
            "angle_deg": 0.0,
            "pattern": "full",
            "is_true_connection": True,
        },
        {
            "name": "shadow_diagonal_30deg_22px",
            "category": "shadow",
            "gap_px": 22.0,
            "angle_deg": 30.0,
            "pattern": "full",
            "is_true_connection": True,
        },
        {
            "name": "shadow_sparse_offset_20px",
            "category": "shadow",
            "gap_px": 20.0,
            "angle_deg": 0.0,
            "pattern": "sparse_offset",
            "is_true_connection": True,
        },

        # --- Canopy Scenarios ---
        {
            "name": "canopy_collinear_horiz_20px",
            "category": "canopy",
            "gap_px": 20.0,
            "angle_deg": 0.0,
            "pattern": "full",
            "is_true_connection": True,
        },
        {
            "name": "canopy_diagonal_45deg_20px",
            "category": "canopy",
            "gap_px": 20.0,
            "angle_deg": 45.0,
            "pattern": "full",
            "is_true_connection": True,
        },
        {
            "name": "canopy_sparse_boundary_22px",
            "category": "canopy",
            "gap_px": 22.0,
            "angle_deg": 0.0,
            "pattern": "sparse_boundary",
            "is_true_connection": True,
        },

        # --- Negative Controls ---
        {
            "name": "neg_misaligned_turn_45deg_shadow",
            "category": "negative_control",
            "gap_px": 20.0,
            "angle_deg": 0.0,
            "turn_deg": 45.0,
            "has_shadow": True,
            "is_true_connection": False,
        },
        {
            "name": "neg_parallel_offset_roads_canopy",
            "category": "negative_control",
            "gap_px": 15.0,
            "angle_deg": 0.0,
            "lateral_shift_px": 12.0,
            "has_canopy": True,
            "is_true_connection": False,
        },
        {
            "name": "neg_oversized_gap_36px_shadow",
            "category": "negative_control",
            "gap_px": 36.0,
            "angle_deg": 0.0,
            "has_shadow": True,
            "is_true_connection": False,
        },
        {
            "name": "neg_zero_evidence_clear_sky_22px",
            "category": "negative_control",
            "gap_px": 22.0,
            "angle_deg": 0.0,
            "is_true_connection": False,
        },
    ]


def run_configuration_on_scenario(
    scenario_data: Dict[str, Any],
    dilation_kernel_size: int,
    dilation_iterations: int,
) -> Dict[str, Any]:
    """Runs gap recovery on one scenario with a specific mask dilation configuration."""
    G = scenario_data["G"].copy()
    prob_map = scenario_data["prob_map"]
    rgb = scenario_data["rgb"]
    gt_u, gt_v = scenario_data["gt_edge"]
    is_true = scenario_data["is_true_connection"]

    # 1. Compute raw mask (0 iterations)
    raw_mask = photometric_occlusion_mask(rgb, dilation_kernel_size=dilation_kernel_size, dilation_iterations=0)
    raw_area = int(raw_mask.sum())

    # 2. Compute configured mask
    active_mask = photometric_occlusion_mask(
        rgb,
        dilation_kernel_size=dilation_kernel_size,
        dilation_iterations=dilation_iterations,
    )
    dilated_area = int(active_mask.sum())
    area_delta_pct = round(((dilated_area - raw_area) / max(1, raw_area)) * 100.0, 2)

    # 3. Initial components
    initial_components = nx.number_connected_components(G)

    # 4. Run confidence-gated gap healing
    t0 = time.perf_counter()
    G_out = confidence_gated_gap_healing(
        G,
        prob_map,
        max_gap_px=15.0,
        min_gap_confidence=0.15,
        enable_occlusion_recovery=True,
        max_occlusion_gap_px=30.0,
        min_occlusion_confidence=0.05,
        occlusion_heading_threshold=0.75,
        max_lateral_offset_px=4.0,
        occlusion_mask=active_mask,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    final_components = nx.number_connected_components(G_out)
    components_reduced = initial_components - final_components

    healed_gt = G_out.has_edge(gt_u, gt_v)
    edge_data = G_out[gt_u][gt_v] if healed_gt else {}
    sinuosity = float(edge_data.get("sinuosity", 1.0)) if healed_gt else None
    recovery_type = edge_data.get("recovery_type") if healed_gt else None

    # Total edges added
    total_added_edges = sum(1 for _, _, d in G_out.edges(data=True) if d.get("healed", False))

    # Evaluation relative to ground truth
    # If is_true is True: healed_gt is TP, not healed is FN
    # If is_true is False: healed_gt (or any edge added) is FP, not healed is TN
    tp = 1 if (is_true and healed_gt) else 0
    fn = 1 if (is_true and not healed_gt) else 0
    fp = 1 if ((not is_true) and total_added_edges > 0) else 0

    return {
        "raw_mask_area_px": raw_area,
        "dilated_mask_area_px": dilated_area,
        "area_delta_pct": area_delta_pct,
        "initial_components": initial_components,
        "final_components": final_components,
        "components_reduced": components_reduced,
        "healed": healed_gt,
        "recovery_type": recovery_type,
        "sinuosity": round(sinuosity, 4) if sinuosity is not None else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "elapsed_ms": elapsed_ms,
    }


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes aggregate TP, FP, FN, precision, recall, false_bridge_rate, and components."""
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    total_recovered = tp + fp
    total_true = tp + fn

    precision = round(tp / total_recovered, 4) if total_recovered > 0 else 1.0
    recall = round(tp / total_true, 4) if total_true > 0 else 1.0
    false_bridge_rate = round(fp / total_recovered, 4) if total_recovered > 0 else 0.0

    mean_comp_reduced = round(sum(r["components_reduced"] for r in records) / max(1, len(records)), 3)
    mean_raw_area = round(sum(r["raw_mask_area_px"] for r in records) / max(1, len(records)), 1)
    mean_dilated_area = round(sum(r["dilated_mask_area_px"] for r in records) / max(1, len(records)), 1)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_bridge_rate": false_bridge_rate,
        "total_connections_added": total_recovered,
        "mean_components_reduced": mean_comp_reduced,
        "mean_raw_mask_area": mean_raw_area,
        "mean_dilated_mask_area": mean_dilated_area,
    }


def run_benchmark(output_dir: Optional[Path] = None, overwrite: bool = False) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = WORKSPACE_ROOT / "outputs" / "mask_dilation_benchmark"
    output_dir = Path(output_dir)

    out_file = output_dir / "dilation_summary.json"
    if out_file.exists() and not overwrite:
        raise FileExistsError(
            f"Benchmark output {out_file} already exists. Pass --overwrite or choose another directory."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios_defs = define_scenarios()
    scenarios_data = [build_scenario_data(sc) for sc in scenarios_defs]

    configs = [
        {"name": "Baseline (0 iter)", "kernel_size": 3, "iterations": 0},
        {"name": "Dilation 3x3 (1 iter)", "kernel_size": 3, "iterations": 1},
        {"name": "Dilation 3x3 (2 iter)", "kernel_size": 3, "iterations": 2},
    ]

    all_results = {cfg["name"]: [] for cfg in configs}

    print("\n" + "=" * 125)
    print(f"ROUTE RESILIENCE - PRIORITY 2: SHADOW & CANOPY MASK DILATION BENCHMARK")
    print("=" * 125)

    for sc in scenarios_data:
        for cfg in configs:
            res = run_configuration_on_scenario(
                sc,
                dilation_kernel_size=cfg["kernel_size"],
                dilation_iterations=cfg["iterations"],
            )
            res["scenario_name"] = sc["name"]
            res["category"] = sc["category"]
            res["is_true_connection"] = sc["is_true_connection"]
            all_results[cfg["name"]].append(res)

    # Detailed Table
    print(f"{'Scenario Name':<34} {'Category':<16} | {'Cfg':<15} {'Raw Area':>9} {'Dil Area':>9} {'Exp %':>7} | {'CC Red':>7} {'Healed':>7} {'TP':>3} {'FP':>3} {'FN':>3}")
    print("-" * 125)

    for sc in scenarios_data:
        s_name = sc["name"]
        cat = sc["category"]
        for cfg in configs:
            r = next(item for item in all_results[cfg["name"]] if item["scenario_name"] == s_name)
            cfg_label = cfg["name"]
            print(
                f"{s_name:<34} {cat:<16} | {cfg_label:<15} {r['raw_mask_area_px']:>9} {r['dilated_mask_area_px']:>9} {r['area_delta_pct']:>6.1f}% | "
                f"{r['components_reduced']:>7} {str(r['healed']):>7} {r['tp']:>3} {r['fp']:>3} {r['fn']:>3}"
            )
        print("-" * 125)

    # Category and Aggregate Summaries
    categories = ["shadow", "canopy", "negative_control"]
    summary_by_config = {}

    print("\n" + "=" * 125)
    print(f"AGGREGATE PERFORMANCE & TRADE-OFF METRICS")
    print("=" * 125)
    print(f"{'Configuration':<24} {'Subset':<18} | {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Recall':>7} {'FBR':>7} | {'Mean Red CC':>12} {'Mean Dil Area':>14}")
    print("-" * 125)

    for cfg in configs:
        cfg_name = cfg["name"]
        cfg_records = all_results[cfg_name]
        summary_by_config[cfg_name] = {"subsets": {}, "total": compute_metrics(cfg_records)}

        for cat in categories:
            cat_records = [r for r in cfg_records if r["category"] == cat]
            cat_metrics = compute_metrics(cat_records)
            summary_by_config[cfg_name]["subsets"][cat] = cat_metrics

            print(
                f"{cfg_name:<24} {cat:<18} | {cat_metrics['tp']:>4} {cat_metrics['fp']:>4} {cat_metrics['fn']:>4} "
                f"{cat_metrics['precision']:>7.3f} {cat_metrics['recall']:>7.3f} {cat_metrics['false_bridge_rate']:>7.3f} | "
                f"{cat_metrics['mean_components_reduced']:>12.2f} {cat_metrics['mean_dilated_mask_area']:>14.1f}"
            )

        tot = summary_by_config[cfg_name]["total"]
        print(
            f"{cfg_name:<24} {'OVERALL':<18} | {tot['tp']:>4} {tot['fp']:>4} {tot['fn']:>4} "
            f"{tot['precision']:>7.3f} {tot['recall']:>7.3f} {tot['false_bridge_rate']:>7.3f} | "
            f"{tot['mean_components_reduced']:>12.2f} {tot['mean_dilated_mask_area']:>14.1f}"
        )
        print("-" * 125)

    output_payload = {
        "configurations": configs,
        "scenarios": [sc["name"] for sc in scenarios_data],
        "detailed_results": all_results,
        "summary": summary_by_config,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[OK] Mask Dilation Benchmark completed safely.")
    print(f"Results written to: {out_file}\n")
    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate mask dilation configurations.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    run_benchmark(output_dir=args.output_dir, overwrite=args.overwrite)
