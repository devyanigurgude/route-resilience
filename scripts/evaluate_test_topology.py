"""Evaluate test-set topological accuracy and Average Path Length (APL) error.

This script computes PS-4 topological validation metrics between the predicted
healed graphs (G_healed) from Phase III artifacts and ground-truth road graphs
(G_gt) extracted from the ground-truth binary masks.

All metrics are strictly evaluated in pixel-space units without fabricating
fictitious geographic coordinates, spatial transforms, or CRS metadata.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.evaluate_topology import simplified_apls
from route_resilience.graph_extraction import mask_to_skeleton, skeleton_to_graph
from route_resilience.phase3_pipeline import load_graph_json


DATASET_DIR = PROJECT_ROOT / "RouteResilience"
SPLIT_FILE = DATASET_DIR / "data_split.csv"
MASK_DIR = DATASET_DIR / "masks_png"
TEST_BATCH_DIR = PROJECT_ROOT / "outputs" / "test_batch"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "topology"


def extract_gt_graph(
    mask_path: Path, min_edge_length_px: float = 3.0
) -> Optional[nx.Graph]:
    """Load GT binary mask, skeletonize, and convert to a pixel-space NetworkX graph."""
    if not mask_path.exists():
        return None

    mask = np.array(Image.open(mask_path).convert("L")) > 0
    if mask.sum() == 0:
        return None

    skeleton = mask_to_skeleton(mask)
    if skeleton.sum() == 0:
        return None

    G_gt = skeleton_to_graph(
        skeleton, prob_map=None, min_edge_length_px=min_edge_length_px
    )
    return G_gt


def evaluate_single_image(
    filename: str,
    road_percentage: float,
    snap_radius: float = 10.0,
    n_pairs: int = 200,
    seed: int = 42,
    min_edge_length_px: float = 3.0,
) -> Dict[str, Any]:
    """Evaluate topology score & path length error for one test image."""
    image_id = Path(filename).stem
    mask_path = MASK_DIR / filename

    record = {
        "filename": filename,
        "image_id": image_id,
        "road_percentage": float(road_percentage),
        "status": "skipped",
        "skip_reason": None,
        "gt_nodes": 0,
        "gt_edges": 0,
        "pred_nodes": 0,
        "pred_edges": 0,
        "topology_score": None,
        "path_length_error": None,
        "miss_rate": None,
        "n_pairs_evaluated": 0,
    }

    if not mask_path.exists():
        record["skip_reason"] = "gt_mask_file_not_found"
        return record

    G_gt = extract_gt_graph(mask_path, min_edge_length_px=min_edge_length_px)
    if G_gt is None or G_gt.number_of_nodes() == 0:
        record["skip_reason"] = "gt_mask_empty_or_no_skeleton"
        return record

    record["gt_nodes"] = G_gt.number_of_nodes()
    record["gt_edges"] = G_gt.number_of_edges()

    if G_gt.number_of_nodes() < 2:
        record["skip_reason"] = "gt_insufficient_nodes (<2)"
        return record

    pred_graph_path = (
        TEST_BATCH_DIR / image_id / "phase3" / "graph" / "graph_healed.json"
    )
    if not pred_graph_path.exists():
        # Fallback to direct healed_graph.json if phase3 subdirectory is absent
        pred_graph_path = TEST_BATCH_DIR / image_id / "healed_graph.json"

    if not pred_graph_path.exists():
        record["skip_reason"] = "pred_graph_not_found"
        return record

    try:
        G_pred = load_graph_json(pred_graph_path)
    except Exception as e:
        record["skip_reason"] = f"pred_graph_load_error: {e}"
        return record

    record["pred_nodes"] = G_pred.number_of_nodes()
    record["pred_edges"] = G_pred.number_of_edges()

    if G_pred.number_of_nodes() < 2:
        record["skip_reason"] = "pred_insufficient_nodes (<2)"
        return record

    # Run standard simplified_apls from evaluate_topology.py
    apls_result = simplified_apls(
        G_pred=G_pred,
        G_gt=G_gt,
        pos_attr="pos",
        weight="length_px",
        snap_radius=snap_radius,
        n_pairs=n_pairs,
        seed=seed,
    )

    score = float(apls_result["score"])
    record["status"] = "evaluated"
    record["topology_score"] = score
    record["path_length_error"] = float(1.0 - score)
    record["miss_rate"] = float(apls_result["miss_rate"])
    record["n_pairs_evaluated"] = int(apls_result["n_pairs_evaluated"])
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate topological accuracy / Average Path Length error on the test split."
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of test images to evaluate (default: all).",
    )
    parser.add_argument(
        "--snap-radius",
        type=float,
        default=10.0,
        help="Pixel snap radius for node matching (default: 10.0 px).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=200,
        help="Number of node pairs to sample for APLS per image (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--min-edge-length-px",
        type=float,
        default=3.0,
        help="Minimum edge length in pixels for skeleton pruning (default: 3.0 px).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ROUTE RESILIENCE - TEST SET TOPOLOGY & PATH LENGTH EVALUATION")
    print("=" * 70)

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")

    df = pd.read_csv(SPLIT_FILE)
    test_df = df[df["split"] == "test"].copy()

    total_test_found = len(test_df)
    print(f"Total test split images found: {total_test_found}")

    if args.max_images is not None:
        test_df = test_df.iloc[: args.max_images]
        print(f"Evaluating subset: {len(test_df)} images")

    results = []
    print(f"Snap radius: {args.snap_radius} px | Sample pairs: {args.n_pairs}")
    print("\nProcessing images...")

    for index, row in enumerate(test_df.itertuples(index=False), start=1):
        filename = row.filename
        road_percentage = getattr(row, "road_percentage", 0.0)

        record = evaluate_single_image(
            filename=filename,
            road_percentage=road_percentage,
            snap_radius=args.snap_radius,
            n_pairs=args.n_pairs,
            seed=args.seed,
            min_edge_length_px=args.min_edge_length_px,
        )
        results.append(record)

        if index % 100 == 0 or index == len(test_df):
            evaluated_so_far = sum(1 for r in results if r["status"] == "evaluated")
            print(f"  [{index}/{len(test_df)}] Evaluated: {evaluated_so_far}, Skipped: {index - evaluated_so_far}")

    results_df = pd.DataFrame(results)

    # Separate evaluated vs skipped
    evaluated_df = results_df[results_df["status"] == "evaluated"].copy()
    skipped_df = results_df[results_df["status"] == "skipped"].copy()

    num_evaluated = len(evaluated_df)
    num_skipped = len(skipped_df)
    skip_breakdown = skipped_df["skip_reason"].value_counts().to_dict()

    total_od_pairs = (
        int(evaluated_df["n_pairs_evaluated"].sum()) if num_evaluated > 0 else 0
    )

    mean_score = (
        float(evaluated_df["topology_score"].mean()) if num_evaluated > 0 else 0.0
    )
    median_score = (
        float(evaluated_df["topology_score"].median()) if num_evaluated > 0 else 0.0
    )
    std_score = (
        float(evaluated_df["topology_score"].std()) if num_evaluated > 0 else 0.0
    )

    mean_error = (
        float(evaluated_df["path_length_error"].mean()) if num_evaluated > 0 else 1.0
    )
    median_error = (
        float(evaluated_df["path_length_error"].median()) if num_evaluated > 0 else 1.0
    )
    std_error = (
        float(evaluated_df["path_length_error"].std()) if num_evaluated > 0 else 0.0
    )

    mean_miss = (
        float(evaluated_df["miss_rate"].mean()) if num_evaluated > 0 else 1.0
    )
    median_miss = (
        float(evaluated_df["miss_rate"].median()) if num_evaluated > 0 else 1.0
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "test_topology_results.csv"
    results_df.to_csv(csv_path, index=False)

    summary = {
        "coordinate_system": "pixel-space (honest graph proxy; no fabricated GIS coordinates)",
        "dataset": "RouteResilience",
        "split": "test",
        "total_test_images_found": total_test_found,
        "images_processed": len(test_df),
        "images_evaluated": num_evaluated,
        "images_skipped": num_skipped,
        "skip_breakdown": skip_breakdown,
        "total_od_pairs_evaluated": total_od_pairs,
        "metrics": {
            "mean_topology_score": mean_score,
            "median_topology_score": median_score,
            "std_topology_score": std_score,
            "mean_path_length_error": mean_error,
            "median_path_length_error": median_error,
            "std_path_length_error": std_error,
            "mean_miss_rate": mean_miss,
            "median_miss_rate": median_miss,
        },
        "parameters": {
            "snap_radius_px": args.snap_radius,
            "n_pairs_per_image": args.n_pairs,
            "random_seed": args.seed,
            "min_edge_length_px": args.min_edge_length_px,
        },
    }

    json_path = OUTPUT_DIR / "topology_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("TOPOLOGY EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total test images found  : {total_test_found}")
    print(f"Total images processed    : {len(test_df)}")
    print(f"Successfully evaluated   : {num_evaluated} ({num_evaluated/len(test_df)*100:.1f}%)")
    print(f"Skipped images           : {num_skipped} ({num_skipped/len(test_df)*100:.1f}%)")
    if skip_breakdown:
        print("Skip breakdown:")
        for reason, count in skip_breakdown.items():
            print(f"  - {reason}: {count}")

    print("\n" + "-" * 70)
    print("SUMMARY METRICS")
    print("-" * 70)
    print(f"Total evaluated OD pairs : {total_od_pairs}")
    print(f"Topology Score (APLS)    : Mean = {mean_score:.4f} | Median = {median_score:.4f} | Std = {std_score:.4f}")
    print(f"Path Length Error        : Mean = {mean_error:.4f} | Median = {median_error:.4f} | Std = {std_error:.4f}")
    print(f"Miss Rate                : Mean = {mean_miss:.4f} | Median = {median_miss:.4f}")

    print("\nSaved artifacts:")
    print(f"  CSV  : {csv_path}")
    print(f"  JSON : {json_path}")


if __name__ == "__main__":
    main()

