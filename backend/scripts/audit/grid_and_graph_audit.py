import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
IMAGE_DIR = str(ROOT_DIR / "RouteResilience" / "images_png")
MASK_DIR = str(ROOT_DIR / "RouteResilience" / "masks_png")
SPLIT_CSV = str(ROOT_DIR / "RouteResilience" / "data_split.csv")

def main():
    print("Loading data_split.csv...", flush=True)
    df = pd.read_csv(SPLIT_CSV)
    df["id"] = df["filename"].str.replace(".png", "", regex=False).astype(int)

    existing_ids = set(df["id"].values)
    all_sorted_ids = sorted(list(existing_ids))
    id_to_split = dict(zip(df["id"], df["split"]))

    # -------------------------------------------------------------
    # TASK 1: Find grid width W (offsets k in 2..300)
    # Pre-extract top/bottom borders of images to make search fast!
    # -------------------------------------------------------------
    print("Pre-loading edge strips for grid width search...", flush=True)
    # To keep memory light and speed high:
    # Only need top row and bottom row for bottom-vs-top edge difference
    # img[-1, :, :] (bottom) and img[0, :, :] (top)

    bottom_edges = {}
    top_edges = {}

    # Let's cache top and bottom rows as float32 (256, 3)
    for tid in all_sorted_ids:
        p = os.path.join(IMAGE_DIR, f"{tid}.png")
        img = cv2.imread(p)
        if img is not None:
            bottom_edges[tid] = img[-1, :, :].astype(np.float32)
            top_edges[tid] = img[0, :, :].astype(np.float32)

    print(f"Loaded edge rows for {len(bottom_edges)} tiles.", flush=True)

    rng = np.random.RandomState(42)
    # Compute control mean: random pairs with distance > 350
    rand_diffs = []
    sampled_control_ids = rng.choice(all_sorted_ids, size=min(1000, len(all_sorted_ids)), replace=False)
    for tid in sampled_control_ids:
        other_candidates = [x for x in all_sorted_ids if abs(x - tid) > 350]
        if other_candidates:
            oid = rng.choice(other_candidates)
            diff = np.abs(bottom_edges[tid] - top_edges[oid]).mean()
            rand_diffs.append(diff)
    control_mean = float(np.mean(rand_diffs))
    print(f"Control Random Pair Vertical Difference Mean: {control_mean:.2f}", flush=True)

    print("Testing offsets k in 2..300 (sampling up to 500 valid pairs per k)...", flush=True)
    offset_results = []

    for k in range(2, 301):
        # find all valid pairs (i, i+k) where both exist
        valid_pairs = [i for i in all_sorted_ids if (i + k) in bottom_edges]
        if len(valid_pairs) == 0:
            continue

        if len(valid_pairs) > 500:
            chosen = rng.choice(valid_pairs, size=500, replace=False)
        else:
            chosen = valid_pairs

        diffs = [np.abs(bottom_edges[i] - top_edges[i + k]).mean() for i in chosen]
        mean_diff = float(np.mean(diffs))
        offset_results.append((k, mean_diff, len(chosen)))

    offset_results.sort(key=lambda x: x[1])

    print("\n--- TOP 10 OFFSETS WITH LOWEST VERTICAL DIFFERENCE ---", flush=True)
    for rank, (k, mdiff, n_pairs) in enumerate(offset_results[:10], 1):
        print(f"Rank {rank}: Offset k = {k:3d} | Mean Bottom-vs-Top Diff = {mdiff:.2f} (sampled {n_pairs} pairs)", flush=True)
    print(f"Control Mean: {control_mean:.2f}", flush=True)

    best_W = offset_results[0][0]
    print(f"\nIdentified Candidate Grid Width W = {best_W}", flush=True)

    # -------------------------------------------------------------
    # TASK 2: Build Adjacency Graph
    # Horizontal: i to i+1 if right/left diff < 8 (or road Jaccard > 0.2)
    # Vertical: i to i+W if bottom/top diff < 8
    # -------------------------------------------------------------
    print(f"\nBuilding adjacency graph using W = {best_W}...", flush=True)

    # Pre-load left/right edges and masks for horizontal connections
    right_edges = {}
    left_edges = {}
    mask_right = {}
    mask_left = {}

    for tid in all_sorted_ids:
        img_p = os.path.join(IMAGE_DIR, f"{tid}.png")
        mask_p = os.path.join(MASK_DIR, f"{tid}.png")
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if img is not None and mask is not None:
            right_edges[tid] = img[:, -1, :].astype(np.float32)
            left_edges[tid] = img[:, 0, :].astype(np.float32)
            mask_right[tid] = (mask[:, -1] == 255)
            mask_left[tid] = (mask[:, 0] == 255)

    adj = {tid: set() for tid in all_sorted_ids}

    horiz_edges_count = 0
    vert_edges_count = 0

    for i in all_sorted_ids:
        # Check horizontal i -> i+1
        if (i + 1) in right_edges:
            diff_h = np.abs(right_edges[i] - left_edges[i + 1]).mean()

            mr = mask_right[i]
            ml = mask_left[i + 1]
            inter = (mr & ml).sum()
            union = (mr | ml).sum()
            jaccard = (inter / union) if union > 0 else 0.0

            if diff_h < 8.0 or jaccard > 0.2:
                adj[i].add(i + 1)
                adj[i + 1].add(i)
                horiz_edges_count += 1

        # Check vertical i -> i+W
        if (i + best_W) in bottom_edges:
            diff_v = np.abs(bottom_edges[i] - top_edges[i + best_W]).mean()
            if diff_v < 8.0:
                adj[i].add(i + best_W)
                adj[i + best_W].add(i)
                vert_edges_count += 1

    print(f"Graph constructed: {len(all_sorted_ids)} nodes, {horiz_edges_count} horizontal edges, {vert_edges_count} vertical edges.", flush=True)

    # Connected Components (BFS)
    visited = set()
    components = []

    for tid in all_sorted_ids:
        if tid not in visited:
            comp = []
            queue = [tid]
            visited.add(tid)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)

    components.sort(key=len, reverse=True)
    comp_sizes = [len(c) for c in components]
    singletons = sum(1 for s in comp_sizes if s == 1)

    print("\n--- CONNECTED COMPONENTS SUMMARY ---", flush=True)
    print(f"Total Connected Components: {len(components)}")
    print(f"Top 10 Largest Component Sizes: {comp_sizes[:10]}")
    print(f"Number of Singletons (isolated tiles): {singletons} ({singletons / len(all_sorted_ids) * 100:.2f}%)")
    print(f"Largest Component captures: {comp_sizes[0]} tiles ({comp_sizes[0] / len(all_sorted_ids) * 100:.2f}%)")

    # -------------------------------------------------------------
    # TASK 3: Longest 10 runs of missing indices in 1..10634
    # -------------------------------------------------------------
    all_possible = set(range(1, 10635))
    missing_ids = sorted(list(all_possible - existing_ids))

    runs = []
    if missing_ids:
        cur_start = missing_ids[0]
        cur_prev = missing_ids[0]
        for m in missing_ids[1:]:
            if m == cur_prev + 1:
                cur_prev = m
            else:
                runs.append((cur_start, cur_prev, cur_prev - cur_start + 1))
                cur_start = m
                cur_prev = m
        runs.append((cur_start, cur_prev, cur_prev - cur_start + 1))

    runs_df = pd.DataFrame(runs, columns=["start_id", "end_id", "length"]).sort_values("length", ascending=False)

    print("\n--- TOP 10 LONGEST RUNS OF MISSING INDICES (1..10634) ---", flush=True)
    top10_runs = runs_df.head(10).reset_index(drop=True)
    for idx, row in top10_runs.iterrows():
        print(f"Run {idx+1:2d}: Start ID {row['start_id']:5d} -> End ID {row['end_id']:5d} | Missing Length = {row['length']:2d} tiles", flush=True)

    # -------------------------------------------------------------
    # TASK 4: Count Val/Test tiles with at least one graph neighbor in Train
    # -------------------------------------------------------------
    train_ids = set(df[df["split"] == "train"]["id"].values)
    val_ids = set(df[df["split"] == "validation"]["id"].values)
    test_ids = set(df[df["split"] == "test"]["id"].values)

    val_with_train_neighbor = 0
    for vid in val_ids:
        if any(nb in train_ids for nb in adj[vid]):
            val_with_train_neighbor += 1

    test_with_train_neighbor = 0
    for tid in test_ids:
        if any(nb in train_ids for nb in adj[tid]):
            test_with_train_neighbor += 1

    print("\n--- GRAPH-NEIGHBOR LEAKAGE AUDIT ---", flush=True)
    print(f"Validation tiles with >= 1 adjacency neighbor in Train: {val_with_train_neighbor} / {len(val_ids)} ({val_with_train_neighbor / len(val_ids) * 100:.2f}%)")
    print(f"Test tiles with >= 1 adjacency neighbor in Train:       {test_with_train_neighbor} / {len(test_ids)} ({test_with_train_neighbor / len(test_ids) * 100:.2f}%)")

if __name__ == "__main__":
    main()
