import os
import cv2
import numpy as np
import pandas as pd

IMAGE_DIR = "RouteResilience/images_png"
MASK_DIR = "RouteResilience/masks_png"
SPLIT_CSV = "RouteResilience/data_split.csv"

def run_adjacency_test():
    print("Loading data_split.csv...")
    df = pd.read_csv(SPLIT_CSV)
    df["id"] = df["filename"].str.replace(".png", "", regex=False).astype(int)

    existing_ids = set(df["id"].values)
    all_sorted_ids = sorted(list(existing_ids))

    # 1. Test consecutive pairs: (i, i+1) where both exist
    consecutive_pairs = [(i, i + 1) for i in all_sorted_ids if (i + 1) in existing_ids]
    print(f"Total consecutive pairs (i, i+1): {len(consecutive_pairs)}")

    # Sample up to 1000 pairs for speed if large, or evaluate all
    eval_pairs = consecutive_pairs[:1000] if len(consecutive_pairs) > 1000 else consecutive_pairs
    print(f"Evaluating {len(eval_pairs)} consecutive pairs...")

    adj_horizontal_diffs = []
    adj_vertical_diffs = []
    adj_min_border_diffs = []
    adj_road_match_horiz = []
    adj_road_match_vert = []

    for id_a, id_b in eval_pairs:
        path_a = os.path.join(IMAGE_DIR, f"{id_a}.png")
        path_b = os.path.join(IMAGE_DIR, f"{id_b}.png")
        mask_path_a = os.path.join(MASK_DIR, f"{id_a}.png")
        mask_path_b = os.path.join(MASK_DIR, f"{id_b}.png")

        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)

        # Horizontal boundary: Right edge of A vs Left edge of B
        diff_h = np.abs(img_a[:, -1, :].astype(float) - img_b[:, 0, :].astype(float)).mean()
        # Vertical boundary: Bottom edge of A vs Top edge of B
        diff_v = np.abs(img_a[-1, :, :].astype(float) - img_b[0, :, :].astype(float)).mean()

        adj_horizontal_diffs.append(diff_h)
        adj_vertical_diffs.append(diff_v)
        adj_min_border_diffs.append(min(diff_h, diff_v))

        # Check road masks
        mask_a = cv2.imread(mask_path_a, cv2.IMREAD_GRAYSCALE) == 255
        mask_b = cv2.imread(mask_path_b, cv2.IMREAD_GRAYSCALE) == 255

        # Road match on borders: how many positions both have road vs either has road
        road_h_intersect = (mask_a[:, -1] & mask_b[:, 0]).sum()
        road_h_union = (mask_a[:, -1] | mask_b[:, 0]).sum()
        road_v_intersect = (mask_a[-1, :] & mask_b[0, :]).sum()
        road_v_union = (mask_a[-1, :] | mask_b[0, :]).sum()

        adj_road_match_horiz.append(road_h_intersect / road_h_union if road_h_union > 0 else np.nan)
        adj_road_match_vert.append(road_v_intersect / road_v_union if road_v_union > 0 else np.nan)

    # 2. Control group: 1000 randomly chosen non-adjacent pairs
    rng = np.random.RandomState(42)
    rand_pairs = []
    while len(rand_pairs) < len(eval_pairs):
        idx_a = rng.choice(all_sorted_ids)
        idx_b = rng.choice(all_sorted_ids)
        if abs(idx_a - idx_b) > 10:
            rand_pairs.append((idx_a, idx_b))

    rand_horizontal_diffs = []
    rand_vertical_diffs = []
    rand_min_border_diffs = []
    rand_road_match_horiz = []
    rand_road_match_vert = []

    for id_a, id_b in rand_pairs:
        path_a = os.path.join(IMAGE_DIR, f"{id_a}.png")
        path_b = os.path.join(IMAGE_DIR, f"{id_b}.png")
        mask_path_a = os.path.join(MASK_DIR, f"{id_a}.png")
        mask_path_b = os.path.join(MASK_DIR, f"{id_b}.png")

        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)

        diff_h = np.abs(img_a[:, -1, :].astype(float) - img_b[:, 0, :].astype(float)).mean()
        diff_v = np.abs(img_a[-1, :, :].astype(float) - img_b[0, :, :].astype(float)).mean()

        rand_horizontal_diffs.append(diff_h)
        rand_vertical_diffs.append(diff_v)
        rand_min_border_diffs.append(min(diff_h, diff_v))

        mask_a = cv2.imread(mask_path_a, cv2.IMREAD_GRAYSCALE) == 255
        mask_b = cv2.imread(mask_path_b, cv2.IMREAD_GRAYSCALE) == 255

        road_h_intersect = (mask_a[:, -1] & mask_b[:, 0]).sum()
        road_h_union = (mask_a[:, -1] | mask_b[:, 0]).sum()
        road_v_intersect = (mask_a[-1, :] & mask_b[0, :]).sum()
        road_v_union = (mask_a[-1, :] | mask_b[0, :]).sum()

        rand_road_match_horiz.append(road_h_intersect / road_h_union if road_h_union > 0 else np.nan)
        rand_road_match_vert.append(road_v_intersect / road_v_union if road_v_union > 0 else np.nan)

    print("\n--- RESULTS ---")
    print(f"Adjacent Pairs (i, i+1) - Right vs Left Edge Difference: Mean = {np.mean(adj_horizontal_diffs):.2f}, Median = {np.median(adj_horizontal_diffs):.2f}")
    print(f"Adjacent Pairs (i, i+1) - Bottom vs Top Edge Difference:  Mean = {np.mean(adj_vertical_diffs):.2f}, Median = {np.median(adj_vertical_diffs):.2f}")
    print(f"Adjacent Pairs (i, i+1) - Min(H, V) Difference:           Mean = {np.mean(adj_min_border_diffs):.2f}, Median = {np.median(adj_min_border_diffs):.2f}")

    print(f"Random Pairs (Control) - Right vs Left Edge Difference:   Mean = {np.mean(rand_horizontal_diffs):.2f}, Median = {np.median(rand_horizontal_diffs):.2f}")
    print(f"Random Pairs (Control) - Bottom vs Top Edge Difference:    Mean = {np.mean(rand_vertical_diffs):.2f}, Median = {np.median(rand_vertical_diffs):.2f}")
    print(f"Random Pairs (Control) - Min(H, V) Difference:             Mean = {np.mean(rand_min_border_diffs):.2f}, Median = {np.median(rand_min_border_diffs):.2f}")

    valid_adj_road_h = [x for x in adj_road_match_horiz if not np.isnan(x)]
    valid_rand_road_h = [x for x in rand_road_match_horiz if not np.isnan(x)]
    valid_adj_road_v = [x for x in adj_road_match_vert if not np.isnan(x)]
    valid_rand_road_v = [x for x in rand_road_match_vert if not np.isnan(x)]

    print(f"Adjacent Road Border Alignment Jaccard (Horiz): Mean = {np.mean(valid_adj_road_h):.4f} (n={len(valid_adj_road_h)})")
    print(f"Random Road Border Alignment Jaccard (Horiz):   Mean = {np.mean(valid_rand_road_h):.4f} (n={len(valid_rand_road_h)})")
    print(f"Adjacent Road Border Alignment Jaccard (Vert):  Mean = {np.mean(valid_adj_road_v):.4f} (n={len(valid_adj_road_v)})")
    print(f"Random Road Border Alignment Jaccard (Vert):    Mean = {np.mean(valid_rand_road_v):.4f} (n={len(valid_rand_road_v)})")

if __name__ == "__main__":
    run_adjacency_test()
