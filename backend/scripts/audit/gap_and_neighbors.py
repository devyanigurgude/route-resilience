import os
import pandas as pd
import numpy as np

def analyze_gaps_and_neighbors():
    df = pd.read_csv("RouteResilience/data_split.csv")
    df["id"] = df["filename"].str.replace(".png", "", regex=False).astype(int)
    existing_ids = set(df["id"].values)

    # 1. Gap Structure in 1..10634
    all_possible = set(range(1, 10635))
    missing_ids = sorted(list(all_possible - existing_ids))
    print(f"Total range: 1..10634 (10,634 potential tiles)")
    print(f"Existing tiles: {len(existing_ids)}")
    print(f"Missing tiles: {len(missing_ids)} ({len(missing_ids)/10634*100:.2f}%)")

    # Identify contiguous runs of missing IDs
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

    print(f"\nTotal contiguous missing runs: {len(runs)}")
    runs_df = pd.DataFrame(runs, columns=["start_id", "end_id", "length"])
    print("Missing run length summary:")
    print(runs_df["length"].describe())
    print("\nTop 15 longest missing runs:")
    print(runs_df.sort_values("length", ascending=False).head(15).to_string(index=False))

    print("\nFrequency of run lengths:")
    print(runs_df["length"].value_counts().head(10).to_string())

    # 2. Cross-Split Neighbor Count (+/- 5 indices)
    train_ids = set(df[df["split"] == "train"]["id"].values)
    val_ids = set(df[df["split"] == "validation"]["id"].values)
    test_ids = set(df[df["split"] == "test"]["id"].values)

    def count_neighbors(source_ids, neighbor_distance=5):
        has_train_neighbor = 0
        total = len(source_ids)
        for s_id in source_ids:
            # check if any id in [s_id - distance, s_id + distance] (excluding s_id) is in train
            neighbors = [s_id + d for d in range(-neighbor_distance, neighbor_distance + 1) if d != 0]
            if any(n in train_ids for n in neighbors):
                has_train_neighbor += 1
        return has_train_neighbor, total

    val_has_train_5, val_tot = count_neighbors(val_ids, neighbor_distance=5)
    test_has_train_5, test_tot = count_neighbors(test_ids, neighbor_distance=5)

    val_has_train_1, _ = count_neighbors(val_ids, neighbor_distance=1)
    test_has_train_1, _ = count_neighbors(test_ids, neighbor_distance=1)

    print("\n--- CROSS-SPLIT NEIGHBOR COUNT IN CURRENT (RANDOM STRATIFIED) SPLIT ---")
    print(f"Validation tiles with a Train neighbor within +/- 1 index: {val_has_train_1} / {val_tot} ({val_has_train_1/val_tot*100:.2f}%)")
    print(f"Test tiles with a Train neighbor within +/- 1 index:       {test_has_train_1} / {test_tot} ({test_has_train_1/test_tot*100:.2f}%)")
    print(f"Validation tiles with a Train neighbor within +/- 5 indices: {val_has_train_5} / {val_tot} ({val_has_train_5/val_tot*100:.2f}%)")
    print(f"Test tiles with a Train neighbor within +/- 5 indices:       {test_has_train_5} / {test_tot} ({test_has_train_5/test_tot*100:.2f}%)")

    # 3. What would it be under a pure contiguous block split (70% train, 15% val, 15% test)?
    # In a contiguous block split along sorted existing IDs:
    # Train: 0 .. 3942, Val: 3943 .. 4787, Test: 4788 .. 5633
    # Only the boundary tiles (at most 5 tiles on each side of the partition) could have a neighbor in another split!
    print("\n--- THEORETICAL CONTIGUOUS BLOCK SPLIT COMPARISON ---")
    print("In a contiguous block split (e.g. Train=[1..N_train], Val=[N_train+1..N_val], Test=[N_val+1..N_total]):")
    print(f"Validation tiles with a Train neighbor within +/- 5 indices: at most 5 tiles ({(5/val_tot)*100:.2f}%)")
    print(f"Test tiles with a Train neighbor within +/- 5 indices: 0 tiles (separated by Validation block) or at most 5 tiles")

if __name__ == "__main__":
    analyze_gaps_and_neighbors()
