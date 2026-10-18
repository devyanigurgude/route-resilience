"""
RouteResilience/create_group_split.py
--------------------------------------
Assign connected components (built from the pixel-adjacency graph) to
train / validation / test splits, keeping entire components together so
that no two adjacent tiles land in different splits.

Split target: ~70% train, ~15% val, ~15% test (by tile count).
Tiles within the same component always go to the same split.

Algorithm
---------
1. Build a pixel-adjacency graph (horizontal i->i+1 and vertical i->i+W,
   with loose thresholds: RGB diff < 12 OR road Jaccard > 0.1 for horizontal,
   RGB diff < 12 for vertical).
2. Find connected components and sort by size (largest first) so that the
   greedy allocator has the most constrained pieces to place first.
3. Greedily assign each component to the split that is furthest from its
   target tile count, with a secondary sort on road-coverage imbalance.
4. Save as data_split_grouped.csv (does NOT overwrite data_split.csv).

Outputs
-------
RouteResilience/data_split_grouped.csv  columns: filename, road_percentage,
                                                  split, group_id
"""
import os
import sys
import random
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

# ── paths ────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent          # RouteResilience/
IMAGE_DIR = ROOT_DIR / "images_png"
MASK_DIR  = ROOT_DIR / "masks_png"
IN_CSV    = ROOT_DIR / "data_split.csv"          # read only
OUT_CSV   = ROOT_DIR / "data_split_grouped.csv"  # new file

W        = 23        # candidate grid width (from Phase 2 audit)
SEED     = 42
TARGETS  = {"train": 0.70, "validation": 0.15, "test": 0.15}

# Loose thresholds (same as loose_graph_and_compare.py)
H_DIFF   = 12.0
H_JAC    = 0.10
V_DIFF   = 12.0


# ── helpers ──────────────────────────────────────────────────────────────────
def load_edges(ids):
    left_e, right_e, top_e, bot_e, mask_l, mask_r = {}, {}, {}, {}, {}, {}
    for tid in ids:
        img  = cv2.imread(str(IMAGE_DIR / f"{tid}.png"))
        mask = cv2.imread(str(MASK_DIR  / f"{tid}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        img = img.astype(np.float32)
        right_e[tid] = img[:, -1, :]
        left_e[tid]  = img[:, 0, :]
        bot_e[tid]   = img[-1, :, :]
        top_e[tid]   = img[0, :, :]
        mask_r[tid]  = (mask[:, -1] == 255)
        mask_l[tid]  = (mask[:, 0]  == 255)
    return left_e, right_e, top_e, bot_e, mask_l, mask_r


def build_graph(ids_set, left_e, right_e, top_e, bot_e, mask_l, mask_r):
    adj = defaultdict(set)
    for i in sorted(ids_set):
        j = i + 1
        if j in ids_set and i in right_e and j in left_e:
            diff = np.abs(right_e[i] - left_e[j]).mean()
            inter = (mask_r[i] & mask_l[j]).sum()
            union = (mask_r[i] | mask_l[j]).sum()
            jac = inter / union if union > 0 else 0.0
            if diff < H_DIFF or jac > H_JAC:
                adj[i].add(j); adj[j].add(i)
        j = i + W
        if j in ids_set and i in bot_e and j in top_e:
            if np.abs(bot_e[i] - top_e[j]).mean() < V_DIFF:
                adj[i].add(j); adj[j].add(i)
    return adj


def connected_components(ids, adj):
    visited, comps = set(), []
    for tid in sorted(ids):
        if tid not in visited:
            comp, q = [], deque([tid])
            visited.add(tid)
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            comps.append(comp)
    return comps


def greedy_assign(comps, id_to_road, n_total, targets, seed):
    """
    Greedy: iterate components largest-first. Assign to whichever split is
    most under-quota (by tile count). Break ties by road-coverage balance.
    """
    rng   = random.Random(seed)
    sizes = {"train": 0, "validation": 0, "test": 0}
    roads = {"train": 0.0, "validation": 0.0, "test": 0.0}
    counts = {"train": 0, "validation": 0, "test": 0}  # component counts

    comp_assignment = {}   # comp_idx -> split
    split_names     = list(targets.keys())

    sorted_comps = sorted(enumerate(comps), key=lambda x: -len(x[1]))
    # Add tiny random tiebreak so determinism is seeded
    sorted_comps = [(idx, c, rng.random()) for idx, c in sorted_comps]
    sorted_comps.sort(key=lambda x: (-len(x[1]), x[2]))

    for idx, comp, _ in sorted_comps:
        # Score each split: how far below its target tile count is it?
        # Pick the split most hungry for tiles.
        def hunger(s):
            want = targets[s] * n_total
            have = sizes[s]
            return want - have

        best = max(split_names, key=lambda s: (hunger(s), -roads[s] / max(sizes[s], 1)))
        comp_assignment[idx] = best
        comp_size  = len(comp)
        comp_road  = sum(id_to_road[t] for t in comp)
        sizes[best] += comp_size
        roads[best] += comp_road
        counts[best] += 1

    return comp_assignment


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Reading data_split.csv ...", flush=True)
    df = pd.read_csv(str(IN_CSV))
    df["id"] = df["filename"].str.replace(".png", "", regex=False).astype(int)
    id_to_road = dict(zip(df["id"], df["road_percentage"]))
    existing   = set(df["id"].values)
    n_total    = len(existing)

    print(f"  {n_total} tiles loaded.", flush=True)
    print("Loading pixel edge strips ...", flush=True)
    left_e, right_e, top_e, bot_e, mask_l, mask_r = load_edges(existing)
    print(f"  Done ({len(left_e)} tiles have valid images).", flush=True)

    print("Building adjacency graph ...", flush=True)
    adj = build_graph(existing, left_e, right_e, top_e, bot_e, mask_l, mask_r)
    print("Finding connected components ...", flush=True)
    comps = connected_components(existing, adj)
    sizes = sorted([len(c) for c in comps], reverse=True)
    singletons = sum(1 for s in sizes if s == 1)
    print(f"  {len(comps)} components | singletons: {singletons} | largest: {sizes[0]}", flush=True)
    print(f"  Top 10 sizes: {sizes[:10]}", flush=True)

    print("Greedy component assignment to splits ...", flush=True)
    comp_assignment = greedy_assign(comps, id_to_road, n_total, TARGETS, SEED)

    # Build output dataframe
    rows = []
    for comp_idx, comp in enumerate(comps):
        split = comp_assignment[comp_idx]
        for tid in comp:
            rows.append({
                "filename":       f"{tid}.png",
                "road_percentage": id_to_road[tid],
                "split":           split,
                "group_id":        comp_idx,
            })

    out_df = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    out_df.to_csv(str(OUT_CSV), index=False)
    print(f"\nSaved: {OUT_CSV}", flush=True)

    # ── Verification ─────────────────────────────────────────────────────────
    print("\n=== VERIFICATION ===", flush=True)
    for s in ["train", "validation", "test"]:
        sub = out_df[out_df["split"] == s]["road_percentage"]
        print(f"\nSplit: {s}")
        print(f"  Count   : {len(sub)}")
        print(f"  Mean    : {sub.mean():.4f}%")
        print(f"  Median  : {sub.median():.4f}%")
        print(f"  Q1 / Q3 : {sub.quantile(0.25):.4f}% / {sub.quantile(0.75):.4f}%")
        print(f"  Zero-road: {(sub == 0).sum()} ({(sub == 0).mean()*100:.2f}%)")

    # Cross-split edges must be 0
    id_to_new_split = dict(zip(out_df["filename"].str.replace(".png","",regex=False).astype(int),
                               out_df["split"]))
    crossing = 0
    for i in sorted(existing):
        for j in adj[i]:
            if j > i and id_to_new_split.get(i) != id_to_new_split.get(j):
                crossing += 1
    print(f"\nAdjacency edges crossing split boundaries (MUST BE 0): {crossing}")
    if crossing == 0:
        print("  PASS — no edge crosses a split boundary.")
    else:
        print("  FAIL — component assignment has cross-split leakage!")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
