"""
scripts/audit/loose_graph_and_compare.py
Read-only audit: rebuild adjacency graph with looser thresholds, compare old vs new
cross-split edge counts. Does NOT write any new data files.
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
IMAGE_DIR = str(ROOT_DIR / "RouteResilience" / "images_png")
MASK_DIR  = str(ROOT_DIR / "RouteResilience" / "masks_png")
SPLIT_CSV = str(ROOT_DIR / "RouteResilience" / "data_split.csv")
GROUP_CSV = str(ROOT_DIR / "RouteResilience" / "data_split_grouped.csv")

W = 23  # candidate grid width from previous audit

def load_edges(ids):
    """Load left/right/top/bottom pixel strips and mask borders for all tile IDs."""
    left_e, right_e, top_e, bot_e = {}, {}, {}, {}
    mask_l, mask_r = {}, {}
    for tid in ids:
        img  = cv2.imread(os.path.join(IMAGE_DIR, f"{tid}.png"))
        mask = cv2.imread(os.path.join(MASK_DIR,  f"{tid}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        img  = img.astype(np.float32)
        left_e[tid]  = img[:, 0, :]
        right_e[tid] = img[:, -1, :]
        top_e[tid]   = img[0, :, :]
        bot_e[tid]   = img[-1, :, :]
        mask_l[tid]  = (mask[:, 0]  == 255)
        mask_r[tid]  = (mask[:, -1] == 255)
    return left_e, right_e, top_e, bot_e, mask_l, mask_r


def build_graph(ids_set, left_e, right_e, top_e, bot_e, mask_l, mask_r,
                h_diff_thresh, h_jac_thresh, v_diff_thresh):
    """Build undirected adjacency dict and return edges list too."""
    adj   = defaultdict(set)
    edges = []          # (a, b, direction)
    ids_list = sorted(ids_set)
    for i in ids_list:
        # Horizontal: i <-> i+1
        j = i + 1
        if j in ids_set and i in right_e and j in left_e:
            diff_h = np.abs(right_e[i] - left_e[j]).mean()
            inter  = (mask_r[i] & mask_l[j]).sum()
            union  = (mask_r[i] | mask_l[j]).sum()
            jac    = inter / union if union > 0 else 0.0
            if diff_h < h_diff_thresh or jac > h_jac_thresh:
                adj[i].add(j); adj[j].add(i)
                edges.append((i, j, "H"))
        # Vertical: i <-> i+W
        j = i + W
        if j in ids_set and i in bot_e and j in top_e:
            diff_v = np.abs(bot_e[i] - top_e[j]).mean()
            if diff_v < v_diff_thresh:
                adj[i].add(j); adj[j].add(i)
                edges.append((i, j, "V"))
    return adj, edges


def connected_components(ids, adj):
    visited = set()
    comps   = []
    for tid in ids:
        if tid not in visited:
            comp = []
            q    = deque([tid])
            visited.add(tid)
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def cross_split_edges(edges, id_to_split):
    crossing = 0
    for a, b, _ in edges:
        if id_to_split.get(a) != id_to_split.get(b):
            crossing += 1
    return crossing


def report_components(comps, label=""):
    sizes     = [len(c) for c in comps]
    singletons = sum(1 for s in sizes if s == 1)
    total      = sum(sizes)
    print(f"\n--- COMPONENT SUMMARY ({label}) ---")
    print(f"  Total components : {len(comps)}")
    print(f"  Singletons       : {singletons} ({singletons/total*100:.2f}%)")
    print(f"  Largest component: {sizes[0]} tiles ({sizes[0]/total*100:.2f}%)")
    print(f"  Top 10 sizes     : {sizes[:10]}")


def main():
    print("Loading CSV and edge strips...", flush=True)
    df = pd.read_csv(SPLIT_CSV)
    df["id"] = df["filename"].str.replace(".png", "", regex=False).astype(int)
    existing_ids = set(df["id"].values)
    id_to_split  = dict(zip(df["id"], df["split"]))

    left_e, right_e, top_e, bot_e, mask_l, mask_r = load_edges(existing_ids)
    print(f"Edge strips loaded for {len(left_e)} tiles.", flush=True)

    # ------------------------------------------------------------------
    # TASK 1 – Rebuild graph with LOOSE thresholds
    # ------------------------------------------------------------------
    print("\n=== TASK 1: LOOSE GRAPH (diff<12, Jac>0.1) ===", flush=True)
    adj_loose, edges_loose = build_graph(
        existing_ids, left_e, right_e, top_e, bot_e, mask_l, mask_r,
        h_diff_thresh=12.0, h_jac_thresh=0.1, v_diff_thresh=12.0
    )
    h_edges = sum(1 for _, _, d in edges_loose if d == "H")
    v_edges = sum(1 for _, _, d in edges_loose if d == "V")
    print(f"  Edges: {h_edges} horizontal, {v_edges} vertical, {len(edges_loose)} total")
    comps_loose = connected_components(sorted(existing_ids), adj_loose)
    report_components(comps_loose, "LOOSE")

    # ------------------------------------------------------------------
    # TASK 4 – Old split cross-split edges (LOOSE graph, for fair comparison)
    # ------------------------------------------------------------------
    print("\n=== TASK 4: OLD SPLIT CROSS-SPLIT EDGES (LOOSE graph) ===", flush=True)
    crossing_old = cross_split_edges(edges_loose, id_to_split)
    total_edges  = len(edges_loose)
    print(f"  Edges crossing train/val/test in OLD split: {crossing_old} / {total_edges} "
          f"({crossing_old/total_edges*100:.2f}%)")

    # ------------------------------------------------------------------
    # Load grouped split if it exists (written by create_group_split.py)
    # ------------------------------------------------------------------
    if os.path.exists(GROUP_CSV):
        print("\n=== TASK 4 (cont): NEW GROUPED SPLIT CROSS-SPLIT EDGES (LOOSE graph) ===", flush=True)
        dg = pd.read_csv(GROUP_CSV)
        dg["id"] = dg["filename"].str.replace(".png", "", regex=False).astype(int)
        id_to_split_new = dict(zip(dg["id"], dg["split"]))
        crossing_new = cross_split_edges(edges_loose, id_to_split_new)
        print(f"  Edges crossing train/val/test in NEW (grouped) split: {crossing_new} / {total_edges} "
              f"({crossing_new/total_edges*100:.2f}%)")
    else:
        print("\nNOTE: data_split_grouped.csv not found; run create_group_split.py first.")

    print("\nDone.", flush=True)
    # Return data for use by create_group_split (dump components to stdout for reference)
    return comps_loose


if __name__ == "__main__":
    main()
