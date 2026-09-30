"""
EXP-2.1 Corridor Target Module
================================
Centerline vs Dilated Corridor Target experiment.

This module provides:
  1. DilatedCorridorDataset  — wraps RoadDataset and dynamically expands the
     1-2 px ground-truth centerline mask to a ~5 px corridor *during training
     only*. The original masks on disk are NEVER modified.
  2. skeletonize_prediction  — post-inference step for corridor models: thin
     the predicted corridor back to a 1-px skeleton before graph evaluation,
     so both models are compared in the same representation.
  3. TopologyEvaluator       — per-image connectivity + APLS + OD-reachability
     metrics, used identically for model A and model B.
  4. VisualComparison        — saves the RGB → GT → prediction → skeleton → graph
     panel required by the experiment spec.

Design constraints (from experiment spec):
  - Original masks on disk: NEVER touched.
  - Dilation: computed on-the-fly inside __getitem__, discarded after the batch.
  - Val / test loaders always use the original 1-2 px centerline target so that
    IoU / Dice are directly comparable between A and B.
  - Corridor model's inference path skeletonizes the predicted mask before
    graph extraction (same pipeline as model A, just with that extra step).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.measure import label as sk_label
from skimage.morphology import disk, skeletonize
from torch.utils.data import Dataset

from route_resilience.data import RoadDataset, build_transforms, preprocess_rgb_image
from route_resilience.graph_extraction import (
    connectivity_ratio,
    mask_to_skeleton,
    skeleton_to_graph,
)
from route_resilience.evaluate_topology import simplified_apls


# ---------------------------------------------------------------------------
# 1. Dilated Corridor Dataset
# ---------------------------------------------------------------------------

class DilatedCorridorDataset(Dataset):
    """
    Training-time wrapper over RoadDataset.

    The original mask (1–2 px centerline) is loaded from disk exactly as
    RoadDataset does. The augmentation pipeline is applied first (so geometric
    transforms are consistent). Then the binary mask is dilated with a disk
    of radius `dilation_radius` pixels to create the ~5 px corridor target
    that is passed to the loss function.

    Val / test: Use the plain RoadDataset (no dilation) so evaluation targets
    are identical to model-A baselines and results are directly comparable.

    Parameters
    ----------
    base_dataset : RoadDataset
        Already-constructed dataset (train split, with augmentation).
    dilation_radius : int
        Morphological dilation radius in pixels. Default 2 gives a ~5 px
        corridor (1-2 px center ± 2 px on each side).
    """

    def __init__(self, base_dataset: RoadDataset, dilation_radius: int = 2):
        self.base = base_dataset
        self.selem = disk(dilation_radius)  # structuring element, precomputed once

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.base[idx]  # mask: (1, H, W) float32 {0, 1}

        # Dilate on the CPU numpy side — mask is float32 (0/1)
        mask_np = mask.squeeze(0).numpy().astype(bool)  # (H, W) bool
        dilated = binary_dilation(mask_np, structure=self.selem)  # (H, W) bool
        dilated_mask = torch.tensor(dilated.astype(np.float32)).unsqueeze(0)

        return image, dilated_mask


# ---------------------------------------------------------------------------
# 2. Skeletonize prediction (corridor inference post-processing)
# ---------------------------------------------------------------------------

def skeletonize_prediction(pred_mask: np.ndarray) -> np.ndarray:
    """
    Thin a predicted binary corridor mask back to a 1-px skeleton.

    Parameters
    ----------
    pred_mask : np.ndarray
        2-D binary mask (bool or uint8), road=1.

    Returns
    -------
    np.ndarray
        Skeletonized binary mask (bool).
    """
    return skeletonize(pred_mask.astype(bool))


# ---------------------------------------------------------------------------
# 3. Topology Evaluator
# ---------------------------------------------------------------------------

class TopologyEvaluator:
    """
    Per-image topology metrics for EXP-2.1.

    Metrics computed:
      - n_components       : connected components in the predicted skeleton
      - largest_cc_ratio   : fraction of skeleton pixels in the largest CC
      - apls_score         : simplified APLS (path-length similarity)
      - miss_rate          : fraction of GT OD pairs with no matched route
      - od_reachability    : fraction of sampled OD pairs that are connected
                             in the predicted graph (regardless of path length)

    Parameters
    ----------
    snap_radius : float  pixel radius for GT→pred node matching
    n_pairs     : int    OD pairs sampled per image
    seed        : int    RNG seed for reproducibility
    min_edge_px : float  spur pruning threshold
    """

    def __init__(
        self,
        snap_radius: float = 10.0,
        n_pairs: int = 200,
        seed: int = 42,
        min_edge_px: float = 3.0,
    ):
        self.snap_radius = snap_radius
        self.n_pairs = n_pairs
        self.seed = seed
        self.min_edge_px = min_edge_px

    # ------------------------------------------------------------------
    def evaluate(
        self,
        pred_binary: np.ndarray,
        gt_binary: np.ndarray,
        prob_map: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Parameters
        ----------
        pred_binary : (H, W) uint8/bool  — binary prediction (already
                      skeletonized if corridor model)
        gt_binary   : (H, W) uint8/bool  — original GT centerline mask
        prob_map    : (H, W) float32     — raw sigmoid probability map
                      (for confidence-gated graph; None → confidence=1.0)

        Returns
        -------
        dict with all topology metrics.
        """
        result: dict = {}

        # ---- Predicted graph ------------------------------------------
        pred_skel = mask_to_skeleton(pred_binary)
        result["pred_road_px"] = int(pred_binary.sum())
        result["pred_skel_px"] = int(pred_skel.sum())

        if pred_skel.sum() == 0:
            return self._empty_result()

        result["n_components"] = self._count_components(pred_skel)
        result["largest_cc_ratio"] = float(connectivity_ratio(pred_skel))

        G_pred = skeleton_to_graph(pred_skel, prob_map=prob_map, min_edge_length_px=self.min_edge_px)
        result["pred_nodes"] = G_pred.number_of_nodes()
        result["pred_edges"] = G_pred.number_of_edges()

        # ---- GT graph -------------------------------------------------
        gt_skel = mask_to_skeleton(gt_binary)
        if gt_skel.sum() == 0 or G_pred.number_of_nodes() < 2:
            result.update({
                "apls_score": 0.0,
                "miss_rate": 1.0,
                "od_reachability": 0.0,
                "n_pairs_evaluated": 0,
            })
            return result

        G_gt = skeleton_to_graph(gt_skel, prob_map=None, min_edge_length_px=self.min_edge_px)
        result["gt_nodes"] = G_gt.number_of_nodes()
        result["gt_edges"] = G_gt.number_of_edges()

        if G_gt.number_of_nodes() < 2:
            result.update({
                "apls_score": 0.0,
                "miss_rate": 1.0,
                "od_reachability": 0.0,
                "n_pairs_evaluated": 0,
            })
            return result

        apls = simplified_apls(
            G_pred=G_pred,
            G_gt=G_gt,
            snap_radius=self.snap_radius,
            n_pairs=self.n_pairs,
            seed=self.seed,
        )
        result["apls_score"] = float(apls["score"])
        result["miss_rate"] = float(apls["miss_rate"])
        result["n_pairs_evaluated"] = int(apls["n_pairs_evaluated"])

        # OD reachability: fraction of GT-sampled pairs reachable in G_pred
        result["od_reachability"] = self._od_reachability(G_gt, G_pred)

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _count_components(skeleton: np.ndarray) -> int:
        labeled = sk_label(skeleton, connectivity=2)
        sizes = np.bincount(labeled.ravel())[1:]
        return int(len(sizes))

    # ------------------------------------------------------------------
    def _od_reachability(self, G_gt: nx.Graph, G_pred: nx.Graph) -> float:
        """
        Fraction of sampled GT origin-destination pairs that are reachable
        (i.e. connected by ANY path) in G_pred, after snapping GT nodes to
        their nearest pred node within snap_radius.
        """
        rng = np.random.default_rng(self.seed)
        gt_nodes = list(G_gt.nodes())
        if len(gt_nodes) < 2:
            return 0.0

        pred_positions = {
            n: np.array(G_pred.nodes[n]["pos"]) for n in G_pred.nodes()
        }

        def nearest(gt_pos):
            if not pred_positions:
                return None
            dists = {n: np.linalg.norm(np.array(gt_pos) - p) for n, p in pred_positions.items()}
            best = min(dists, key=dists.get)
            return best if dists[best] <= self.snap_radius else None

        k = min(self.n_pairs, len(gt_nodes) ** 2)
        pairs = rng.choice(len(gt_nodes), size=(k, 2), replace=True)
        reachable = total = 0
        for i, j in pairs:
            if i == j:
                continue
            total += 1
            pa = nearest(G_gt.nodes[gt_nodes[i]]["pos"])
            pb = nearest(G_gt.nodes[gt_nodes[j]]["pos"])
            if pa is None or pb is None:
                continue
            if nx.has_path(G_pred, pa, pb):
                reachable += 1

        return float(reachable / total) if total > 0 else 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _empty_result() -> dict:
        return {
            "pred_road_px": 0,
            "pred_skel_px": 0,
            "n_components": 0,
            "largest_cc_ratio": 0.0,
            "pred_nodes": 0,
            "pred_edges": 0,
            "gt_nodes": 0,
            "gt_edges": 0,
            "apls_score": 0.0,
            "miss_rate": 1.0,
            "od_reachability": 0.0,
            "n_pairs_evaluated": 0,
        }


# ---------------------------------------------------------------------------
# 4. Visual comparison saver
# ---------------------------------------------------------------------------

def save_comparison_panel(
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    pred_skeleton: np.ndarray,
    G_pred: Optional[nx.Graph],
    save_path: str,
    title: str = "",
) -> None:
    """
    Save a 5-panel figure: RGB | GT | Prediction | Skeleton | Graph overlay.

    Parameters
    ----------
    rgb          : (H, W, 3) uint8
    gt_mask      : (H, W) binary
    pred_mask    : (H, W) binary
    pred_skeleton: (H, W) binary
    G_pred       : NetworkX graph (for edge overlay) or None
    save_path    : output file path (.png)
    title        : suptitle string
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return  # matplotlib not installed — silently skip visualisation

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(title, fontsize=10)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=8)
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT Centerline", fontsize=8)
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction", fontsize=8)
    axes[2].axis("off")

    axes[3].imshow(pred_skeleton, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Skeleton", fontsize=8)
    axes[3].axis("off")

    # Panel 5: graph overlay on RGB
    axes[4].imshow(rgb)
    if G_pred is not None and G_pred.number_of_edges() > 0:
        for u, v, data in G_pred.edges(data=True):
            pts = data.get("pts")
            if pts is not None and len(pts) > 1:
                cols = pts[:, 1]
                rows = pts[:, 0]
                color = "red" if data.get("healed", False) else "lime"
                axes[4].plot(cols, rows, color=color, linewidth=0.8, alpha=0.85)
    axes[4].set_title("Graph Overlay\n(green=direct, red=healed)", fontsize=8)
    axes[4].axis("off")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

