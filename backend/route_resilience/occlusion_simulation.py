"""Controlled synthetic occlusion simulation for satellite road imagery.

This module provides reproducible, parameterized occlusion patterns placed
over known road locations to evaluate segmentation resilience and graph
recovery under controlled obstruction:
  1. Rectangular block occlusions (e.g., simulating large buildings, construction, or data dropouts).
  2. Irregular organic patches (e.g., simulating dense tree canopy clusters or cloud shadows).
  3. Transverse linear swaths (e.g., simulating overpasses, bridge shadows, or canopy corridors).

All functions strictly preserve the ground-truth road mask untouched.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import disk


def generate_road_centered_occlusions(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pattern: str = "rectangular_block",
    num_occlusions: int = 2,
    size_range: Tuple[int, int] = (16, 26),
    fill_mode: str = "shadow",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Applies controlled synthetic occlusions over known ground-truth road pixels.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input RGB image of shape (H, W, 3) and dtype uint8.
    gt_mask : np.ndarray
        Ground-truth road mask of shape (H, W), non-zero where roads exist.
        The ground truth is NEVER modified.
    pattern : str
        One of "rectangular_block", "irregular_patch", "linear_transverse".
    num_occlusions : int
        Number of occlusion cutouts to place along road segments.
    size_range : Tuple[int, int]
        Min and max occlusion extent in pixels.
    fill_mode : str
        "shadow" (dark shadow tone ~35-45), "canopy" (vegetation green),
        "mean" (local background mean), or "black" (0).
    seed : int
        RNG seed for strict reproducibility.

    Returns
    -------
    occluded_rgb : np.ndarray
        Occluded RGB image of shape (H, W, 3), dtype uint8.
    occlusion_mask : np.ndarray
        Boolean array of shape (H, W), True where occlusion was stamped.
    metadata : Dict
        Parameters, centroid locations, and affected road pixel count.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image of shape (H, W, 3), got {image_rgb.shape}")
    if gt_mask.shape[:2] != image_rgb.shape[:2]:
        raise ValueError(f"gt_mask shape {gt_mask.shape} does not match image {image_rgb.shape[:2]}")

    H, W, _ = image_rgb.shape
    occluded_rgb = image_rgb.copy()
    occlusion_mask = np.zeros((H, W), dtype=bool)

    rng = np.random.default_rng(seed)
    road_coords = np.argwhere(gt_mask > 0)
    if len(road_coords) == 0:
        return occluded_rgb, occlusion_mask, {"pattern": pattern, "num_placed": 0, "affected_pixels": 0}

    # Avoid image border margins
    margin = max(size_range[1], 10)
    valid_indices = [
        idx for idx, (r, c) in enumerate(road_coords)
        if margin <= r < H - margin and margin <= c < W - margin
    ]
    if not valid_indices:
        valid_indices = list(range(len(road_coords)))

    # Select well-spaced anchor points along the road
    chosen_anchors = []
    shuffled_idx = rng.permutation(valid_indices)
    min_separation = float(size_range[1] * 1.5)

    for idx in shuffled_idx:
        cand_pt = road_coords[idx]
        if all(np.linalg.norm(cand_pt - prev) >= min_separation for prev in chosen_anchors):
            chosen_anchors.append(cand_pt)
            if len(chosen_anchors) >= num_occlusions:
                break

    if not chosen_anchors and len(road_coords) > 0:
        chosen_anchors = [road_coords[rng.choice(len(road_coords))]]

    # Choose fill color
    if fill_mode == "shadow":
        fill_color = np.array([38, 42, 46], dtype=np.uint8)
    elif fill_mode == "canopy":
        fill_color = np.array([45, 68, 38], dtype=np.uint8)
    elif fill_mode == "cloud":
        fill_color = np.array([232, 236, 240], dtype=np.uint8)
    elif fill_mode == "black":
        fill_color = np.array([0, 0, 0], dtype=np.uint8)
    elif fill_mode == "mean":
        bg_pixels = image_rgb[gt_mask == 0]
        if len(bg_pixels) > 0:
            fill_color = np.clip(bg_pixels.mean(axis=0), 0, 255).astype(np.uint8)
        else:
            fill_color = np.array([40, 40, 40], dtype=np.uint8)
    else:
        fill_color = np.array([35, 38, 42], dtype=np.uint8)

    records = []
    for anchor in chosen_anchors:
        ar, ac = int(anchor[0]), int(anchor[1])
        sz = int(rng.integers(size_range[0], size_range[1] + 1))

        if pattern == "rectangular_block":
            half_h = sz // 2
            half_w = int(rng.integers(size_range[0], size_range[1] + 1)) // 2
            r0 = max(0, ar - half_h)
            r1 = min(H, ar + half_h + 1)
            c0 = max(0, ac - half_w)
            c1 = min(W, ac + half_w + 1)
            occlusion_mask[r0:r1, c0:c1] = True
            records.append({"type": pattern, "center": (ar, ac), "bbox": (r0, c0, r1, c1)})

        elif pattern in ("irregular_patch", "procedural_cloud", "canopy_cluster"):
            # Organic patch constructed by multiple dilated disks
            patch = np.zeros((H, W), dtype=bool)
            num_subparts = int(rng.integers(4, 7)) if pattern == "procedural_cloud" else int(rng.integers(3, 6))
            radius = max(4, sz // 2) if pattern == "procedural_cloud" else max(3, sz // 3)
            for _ in range(num_subparts):
                dr = int(rng.integers(-radius, radius + 1))
                dc = int(rng.integers(-radius, radius + 1))
                cr, cc = int(np.clip(ar + dr, 0, H - 1)), int(np.clip(ac + dc, 0, W - 1))
                patch[cr, cc] = True
            dilated = binary_dilation(patch, structure=disk(radius))
            occlusion_mask |= dilated
            if pattern == "procedural_cloud" and fill_mode != "black":
                fill_color = np.array([232, 236, 240], dtype=np.uint8)
            elif pattern == "canopy_cluster" and fill_mode != "black":
                fill_color = np.array([45, 68, 38], dtype=np.uint8)
            records.append({"type": pattern, "center": (ar, ac), "radius": radius})

        elif pattern == "linear_transverse":
            # Perpendicular cut across the road corridor
            thickness = max(8, sz // 2)
            length = sz + 8
            # Estimate road tangent locally from gt_mask
            local_pts = road_coords[np.linalg.norm(road_coords - anchor, axis=1) <= 10]
            if len(local_pts) >= 4:
                cov = np.cov((local_pts - anchor).T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                tangent = eigvecs[:, -1]
                normal = np.array([-tangent[1], tangent[0]])
            else:
                normal = np.array([1.0, 0.0])

            # Draw transverse strip along normal vector
            strip_pts = []
            for t in np.linspace(-length / 2, length / 2, num=int(length * 2)):
                for w in np.linspace(-thickness / 2, thickness / 2, num=int(thickness * 2)):
                    pr = int(np.clip(round(ar + t * normal[0] + w * tangent[0] if len(local_pts) >= 4 else ar + t), 0, H - 1))
                    pc = int(np.clip(round(ac + t * normal[1] + w * tangent[1] if len(local_pts) >= 4 else ac + w), 0, W - 1))
                    strip_pts.append((pr, pc))
            strip_arr = np.array(strip_pts)
            occlusion_mask[strip_arr[:, 0], strip_arr[:, 1]] = True
            records.append({"type": pattern, "center": (ar, ac), "thickness": thickness, "length": length})

    # Apply occlusion to the image copy with subtle edge softness/noise
    noise = rng.normal(0, 3.0, (H, W, 3))
    noisy_fill = np.clip(fill_color.astype(float) + noise, 0, 255).astype(np.uint8)
    occluded_rgb[occlusion_mask] = noisy_fill[occlusion_mask]

    affected_road_pixels = int(np.sum(occlusion_mask & (gt_mask > 0)))

    metadata = {
        "pattern": pattern,
        "num_occlusions_requested": num_occlusions,
        "num_occlusions_placed": len(chosen_anchors),
        "occlusion_pixel_count": int(occlusion_mask.sum()),
        "affected_road_pixels": affected_road_pixels,
        "fill_mode": fill_mode,
        "anchors": [(int(r), int(c)) for r, c in chosen_anchors],
        "details": records,
    }
    return occluded_rgb, occlusion_mask, metadata
