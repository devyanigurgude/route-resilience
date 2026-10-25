"""Real-image segmentation inference connected to the pixel-space graph pipeline.

This module connects checkpoint-backed inference to the pixel-space graph
pipeline, retaining native image coordinates and reporting per-image extraction
diagnostics. Graph distances and positions remain pixel units; they are not
geographic distances or travel times.
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import preprocess_rgb_image
from .evaluate_topology import graph_summary
from .graph_extraction import (
    attach_width,
    clean_road_mask,
    confidence_gated_gap_healing,
    connectivity_ratio,
    mask_to_skeleton,
    skeleton_to_graph,
)
from .models import build_model
from .train import _torch_load_checkpoint


def load_model_for_inference(checkpoint_path, model_cfg, loss_name: str, device):
    """Load a full, compatible checkpoint without downloading encoder weights.

    A graph run needs the learned checkpoint weights, not a fresh ImageNet
    initialization. Legacy weights-only files are rejected because their model
    configuration cannot be verified safely.
    """
    checkpoint = _torch_load_checkpoint(checkpoint_path, device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Inference requires a full checkpoint with model_state_dict and config.")
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("Checkpoint has no saved configuration; refusing an unverifiable load.")

    saved_model = saved_config.get("model")
    saved_loss = saved_config.get("train", {}).get("loss")
    if saved_model != asdict(model_cfg) or saved_loss != loss_name:
        raise ValueError(
            "Checkpoint configuration is incompatible with the requested model or loss. "
            f"checkpoint model={saved_model}, loss={saved_loss}; "
            f"requested model={asdict(model_cfg)}, loss={loss_name}"
        )

    # The checkpoint supplies all learned encoder weights. Passing None avoids
    # an unnecessary ImageNet download during inference.
    model = build_model(replace(model_cfg, encoder_weights=None)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_probability_map(model, rgb_image: np.ndarray, device) -> np.ndarray:
    """Predict at native tile scale and return a float32 HxW probability map."""
    rgb_image = np.asarray(rgb_image)
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {rgb_image.shape}")

    height, width = rgb_image.shape[:2]
    tile_size = 256
    if height <= tile_size and width <= tile_size:
        padded = _pad_image_to_tile(rgb_image, tile_size)
        probability = _predict_tile(model, padded, device)
        return probability[:height, :width]

    stride = tile_size - tile_size // 8
    row_starts = _tile_starts(height, tile_size, stride)
    col_starts = _tile_starts(width, tile_size, stride)
    probability_sum = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint16)

    for row in row_starts:
        for col in col_starts:
            tile = rgb_image[row:min(row + tile_size, height), col:min(col + tile_size, width)]
            tile_height, tile_width = tile.shape[:2]
            padded = _pad_image_to_tile(tile, tile_size)
            probability = _predict_tile(model, padded, device)[:tile_height, :tile_width]
            probability_sum[row:row + tile_height, col:col + tile_width] += probability
            coverage[row:row + tile_height, col:col + tile_width] += 1

    return probability_sum / coverage


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final_start = max(0, length - tile_size)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _pad_image_to_tile(image: np.ndarray, tile_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    pad_height = max(0, tile_size - height)
    pad_width = max(0, tile_size - width)
    if not pad_height and not pad_width:
        return image
    mode = "reflect" if height > 1 and width > 1 else "edge"
    return np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode=mode)


def _predict_tile(model, rgb_tile: np.ndarray, device) -> np.ndarray:
    image = preprocess_rgb_image(rgb_tile).unsqueeze(0).to(device)
    logits = model(image)
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 1:
        raise ValueError(f"Expected model logits with shape [1, 1, H, W], got {tuple(logits.shape)}")
    return torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def _metrics(skeleton: np.ndarray, graph: nx.Graph) -> Dict[str, Any]:
    return {
        "skeleton_connectivity_ratio": connectivity_ratio(skeleton),
        **graph_summary(graph),
    }


def _graph_quality(graph: nx.Graph) -> Dict[str, Any]:
    component_sizes = sorted(
        (len(component) for component in nx.connected_components(graph)),
        reverse=True,
    )
    return {
        "component_sizes": component_sizes,
        "isolated_node_count": sum(1 for _, degree in graph.degree() if degree == 0),
        "endpoint_count": sum(1 for _, degree in graph.degree() if degree == 1),
        "junction_count": sum(1 for _, degree in graph.degree() if degree >= 3),
        "routable_component_count": sum(size > 1 for size in component_sizes),
    }


def _numeric_deltas(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: after[key] - before[key]
        for key in before
        if isinstance(before[key], (int, float)) and isinstance(after.get(key), (int, float))
    }


def cloud_occlusion_mask(rgb_image: np.ndarray) -> np.ndarray:
    """Detect bright, low-saturation cloud cover in an RGB satellite image.

    This is intentionally conservative: it is only a spatial gate for the
    long-gap geometric test, never a road detector on its own.

    The strict core (luminance >= 0.68, saturation <= 0.30) is dilated by
    a modest radius to absorb semi-transparent cloud edges and thin haze
    that sit just below the luminance threshold.  These margins still
    occlude road pixels but would otherwise create a gap in the cloud
    mask, causing the coverage check to reject a valid cloud bridge.
    """
    from scipy.ndimage import binary_dilation

    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {rgb.shape}")
    channel_max = rgb.max(axis=2)
    channel_min = rgb.min(axis=2)
    saturation = (channel_max - channel_min) / np.maximum(channel_max, 1e-6)
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    # Strict core: bright, achromatic cloud pixels
    core = (luminance >= 0.68) & (saturation <= 0.30)
    # Halo ring: slightly dimmer or more saturated cloud edges
    halo = (luminance >= 0.52) & (saturation <= 0.42)
    # Dilate the core by ~6 px, then intersect with the halo criterion
    # so only cloud-adjacent semi-transparent pixels are absorbed.
    dilated_core = binary_dilation(core, iterations=6)
    return core | (dilated_core & halo)


def dilate_mask(
    mask: Optional[np.ndarray],
    kernel_size: int = 3,
    iterations: int = 1,
) -> Optional[np.ndarray]:
    """Applies morphological dilation with a square structuring element.

    Preserves the source mask unchanged. If iterations == 0 or kernel_size == 1,
    returns a copy of the original mask.

    Parameters
    ----------
    mask : Optional[np.ndarray]
        2D boolean array to dilate. If None, returns None.
    kernel_size : int
        Size of the square structuring element. Must be a positive odd integer (1, 3, 5, ...).
    iterations : int
        Number of dilation iterations. Must be non-negative.
    """
    if mask is None:
        return None
    if iterations < 0:
        raise ValueError(f"Dilation iterations must be non-negative, got {iterations}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"Dilation kernel_size must be a positive odd integer, got {kernel_size}")
    if iterations == 0 or kernel_size == 1:
        return np.asarray(mask, dtype=bool).copy()

    from scipy.ndimage import binary_dilation

    mask_bool = np.asarray(mask, dtype=bool)
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    return binary_dilation(mask_bool, structure=structure, iterations=iterations)


def shadow_occlusion_mask(
    rgb_image: np.ndarray,
    dilation_kernel_size: int = 3,
    dilation_iterations: int = 0,
) -> np.ndarray:
    """Detect deep shadow candidate regions in an RGB satellite image.

    Conservative spatial gate for occlusion-aware continuity.
    Identifies dark regions where luminance and all RGB channels are low.
    """
    if rgb_image is None:
        return None
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {rgb.shape}")
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    channel_max = rgb.max(axis=2)
    base_mask = (luminance <= 0.16) & (channel_max <= 0.22)
    if dilation_iterations > 0:
        return dilate_mask(base_mask, kernel_size=dilation_kernel_size, iterations=dilation_iterations)
    return base_mask


def canopy_occlusion_mask(
    rgb_image: np.ndarray,
    dilation_kernel_size: int = 3,
    dilation_iterations: int = 0,
) -> np.ndarray:
    """Detect dense tree canopy vegetation candidate regions in an RGB satellite image.

    Conservative spatial gate for occlusion-aware continuity.
    Uses Excess Green Index (ExG = 2G - R - B) and green-dominance.
    """
    if rgb_image is None:
        return None
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {rgb.shape}")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2.0 * g - r - b
    base_mask = (exg >= 0.08) & (g > r * 1.08) & (g > b * 1.05) & (g <= 0.70)
    if dilation_iterations > 0:
        return dilate_mask(base_mask, kernel_size=dilation_kernel_size, iterations=dilation_iterations)
    return base_mask


def photometric_occlusion_mask(
    rgb_image: np.ndarray,
    dilation_kernel_size: int = 3,
    dilation_iterations: int = 0,
) -> np.ndarray:
    """Combines conservative cloud, shadow, and tree canopy candidate masks."""
    if rgb_image is None:
        return None
    cloud = cloud_occlusion_mask(rgb_image)
    shadow = shadow_occlusion_mask(
        rgb_image,
        dilation_kernel_size=dilation_kernel_size,
        dilation_iterations=dilation_iterations,
    )
    canopy = canopy_occlusion_mask(
        rgb_image,
        dilation_kernel_size=dilation_kernel_size,
        dilation_iterations=dilation_iterations,
    )
    return cloud | shadow | canopy


def graph_from_probability_map(
    probability_map, threshold: float, graph_cfg, rgb_image: np.ndarray | None = None
) -> Dict[str, Any]:
    """Threshold a real prediction and run the existing pixel-space graph stages."""
    probability_map = np.asarray(probability_map, dtype=np.float32)
    if probability_map.ndim != 2:
        raise ValueError(f"Expected an HxW probability map, got shape {probability_map.shape}")
    if probability_map.size == 0 or not np.isfinite(probability_map).all():
        raise ValueError("Probability map must be non-empty and contain only finite values")
    if np.any((probability_map < 0.0) | (probability_map > 1.0)):
        raise ValueError("Probability map values must be in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    threshold_mask = probability_map >= threshold
    binary_mask = clean_road_mask(
        threshold_mask,
        min_component_pixels=max(1, int(np.ceil(graph_cfg.min_edge_length_px))),
    )
    skeleton = mask_to_skeleton(binary_mask)
    graph_before = skeleton_to_graph(
        skeleton,
        prob_map=probability_map,
        min_edge_length_px=graph_cfg.min_edge_length_px,
    )
    graph_before = attach_width(graph_before, binary_mask)

    dilation_kernel_size = getattr(graph_cfg, "mask_dilation_kernel_size", 3)
    dilation_iters = (
        getattr(graph_cfg, "mask_dilation_iterations", 1)
        if getattr(graph_cfg, "enable_mask_dilation", False)
        else 0
    )
    cloud_mask = cloud_occlusion_mask(rgb_image) if rgb_image is not None else None
    photometric_mask = (
        photometric_occlusion_mask(
            rgb_image,
            dilation_kernel_size=dilation_kernel_size,
            dilation_iterations=dilation_iters,
        )
        if rgb_image is not None
        else None
    )
    graph_after = confidence_gated_gap_healing(
        graph_before.copy(),
        probability_map,
        max_gap_px=graph_cfg.max_gap_px,
        min_gap_confidence=graph_cfg.min_gap_confidence,
        tangent_dot_threshold=graph_cfg.tangent_dot_threshold,
        continuity_penalty_weight=graph_cfg.continuity_penalty_weight,
        min_component_len=getattr(graph_cfg, "min_component_len", 0.0),
        min_total_component_len=getattr(graph_cfg, "min_total_component_len", 0.0),
        max_sinuosity=getattr(graph_cfg, "max_sinuosity", 1.5),
        min_flank_contrast=getattr(graph_cfg, "min_flank_contrast", None),
        filter_tile_borders=getattr(graph_cfg, "filter_tile_borders", False),
        enable_occlusion_recovery=getattr(graph_cfg, "enable_occlusion_recovery", False),
        max_occlusion_gap_px=getattr(graph_cfg, "max_occlusion_gap_px", 30.0),
        min_occlusion_confidence=getattr(graph_cfg, "min_occlusion_confidence", 0.05),
        occlusion_heading_threshold=getattr(graph_cfg, "occlusion_heading_threshold", 0.75),
        max_lateral_offset_px=getattr(graph_cfg, "max_lateral_offset_px", 4.0),
        occlusion_mask=photometric_mask,
        cloud_mask=cloud_mask,
        enable_cloud_occlusion_recovery=getattr(graph_cfg, "enable_cloud_occlusion_recovery", False),
        max_cloud_occlusion_gap_px=getattr(graph_cfg, "max_cloud_occlusion_gap_px", 240.0),
        cloud_heading_threshold=getattr(graph_cfg, "cloud_heading_threshold", 0.78),
        cloud_min_component_len=getattr(graph_cfg, "cloud_min_component_len", 20.0),
        cloud_min_coverage=getattr(graph_cfg, "cloud_min_coverage", 0.55),
        cloud_max_sinuosity=getattr(graph_cfg, "cloud_max_sinuosity", 1.35),
    )

    before = _metrics(skeleton, graph_before)
    after = _metrics(skeleton, graph_after)
    graph_quality = _graph_quality(graph_after)
    road_pixel_count = int(binary_mask.sum())
    warnings = []
    if road_pixel_count == 0:
        warnings.append(
            "No pixels exceeded the road threshold; check image quality and model-domain suitability."
        )
    if graph_after.number_of_edges() == 0:
        warnings.append(
            "No traversable road edge was extracted; inspect the probability map and road mask."
        )
    if after["n_components"] > 1:
        warnings.append(
            f"The extracted network has {after['n_components']} components after evidence-gated healing. "
            "These may be genuine separate roads or segmentation gaps; inspect the overlay. "
            "No unsupported links were added."
        )
    if graph_quality["isolated_node_count"] > 0:
        warnings.append(
            f"The extracted network contains {graph_quality['isolated_node_count']} isolated nodes "
            "that cannot participate in a route."
        )

    graph_status = (
        "insufficient"
        if graph_quality["routable_component_count"] == 0
        else "fragmented"
        if after["n_components"] > 1
        else "connected"
    )
    summary = {
        "threshold": float(threshold),
        "image_dimensions": {
            "height": int(probability_map.shape[0]),
            "width": int(probability_map.shape[1]),
        },
        "coordinate_system": "pixel-space road-network proxy; not geographic distance or travel time",
        "mask_quality": {
            "thresholded_road_pixel_count": int(threshold_mask.sum()),
            "road_pixel_count": road_pixel_count,
            "filtered_noise_pixel_count": int(threshold_mask.sum() - road_pixel_count),
            "road_fraction": float(binary_mask.mean()),
            "mean_probability": float(probability_map.mean()),
            "probability_std": float(probability_map.std()),
        },
        "graph_quality": graph_quality,
        "graph_status": graph_status,
        "warnings": warnings,
        "before_healing": before,
        "after_healing": after,
        "delta_after_minus_before": _numeric_deltas(before, after),
        "healed_edge_count": sum(
            1 for _, _, data in graph_after.edges(data=True) if data.get("healed", False)
        ),
        "occlusion_recovered_edge_count": sum(
            1 for _, _, data in graph_after.edges(data=True) if data.get("occlusion_recovered", False)
        ),
        "candidate_occlusions": graph_after.graph.get("candidate_occlusions", []),
        "candidate_occlusion_count": len(graph_after.graph.get("candidate_occlusions", [])),
    }
    return {
        "probability_map": probability_map,
        "binary_mask": binary_mask,
        "skeleton": skeleton,
        "graph_before": graph_before,
        "graph_after": graph_after,
        "G_extracted": graph_before,
        "G_healed": graph_after,
        "summary": summary,
    }


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def graph_to_jsonable(graph: nx.Graph) -> Dict[str, Any]:
    """Serialize graph IDs, attributes, and candidate occlusions safely."""
    payload = {
        "coordinate_system": "pixel-space",
        "nodes": [
            {"id": _json_safe(node), "attributes": _json_safe(data)}
            for node, data in graph.nodes(data=True)
        ],
        "edges": [
            {"source": _json_safe(u), "target": _json_safe(v), "attributes": _json_safe(data)}
            for u, v, data in graph.edges(data=True)
        ],
    }
    candidates = graph.graph.get("candidate_occlusions", [])
    if candidates:
        payload["candidate_occlusions"] = _json_safe(candidates)
    return payload


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).convert("RGB")


def _graph_overlay(rgb_image: np.ndarray, graph: nx.Graph) -> Image.Image:
    canvas = Image.fromarray(rgb_image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    # 1. Candidate occlusions (dashed purple lines)
    for cand in graph.graph.get("candidate_occlusions", []):
        if not cand.get("recovered", False):
            pu = np.array(cand["pos_source"], dtype=float)
            pv = np.array(cand["pos_target"], dtype=float)
            n_steps = max(3, int(cand.get("gap_px", 10.0) / 3.0))
            pts = np.linspace(pu, pv, n_steps)
            for k in range(0, len(pts) - 1, 2):
                p1 = (int(pts[k][1]), int(pts[k][0]))
                p2 = (int(pts[k + 1][1]), int(pts[k + 1][0]))
                draw.line([p1, p2], fill=(168, 85, 247), width=1)

    # 2. Graph edges
    for _, _, data in graph.edges(data=True):
        points = [(int(col), int(row)) for row, col in np.asarray(data["pts"])]
        if data.get("occlusion_recovered", False):
            color = (16, 185, 129)  # Emerald green for occlusion-recovered continuity
            width = 2
        elif data.get("healed", False):
            color = (255, 220, 0)    # Amber yellow for standard healed gap
            width = 1
        else:
            color = (255, 0, 0)      # Red for directly detected road
            width = 1
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)

    # 3. Nodes
    for _, data in graph.nodes(data=True):
        row, col = data["pos"]
        draw.ellipse((int(col) - 2, int(row) - 2, int(col) + 2, int(row) + 2), fill=(0, 255, 255))
    return canvas


def save_artifacts(result: Dict[str, Any], rgb_image: np.ndarray, output_dir) -> Path:
    """Save required rasters, graphs, metrics, and a compact technical overview."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "probability.npy", result["probability_map"])
    _mask_image(result["binary_mask"]).save(output_dir / "binary_mask.png")
    _mask_image(result["skeleton"]).save(output_dir / "skeleton.png")
    _graph_overlay(rgb_image, result["graph_after"]).save(output_dir / "repaired_network.png")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result["summary"]), handle, indent=2)
    for filename, graph_key in (("graph.json", "graph_before"), ("healed_graph.json", "graph_after")):
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(graph_to_jsonable(result[graph_key]), handle, indent=2)

    panels = [
        Image.fromarray(rgb_image).convert("RGB"),
        _mask_image(result["binary_mask"]),
        _mask_image(result["skeleton"]),
        _graph_overlay(rgb_image, result["graph_after"]),
    ]
    labels = [
        "Original image",
        "Binary road mask",
        "Skeleton",
        "Healed graph (green=occlusion recovered, yellow=healed)",
    ]
    width, height = panels[0].size
    figure = Image.new("RGB", (width * 2, height * 2), "white")
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x, y = (index % 2) * width, (index // 2) * height
        figure.paste(panel, (x, y))
        ImageDraw.Draw(figure).rectangle((x, y, x + min(width, len(label) * 7 + 6), y + 14), fill="white")
        ImageDraw.Draw(figure).text((x + 3, y + 2), label, fill="black")
    figure.save(output_dir / "visualization.png")
    return output_dir


def load_rgb_image(image_path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
