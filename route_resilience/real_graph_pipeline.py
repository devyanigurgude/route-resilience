"""Real-image segmentation inference connected to the pixel-space graph pipeline.

This module intentionally keeps the existing graph algorithms unchanged. Its
job is only to bridge a checkpoint-loaded segmentation model to their expected
probability-map and binary-mask inputs. Graph distances and positions remain in
pixel units; they are not geographic distances or travel times.
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import preprocess_rgb_image
from .evaluate_topology import graph_summary
from .graph_extraction import (
    attach_width,
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
    """Return deterministic sigmoid probabilities as a float32 HxW NumPy array."""
    image = preprocess_rgb_image(rgb_image).unsqueeze(0).to(device)
    logits = model(image)
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 1:
        raise ValueError(f"Expected model logits with shape [1, 1, H, W], got {tuple(logits.shape)}")
    return torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def _metrics(skeleton: np.ndarray, graph: nx.Graph) -> Dict[str, Any]:
    return {
        "skeleton_connectivity_ratio": connectivity_ratio(skeleton),
        **graph_summary(graph),
    }


def _numeric_deltas(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: after[key] - before[key]
        for key in before
        if isinstance(before[key], (int, float)) and isinstance(after.get(key), (int, float))
    }


def graph_from_probability_map(probability_map, threshold: float, graph_cfg) -> Dict[str, Any]:
    """Threshold a real prediction and run the existing pixel-space graph stages."""
    probability_map = np.asarray(probability_map, dtype=np.float32)
    if probability_map.ndim != 2:
        raise ValueError(f"Expected an HxW probability map, got shape {probability_map.shape}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    binary_mask = probability_map >= threshold
    skeleton = mask_to_skeleton(binary_mask)
    graph_before = skeleton_to_graph(
        skeleton,
        prob_map=probability_map,
        min_edge_length_px=graph_cfg.min_edge_length_px,
    )
    graph_before = attach_width(graph_before, binary_mask)
    graph_after = confidence_gated_gap_healing(
        graph_before.copy(),
        probability_map,
        max_gap_px=graph_cfg.max_gap_px,
        min_gap_confidence=graph_cfg.min_gap_confidence,
    )

    before = _metrics(skeleton, graph_before)
    after = _metrics(skeleton, graph_after)
    summary = {
        "threshold": float(threshold),
        "coordinate_system": "pixel-space road-network proxy; not geographic distance or travel time",
        "before_healing": before,
        "after_healing": after,
        "delta_after_minus_before": _numeric_deltas(before, after),
        "healed_edge_count": sum(
            1 for _, _, data in graph_after.edges(data=True) if data.get("healed", False)
        ),
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
    """Serialize graph IDs and all existing node/edge attributes safely."""
    return {
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


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).convert("RGB")


def _graph_overlay(rgb_image: np.ndarray, graph: nx.Graph) -> Image.Image:
    canvas = Image.fromarray(rgb_image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for _, _, data in graph.edges(data=True):
        points = [(int(col), int(row)) for row, col in np.asarray(data["pts"])]
        color = (255, 220, 0) if data.get("healed", False) else (255, 0, 0)
        if len(points) >= 2:
            draw.line(points, fill=color, width=1)
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
    labels = ["Original image", "Binary road mask", "Skeleton", "Healed graph (yellow=healed)"]
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
