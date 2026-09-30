"""Run a trained segmentation checkpoint through the existing pixel-space graph pipeline."""

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from route_resilience.config import Config
from route_resilience.real_graph_pipeline import (
    graph_from_probability_map,
    load_model_for_inference,
    load_rgb_image,
    predict_probability_map,
    save_artifacts,
)
from route_resilience.train import get_device


def resolve_threshold(value, checkpoint):
    """Accept a numeric val-selected threshold or its saved JSON artifact."""
    try:
        threshold = float(value)
    except ValueError:
        with open(value, encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact.get("selection_split") != "validation":
            raise ValueError("Threshold artifact must state selection_split='validation'.")
        threshold = artifact.get("threshold")
        if artifact.get("config") != checkpoint.get("config"):
            raise ValueError("Threshold artifact configuration does not match the checkpoint.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return threshold


def validate_graph_config(checkpoint, graph_cfg):
    """Refuse graph settings that differ from the checkpointed run."""
    saved = checkpoint.get("config", {}).get("graph")
    if saved is not None and saved != asdict(graph_cfg):
        raise ValueError(
            "Graph configuration differs from the checkpointed run; refusing an "
            "unverifiable real-image graph extraction."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Infer one image and build a pixel-space healed road graph."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--arch", required=True, choices=["unet", "unet_smp", "deeplabv3plus"])
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--encoder_weights", required=True, choices=["imagenet", "none"])
    parser.add_argument("--loss", default="focal_dice", choices=["bce_dice", "focal_dice"])
    parser.add_argument("--threshold", required=True,
                        help="Validation-selected numeric threshold, or its threshold_*.json artifact.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = Config()
    cfg.model.arch = args.arch
    cfg.model.encoder_name = args.encoder
    cfg.model.encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights
    cfg.train.loss = args.loss
    device = get_device(args.device)

    model, checkpoint = load_model_for_inference(
        args.checkpoint, cfg.model, cfg.train.loss, device
    )
    validate_graph_config(checkpoint, cfg.graph)
    threshold = resolve_threshold(args.threshold, checkpoint)
    rgb_image = load_rgb_image(args.image)
    probability_map = predict_probability_map(model, rgb_image, device)
    result = graph_from_probability_map(probability_map, threshold, cfg.graph)
    save_artifacts(result, rgb_image, args.output_dir)

    print(f"Saved pixel-space graph artifacts to {args.output_dir}")
    print(result["summary"])


if __name__ == "__main__":
    main()
