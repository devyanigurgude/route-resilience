from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from route_resilience.config import Config
from route_resilience.real_graph_pipeline import (
    load_model_for_inference,
    predict_probability_map,
)
from route_resilience.train import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "RouteResilience"
SPLIT_FILE = DATASET_DIR / "data_split.csv"
IMAGE_DIR = DATASET_DIR / "images_png"
MASK_DIR = DATASET_DIR / "masks_png"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth"

THRESHOLD = 0.5


def calculate_metrics(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    union = tp + fp + fn

    iou = tp / union if union > 0 else 1.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return iou, dice, precision, recall


def main():
    print("=" * 60)
    print("ROUTE RESILIENCE - FINAL TEST SEGMENTATION EVALUATION")
    print("=" * 60)

    df = pd.read_csv(SPLIT_FILE)
    test_df = df[df["split"] == "test"].copy()

    print(f"Test images: {len(test_df)}")
    print(f"Threshold: {THRESHOLD}")

    cfg = Config()
    cfg.model.arch = "unet_smp"
    cfg.model.encoder_name = "resnet34"
    cfg.model.encoder_weights = "imagenet"
    cfg.train.loss = "focal_dice"

    device = get_device("cpu")

    print(f"Device: {device}")
    print("\nLoading trained model...")

    model, checkpoint = load_model_for_inference(
        CHECKPOINT,
        cfg.model,
        cfg.train.loss,
        device,
    )

    print("Model loaded successfully!")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Best validation IoU: {checkpoint.get('best_val_iou')}")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    results = []

    for index, row in enumerate(test_df.itertuples(index=False), start=1):

        filename = row.filename

        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / filename

        try:
            image = np.array(Image.open(image_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L")) > 0

            probability_map = predict_probability_map(
                model,
                image,
                device,
            )

            prediction = probability_map >= THRESHOLD

            pred = prediction.astype(bool)
            target = mask.astype(bool)

            tp = np.logical_and(pred, target).sum()
            fp = np.logical_and(pred, ~target).sum()
            fn = np.logical_and(~pred, target).sum()

            total_tp += tp
            total_fp += fp
            total_fn += fn

            iou, dice, precision, recall = calculate_metrics(
                prediction,
                mask,
            )

            results.append({
                "filename": filename,
                "IoU": iou,
                "Dice": dice,
                "Precision": precision,
                "Recall": recall,
            })

            if index % 50 == 0 or index == len(test_df):
                print(f"Processed {index}/{len(test_df)}")

        except Exception as e:
            print(f"FAILED: {filename} -> {e}")

    # Dataset-level metrics
    union = total_tp + total_fp + total_fn

    dataset_iou = total_tp / union if union > 0 else 1.0

    dataset_dice = (
        2 * total_tp / (2 * total_tp + total_fp + total_fn)
        if (2 * total_tp + total_fp + total_fn) > 0
        else 1.0
    )

    dataset_precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    dataset_recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )

    results_df = pd.DataFrame(results)

    output_dir = PROJECT_ROOT / "outputs" / "test_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "segmentation_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Images evaluated : {len(results_df)}")
    print(f"IoU              : {dataset_iou:.4f}")
    print(f"Dice / F1        : {dataset_dice:.4f}")
    print(f"Precision        : {dataset_precision:.4f}")
    print(f"Recall           : {dataset_recall:.4f}")

    print("\nResults saved to:")
    print(results_path)


if __name__ == "__main__":
    main()
    