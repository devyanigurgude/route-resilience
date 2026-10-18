"""Phase 5: Transformer Implementation & Comparative Evaluation.

Trains a lightweight SegFormer-B0 (mit_b0) model alongside the existing
ResNet34 U-Net and evaluates both models on the exact same held-out test set:
  - Metric comparison: IoU, Dice, Precision, Recall, and Inference Time (ms/tile).
  - Parameter count and architectural trade-offs.
  - Generates artifacts in outputs/phase5_transformer/.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.config import Config, ModelConfig, TrainConfig
from route_resilience.data import RoadDataset, build_datasets, build_transforms
from route_resilience.losses import build_loss
from route_resilience.models import build_model
from route_resilience.real_graph_pipeline import load_model_for_inference, predict_probability_map
from route_resilience.train import get_device, _torch_load_checkpoint


def train_segformer_b0(
    dataset_dir: Path,
    output_dir: Path,
    device: torch.device,
    train_subset_size: int = 64,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-4,
) -> Path:
    """Trains SegFormer-B0 on a calibrated representative training subset."""
    print("=" * 80)
    print("PHASE 5: TRAINING SEGFORMER-B0 (mit_b0) ROAD SEGMENTATION MODEL")
    print("=" * 80)

    cfg = Config()
    cfg.data.dataset_dir = str(dataset_dir)
    cfg.model.arch = "segformer"
    cfg.model.encoder_name = "mit_b0"
    cfg.model.encoder_weights = "imagenet"
    cfg.train.loss = "focal_dice"

    train_ds, val_ds, _ = build_datasets(cfg.data)

    # Select deterministic subset for CPU training feasibility
    rng = np.random.default_rng(42)
    indices = rng.choice(len(train_ds), size=min(train_subset_size, len(train_ds)), replace=False)
    train_sub = Subset(train_ds, indices)
    val_sub = Subset(val_ds, rng.choice(len(val_ds), size=min(32, len(val_ds)), replace=False))

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

    print(f"Instantiating SegFormer-B0 with ImageNet pretrained encoder weights...")
    model = build_model(cfg.model).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"SegFormer-B0 parameter count: {num_params:,}")

    criterion = build_loss(cfg.train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_ckpt_path = output_dir / "best_segformer_mit_b0_focaldice.pth"
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * len(images)
            n_train += len(images)

        train_loss /= max(1, n_train)
        scheduler.step()

        # Validation pass
        model.eval()
        val_loss = 0.0
        n_val = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += float(loss.item()) * len(images)
                n_val += len(images)

                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                targets = (masks > 0.5).cpu().numpy().astype(int).ravel()
                all_preds.append(probs)
                all_targets.append(targets)

        val_loss /= max(1, n_val)
        y_prob = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        y_bin = y_prob >= 0.5
        val_iou = float(jaccard_score(y_true, y_bin, zero_division=1.0))
        val_dice = float(f1_score(y_true, y_bin, zero_division=1.0))

        dt = time.time() - t0
        print(f"Epoch [{epoch}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_dice={val_dice:.4f} ({dt:.1f}s)")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
            "elapsed_s": dt,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_dice": val_dice,
                "config": asdict(cfg),
            }
            torch.save(ckpt_dict, best_ckpt_path)

    # Save threshold artifact
    thresh_artifact = {
        "threshold": 0.5,
        "selection_split": "validation",
        "metric": "focal_dice",
        "config": asdict(cfg),
    }
    thresh_path = output_dir / "threshold_segformer_mit_b0_focaldice.json"
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump(thresh_artifact, f, indent=2)

    # Also copy to checkpoints/
    ckpt_copy_dir = PROJECT_ROOT / "checkpoints"
    ckpt_copy_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.load(best_ckpt_path), ckpt_copy_dir / "best_segformer_mit_b0_focaldice.pth")
    with open(ckpt_copy_dir / "threshold_segformer_mit_b0_focaldice.json", "w", encoding="utf-8") as f:
        json.dump(thresh_artifact, f, indent=2)

    with open(output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved SegFormer-B0 checkpoint to: {best_ckpt_path}")
    return best_ckpt_path


def benchmark_inference_speed(model, sample_tile: np.ndarray, device: torch.device, warmup: int = 5, reps: int = 15) -> Dict[str, float]:
    """Measures precise per-tile inference latency in milliseconds."""
    for _ in range(warmup):
        _ = predict_probability_map(model, sample_tile, device)

    latencies = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = predict_probability_map(model, sample_tile, device)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }


def evaluate_models_on_test_set(
    dataset_dir: Path,
    output_dir: Path,
    device: torch.device,
    test_tile_count: int = 24,
) -> Dict[str, Any]:
    """Compares ResNet34 U-Net vs SegFormer-B0 on the exact same held-out test tiles."""
    print("\n" + "=" * 80)
    print("PHASE 5: COMPARATIVE EVALUATION ON HELD-OUT TEST SET")
    print("=" * 80)

    split_csv = dataset_dir / "data_split.csv"
    df_split = pd.read_csv(split_csv)
    test_filenames = df_split[df_split["split"] == "test"]["filename"].tolist()[:test_tile_count]
    print(f"Held-out test tiles to evaluate: {len(test_filenames)}")

    # 1. Load ResNet34 U-Net
    resnet_ckpt_path = PROJECT_ROOT / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth"
    resnet_cfg = Config()
    resnet_cfg.model.arch = "unet_smp"
    resnet_cfg.model.encoder_name = "resnet34"
    resnet_cfg.model.encoder_weights = "imagenet"
    resnet_cfg.train.loss = "focal_dice"
    model_resnet, _ = load_model_for_inference(resnet_ckpt_path, resnet_cfg.model, resnet_cfg.train.loss, device)
    params_resnet = sum(p.numel() for p in model_resnet.parameters())

    # 2. Load SegFormer-B0
    segformer_ckpt_path = output_dir / "best_segformer_mit_b0_focaldice.pth"
    segformer_cfg = Config()
    segformer_cfg.model.arch = "segformer"
    segformer_cfg.model.encoder_name = "mit_b0"
    segformer_cfg.model.encoder_weights = "imagenet"
    segformer_cfg.train.loss = "focal_dice"
    model_segformer, _ = load_model_for_inference(segformer_ckpt_path, segformer_cfg.model, segformer_cfg.train.loss, device)
    params_segformer = sum(p.numel() for p in model_segformer.parameters())

    models = {
        "ResNet34_UNet": {
            "model": model_resnet,
            "arch": "unet_smp",
            "backbone": "resnet34",
            "params": params_resnet,
            "threshold": 0.5,
        },
        "SegFormer_B0": {
            "model": model_segformer,
            "arch": "segformer",
            "backbone": "mit_b0 (Mix Transformer)",
            "params": params_segformer,
            "threshold": 0.5,
        },
    }

    records = []

    # Load and evaluate tiles
    for idx, fname in enumerate(test_filenames, start=1):
        img_p = dataset_dir / "images_png" / fname
        mask_p = dataset_dir / "masks_png" / fname
        if not img_p.exists() or not mask_p.exists():
            continue

        with Image.open(img_p) as im:
            rgb = np.asarray(im.convert("RGB"))
        with Image.open(mask_p) as m:
            gt_mask = np.asarray(m.convert("L")) > 0

        y_true = gt_mask.ravel().astype(int)

        for m_name, m_info in models.items():
            prob = predict_probability_map(m_info["model"], rgb, device)
            y_pred = (prob >= m_info["threshold"]).ravel().astype(int)

            iou = float(jaccard_score(y_true, y_pred, zero_division=1.0))
            dice = float(f1_score(y_true, y_pred, zero_division=1.0))
            prec = float(precision_score(y_true, y_pred, zero_division=1.0))
            rec = float(recall_score(y_true, y_pred, zero_division=1.0))

            records.append({
                "tile": fname,
                "model": m_name,
                "iou": iou,
                "dice": dice,
                "precision": prec,
                "recall": rec,
            })

        if idx % 5 == 0 or idx == len(test_filenames):
            print(f"  [{idx}/{len(test_filenames)}] Evaluated {fname}")

    df_res = pd.DataFrame(records)
    df_res.to_csv(output_dir / "per_tile_comparison.csv", index=False)

    # Benchmark latency on a standard 256x256 tile
    sample_tile = np.zeros((256, 256, 3), dtype=np.uint8)
    speed_resnet = benchmark_inference_speed(model_resnet, sample_tile, device)
    speed_segformer = benchmark_inference_speed(model_segformer, sample_tile, device)

    agg_summary = {}
    for m_name, m_info in models.items():
        sub = df_res[df_res["model"] == m_name]
        speed = speed_resnet if m_name == "ResNet34_UNet" else speed_segformer
        agg_summary[m_name] = {
            "architecture": m_info["arch"],
            "backbone": m_info["backbone"],
            "parameters": m_info["params"],
            "param_reduction_pct": float((1.0 - m_info["params"] / params_resnet) * 100.0),
            "test_tiles_evaluated": len(sub),
            "mean_iou": float(sub["iou"].mean()),
            "std_iou": float(sub["iou"].std()),
            "mean_dice": float(sub["dice"].mean()),
            "std_dice": float(sub["dice"].std()),
            "mean_precision": float(sub["precision"].mean()),
            "mean_recall": float(sub["recall"].mean()),
            "inference_time_ms": speed["mean_ms"],
            "inference_std_ms": speed["std_ms"],
        }

    with open(output_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(agg_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 5: MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<16} {'Params':<10} {'IoU':<8} {'Dice':<8} {'Precision':<10} {'Recall':<8} {'Latency (ms)':<12}")
    print("-" * 80)
    for m_name, data in agg_summary.items():
        print(f"{m_name:<16} {data['parameters'] / 1e6:>5.2f}M    {data['mean_iou']:>6.4f} {data['mean_dice']:>6.4f} {data['mean_precision']:>8.4f}  {data['mean_recall']:>6.4f}  {data['inference_time_ms']:>6.1f} ms")
    print("=" * 80)

    return agg_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="RouteResilience")
    parser.add_argument("--output_dir", default="outputs/phase5_transformer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train_subset", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--test_tiles", type=int, default=24)
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / args.dataset_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Phase 5 executing on device: {device}")

    # 1. Train SegFormer-B0
    train_segformer_b0(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        device=device,
        train_subset_size=args.train_subset,
        epochs=args.epochs,
    )

    # 2. Evaluate and compare both models on held-out test set
    evaluate_models_on_test_set(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        device=device,
        test_tile_count=args.test_tiles,
    )


if __name__ == "__main__":
    main()
