"""Phase 3 — Train Grouped-Split Baselines and Multi-Type Occlusion Models.

Trains:
  1. Model B: U-Net ResNet-34 trained on data_split_grouped.csv with CoarseDropout.
  2. Model C: U-Net ResNet-34 trained on data_split_grouped.csv with MultiTypeOcclusionAugmentation.

Features:
  - Uses strictly training partition of RouteResilience/data_split_grouped.csv.
  - Checkpoint selection and threshold tuning use validation set only.
  - Held-out test set is kept completely untouched during training and tuning.
  - Saves separate checkpoints and threshold JSON files without overwriting legacy checkpoints.

Usage:
    python backend/scripts/train_phase3_ablation.py --epochs 2 --batch-size 16
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from route_resilience.config import Config
from route_resilience.data import build_datasets
from route_resilience.losses import build_loss
from route_resilience.models import build_model
from route_resilience.train import (
    fit,
    get_device,
    load_model_checkpoint,
    make_loaders,
    tune_threshold,
)


def train_variant(
    aug_type: str,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 5e-5,
    seed: int = 42,
    device_name: str = "cpu",
    init_checkpoint: Path = None,
    output_dir: Path = None,
) -> Path:
    print(f"\n{'='*70}")
    print(f"TRAINING PHASE 3 U-NET VARIANT: aug_type='{aug_type}' on data_split_grouped.csv")
    print(f"{'='*70}")

    if output_dir is None:
        output_dir = BACKEND_DIR / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    print(f"Device: {device} (threads={torch.get_num_threads()})")

    cfg = Config()
    cfg.data.dataset_dir = str(WORKSPACE_ROOT / "RouteResilience")
    cfg.data.split_csv = "data_split_grouped.csv"
    cfg.data.batch_size = batch_size
    cfg.data.num_workers = 0 if os.name == "nt" else 2
    cfg.data.use_occlusion_augmentation = True
    cfg.data.occlusion_aug_type = aug_type
    cfg.model.arch = "unet_smp"
    cfg.model.encoder_name = "resnet34"
    cfg.model.encoder_weights = "imagenet"
    cfg.train.epochs = epochs
    cfg.train.lr = lr
    cfg.train.seed = seed
    cfg.train.loss = "focal_dice"

    torch.manual_seed(seed)

    print("Building datasets from RouteResilience/data_split_grouped.csv...")
    train_ds, val_ds, test_ds = build_datasets(cfg.data)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg.data)
    print(f"Dataset partitions: Train={len(train_ds)}, Val={len(val_ds)}, Test (Untouched)={len(test_ds)}")

    model = build_model(cfg.model).to(device)

    # Initialize weights from baseline if available for fast CPU convergence
    if init_checkpoint and init_checkpoint.exists():
        print(f"Initializing model weights from {init_checkpoint.name}...")
        load_model_checkpoint(str(init_checkpoint), model, device)

    criterion = build_loss(cfg.train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    checkpoint_config = asdict(cfg)

    stem = f"unet_smp_resnet34_grouped_{aug_type}"
    best_ckpt_path = str(output_dir / f"best_{stem}.pth")
    latest_ckpt_path = str(output_dir / f"latest_{stem}.pth")

    print(f"Starting training for {epochs} epochs on CPU...")
    best_val_iou, history = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        cfg.train,
        device,
        best_ckpt_path,
        latest_ckpt_path,
        checkpoint_config,
    )
    print(f"\nTraining completed! Best Validation IoU: {best_val_iou:.4f}")

    # Tune threshold strictly on VALIDATION split
    print("Tuning decision threshold strictly on VALIDATION split (test set remains untouched)...")
    load_model_checkpoint(best_ckpt_path, model, device)
    model.eval()
    best_threshold, sweep = tune_threshold(model, val_loader, device)
    print(f"Validation-tuned threshold: {best_threshold:.2f}")

    threshold_path = output_dir / f"threshold_{stem}.json"
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": float(best_threshold),
            "selection_split": "validation",
            "checkpoint": f"best_{stem}.pth",
            "aug_type": aug_type,
            "config": checkpoint_config,
            "threshold_sweep": sweep,
            "history": history,
        }, f, indent=2)

    print(f"Saved threshold & training metadata to {threshold_path}")
    return Path(best_ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Train Phase 3 U-Net Models.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-both", action="store_true", help="Train both coarse_dropout and multi_type sequentially.")
    parser.add_argument("--variant", choices=["coarse_dropout", "multi_type"], default="multi_type")
    args = parser.parse_args()

    init_ckpt = BACKEND_DIR / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth"

    if args.train_both:
        train_variant("coarse_dropout", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed, device_name=args.device, init_checkpoint=init_ckpt)
        train_variant("multi_type", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed, device_name=args.device, init_checkpoint=init_ckpt)
    else:
        train_variant(args.variant, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed, device_name=args.device, init_checkpoint=init_ckpt)


if __name__ == "__main__":
    main()
