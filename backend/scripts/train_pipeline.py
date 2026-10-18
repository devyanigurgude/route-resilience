"""
End-to-end training script -- run this on Colab/a GPU machine with the
actual dataset in place. This replaces the ad-hoc notebook cells: same
config, same loss, same augmentation, same epoch budget for every
architecture, so a comparison between them is actually valid.

Usage:
    python scripts/train_pipeline.py --arch unet
    python scripts/train_pipeline.py --arch deeplabv3plus
    python scripts/train_pipeline.py --arch unet_smp --encoder resnet34

Requires: torch, torchvision, segmentation-models-pytorch, albumentations
    pip install torch torchvision segmentation-models-pytorch albumentations --break-system-packages
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from route_resilience.config import Config
from route_resilience.data import build_datasets
from route_resilience.losses import build_loss
from route_resilience.models import build_model
from route_resilience.train import (
    evaluate_test,
    fit,
    get_device,
    load_model_checkpoint,
    load_resume_checkpoint,
    make_loaders,
    tune_threshold,
)


def checkpoint_stem(model_cfg, loss_name):
    """Name checkpoints after the model that is actually instantiated."""
    loss_label = loss_name.replace("_", "")
    if model_cfg.arch == "unet":
        return f"mc_dropout_unet_{loss_label}"
    return f"{model_cfg.arch}_{model_cfg.encoder_name}_{loss_label}"


def print_model_summary(model, model_cfg, train_cfg, use_occlusion_augmentation):
    """Print enough detail to make an accidental architecture mismatch obvious."""
    class_name = f"{type(model).__module__}.{type(model).__name__}"
    if model_cfg.arch == "unet":
        encoder = "none (custom U-Net)"
        encoder_weights = "n/a"
        dropout = model_cfg.dropout_p
    else:
        encoder = model_cfg.encoder_name
        encoder_weights = model_cfg.encoder_weights
        dropout = "none (stock segmentation_models_pytorch model)"

    print("Model configuration:")
    print(f"  architecture class: {class_name}")
    print(f"  encoder: {encoder}")
    print(f"  encoder weights: {encoder_weights}")
    print(f"  loss: {train_cfg.loss}")
    print(f"  occlusion augmentation: {use_occlusion_augmentation}")
    print(f"  dropout: {dropout}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="/content/RouteResilience")
    parser.add_argument("--arch", default="unet", choices=["unet", "unet_smp", "deeplabv3plus", "segformer"])
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder_weights", default="imagenet", choices=["imagenet", "none"],
                        help="Encoder initialization for SMP architectures.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 2)
    parser.add_argument("--loss", default="focal_dice", choices=["bce_dice", "focal_dice"])
    parser.add_argument("--no_occlusion_aug", action="store_true",
                         help="Disable synthetic occlusion augmentation -- use only for an "
                              "explicit with/without-occlusion-aug ablation, not by default.")
    parser.add_argument("--ckpt_dir", default="./checkpoints")
    parser.add_argument("--resume", default=None,
                        help="Path to a latest_*.pth checkpoint from an interrupted run.")
    args = parser.parse_args()

    cfg = Config()
    cfg.data.dataset_dir = args.dataset_dir
    cfg.data.use_occlusion_augmentation = not args.no_occlusion_aug
    cfg.model.arch = args.arch
    if args.arch == "segformer" and args.encoder == "resnet34":
        cfg.model.encoder_name = "mit_b0"
    else:
        cfg.model.encoder_name = args.encoder

    cfg.model.encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights
    cfg.train.epochs = args.epochs
    cfg.train.patience = args.patience
    cfg.data.num_workers = args.num_workers
    cfg.train.loss = args.loss

    os.makedirs(args.ckpt_dir, exist_ok=True)
    run_name = checkpoint_stem(cfg.model, cfg.train.loss)
    best_ckpt_path = os.path.join(args.ckpt_dir, f"best_{run_name}.pth")
    latest_ckpt_path = os.path.join(args.ckpt_dir, f"latest_{run_name}.pth")

    device = get_device(cfg.device)
    print(f"Device: {device}")

    train_ds, val_ds, test_ds = build_datasets(cfg.data)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg.data)
    print(f"Train/val/test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    import torch  # local import so this file can be *parsed* without torch installed

    model = build_model(cfg.model).to(device)
    print_model_summary(model, cfg.model, cfg.train, cfg.data.use_occlusion_augmentation)
    criterion = build_loss(cfg.train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    checkpoint_config = asdict(cfg)

    start_epoch = 0
    best_val_iou = 0.0
    epochs_without_improvement = 0
    history = None
    if args.resume:
        resume = load_resume_checkpoint(args.resume, model, optimizer, scheduler, device)
        if resume["config"] != checkpoint_config:
            raise ValueError(
                "Resume configuration differs from the checkpoint. Use the same "
                "architecture, data, augmentation, and training arguments."
            )
        start_epoch = resume["epoch"]
        best_val_iou = resume["best_val_iou"]
        epochs_without_improvement = resume["epochs_without_improvement"]
        history = resume.get("history")
        print(f"Resuming after epoch {start_epoch}: best val IoU={best_val_iou:.4f}, "
              f"epochs without improvement={epochs_without_improvement}")

    best_val_iou, history = fit(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        cfg.train, device, best_ckpt_path, latest_ckpt_path, checkpoint_config,
        start_epoch=start_epoch, best_val_iou=best_val_iou,
        epochs_without_improvement=epochs_without_improvement, history=history,
    )
    print(f"\nBest validation IoU: {best_val_iou:.4f}")

    load_model_checkpoint(best_ckpt_path, model, device)
    model.eval()

    best_threshold, sweep = tune_threshold(model, val_loader, device)
    print(f"Best threshold (tuned on VALIDATION, not test): {best_threshold:.2f}")

    # Keep the validation-selected operating point next to its checkpoint so
    # downstream real-image inference never has to tune on the target image.
    threshold_path = os.path.join(args.ckpt_dir, f"threshold_{run_name}.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": best_threshold,
            "selection_split": "validation",
            "checkpoint": os.path.basename(best_ckpt_path),
            "config": checkpoint_config,
            "threshold_sweep": sweep,
        }, f, indent=2)
    print(f"Saved validation-selected threshold to {threshold_path}")

    test_metrics = evaluate_test(model, test_loader, device, best_threshold)
    print("\n===== FINAL TEST METRICS (val-tuned threshold) =====")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    results_path = os.path.join(args.ckpt_dir, f"results_{args.arch}_{args.encoder}.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": {"arch": args.arch, "encoder": args.encoder, "loss": args.loss,
                       "occlusion_aug": cfg.data.use_occlusion_augmentation,
                       "epochs_run": len(history["train_loss"])},
            "best_val_iou": best_val_iou,
            "threshold_sweep": sweep,
            "test_metrics": test_metrics,
        }, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
