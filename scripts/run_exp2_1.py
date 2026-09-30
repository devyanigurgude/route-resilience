"""
EXP-2.1 — Centerline vs Dilated Corridor Target
================================================

Run controlled A/B experiment:
  A. Model trained on original 1–2 px centerline ground truth
  B. Model trained on dynamically-dilated ~5 px corridor target
     (dilation applied only during training; original masks untouched)

Both models use:
  - same architecture  : unet_smp / resnet34 / imagenet
  - same dataset split : data_split.csv
  - same optimizer     : AdamW, lr=1e-4, wd=1e-4 + CosineAnnealingLR
  - same augmentation  : flip / rotate / brightness / occlusion
  - same loss          : FocalDiceLoss (alpha=0.75, gamma=2.0) from Phase 1
  - same epoch budget  : up to --epochs, early-stopping patience --patience
  - same threshold     : tuned on validation split (not test)

Inference:
  - Model A: prediction → skeleton → graph
  - Model B: prediction → skeletonize corridor → skeleton → graph

Metrics (identical evaluation harness for A and B):
  Pixel : IoU, Dice, Precision, Recall  (evaluated against GT centerline)
  Graph : n_components, largest_cc_ratio, APLS, miss_rate, OD reachability

Usage:
    python scripts/run_exp2_1.py --dataset_dir ./RouteResilience
    python scripts/run_exp2_1.py --dataset_dir ./RouteResilience --epochs 30 --patience 7 --max_visuals 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.config import Config
from route_resilience.data import RoadDataset, build_datasets, build_transforms
from route_resilience.exp2_corridor import (
    DilatedCorridorDataset,
    TopologyEvaluator,
    save_comparison_panel,
    skeletonize_prediction,
)
from route_resilience.graph_extraction import mask_to_skeleton, skeleton_to_graph
from route_resilience.losses import build_loss
from route_resilience.models import build_model
from route_resilience.train import (
    fit,
    get_device,
    load_model_checkpoint,
    load_resume_checkpoint,
    make_loaders,
    tune_threshold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _checkpoint_stem(variant: str) -> str:
    return f"exp2_1_{variant}_unet_smp_resnet34_focaldice"


def _make_cfg(args) -> Config:
    cfg = Config()
    cfg.data.dataset_dir = args.dataset_dir
    cfg.data.use_occlusion_augmentation = True  # required — do not change
    cfg.data.num_workers = args.num_workers
    cfg.model.arch = "unet_smp"
    cfg.model.encoder_name = "resnet34"
    cfg.model.encoder_weights = "imagenet"
    cfg.train.epochs = args.epochs
    cfg.train.patience = args.patience
    cfg.train.loss = "focal_dice"
    cfg.train.focal_alpha = 0.75
    cfg.train.focal_gamma = 2.0
    cfg.train.lr = 1e-4
    cfg.train.weight_decay = 1e-4
    cfg.train.seed = 42
    return cfg


def _seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _Tee:
    """Mirror a text stream to the console and the experiment log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


# ---------------------------------------------------------------------------
# Pixel metrics — evaluated against GT CENTERLINE for both models
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_pixel(model, loader, device, threshold) -> Dict:
    model.eval()
    all_preds, all_targets = [], []
    for images, masks in loader:
        images = images.to(device)
        probs = torch.sigmoid(model(images))
        preds = (probs >= threshold).float().cpu().numpy().flatten()
        targs = masks.numpy().flatten()
        all_preds.append(preds)
        all_targets.append(targs)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return {
        "iou": float(jaccard_score(all_targets, all_preds, zero_division=0)),
        "dice": float(f1_score(all_targets, all_preds, zero_division=0)),
        "precision": float(precision_score(all_targets, all_preds, zero_division=0)),
        "recall": float(recall_score(all_targets, all_preds, zero_division=0)),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# Topology evaluation — full test-set sweep
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_topology(
    model,
    test_loader,
    device,
    threshold,
    variant: str,
    cfg: Config,
    output_dir: Path,
    max_visuals: int,
    dilation_radius: int,
) -> Dict:
    """
    For each test image:
      1. Run model inference → probability map → binary prediction
      2. For corridor (B): skeletonize the predicted corridor first
      3. Build graph from skeleton
      4. Compute topology metrics
      5. Save visual panel (up to max_visuals panels)

    Returns aggregated topology metrics dict.
    """
    evaluator = TopologyEvaluator(snap_radius=10.0, n_pairs=200, seed=42, min_edge_px=3.0)
    is_corridor = (variant == "B")

    vis_dir = output_dir / "visuals" / variant
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path(cfg.data.dataset_dir) / cfg.data.image_subdir
    mask_dir = Path(cfg.data.dataset_dir) / cfg.data.mask_subdir

    # We need filenames — rebuild from the test dataframe
    import pandas as pd
    split_path = Path(cfg.data.dataset_dir) / cfg.data.split_csv
    test_df = pd.read_csv(split_path)
    test_df = test_df[test_df["split"] == "test"].reset_index(drop=True)

    model.eval()
    topo_records = []
    vis_count = 0

    for batch_idx, (images, gt_masks) in enumerate(test_loader):
        images_dev = images.to(device)
        probs_batch = torch.sigmoid(model(images_dev))  # (B,1,H,W)
        batch_size = images.shape[0]

        for b in range(batch_size):
            global_idx = batch_idx * test_loader.batch_size + b
            if global_idx >= len(test_df):
                break

            filename = test_df.iloc[global_idx]["filename"]
            prob_map = probs_batch[b, 0].cpu().numpy()  # (H, W)
            pred_binary = (prob_map >= threshold).astype(np.uint8)
            gt_binary = gt_masks[b, 0].numpy().astype(np.uint8)  # original centerline

            # Corridor post-processing: skeletonize corridor before graph
            if is_corridor:
                pred_for_graph = skeletonize_prediction(pred_binary).astype(np.uint8)
            else:
                pred_for_graph = pred_binary

            # Topology metrics
            topo = evaluator.evaluate(pred_for_graph, gt_binary, prob_map=prob_map)
            topo["filename"] = filename
            topo["variant"] = variant
            topo_records.append(topo)

            # Visual panel (first max_visuals samples)
            if vis_count < max_visuals:
                try:
                    rgb_img = np.array(
                        Image.open(image_dir / filename).convert("RGB"), dtype=np.uint8
                    )
                except Exception:
                    rgb_img = np.zeros((256, 256, 3), dtype=np.uint8)

                pred_skel = mask_to_skeleton(pred_for_graph)

                # Build graph for overlay (use prob_map for confidence)
                G_pred = None
                try:
                    G_pred = skeleton_to_graph(pred_skel, prob_map=prob_map, min_edge_length_px=3.0)
                except Exception:
                    pass

                panel_path = str(vis_dir / f"panel_{global_idx:04d}_{Path(filename).stem}.png")
                save_comparison_panel(
                    rgb=rgb_img,
                    gt_mask=gt_binary,
                    pred_mask=pred_binary,
                    pred_skeleton=pred_skel.astype(np.uint8),
                    G_pred=G_pred,
                    save_path=panel_path,
                    title=f"EXP-2.1 Model {variant} | {filename} | thresh={threshold:.2f}",
                )
                vis_count += 1

    if not topo_records:
        return {"error": "no_topology_records"}

    # Aggregate
    def _mean(key):
        vals = [r[key] for r in topo_records if key in r and r[key] is not None]
        return float(np.mean(vals)) if vals else 0.0

    def _median(key):
        vals = [r[key] for r in topo_records if key in r and r[key] is not None]
        return float(np.median(vals)) if vals else 0.0

    agg = {
        "n_images": len(topo_records),
        "n_components_mean": _mean("n_components"),
        "n_components_median": _median("n_components"),
        "largest_cc_ratio_mean": _mean("largest_cc_ratio"),
        "apls_score_mean": _mean("apls_score"),
        "apls_score_median": _median("apls_score"),
        "miss_rate_mean": _mean("miss_rate"),
        "od_reachability_mean": _mean("od_reachability"),
        "od_reachability_median": _median("od_reachability"),
    }

    # Save per-image records
    import pandas as pd
    records_df = pd.DataFrame(topo_records)
    records_df.to_csv(output_dir / f"topo_per_image_{variant}.csv", index=False)

    return agg


# ---------------------------------------------------------------------------
# Single training + evaluation run
# ---------------------------------------------------------------------------

def run_variant(
    variant: str,
    cfg: Config,
    device: torch.device,
    ckpt_dir: Path,
    output_dir: Path,
    max_visuals: int,
    dilation_radius: int,
    skip_training: bool = False,
) -> Dict:
    """Train one variant (A or B) and return all metrics."""
    print(f"\n{'='*70}")
    print(f"  EXP-2.1 — Variant {variant} ({'Centerline' if variant=='A' else 'Dilated Corridor'})")
    print(f"{'='*70}")

    _seed_everything(cfg.train.seed)

    stem = _checkpoint_stem(variant)
    best_ckpt = ckpt_dir / f"best_{stem}.pth"
    latest_ckpt = ckpt_dir / f"latest_{stem}.pth"

    # ---- Build datasets -----------------------------------------------
    # Both variants share the same val/test datasets (GT centerline targets)
    # so pixel metrics and topology are directly comparable.
    train_ds_base, val_ds, test_ds = build_datasets(cfg.data)

    if variant == "B":
        # Wrap train split with dynamic dilation
        train_ds = DilatedCorridorDataset(train_ds_base, dilation_radius=dilation_radius)
        print(f"  Corridor dilation radius : {dilation_radius} px (disk)")
        print(f"  Train targets            : dynamically dilated ~{1 + 2*dilation_radius} px corridor")
    else:
        train_ds = train_ds_base
        print(f"  Train targets            : original 1–2 px centerline (unchanged)")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )
    # Val/test always use the plain centerline targets
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )
    test_loader_plain = DataLoader(
        test_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )

    print(f"  Train / val / test sizes : {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # ---- Model, optimizer, loss ---------------------------------------
    model = build_model(cfg.model).to(device)
    criterion = build_loss(cfg.train)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    checkpoint_config = asdict(cfg)
    checkpoint_config["exp2_1_variant"] = variant

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model params             : {param_count:,}")
    print(f"  Loss                     : {cfg.train.loss} (alpha={cfg.train.focal_alpha}, gamma={cfg.train.focal_gamma})")
    print(f"  Optimizer                : AdamW lr={cfg.train.lr} wd={cfg.train.weight_decay}")

    # ---- Training -----------------------------------------------------
    t0 = time.time()
    if skip_training and (best_ckpt.exists() or latest_ckpt.exists()):
        checkpoint_to_load = best_ckpt if best_ckpt.exists() else latest_ckpt
        print(f"  Skipping training — loading existing checkpoint: {checkpoint_to_load}")
        load_model_checkpoint(str(checkpoint_to_load), model, device)
        best_val_iou = 0.0  # unknown without training
        history = {}
    else:
        resume_state = {}
        if latest_ckpt.exists():
            resume_state = load_resume_checkpoint(
                str(latest_ckpt), model, optimizer, scheduler, device
            )
            if resume_state.get("config") != checkpoint_config:
                raise ValueError(
                    "Latest EXP-2.1 checkpoint configuration does not match this run. "
                    "Resume with the original EXP-2.1 settings."
                )
            print(
                f"  Resuming Variant {variant} from completed epoch "
                f"{resume_state['epoch']}/{cfg.train.epochs}"
            )

        def _progress(epoch, train_loss, val_loss, val_iou, val_dice):
            elapsed = time.time() - t0
            print(
                f"  Progress | variant={variant} | epoch={epoch}/{cfg.train.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"IoU={val_iou:.4f} | Dice={val_dice:.4f} | elapsed={elapsed:.1f}s"
            )

        best_val_iou, history = fit(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            cfg.train, device,
            best_ckpt_path=str(best_ckpt),
            latest_ckpt_path=str(latest_ckpt),
            checkpoint_config=checkpoint_config,
            start_epoch=resume_state.get("epoch", 0),
            best_val_iou=resume_state.get("best_val_iou", 0.0),
            epochs_without_improvement=resume_state.get("epochs_without_improvement", 0),
            history=resume_state.get("history"),
            progress_callback=_progress,
        )
        print(f"\n  Training done — best val IoU: {best_val_iou:.4f}")
        checkpoint_to_load = best_ckpt if best_ckpt.exists() else latest_ckpt
        load_model_checkpoint(str(checkpoint_to_load), model, device)

    train_time_s = time.time() - t0

    # ---- Threshold tuning (validation split, GT centerline) -----------
    best_threshold, sweep = tune_threshold(model, val_loader, device)
    print(f"  Best threshold (val-tuned): {best_threshold:.2f}")

    # Save threshold
    thresh_path = output_dir / f"threshold_{variant}.json"
    with open(thresh_path, "w") as f:
        json.dump({"threshold": best_threshold, "sweep": sweep, "variant": variant}, f, indent=2)

    # ---- Pixel metrics (test set, GT centerline) ----------------------
    pixel_metrics = _evaluate_pixel(model, test_loader_plain, device, best_threshold)
    print(f"\n  Pixel metrics (vs GT centerline):")
    for k, v in pixel_metrics.items():
        print(f"    {k:12s}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # ---- Topology metrics (test set) ----------------------------------
    print(f"\n  Computing topology metrics ({'with corridor skeletonization' if variant=='B' else 'standard'})...")
    topo_metrics = _evaluate_topology(
        model=model,
        test_loader=test_loader_plain,
        device=device,
        threshold=best_threshold,
        variant=variant,
        cfg=cfg,
        output_dir=output_dir,
        max_visuals=max_visuals,
        dilation_radius=dilation_radius,
    )
    print(f"  Topology metrics:")
    for k, v in topo_metrics.items():
        if isinstance(v, float):
            print(f"    {k:30s}: {v:.4f}")
        else:
            print(f"    {k:30s}: {v}")

    return {
        "variant": variant,
        "target": "centerline" if variant == "A" else f"dilated_corridor_r{dilation_radius}",
        "best_val_iou": float(best_val_iou),
        "train_time_seconds": float(train_time_s),
        "epochs_run": len(history.get("train_loss", [])),
        "pixel_metrics": pixel_metrics,
        "topology_metrics": topo_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EXP-2.1: Centerline vs Dilated Corridor Target experiment"
    )
    parser.add_argument("--dataset_dir", default="./RouteResilience",
                        help="Path to the RouteResilience dataset directory.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs per variant (default: 50).")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stopping patience (default: 10).")
    parser.add_argument("--dilation_radius", type=int, default=2,
                        help="Disk radius for corridor dilation (default: 2 → ~5 px corridor).")
    parser.add_argument("--max_visuals", type=int, default=30,
                        help="Max visual comparison panels to save per variant.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (default: 0 for Windows compatibility).")
    parser.add_argument("--ckpt_dir", default="./checkpoints/exp2_1",
                        help="Directory for model checkpoints.")
    parser.add_argument("--output_dir", default="./outputs/exp2_1",
                        help="Output directory for results and visuals.")
    parser.add_argument("--variants", default="AB",
                        help="Which variants to run: A, B, or AB (default: AB).")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and load existing best checkpoints (evaluation only).")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "exp2_1.log"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, log_file)
    sys.stderr = _Tee(original_stderr, log_file)
    print(f"\n{'='*70}\nEXP-2.1 run started: {time.strftime('%Y-%m-%d %H:%M:%S')}\nLog: {log_path}\n{'='*70}")

    cfg = _make_cfg(args)
    device = get_device(cfg.device)
    print(f"Device: {device}")

    print("\n" + "="*70)
    print("  EXP-2.1 CONFIGURATION")
    print("="*70)
    print(f"  Architecture  : unet_smp / resnet34 / imagenet")
    print(f"  Loss          : FocalDiceLoss (alpha=0.75, gamma=2.0)")
    print(f"  Optimizer     : AdamW lr=1e-4 wd=1e-4 + CosineAnnealingLR")
    print(f"  Augmentation  : flip + rotate + brightness + occlusion (ON)")
    print(f"  Epochs budget : {args.epochs}, patience={args.patience}")
    print(f"  Corridor dil  : disk(r={args.dilation_radius}) -> ~{1+2*args.dilation_radius} px wide")
    print(f"  Dataset       : {args.dataset_dir}")
    print(f"  Output        : {output_dir}")

    all_results = []

    if "A" in args.variants:
        result_a = run_variant(
            variant="A",
            cfg=cfg,
            device=device,
            ckpt_dir=ckpt_dir,
            output_dir=output_dir,
            max_visuals=args.max_visuals,
            dilation_radius=args.dilation_radius,
            skip_training=args.skip_training,
        )
        all_results.append(result_a)

    if "B" in args.variants:
        result_b = run_variant(
            variant="B",
            cfg=cfg,
            device=device,
            ckpt_dir=ckpt_dir,
            output_dir=output_dir,
            max_visuals=args.max_visuals,
            dilation_radius=args.dilation_radius,
            skip_training=args.skip_training,
        )
        all_results.append(result_b)

    # ---- Print comparison table ---------------------------------------
    if len(all_results) == 2:
        ra, rb = all_results[0], all_results[1]
        print("\n" + "="*70)
        print("  EXP-2.1 FINAL COMPARISON REPORT")
        print("="*70)

        def _delta(key, sub=None):
            va = ra[sub][key] if sub else ra[key]
            vb = rb[sub][key] if sub else rb[key]
            delta = vb - va
            sign = "+" if delta >= 0 else ""
            return f"{va:.4f}  →  {vb:.4f}  ({sign}{delta:+.4f})"

        print(f"\n  {'Metric':<36}  {'Model A (centerline)':<14}  {'Model B (corridor)':<14}  {'Δ (B−A)'}")
        print(f"  {'-'*70}")

        pix_keys = ["iou", "dice", "precision", "recall"]
        for k in pix_keys:
            va = ra["pixel_metrics"][k]
            vb = rb["pixel_metrics"][k]
            d = vb - va
            print(f"  {k:<36}  {va:.4f}          {vb:.4f}          {'+' if d>=0 else ''}{d:+.4f}")

        print()
        topo_keys = [
            ("n_components_mean", "n_components (mean, lower=better)"),
            ("largest_cc_ratio_mean", "largest_cc_ratio (higher=better)"),
            ("apls_score_mean", "APLS score (higher=better)"),
            ("miss_rate_mean", "miss_rate (lower=better)"),
            ("od_reachability_mean", "OD reachability (higher=better)"),
        ]
        for key, label in topo_keys:
            va = ra["topology_metrics"].get(key, float("nan"))
            vb = rb["topology_metrics"].get(key, float("nan"))
            d = vb - va
            print(f"  {label:<36}  {va:.4f}          {vb:.4f}          {'+' if d>=0 else ''}{d:+.4f}")

        print(f"\n  {'Epochs run':<36}  {ra['epochs_run']:<14}  {rb['epochs_run']:<14}")
        print(f"  {'Train time (s)':<36}  {ra['train_time_seconds']:<14.0f}  {rb['train_time_seconds']:<14.0f}")

        # ---- Winner recommendation ------------------------------------
        print("\n" + "-"*70)
        print("  PRIMARY DECISION CRITERION: network topology, NOT IoU/Dice")
        print("-"*70)

        a_topo = ra["topology_metrics"]
        b_topo = rb["topology_metrics"]

        topo_wins_b = 0
        topo_wins_a = 0

        # Higher APLS → better
        if b_topo.get("apls_score_mean", 0) > a_topo.get("apls_score_mean", 0):
            topo_wins_b += 1
        else:
            topo_wins_a += 1

        # Lower miss rate → better
        if b_topo.get("miss_rate_mean", 1) < a_topo.get("miss_rate_mean", 1):
            topo_wins_b += 1
        else:
            topo_wins_a += 1

        # Higher OD reachability → better
        if b_topo.get("od_reachability_mean", 0) > a_topo.get("od_reachability_mean", 0):
            topo_wins_b += 1
        else:
            topo_wins_a += 1

        # Higher largest_cc_ratio → better
        if b_topo.get("largest_cc_ratio_mean", 0) > a_topo.get("largest_cc_ratio_mean", 0):
            topo_wins_b += 1
        else:
            topo_wins_a += 1

        # Lower n_components → better (fewer fragments)
        if b_topo.get("n_components_mean", 999) < a_topo.get("n_components_mean", 999):
            topo_wins_b += 1
        else:
            topo_wins_a += 1

        print(f"\n  Topology score (# of 5 criteria won): A={topo_wins_a}  B={topo_wins_b}")

        if topo_wins_b > topo_wins_a:
            winner = "B (Dilated Corridor)"
            rationale = "Corridor training improves road connectivity metrics."
        elif topo_wins_a > topo_wins_b:
            winner = "A (Centerline)"
            rationale = "Centerline training already produces better-connected topology."
        else:
            winner = "TIE — require manual inspection of visual panels"
            rationale = "Both models equally split the topology criteria."

        print(f"  Recommended next baseline: {winner}")
        print(f"  Rationale: {rationale}")

    # ---- Save full results JSON ----------------------------------------
    report = {
        "experiment": "EXP-2.1",
        "description": "Centerline vs Dilated Corridor Target",
        "config": {
            "arch": "unet_smp",
            "encoder": "resnet34",
            "encoder_weights": "imagenet",
            "loss": "focal_dice",
            "focal_alpha": 0.75,
            "focal_gamma": 2.0,
            "optimizer": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "augmentation": "flip+rotate+brightness+occlusion",
            "dilation_radius": args.dilation_radius,
            "corridor_width_px": 1 + 2 * args.dilation_radius,
            "epochs_budget": args.epochs,
            "patience": args.patience,
            "seed": 42,
            "threshold_tuning": "validation split",
        },
        "variants": all_results,
    }

    report_path = output_dir / "exp2_1_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Full report saved to: {report_path}")
    print(f"  Visual panels in    : {output_dir / 'visuals'}")
    print("\nDO NOT PROCEED TO PHASE 3 — EXP-2.1 complete.")


if __name__ == "__main__":
    main()
