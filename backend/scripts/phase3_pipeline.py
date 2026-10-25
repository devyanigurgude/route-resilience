"""Phase 3 — Sequential train + evaluate pipeline.

Runs entirely in one process:
  1. Train Model B (coarse_dropout, grouped split)
  2. Train Model C (multi_type, grouped split)
  3. Run 3-way ablation evaluation (Model A vs B vs C)
  4. Write summary JSON + per-tile CSV

Designed to run unattended. All stdout/stderr goes to the terminal.
Usage (from workspace root):
    python backend/scripts/phase3_pipeline.py --epochs 2 --batch-size 16

Safety guarantees:
  - Never overwrites best_unet_smp_resnet34_focaldice.pth (Model A).
  - Never touches the test set during training or threshold tuning.
  - All outputs go to backend/outputs/phase3_ablation/.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# ─── training imports ──────────────────────────────────────────────────────────
from dataclasses import asdict
import numpy as np
import pandas as pd
from PIL import Image
import networkx as nx

from route_resilience.config import Config, GraphConfig, ModelConfig
from route_resilience.data import build_datasets
from route_resilience.evaluate_topology import simplified_apls
from route_resilience.graph_extraction import mask_to_skeleton, skeleton_to_graph
from route_resilience.losses import build_loss
from route_resilience.models import build_model
from route_resilience.occlusion_simulation import generate_road_centered_occlusions
from route_resilience.real_graph_pipeline import (
    graph_from_probability_map,
    load_model_for_inference,
    load_rgb_image,
    predict_probability_map,
)
from route_resilience.topology_matching import evaluate_connection_recovery
from route_resilience.train import (
    fit,
    get_device,
    load_model_checkpoint,
    make_loaders,
    tune_threshold,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_variant(
    aug_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device,
    init_checkpoint: Path,
    checkpoint_dir: Path,
) -> dict:
    """Train one variant. Returns a dict with checkpoint_path, threshold, history."""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"TRAINING: aug_type='{aug_type}'  epochs={epochs}  lr={lr}  seed={seed}")
    print(bar)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"unet_smp_resnet34_grouped_{aug_type}"
    best_ckpt = str(checkpoint_dir / f"best_{stem}.pth")
    latest_ckpt = str(checkpoint_dir / f"latest_{stem}.pth")

    cfg = Config()
    cfg.data.dataset_dir = str(WORKSPACE_ROOT / "RouteResilience")
    cfg.data.split_csv = "data_split_grouped.csv"
    cfg.data.batch_size = batch_size
    cfg.data.num_workers = 0
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
    train_ds, val_ds, test_ds = build_datasets(cfg.data)
    train_loader, val_loader, _ = make_loaders(train_ds, val_ds, test_ds, cfg.data)
    print(f"Partitions: Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)} [UNTOUCHED]")

    model = build_model(cfg.model).to(device)
    if init_checkpoint.exists():
        print(f"  Init weights from: {init_checkpoint.name}")
        load_model_checkpoint(str(init_checkpoint), model, device)

    criterion = build_loss(cfg.train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    best_val_iou, history = fit(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        cfg.train, device, best_ckpt, latest_ckpt, asdict(cfg),
    )
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/60:.1f} min. Best Val IoU={best_val_iou:.4f}")

    # Threshold tuning on VALIDATION only
    print("Tuning threshold on VALIDATION split only...")
    load_model_checkpoint(best_ckpt, model, device)
    model.eval()
    best_threshold, sweep = tune_threshold(model, val_loader, device)
    print(f"Val-tuned threshold: {best_threshold:.2f}")

    threshold_path = checkpoint_dir / f"threshold_{stem}.json"
    payload = {
        "threshold": float(best_threshold),
        "selection_split": "validation",
        "checkpoint": f"best_{stem}.pth",
        "aug_type": aug_type,
        "epochs_trained": len(history["train_loss"]),
        "best_val_iou": float(best_val_iou),
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "elapsed_seconds": round(elapsed, 1),
        "threshold_sweep": sweep,
        "history": history,
    }
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Threshold + metadata saved to {threshold_path}")

    return {
        "checkpoint_path": Path(best_ckpt),
        "threshold": float(best_threshold),
        "best_val_iou": float(best_val_iou),
        "history": history,
        "epochs_trained": len(history["train_loss"]),
        "elapsed_seconds": round(elapsed, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def calc_pixel_metrics(pred_bin, gt_mask):
    pred = pred_bin.astype(bool)
    gt = gt_mask.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    union = tp + fp + fn
    iou = tp / union if union > 0 else (1.0 if gt.sum() == 0 else 0.0)
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else (1.0 if gt.sum() == 0 else 0.0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"iou": iou, "dice": dice, "precision": prec, "recall": rec}


def graph_stats(G):
    if G is None or G.number_of_nodes() == 0:
        return {"node_count": 0, "edge_count": 0, "num_components": 0, "lcr": 0.0}
    comps = list(nx.connected_components(G))
    n = G.number_of_nodes()
    lcr = max(len(c) for c in comps) / n if n > 0 else 0.0
    return {"node_count": n, "edge_count": G.number_of_edges(), "num_components": len(comps), "lcr": float(lcr)}


def gt_graph(gt_mask):
    skel = mask_to_skeleton(gt_mask > 0)
    if skel.sum() == 0:
        return None
    return skeleton_to_graph(skel, prob_map=None, min_edge_length_px=3.0)


def get_test_tiles(dataset_dir, split_csv, sample_size, seed):
    df = pd.read_csv(split_csv)
    test_all = df[df["split"] == "test"].copy()
    valid = test_all[test_all["road_percentage"] >= 0.2].copy()
    if len(valid) <= sample_size:
        return valid
    valid["bin"] = pd.qcut(valid["road_percentage"], q=min(5, sample_size), labels=False, duplicates="drop")
    rows = []
    for _, grp in valid.groupby("bin"):
        n = max(1, int(round(sample_size * len(grp) / len(valid))))
        rows.append(grp.sample(n=min(len(grp), n), random_state=seed))
    return pd.concat(rows).head(sample_size).copy()


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

OCCLUSION_SCENARIOS = [
    {"name": "clean_baseline",       "pattern": None,               "severity": "none",   "num": 0, "size": (0,  0),  "fill": None},
    {"name": "rect_dropout_mild",    "pattern": "rectangular_block","severity": "mild",   "num": 1, "size": (16, 22), "fill": "black"},
    {"name": "rect_dropout_severe",  "pattern": "rectangular_block","severity": "severe", "num": 2, "size": (26, 36), "fill": "black"},
    {"name": "canopy_mild",          "pattern": "canopy_cluster",   "severity": "mild",   "num": 1, "size": (16, 22), "fill": "canopy"},
    {"name": "canopy_severe",        "pattern": "canopy_cluster",   "severity": "severe", "num": 2, "size": (26, 36), "fill": "canopy"},
    {"name": "cloud_mild",           "pattern": "procedural_cloud", "severity": "mild",   "num": 1, "size": (16, 22), "fill": "cloud"},
    {"name": "cloud_severe",         "pattern": "procedural_cloud", "severity": "severe", "num": 2, "size": (26, 36), "fill": "cloud"},
]

GRAPH_CFG = GraphConfig(
    max_gap_px=15.0,
    min_gap_confidence=0.15,
    enable_occlusion_recovery=True,
    max_occlusion_gap_px=30.0,
    min_occlusion_confidence=0.05,
    occlusion_heading_threshold=0.60,
    max_lateral_offset_px=4.0,
    enable_cloud_occlusion_recovery=True,
    max_cloud_occlusion_gap_px=240.0,
    cloud_heading_threshold=0.78,
)


def evaluate_model_on_scenario(model, threshold, test_tiles, dataset_dir, device, seed):
    """Return dict[scenario_name] -> scenario_result_dict and list of per-tile rows."""
    per_tile_rows = []
    by_scenario = {}

    for sc_idx, sc in enumerate(OCCLUSION_SCENARIOS):
        sc_name = sc["name"]
        accum = {
            "iou": [], "dice": [], "precision": [], "recall": [],
            "occ_road_recall": [],
            "unhealed_comp": [], "healed_comp": [], "comp_reduction": [],
            "unhealed_lcr": [], "healed_lcr": [],
            "unhealed_apls": [], "healed_apls": [],
            "tp": [], "fp": [],
        }

        for t_idx, row in enumerate(test_tiles.itertuples(index=False)):
            filename = row.filename
            img_path = dataset_dir / "images_png" / filename
            mask_path = dataset_dir / "masks_png" / filename

            orig_rgb = load_rgb_image(img_path)
            gt_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            G_gt = gt_graph(gt_mask)

            if sc["pattern"] is not None:
                tile_seed = seed + t_idx * 137 + sc_idx * 19
                eval_rgb, occ_mask, _ = generate_road_centered_occlusions(
                    orig_rgb, gt_mask,
                    pattern=sc["pattern"], num_occlusions=sc["num"],
                    size_range=sc["size"], fill_mode=sc["fill"], seed=tile_seed,
                )
            else:
                eval_rgb = orig_rgb
                occ_mask = np.zeros(gt_mask.shape, dtype=bool)

            prob_map = predict_probability_map(model, eval_rgb, device)
            pred_bin = prob_map >= threshold

            pm = calc_pixel_metrics(pred_bin, gt_mask)
            for k in ("iou", "dice", "precision", "recall"):
                accum[k].append(pm[k])

            occ_road_total = int(np.logical_and(gt_mask, occ_mask).sum())
            if occ_road_total > 0:
                occ_road_pred = int(np.logical_and(pred_bin & gt_mask, occ_mask).sum())
                accum["occ_road_recall"].append(occ_road_pred / occ_road_total)
            else:
                accum["occ_road_recall"].append(1.0 if sc["pattern"] is None else 0.0)

            pipe_res = graph_from_probability_map(
                prob_map, rgb_image=eval_rgb, threshold=threshold, graph_cfg=GRAPH_CFG
            )
            G_u = pipe_res["graph_before"]
            G_h = pipe_res["graph_after"]
            su = graph_stats(G_u)
            sh = graph_stats(G_h)
            accum["unhealed_comp"].append(su["num_components"])
            accum["healed_comp"].append(sh["num_components"])
            accum["comp_reduction"].append(su["num_components"] - sh["num_components"])
            accum["unhealed_lcr"].append(su["lcr"])
            accum["healed_lcr"].append(sh["lcr"])

            if G_gt is not None and G_gt.number_of_nodes() >= 2:
                apls_u = simplified_apls(G_u, G_gt)["score"]
                apls_h = simplified_apls(G_h, G_gt)["score"]
            else:
                apls_u = apls_h = 0.0
            accum["unhealed_apls"].append(apls_u)
            accum["healed_apls"].append(apls_h)

            rec = evaluate_connection_recovery(
                G_u, G_h, G_gt, gt_mask,
                occlusion_mask=occ_mask if sc["pattern"] is not None else None,
            )
            tp = rec["true_recovered_connections"]
            fp = rec["false_recovered_connections"]
            accum["tp"].append(tp)
            accum["fp"].append(fp)

            per_tile_rows.append({
                "scenario": sc_name,
                "pattern": sc["pattern"] or "none",
                "severity": sc["severity"],
                "filename": filename,
                "iou": pm["iou"], "dice": pm["dice"],
                "precision": pm["precision"], "recall": pm["recall"],
                "occ_road_recall": accum["occ_road_recall"][-1],
                "unhealed_comp": su["num_components"],
                "healed_comp": sh["num_components"],
                "unhealed_lcr": su["lcr"],
                "healed_lcr": sh["lcr"],
                "unhealed_apls": apls_u,
                "healed_apls": apls_h,
                "tp": tp, "fp": fp,
                "tot_cand": tp + fp,
            })

        # ── Aggregation with both micro and macro precision ──
        sum_tp = int(sum(accum["tp"]))
        sum_fp = int(sum(accum["fp"]))
        sum_cand = sum_tp + sum_fp
        micro_prec = (sum_tp / sum_cand) if sum_cand > 0 else None  # None = no candidates at all

        # Macro-active: only tiles where at least one candidate was evaluated
        active_precs = [tp / (tp + fp) for tp, fp in zip(accum["tp"], accum["fp"]) if (tp + fp) > 0]
        macro_active_prec = float(np.mean(active_precs)) if active_precs else None
        active_tile_count = len(active_precs)

        # Legacy macro (tiles with 0 candidates count as 1.0) — kept for comparability with Phase 2
        legacy_precs = [tp / (tp + fp) if (tp + fp) > 0 else 1.0 for tp, fp in zip(accum["tp"], accum["fp"])]
        legacy_macro_prec = float(np.mean(legacy_precs))

        by_scenario[sc_name] = {
            "mean_iou":              float(np.mean(accum["iou"])),
            "mean_dice":             float(np.mean(accum["dice"])),
            "mean_precision":        float(np.mean(accum["precision"])),
            "mean_recall":           float(np.mean(accum["recall"])),
            "mean_occ_road_recall":  float(np.mean(accum["occ_road_recall"])),
            "mean_unhealed_comp":    float(np.mean(accum["unhealed_comp"])),
            "mean_healed_comp":      float(np.mean(accum["healed_comp"])),
            "mean_comp_reduction":   float(np.mean(accum["comp_reduction"])),
            "mean_unhealed_lcr":     float(np.mean(accum["unhealed_lcr"])),
            "mean_healed_lcr":       float(np.mean(accum["healed_lcr"])),
            "mean_unhealed_apls":    float(np.mean(accum["unhealed_apls"])),
            "mean_healed_apls":      float(np.mean(accum["healed_apls"])),
            "mean_apls_gain":        float(np.mean(accum["healed_apls"]) - np.mean(accum["unhealed_apls"])),
            # ── Recovery precision (3 formulations) ──────────────────────────
            "total_tp":              sum_tp,
            "total_fp":              sum_fp,
            "total_candidates":      sum_cand,
            # Micro-pooled: sum(TP) / sum(TP+FP). Correct event-level precision.
            "micro_recovery_precision":        micro_prec,
            # Macro-active: mean over tiles with at least one candidate. No 1.0 inflation.
            "macro_active_recovery_precision": macro_active_prec,
            "active_tile_count":               active_tile_count,
            # Legacy: mean including 1.0 for empty tiles. Kept for Phase 2 comparability.
            "legacy_macro_recovery_precision": legacy_macro_prec,
            "total_tiles_evaluated":           len(test_tiles),
        }

    return by_scenario, per_tile_rows


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Train B & C, then 3-way ablation evaluation.")
    parser.add_argument("--epochs",      type=int,   default=2,      help="Training epochs per variant")
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      type=str,   default="cpu")
    parser.add_argument("--sample-size", type=int,   default=30,     help="Test tiles for evaluation")
    parser.add_argument("--skip-train",  action="store_true",        help="Skip training; use existing checkpoints")
    args = parser.parse_args()

    device = get_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    print(f"Device: {device}  Threads: {torch.get_num_threads()}")

    checkpoint_dir = BACKEND_DIR / "checkpoints"
    output_dir     = BACKEND_DIR / "outputs" / "phase3_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    init_ckpt = checkpoint_dir / "best_unet_smp_resnet34_focaldice.pth"

    training_meta = {}

    # ── 1. Train Model B ──────────────────────────────────────────────────────
    b_ckpt = checkpoint_dir / "best_unet_smp_resnet34_grouped_coarse_dropout.pth"
    if not args.skip_train or not b_ckpt.exists():
        print("\n[STEP 1/3] Training Model B (coarse_dropout, grouped split)...")
        meta_b = train_variant(
            aug_type="coarse_dropout", epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, seed=args.seed, device=device,
            init_checkpoint=init_ckpt, checkpoint_dir=checkpoint_dir,
        )
        training_meta["model_b"] = meta_b
    else:
        print(f"\n[STEP 1/3] Skipping Model B training — checkpoint exists: {b_ckpt.name}")
        t_path = checkpoint_dir / "threshold_unet_smp_resnet34_grouped_coarse_dropout.json"
        thresh = 0.50
        if t_path.exists():
            with open(t_path) as f:
                thresh = float(json.load(f).get("threshold", thresh))
        training_meta["model_b"] = {"checkpoint_path": b_ckpt, "threshold": thresh, "skipped": True}

    # ── 2. Train Model C ──────────────────────────────────────────────────────
    c_ckpt = checkpoint_dir / "best_unet_smp_resnet34_grouped_multi_type.pth"
    if not args.skip_train or not c_ckpt.exists():
        print("\n[STEP 2/3] Training Model C (multi_type, grouped split)...")
        meta_c = train_variant(
            aug_type="multi_type", epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, seed=args.seed, device=device,
            init_checkpoint=init_ckpt, checkpoint_dir=checkpoint_dir,
        )
        training_meta["model_c"] = meta_c
    else:
        print(f"\n[STEP 2/3] Skipping Model C training — checkpoint exists: {c_ckpt.name}")
        t_path = checkpoint_dir / "threshold_unet_smp_resnet34_grouped_multi_type.json"
        thresh = 0.50
        if t_path.exists():
            with open(t_path) as f:
                thresh = float(json.load(f).get("threshold", thresh))
        training_meta["model_c"] = {"checkpoint_path": c_ckpt, "threshold": thresh, "skipped": True}

    # ── 3. Load all three models for evaluation ───────────────────────────────
    print("\n[STEP 3/3] Running 3-way evaluation (Model A vs B vs C)...")

    model_cfg = ModelConfig(arch="unet_smp", encoder_name="resnet34", encoder_weights="imagenet",
                            in_channels=3, classes=1, dropout_p=0.2)

    models_eval = [
        {
            "id":   "model_a_baseline",
            "name": "Model A – Baseline (random split, coarse_dropout)",
            "ckpt": checkpoint_dir / "best_unet_smp_resnet34_focaldice.pth",
            "tfile": checkpoint_dir / "threshold_unet_smp_resnet34_focaldice.json",
            "default_thresh": 0.50,
        },
        {
            "id":   "model_b_grouped_coarse",
            "name": "Model B – Grouped split, coarse_dropout",
            "ckpt": b_ckpt,
            "tfile": checkpoint_dir / "threshold_unet_smp_resnet34_grouped_coarse_dropout.json",
            "default_thresh": 0.50,
        },
        {
            "id":   "model_c_grouped_multi",
            "name": "Model C – Grouped split, multi_type occlusion",
            "ckpt": c_ckpt,
            "tfile": checkpoint_dir / "threshold_unet_smp_resnet34_grouped_multi_type.json",
            "default_thresh": 0.50,
        },
    ]

    # Confirm checkpoints exist
    for m in models_eval:
        if not m["ckpt"].exists():
            print(f"ERROR: Missing checkpoint {m['ckpt']}. Aborting evaluation.")
            sys.exit(1)

    # Select test tiles (identical for all models)
    dataset_dir = WORKSPACE_ROOT / "RouteResilience"
    split_csv   = dataset_dir / "data_split_grouped.csv"
    test_tiles  = get_test_tiles(dataset_dir, split_csv, args.sample_size, args.seed)
    print(f"Evaluating on {len(test_tiles)} held-out spatial test tiles (seed={args.seed})")

    all_summary = {}
    all_per_tile = []

    for m in models_eval:
        thresh = m["default_thresh"]
        if m["tfile"].exists():
            with open(m["tfile"]) as f:
                thresh = float(json.load(f).get("threshold", thresh))

        print(f"\n  Loading: {m['name']} | ckpt={m['ckpt'].name} | threshold={thresh:.2f}")
        model_obj, meta = load_model_for_inference(m["ckpt"], model_cfg, "focal_dice", device)

        ckpt_epoch = meta.get("epoch", "unknown")
        ckpt_val_iou = meta.get("best_val_iou", None)
        print(f"    Checkpoint epoch={ckpt_epoch}  best_val_iou={ckpt_val_iou}")

        print(f"  Evaluating across {len(OCCLUSION_SCENARIOS)} scenarios × {len(test_tiles)} tiles...")
        by_sc, per_tile = evaluate_model_on_scenario(
            model_obj, thresh, test_tiles, dataset_dir, device, args.seed
        )

        all_summary[m["id"]] = {
            "model_id":           m["id"],
            "model_name":         m["name"],
            "checkpoint":         m["ckpt"].name,
            "checkpoint_epoch":   ckpt_epoch,
            "checkpoint_val_iou": ckpt_val_iou,
            "threshold":          thresh,
            "threshold_source":   "validation_tuned" if m["tfile"].exists() else "default_0.50",
            "test_tiles":         len(test_tiles),
            "seed":               args.seed,
            "scenarios":          by_sc,
        }

        for r in per_tile:
            r["model_id"] = m["id"]
            all_per_tile.append(r)

    # ── 4. Persist outputs ────────────────────────────────────────────────────
    pipeline_meta = {
        "phase": 3,
        "description": "3-way ablation: Model A (baseline) vs Model B (grouped+coarse) vs Model C (grouped+multi_type)",
        "training_meta": {
            k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items() if kk != "history"}
            for k, v in training_meta.items()
        },
        "evaluation_models": [m["id"] for m in models_eval],
        "evaluation_test_tiles": len(test_tiles),
        "evaluation_seed": args.seed,
        "precision_notes": (
            "micro_recovery_precision = sum(TP)/sum(TP+FP) across tiles; tiles with 0 candidates "
            "NOT counted as 100%. macro_active_recovery_precision = mean over tiles where at least "
            "one candidate was evaluated (no 1.0 inflation). legacy_macro_recovery_precision matches "
            "Phase 2 methodology for direct comparability."
        ),
    }

    summary_path = output_dir / "phase3_ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({**pipeline_meta, "results": all_summary}, f, indent=2)
    print(f"\nSummary JSON  → {summary_path}")

    df_per = pd.DataFrame(all_per_tile)
    csv_path = output_dir / "phase3_ablation_per_tile.csv"
    df_per.to_csv(csv_path, index=False)
    print(f"Per-tile CSV  → {csv_path}")

    # ── 5. Print compact results table ────────────────────────────────────────
    KEY_SCENARIOS = ["clean_baseline", "rect_dropout_severe", "canopy_severe", "cloud_severe"]
    print("\n" + "=" * 120)
    print(f"{'PHASE 3 ABLATION — KEY METRICS TABLE':^120}")
    print("=" * 120)
    hdr = f"{'Scenario':<24} {'Model':<18} {'IoU':>6} {'Dice':>6} {'OccRec':>7} {'APLS↑':>7} {'APLSGn':>7} {'ΣTP':>5} {'ΣFP':>5} {'MicroPrc':>9} {'MacroActv':>10}"
    print(hdr)
    print("-" * 120)
    for sc_name in KEY_SCENARIOS:
        for m in models_eval:
            sc = all_summary[m["id"]]["scenarios"].get(sc_name, {})
            if not sc:
                continue
            micro = sc["micro_recovery_precision"]
            macro_a = sc["macro_active_recovery_precision"]
            print(
                f"{sc_name:<24} {m['id']:<18} "
                f"{sc['mean_iou']:>6.4f} {sc['mean_dice']:>6.4f} "
                f"{sc['mean_occ_road_recall']:>7.4f} "
                f"{sc['mean_healed_apls']:>7.4f} {sc['mean_apls_gain']:>+7.4f} "
                f"{sc['total_tp']:>5d} {sc['total_fp']:>5d} "
                f"{(micro if micro is not None else float('nan')):>9.4f} "
                f"{(macro_a if macro_a is not None else float('nan')):>10.4f}"
            )
        print()

    print("=" * 120)
    print(f"\nAll outputs in: {output_dir}")
    print("Phase 3 pipeline complete.")


if __name__ == "__main__":
    main()

