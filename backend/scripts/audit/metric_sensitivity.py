import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt

from route_resilience.config import Config
from route_resilience.data import preprocess_rgb_image
from route_resilience.models import build_model

PROJECT_ROOT = "."
CHECKPOINT_PATH = "checkpoints/best_unet_smp_resnet34_focaldice.pth"
RESULTS_JSON_PATH = "checkpoints/results_unet_smp_resnet34.json"
SPLIT_CSV = "RouteResilience/data_split.csv"
IMAGE_DIR = "RouteResilience/images_png"
MASK_DIR = "RouteResilience/masks_png"

def evaluate_test_metrics_with_tolerance():
    if not os.path.exists(CHECKPOINT_PATH):
        print("CHECKPOINT_NOT_FOUND: checkpoints/best_unet_smp_resnet34_focaldice.pth does not exist.")
        return

    print("Loading test split and checkpoint...")
    df = pd.read_csv(SPLIT_CSV)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Total test tiles: {len(test_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load results JSON to get optimal threshold
    with open(RESULTS_JSON_PATH, "r") as f:
        res = json.load(f)
    threshold = res.get("test_metrics", {}).get("threshold_used", 0.5)
    print(f"Optimal threshold from training run: {threshold}")

    # Build model and load state_dict
    cfg = Config()
    cfg.model.arch = "unet_smp"
    cfg.model.encoder_weights = None  # learned weights are loaded from checkpoint
    model = build_model(cfg.model).to(device)

    from route_resilience.train import _torch_load_checkpoint
    checkpoint = _torch_load_checkpoint(CHECKPOINT_PATH, device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    exact_ious = []
    tol3_ious = []
    exact_precisions = []
    exact_recalls = []
    tol3_precisions = []
    tol3_recalls = []

    print("Running inference across test set...")
    with torch.no_grad():
        for idx in range(len(test_df)):
            fname = test_df.iloc[idx]["filename"]
            img_path = os.path.join(IMAGE_DIR, fname)
            mask_path = os.path.join(MASK_DIR, fname)

            image = Image.open(img_path).convert("RGB")
            mask = np.array(Image.open(mask_path).convert("L")) == 255  # boolean (256, 256)

            tensor = preprocess_rgb_image(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_bin = probs >= threshold  # boolean (256, 256)

            # 1. Exact-pixel metrics
            tp_exact = np.logical_and(pred_bin, mask).sum()
            fp_exact = np.logical_and(pred_bin, ~mask).sum()
            fn_exact = np.logical_and(~pred_bin, mask).sum()
            union_exact = tp_exact + fp_exact + fn_exact

            iou_exact = (tp_exact / union_exact) if union_exact > 0 else 1.0
            p_exact = (tp_exact / (tp_exact + fp_exact)) if (tp_exact + fp_exact) > 0 else 1.0
            r_exact = (tp_exact / (tp_exact + fn_exact)) if (tp_exact + fn_exact) > 0 else 1.0

            exact_ious.append(iou_exact)
            exact_precisions.append(p_exact)
            exact_recalls.append(r_exact)

            # 2. 3-pixel tolerance metrics (buffer/distance transform <= 3px)
            # A predicted road pixel is considered a true positive if within 3px of GT
            # A GT road pixel is considered recalled if within 3px of prediction
            if mask.any() and pred_bin.any():
                dist_to_gt = distance_transform_edt(~mask)
                dist_to_pred = distance_transform_edt(~pred_bin)

                # Matched prediction pixels (within 3px of GT)
                tp_pred = np.logical_and(pred_bin, dist_to_gt <= 3.0).sum()
                # Matched GT pixels (within 3px of prediction)
                tp_gt = np.logical_and(mask, dist_to_pred <= 3.0).sum()

                fp_tol = np.logical_and(pred_bin, dist_to_gt > 3.0).sum()
                fn_tol = np.logical_and(mask, dist_to_pred > 3.0).sum()

                # Standard relaxed precision & recall
                prec_tol = tp_pred / pred_bin.sum() if pred_bin.sum() > 0 else 1.0
                rec_tol = tp_gt / mask.sum() if mask.sum() > 0 else 1.0

                # Relaxed IoU: harmonic or buffer-based union
                # Buffer-based IoU: |Pred_buffer_3 ∩ GT_buffer_3| / |Pred_buffer_3 ∪ GT_buffer_3|
                # or Jaccard from relaxed TP: TP_relaxed / (TP_relaxed + FP_tol + FN_tol)
                tp_avg = (tp_pred + tp_gt) / 2.0
                denom = tp_avg + fp_tol + fn_tol
                iou_tol = tp_avg / denom if denom > 0 else 1.0
            elif not mask.any() and not pred_bin.any():
                iou_tol = 1.0
                prec_tol = 1.0
                rec_tol = 1.0
            else:
                iou_tol = 0.0
                prec_tol = 0.0 if pred_bin.any() else 1.0
                rec_tol = 0.0 if mask.any() else 1.0

            tol3_ious.append(iou_tol)
            tol3_precisions.append(prec_tol)
            tol3_recalls.append(rec_tol)

            if (idx + 1) % 200 == 0 or (idx + 1) == len(test_df):
                print(f"Evaluated {idx + 1}/{len(test_df)} test tiles...")

    print("\n=== METRIC SENSITIVITY RESULTS ===")
    print(f"Test Tiles Count: {len(test_df)}")
    print(f"Threshold Used:   {threshold:.4f}")
    print(f"Exact Mean IoU:          {np.mean(exact_ious):.4f} (Median: {np.median(exact_ious):.4f})")
    print(f"Exact Mean Precision:    {np.mean(exact_precisions):.4f}")
    print(f"Exact Mean Recall:       {np.mean(exact_recalls):.4f}")
    print(f"3-px Tolerance Mean IoU: {np.mean(tol3_ious):.4f} (Median: {np.median(tol3_ious):.4f})")
    print(f"3-px Tolerance Precision:{np.mean(tol3_precisions):.4f}")
    print(f"3-px Tolerance Recall:   {np.mean(tol3_recalls):.4f}")
    print(f"Relative IoU Gain under 3-px Slack: {((np.mean(tol3_ious) - np.mean(exact_ious))/np.mean(exact_ious))*100:.2f}%")

if __name__ == "__main__":
    evaluate_test_metrics_with_tolerance()
