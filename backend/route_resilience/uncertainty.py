"""
MC-Dropout inference and calibration checking.

This is arguably the single most important module in the whole
pipeline (Phase 0, Part 2/3): every downstream confidence-aware graph
and criticality claim depends on these numbers actually meaning what
they claim to mean. Run `expected_calibration_error` BEFORE trusting
any confidence-weighted result downstream (H3, and it gates H4).
"""

from typing import Tuple

import numpy as np
import torch

from .models import enable_mc_dropout


@torch.no_grad()
def mc_dropout_predict(model, image: torch.Tensor, n_samples: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    image: a single preprocessed tensor, shape (C, H, W) or (1, C, H, W).
    Returns (mean_prob, std_prob), each shape (H, W) -- mean is the
    prediction to threshold, std is the per-pixel epistemic uncertainty
    to carry forward into graph confidence.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    device = next(model.parameters()).device
    image = image.to(device)

    enable_mc_dropout(model)  # eval() everywhere except Dropout layers

    samples = []
    for _ in range(n_samples):
        logits = model(image)
        probs = torch.sigmoid(logits)
        samples.append(probs.squeeze(0).squeeze(0).cpu().numpy())

    samples = np.stack(samples, axis=0)  # (n_samples, H, W)
    mean_prob = samples.mean(axis=0)
    std_prob = samples.std(axis=0)

    model.eval()  # restore full eval mode for any subsequent deterministic call
    return mean_prob, std_prob


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """
    Standard binned ECE for a binary segmentation task, computed over
    flattened pixels. Confidence per pixel is taken as max(p, 1-p)
    (i.e. confidence in whatever class was predicted), accuracy is
    whether the thresholded prediction (>0.5) matched the label.

    Returns ECE plus the per-bin table so you can plot a reliability
    diagram rather than trusting a single scalar.
    """
    probs = probs.flatten()
    labels = labels.flatten()

    confidence = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(np.float32)
    correct = (predictions == labels).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_table = []

    n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        count = in_bin.sum()
        if count == 0:
            bin_table.append({"range": (float(lo), float(hi)), "count": 0,
                               "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = confidence[in_bin].mean()
        acc = correct[in_bin].mean()
        ece += (count / n) * abs(avg_conf - acc)
        bin_table.append({"range": (float(lo), float(hi)), "count": int(count),
                           "avg_confidence": float(avg_conf), "accuracy": float(acc)})

    return {"ece": float(ece), "bins": bin_table}
