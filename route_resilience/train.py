"""
Training + evaluation harness. One implementation shared by every
architecture, so a comparison between models changes only cfg.model.arch
and nothing about how training/evaluation is run.

Threshold selection is ALWAYS done on the validation loader
(`tune_threshold`), never on the test loader -- the reviewed notebook
had one run that tuned the decision threshold against test_loader,
which invalidates the test metric it later reported.
"""

import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from torch.utils.data import DataLoader


def get_device(cfg_device: str = "cuda"):
    return torch.device(cfg_device if torch.cuda.is_available() else "cpu")


def make_loaders(train_ds, val_ds, test_ds, cfg):
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def _iou_from_probs(probs, masks, threshold):
    preds = (probs >= threshold).float()
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss, running_iou = 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, masks)
        probs = torch.sigmoid(outputs)
        running_loss += loss.item()
        running_iou += _iou_from_probs(probs, masks, threshold)
    n = len(loader)
    return running_loss / n, running_iou / n


@torch.no_grad()
def validate_with_dice(model, loader, criterion, device, threshold=0.5):
    """Validation loss plus thresholded IoU and Dice for run progress logs."""
    model.eval()
    running_loss = running_iou = running_dice = 0.0
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, masks)
        probs = torch.sigmoid(outputs)
        preds = (probs >= threshold).float()
        intersection = (preds * masks).sum()
        pred_and_target = preds.sum() + masks.sum()
        dice = 1.0 if pred_and_target == 0 else (2 * intersection / pred_and_target).item()
        running_loss += loss.item()
        running_iou += _iou_from_probs(probs, masks, threshold)
        running_dice += dice
    n = len(loader)
    return running_loss / n, running_iou / n, running_dice / n


def _rng_state():
    """Capture RNG state so an interrupted run can continue reproducibly."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _save_checkpoint(path, state):
    """Atomically replace a checkpoint, avoiding a half-written Drive file."""
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _torch_load_checkpoint(path, device):
    """Load trusted local checkpoints on both older and newer PyTorch versions."""
    try:
        # Newer PyTorch defaults to weights_only=True, which cannot restore
        # Python/NumPy RNG state stored in a full training checkpoint.
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        return torch.load(path, map_location=device)


def load_resume_checkpoint(path, model, optimizer, scheduler, device):
    """Restore all training state and return the point at which to continue."""
    checkpoint = _torch_load_checkpoint(path, device)
    required = {
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "epoch", "best_val_iou", "epochs_without_improvement", "config",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            f"{path} is not a resume checkpoint; missing: {', '.join(sorted(missing))}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    rng = checkpoint.get("rng_state")
    if rng is not None:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and "torch_cuda" in rng:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    return checkpoint


def load_model_checkpoint(path, model, device):
    """Load either a new full checkpoint or a legacy weights-only checkpoint."""
    checkpoint = _torch_load_checkpoint(path, device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))


def fit(
    model, train_loader, val_loader, criterion, optimizer, scheduler, cfg, device,
    best_ckpt_path, latest_ckpt_path, checkpoint_config, start_epoch=0,
    best_val_iou=0.0, epochs_without_improvement=0, history=None,
    progress_callback=None,
):
    """
    Fair, shared training loop with early stopping on validation IoU.
    Saves separate best and latest full checkpoints after every epoch.
    `start_epoch` is zero-based (so 39 continues with human epoch 40).
    """
    if history is None:
        history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    else:
        history.setdefault("val_dice", [])

    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou, val_dice = validate_with_dice(
            model, val_loader, criterion, device, cfg.threshold
        )
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        improved = val_iou > best_val_iou
        if improved:
            best_val_iou = val_iou
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch + 1,
            "best_val_iou": best_val_iou,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "config": checkpoint_config,
            "rng_state": _rng_state(),
        }
        if improved:
            _save_checkpoint(best_ckpt_path, checkpoint)
        _save_checkpoint(latest_ckpt_path, checkpoint)

        print(
            f"Epoch [{epoch+1}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_iou={val_iou:.4f} val_dice={val_dice:.4f}"
            f"{'  * best' if improved else ''}"
        )
        if progress_callback is not None:
            progress_callback(epoch + 1, train_loss, val_loss, val_iou, val_dice)

        if epochs_without_improvement >= cfg.patience:
            print(f"Early stopping at epoch {epoch+1} (best val IoU {best_val_iou:.4f})")
            break

    return best_val_iou, history


@torch.no_grad()
def tune_threshold(model, val_loader, device, thresholds=None):
    """
    Sweep thresholds on the VALIDATION loader only. Returns the
    threshold that maximizes IoU plus the full sweep table, so you can
    see how sensitive the choice is rather than trusting one number.
    """
    if thresholds is None:
        thresholds = np.arange(0.20, 0.65, 0.05)

    model.eval()
    results = []
    for t in thresholds:
        tp = fp = fn = 0
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            probs = torch.sigmoid(model(images))
            preds = (probs >= t).float().view(-1)
            m = masks.view(-1)
            tp += ((preds == 1) & (m == 1)).sum().item()
            fp += ((preds == 1) & (m == 0)).sum().item()
            fn += ((preds == 0) & (m == 1)).sum().item()
        iou = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        results.append({"threshold": float(t), "iou": iou, "dice": dice,
                         "precision": precision, "recall": recall})

    best = max(results, key=lambda r: r["iou"])
    return best["threshold"], results


@torch.no_grad()
def evaluate_test(model, test_loader, device, threshold):
    """Final, single evaluation on the held-out test set at the val-tuned threshold."""
    model.eval()
    all_preds, all_targets = [], []
    for images, masks in test_loader:
        images = images.to(device)
        probs = torch.sigmoid(model(images))
        preds = (probs >= threshold).float()
        all_preds.append(preds.cpu().numpy().flatten())
        all_targets.append(masks.numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return {
        "iou": jaccard_score(all_targets, all_preds, zero_division=0),
        "dice": f1_score(all_targets, all_preds, zero_division=0),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "threshold_used": threshold,
    }
