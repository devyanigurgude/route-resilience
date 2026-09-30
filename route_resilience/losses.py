"""
Loss functions. One definition of each, imported everywhere -- the
reviewed notebook had DoubleConv/UNet/FocalDiceLoss each defined two or
three times in different cells, which is how it's easy to accidentally
train against a stale class definition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        if logits.ndim > 1:
            batch_size = logits.shape[0]
            probs_flat = probs.view(batch_size, -1)
            targets_flat = targets.view(batch_size, -1)
            intersection = (probs_flat * targets_flat).sum(dim=1)
            cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
            denominator = torch.clamp(cardinality + self.smooth, min=1e-7)
            dice = (2.0 * intersection + self.smooth) / denominator
            return (1.0 - dice).mean()
        else:
            probs_flat = probs.view(-1)
            targets_flat = targets.view(-1)
            intersection = (probs_flat * targets_flat).sum()
            cardinality = probs_flat.sum() + targets_flat.sum()
            denominator = max(float(cardinality + self.smooth), 1e-7)
            dice = (2.0 * intersection + self.smooth) / denominator
            return 1.0 - dice


class FocalLoss(nn.Module):
    """
    alpha upweights the road (positive) class -- necessary because roads
    are a small minority of pixels in almost any tile. gamma downweights
    already-easy (well-classified) pixels so training focuses on the
    genuinely ambiguous/occluded ones, which is directly relevant to
    this project rather than incidental.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        else:
            alpha_t = 1.0
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.dice(logits, targets)


def build_loss(cfg):
    if cfg.loss == "bce_dice":
        return BCEDiceLoss()
    if cfg.loss == "focal_dice":
        return FocalDiceLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    raise ValueError(f"Unknown loss: {cfg.loss}")
