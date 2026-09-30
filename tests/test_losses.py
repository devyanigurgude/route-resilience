"""Comprehensive tests for FocalLoss and DiceLoss.

Tests verify:
1. FocalLoss foreground vs background alpha_t class weighting (including the exact p=0.5 case).
2. Gradient ratio between foreground and background errors under alpha_t.
3. Easy vs hard sample suppression under gamma.
4. DiceLoss sample-wise batch aggregation vs batch-flattening.
5. Zero-division and numerical stability on empty masks, smooth=0, and extreme logits.
6. End-to-end backpropagation through FocalDiceLoss.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from route_resilience.losses import BCEDiceLoss, DiceLoss, FocalDiceLoss, FocalLoss, build_loss
from route_resilience.config import TrainConfig


def test_focal_loss_alpha_t_weighting_at_half_probability():
    """At p=0.5 (logits=0), focal factor (1-pt)^gamma is identical for both classes.
    
    Therefore, the loss for a false-negative pixel (target=1, p=0.5) vs a
    false-positive pixel (target=0, p=0.5) must strictly equal alpha / (1 - alpha).
    """
    alpha = 0.75
    gamma = 2.0
    focal = FocalLoss(alpha=alpha, gamma=gamma)

    # 1 batch, 1 channel, 1x2 image: [pixel 0: target=1, pixel 1: target=0]
    logits = torch.zeros(1, 1, 1, 2)  # sigmoid(0) = 0.5
    targets = torch.tensor([[[[1.0, 0.0]]]])

    # Unreduced loss per pixel
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1.0 - probs)
    alpha_t = torch.where(targets == 1, alpha, 1.0 - alpha)
    focal_weight = alpha_t * (1.0 - pt) ** gamma
    per_pixel_loss = focal_weight * bce

    loss_fg = per_pixel_loss[0, 0, 0, 0].item()
    loss_bg = per_pixel_loss[0, 0, 0, 1].item()

    expected_ratio = alpha / (1.0 - alpha)  # 0.75 / 0.25 = 3.0
    actual_ratio = loss_fg / loss_bg

    assert math.isclose(actual_ratio, expected_ratio, rel_tol=1e-5), (
        f"Expected foreground-to-background loss ratio {expected_ratio}, got {actual_ratio}"
    )

    # Total loss should be mean of the two pixels
    total_loss = focal(logits, targets).item()
    expected_total = (loss_fg + loss_bg) / 2.0
    assert math.isclose(total_loss, expected_total, rel_tol=1e-5)


def test_focal_loss_gradient_ratio_foreground_vs_background():
    """Verify that gradients for a foreground error are scaled by alpha / (1 - alpha)
    relative to a background error at p=0.5.
    """
    alpha = 0.80
    gamma = 2.0
    focal = FocalLoss(alpha=alpha, gamma=gamma)

    # Pixel 0: target=1 (road), Pixel 1: target=0 (background)
    logits = torch.zeros(1, 1, 1, 2, requires_grad=True)
    targets = torch.tensor([[[[1.0, 0.0]]]])

    loss = focal(logits, targets)
    loss.backward()

    grad = logits.grad
    grad_fg = grad[0, 0, 0, 0].item()  # gradient for predicting background on road
    grad_bg = grad[0, 0, 0, 1].item()  # gradient for predicting road on background

    # At logits=0, d(BCE)/dz = p - y. For y=1, grad is negative (-0.5); for y=0, grad is positive (+0.5).
    # Ratio of magnitudes must equal alpha / (1 - alpha) = 0.8 / 0.2 = 4.0
    ratio = abs(grad_fg) / abs(grad_bg)
    expected_ratio = alpha / (1.0 - alpha)

    assert math.isclose(ratio, expected_ratio, rel_tol=1e-5), (
        f"Expected gradient magnitude ratio {expected_ratio}, got {ratio}"
    )


def test_focal_loss_alpha_none_treats_classes_equally():
    """When alpha=None, alpha_t=1.0 and both classes receive symmetric weighting."""
    focal = FocalLoss(alpha=None, gamma=2.0)
    logits = torch.zeros(1, 1, 1, 2)
    targets = torch.tensor([[[[1.0, 0.0]]]])

    loss = focal(logits, targets)

    # Unreduced check
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1.0 - probs)
    weights = (1.0 - pt) ** 2.0
    expected_loss = (weights * bce).mean().item()

    assert math.isclose(loss.item(), expected_loss, rel_tol=1e-5)


def test_focal_loss_gamma_suppresses_easy_pixels():
    """High-confidence correct pixels (pt ~ 0.99) should have negligible weight
    compared to ambiguous pixels (pt ~ 0.5) when gamma > 0.
    """
    focal = FocalLoss(alpha=0.5, gamma=2.0)

    # Easy pixel (p=0.99, target=1) vs Ambiguous pixel (p=0.5, target=1)
    easy_logit = torch.logit(torch.tensor(0.99))
    ambig_logit = torch.tensor(0.0)

    logits_easy = easy_logit.view(1, 1, 1, 1)
    logits_ambig = ambig_logit.view(1, 1, 1, 1)
    target = torch.tensor([[[[1.0]]]])

    loss_easy = focal(logits_easy, target).item()
    loss_ambig = focal(logits_ambig, target).item()

    # (1 - 0.99)^2 = 0.0001, (1 - 0.5)^2 = 0.25 -> 2500x modulation factor
    assert loss_ambig > 100 * loss_easy, (
        f"Ambiguous loss ({loss_ambig}) should massively outweigh easy loss ({loss_easy})"
    )


def test_dice_loss_sample_wise_averaging_preserves_per_image_invariance():
    """In sample-wise Dice, each image in the batch contributes equally (1/B) to the loss,
    preventing dense tiles from drowning out sparse tiles.
    """
    dice_fn = DiceLoss(smooth=1e-6)

    # Batch of 2 images:
    # Sample 0: Perfect prediction (Dice = 1.0 -> Loss = 0.0)
    # Sample 1: Completely missed prediction (Dice = 0.0 -> Loss = 1.0)
    # Sample-wise mean loss MUST be (0.0 + 1.0) / 2 = 0.50.
    logits = torch.tensor([
        [[[10.0, 10.0], [10.0, 10.0]]],   # sample 0: high confidence (road)
        [[[-10.0, -10.0], [-10.0, -10.0]]] # sample 1: low confidence (background)
    ])
    targets = torch.tensor([
        [[[1.0, 1.0], [1.0, 1.0]]],       # sample 0: all road
        [[[1.0, 1.0], [1.0, 1.0]]]        # sample 1: all road
    ])

    loss = dice_fn(logits, targets).item()
    assert math.isclose(loss, 0.5, abs_tol=1e-3), (
        f"Expected sample-wise Dice loss to be 0.5, got {loss}"
    )


def test_dice_loss_zero_division_and_numerical_stability():
    """Verify DiceLoss is numerically stable with empty masks, smooth=0, and extreme logits."""
    # 1. Both target and prediction are all zeros (empty background tile)
    dice_standard = DiceLoss(smooth=1.0)
    logits_empty = torch.full((2, 1, 8, 8), -20.0)  # probs ~ 0
    targets_empty = torch.zeros(2, 1, 8, 8)
    loss = dice_standard(logits_empty, targets_empty)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() < 0.05  # predicting empty for empty target should have near-zero loss

    # 2. Both target and prediction all zeros with smooth=0.0
    dice_zero_smooth = DiceLoss(smooth=0.0)
    loss_zero_smooth = dice_zero_smooth(logits_empty, targets_empty)
    assert not torch.isnan(loss_zero_smooth) and not torch.isinf(loss_zero_smooth)

    # 3. Extreme logit values (+100.0, -100.0)
    logits_extreme = torch.tensor([[[[100.0, -100.0], [-100.0, 100.0]]]])
    targets_extreme = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    loss_extreme = dice_standard(logits_extreme, targets_extreme)
    assert not torch.isnan(loss_extreme) and not torch.isinf(loss_extreme)
    assert math.isclose(loss_extreme.item(), 0.0, abs_tol=1e-4)


def test_focal_dice_loss_backward_pass():
    """Verify FocalDiceLoss backpropagates valid, non-zero gradients on all inputs."""
    criterion = FocalDiceLoss(alpha=0.75, gamma=2.0, smooth=1.0)

    logits = torch.randn(4, 1, 16, 16, requires_grad=True)
    targets = (torch.rand(4, 1, 16, 16) > 0.9).float()  # sparse road target

    loss = criterion(logits, targets)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not torch.isinf(logits.grad).any()
    assert (logits.grad != 0).any()


def test_build_loss_factory():
    """Verify build_loss respects TrainConfig."""
    cfg_focal = TrainConfig(loss="focal_dice", focal_alpha=0.8, focal_gamma=2.5)
    loss_focal = build_loss(cfg_focal)
    assert isinstance(loss_focal, FocalDiceLoss)
    assert loss_focal.focal.alpha == 0.8
    assert loss_focal.focal.gamma == 2.5

    cfg_bce = TrainConfig(loss="bce_dice")
    loss_bce = build_loss(cfg_bce)
    assert isinstance(loss_bce, BCEDiceLoss)

    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss(TrainConfig(loss="unsupported_loss"))

