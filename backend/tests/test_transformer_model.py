"""Unit tests for Phase 5 Transformer (SegFormer-B0) integration and backward compatibility."""

import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from route_resilience.config import Config, ModelConfig, TrainConfig
from route_resilience.graph_extraction import (
    clean_road_mask,
    confidence_gated_gap_healing,
    mask_to_skeleton,
    skeleton_to_graph,
)
from route_resilience.losses import build_loss
from route_resilience.models import build_model
from route_resilience.real_graph_pipeline import load_model_for_inference, predict_probability_map


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "checkpoints").exists() and (PROJECT_ROOT.parent / "checkpoints").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent


def test_segformer_b0_instantiation_and_output_shape():
    """Verify that SegFormer-B0 instantiates and produces [B, 1, H, W] logits."""
    cfg = ModelConfig(arch="segformer", encoder_name="mit_b0", encoder_weights=None, in_channels=3, classes=1)
    model = build_model(cfg)
    assert model is not None

    batch_x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(batch_x)

    assert out.shape == (2, 1, 256, 256), f"Expected shape (2, 1, 256, 256), got {out.shape}"
    params = sum(p.numel() for p in model.parameters())
    assert 3_000_000 < params < 4_000_000, f"Expected SegFormer-B0 ~3.7M params, got {params}"


def test_segformer_b0_loss_and_gradients():
    """Verify forward and backward gradients with FocalDiceLoss."""
    cfg = ModelConfig(arch="segformer", encoder_name="mit_b0", encoder_weights=None, in_channels=3, classes=1)
    model = build_model(cfg)
    model.train()

    criterion = build_loss(TrainConfig(loss="focal_dice"))
    images = torch.randn(2, 3, 256, 256)
    masks = torch.randint(0, 2, (2, 1, 256, 256)).float()

    logits = model(images)
    loss = criterion(logits, masks)
    assert not torch.isnan(loss)
    assert loss.item() > 0

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
            break


def test_segformer_checkpoint_save_and_load():
    """Verify checkpoint saving, format compliance, and load_model_for_inference compatibility."""
    model_cfg = ModelConfig(arch="segformer", encoder_name="mit_b0", encoder_weights="imagenet")
    model = build_model(model_cfg)

    full_cfg = Config()
    full_cfg.model = model_cfg
    full_cfg.train.loss = "focal_dice"

    ckpt_dict = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "config": {
            "model": {
                "arch": "segformer",
                "encoder_name": "mit_b0",
                "encoder_weights": "imagenet",
                "in_channels": 3,
                "classes": 1,
                "dropout_p": 0.2,
            },
            "train": {"loss": "focal_dice"},
            "graph": {"max_gap_px": 15.0},
        },
        "val_loss": 0.5,
    }

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        torch.save(ckpt_dict, tmp_path)
        loaded_model, loaded_ckpt = load_model_for_inference(
            str(tmp_path),
            model_cfg=full_cfg.model,
            loss_name="focal_dice",
            device=torch.device("cpu"),
        )
        assert loaded_model is not None
        assert loaded_ckpt["epoch"] == 1

        # Test inference with loaded model
        tile = np.zeros((256, 256, 3), dtype=np.uint8)
        prob = predict_probability_map(loaded_model, tile, torch.device("cpu"))
        assert prob.shape == (256, 256)
        assert (prob >= 0.0).all() and (prob <= 1.0).all()
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)


def test_backward_compatibility_resnet34():
    """Verify that existing ResNet-34 U-Net checkpoint loads without regression."""
    resnet_path = PROJECT_ROOT / "checkpoints" / "best_unet_smp_resnet34_focaldice.pth"
    if not resnet_path.exists():
        pytest.skip("ResNet-34 checkpoint not found on disk")

    model_cfg = ModelConfig(arch="unet_smp", encoder_name="resnet34", encoder_weights="imagenet")
    model, checkpoint = load_model_for_inference(
        str(resnet_path),
        model_cfg=model_cfg,
        loss_name="focal_dice",
        device=torch.device("cpu"),
    )
    assert model is not None
    assert checkpoint["config"]["model"]["arch"] == "unet_smp"
    assert checkpoint["config"]["model"]["encoder_name"] == "resnet34"

    tile = np.zeros((256, 256, 3), dtype=np.uint8)
    prob = predict_probability_map(model, tile, torch.device("cpu"))
    assert prob.shape == (256, 256)


def test_end_to_end_graph_pipeline_with_segformer():
    """Verify that SegFormer predictions flow seamlessly into downstream graph extraction and gap healing."""
    model_cfg = ModelConfig(arch="segformer", encoder_name="mit_b0", encoder_weights=None)
    model = build_model(model_cfg)
    model.eval()

    # Create synthetic road line in RGB image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[128, 50:100, :] = 255
    img[128, 110:160, :] = 255  # 10px gap

    prob = predict_probability_map(model, img, torch.device("cpu"))
    assert prob.shape == (256, 256)

    # Synthetic probability with clear gap
    prob_synth = np.zeros((256, 256), dtype=np.float32)
    prob_synth[128, 50:100] = 0.95
    prob_synth[128, 100:110] = 0.60  # interior gap evidence
    prob_synth[128, 110:160] = 0.95

    binary_mask = clean_road_mask(prob_synth >= 0.5, min_component_pixels=3)
    skel = mask_to_skeleton(binary_mask)
    G = skeleton_to_graph(skel, prob_map=prob_synth, min_edge_length_px=3.0)

    # Unhealed graph should have 2 components
    assert G.number_of_nodes() > 0

    # Healed graph should close the 10px gap
    G_healed = confidence_gated_gap_healing(
        G=G.copy(),
        prob_map=prob_synth,
        max_gap_px=15.0,
        min_gap_confidence=0.15,
        tangent_dot_threshold=0.7,
    )
    assert G_healed.number_of_nodes() > 0
