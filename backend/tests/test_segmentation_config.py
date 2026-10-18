"""No-training checks for segmentation model selection and augmentation."""

import sys
import types
import warnings

import numpy as np

# Albumentations 2.0 checks the internet for update metadata at import time.
# That warning is unrelated to transform construction and would make this
# no-network test flaky.
warnings.filterwarnings(
    "ignore", message="Error fetching version info", module="albumentations.check_version"
)

from route_resilience.config import DataConfig, ModelConfig
from route_resilience.data import build_transforms


def test_coarse_dropout_v2_keeps_mask_aligned_without_transform_warnings():
    """Occlude image pixels but deliberately preserve the road target mask."""
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 255
    cfg = DataConfig(
        image_size=64,
        use_occlusion_augmentation=True,
        occlusion_max_holes=3,
        occlusion_max_frac=0.25,
    )
    transform = build_transforms(cfg)
    for op in transform.transforms:
        op.p = 1.0 if op.__class__.__name__ == "CoarseDropout" else 0.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        augmented = transform(image=image, mask=mask)

    assert not caught, [str(w.message) for w in caught]
    assert augmented["image"].shape == image.shape
    assert augmented["mask"].shape == mask.shape
    assert np.any(augmented["image"] != image)
    assert np.array_equal(augmented["mask"], mask)


def test_preprocess_rgb_image_accepts_pil_and_numpy():
    """Verify preprocess_rgb_image safely handles both PIL.Image and numpy array."""
    from PIL import Image
    import torch
    from route_resilience.data import preprocess_rgb_image

    pil_img = Image.new("RGB", (64, 48), color=(100, 150, 200))
    tensor_pil = preprocess_rgb_image(pil_img)
    assert isinstance(tensor_pil, torch.Tensor)
    assert tensor_pil.shape == (3, 48, 64)

    np_img = np.asarray(pil_img)
    tensor_np = preprocess_rgb_image(np_img)
    assert isinstance(tensor_np, torch.Tensor)
    assert tensor_np.shape == (3, 48, 64)
    assert torch.allclose(tensor_pil, tensor_np)


def test_unet_smp_factory_requests_resnet34_imagenet_without_downloading(monkeypatch):
    """Verify exact factory arguments using a local stand-in for SMP."""
    captured = {}

    class FakeUnet:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", types.SimpleNamespace(Unet=FakeUnet))

    from route_resilience.models import build_model

    model = build_model(ModelConfig(arch="unet_smp", encoder_name="resnet34", encoder_weights="imagenet"))
    assert isinstance(model, FakeUnet)
    assert captured == {
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "in_channels": 3,
        "classes": 1,
    }


def test_checkpoint_names_distinguish_custom_and_smp_unets():
    """A custom U-Net checkpoint must never be labelled as ResNet34."""
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).parents[1] / "scripts" / "train_pipeline.py"
    spec = importlib.util.spec_from_file_location("train_pipeline", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.checkpoint_stem(ModelConfig(arch="unet"), "focal_dice") == "mc_dropout_unet_focaldice"
    assert (
        module.checkpoint_stem(
            ModelConfig(arch="unet_smp", encoder_name="resnet34"), "focal_dice"
        )
        == "unet_smp_resnet34_focaldice"
    )
