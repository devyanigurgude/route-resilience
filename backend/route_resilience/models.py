"""
Model builders. One definition per architecture, selected by
cfg.model.arch, so a "U-Net vs DeepLabV3+" experiment differs in
exactly one config field and nothing else.

`unet` (custom, with Dropout2d) is the model used for MC-Dropout
uncertainty estimation -- segmentation_models_pytorch's stock Unet/
DeepLabV3Plus don't have inference-time dropout, and bolting it on by
monkeypatching third-party modules is fragile. `unet_smp` and
`deeplabv3plus` (via segmentation_models_pytorch) are for the plain
architecture-comparison experiment (E1/E2) where uncertainty isn't
being measured.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MCDropoutUNet(nn.Module):
    """
    Custom U-Net with Dropout2d in the bottleneck and decoder blocks.
    Dropout stays active at inference time (see `enable_mc_dropout`
    below) to produce a distribution over predictions -> mean =
    prediction, std = per-pixel uncertainty. This is the CORE
    uncertainty mechanism the whole confidence-aware graph depends on.
    """

    def __init__(self, in_channels: int = 3, classes: int = 1, dropout_p: float = 0.2):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512, dropout_p=dropout_p)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024, dropout_p=dropout_p)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512, dropout_p=dropout_p)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256, dropout_p=dropout_p)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


def enable_mc_dropout(model: nn.Module):
    """
    Put the whole model in eval() (so BatchNorm uses running stats --
    critical, BN in train mode would make each stochastic pass see a
    different batch statistic and corrupt the uncertainty estimate),
    then switch ONLY the Dropout layers back to train() so they keep
    sampling.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
    return model


def build_model(cfg):
    arch = cfg.arch

    if arch == "unet":
        return MCDropoutUNet(
            in_channels=cfg.in_channels,
            classes=cfg.classes,
            dropout_p=cfg.dropout_p,
        )

    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pip install segmentation-models-pytorch --break-system-packages"
        ) from e

    if arch == "unet_smp":
        return smp.Unet(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,  # keep "imagenet" -- see review note
            in_channels=cfg.in_channels,
            classes=cfg.classes,
        )

    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,  # NOT None -- this was the bug
            in_channels=cfg.in_channels,
            classes=cfg.classes,
        )

    if arch in ("segformer", "segformer_smp"):
        return smp.Segformer(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=cfg.classes,
        )

    raise ValueError(f"Unknown arch: {arch}")
