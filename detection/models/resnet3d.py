"""
ResNet3D-50 detector for intracranial aneurysm detection.

Architecture:
    - Standard ResNet-50 with all 2D ops replaced by 3D equivalents.
    - 2-channel input: [vol, cow_mask].
    - Binary head   : aneurysm presence.
    - Multilabel head: 13-zone anatomical localisation.
    - Auxiliary sphere head: coarse segmentation of CoW centroid region.

Pre-training:
    Supports loading Med3D weights (TencentMedicalNet/MedicalNet-Resnet50).
    conv1 is adapted from 1-channel to 2-channel by duplication + scale=0.5.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


# ── Building blocks ───────────────────────────────────────────────────────────

class Bottleneck3D(nn.Module):
    """3D ResNet Bottleneck: 1×1×1 → 3×3×3 → 1×1×1 + residual connection."""

    expansion = 4

    def __init__(
        self,
        inplanes:   int,
        planes:     int,
        stride:     int          = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm3d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


# ── Detector ──────────────────────────────────────────────────────────────────

class ResNet3D50Detector(nn.Module):
    """ResNet3D-50 for simultaneous binary detection and zone localisation.

    Args:
        in_channels : number of input channels (default 2: vol + cow_mask)
        n_zones     : number of anatomical zones (default 13)
        dropout     : dropout rate for classification heads
    """

    def __init__(
        self,
        in_channels: int   = 2,
        n_zones:     int   = 13,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()
        self.inplanes = 64

        # Stem
        self.conv1   = nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm3d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)

        # ResNet stages
        self.layer1 = self._make_layer(64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

        # Auxiliary sphere head (attached after layer2, 512 ch, 1/8 spatial)
        self.aux_head = nn.Sequential(
            nn.Conv3d(512, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 1, 1),
        )

        # Classification heads
        self.gap = nn.AdaptiveAvgPool3d(1)
        feat_dim = 512 * Bottleneck3D.expansion + 1  # 2048 + modality flag

        self.head_binary = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, 1),
        )
        self.head_multilabel = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, n_zones),
        )

        self._init_weights()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck3D.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * Bottleneck3D.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * Bottleneck3D.expansion),
            )
        layers = [Bottleneck3D(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck3D.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck3D(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:        torch.Tensor,
        mod_flag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x        : (B, 2, D, H, W) — [vol, cow_mask]
            mod_flag : (B,) float — 1.0 for MRA, 0.0 for CTA

        Returns:
            out_binary     : (B,) logits for aneurysm presence
            out_multilabel : (B, 13) logits for zone localisation
            sphere_logits  : (B, 1, D', H', W') aux sphere prediction
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        sphere_logits = self.aux_head(x)   # attached after layer2
        x = self.layer3(x)
        x = self.layer4(x)

        pooled = self.gap(x).flatten(1)
        pooled = torch.cat(
            [pooled, mod_flag.to(x.device).view(pooled.shape[0], 1)], dim=1
        )

        return (
            self.head_binary(pooled).squeeze(1),
            self.head_multilabel(pooled),
            sphere_logits,
        )


# ── Med3D weight loader ───────────────────────────────────────────────────────

def load_med3d_weights(
    model:        ResNet3D50Detector,
    weights_path: str | Path | None = None,
) -> ResNet3D50Detector:
    """Load Med3D pretrained weights into the ResNet3D encoder.

    Skips layers with shape mismatches (conv1: 2ch vs original 1ch, heads).
    After loading, conv1 is re-initialised from the 1-channel Med3D weights
    by duplication and scaling by 0.5.

    Download weights from:
        https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50

    Args:
        model        : ResNet3D50Detector instance
        weights_path : path to Med3D resnet_50.pth, or None to skip

    Returns:
        model with loaded (or kaiming-initialised) weights
    """
    if weights_path is None or not Path(weights_path).exists():
        print("⚠  Med3D weights not found — using kaiming initialisation")
        print("   Download from: https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50")
        return model

    ckpt  = torch.load(str(weights_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    m = model.module if hasattr(model, "module") else model
    model_dict = m.state_dict()

    matched, skip_shape, skip_miss = [], [], []
    filtered: dict = {}
    for k, v in state.items():
        if k not in model_dict:
            skip_miss.append(k)
        elif model_dict[k].shape != v.shape:
            skip_shape.append(k)
        else:
            filtered[k] = v
            matched.append(k)

    model_dict.update(filtered)
    m.load_state_dict(model_dict)

    print(f"✓  Med3D weights loaded: {len(matched)} layers")
    print(f"   Shape mismatch (skipped): {len(skip_shape)}  ← conv1 (2ch vs 1ch)")
    print(f"   Missing (skipped)       : {len(skip_miss)}")

    # Adapt conv1: 1ch → 2ch via duplication, scale by 0.5
    if "conv1.weight" in state:
        with torch.no_grad():
            w1 = state["conv1.weight"]          # (64, 1, 7, 7, 7)
            w2 = w1.repeat(1, 2, 1, 1, 1) / 2.0
            m.conv1.weight.copy_(w2)
        print("✓  conv1 adapted: 1ch → 2ch (scale=0.5)")

    return model
