"""
nnU-Net based detector for intracranial aneurysm detection.

Architecture:
    - BasicUNet backbone (MONAI) with CoW segmentator weights as initialisation.
    - Bottleneck features captured via forward hook on the encoder bottleneck.
    - Auxiliary sphere head, binary head, multilabel zone head.

The key idea: instead of generic Med3D pre-training, the encoder is initialised
with weights from the Fine CoW segmentator. This gives the detector a head start
with representations already tuned to CoW anatomy — the exact spatial context
relevant for aneurysm localisation.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet


class nnUNetDetector(nn.Module):
    """nnU-Net style detector using BasicUNet encoder as backbone.

    Args:
        pretrained_path : path to Fine CoW segmentator checkpoint (.pth).
                          Set to None to use random initialisation.
        in_channels     : number of input channels (default 2: vol + cow_mask)
        n_zones         : number of anatomical zones (default 13)
        dropout         : dropout rate for classification heads
    """

    BOTTLENECK_CH = 512

    def __init__(
        self,
        pretrained_path: str | Path | None = None,
        in_channels:     int               = 2,
        n_zones:         int               = 13,
        dropout:         float             = 0.3,
    ) -> None:
        super().__init__()

        self.backbone = BasicUNet(
            spatial_dims = 3,
            in_channels  = in_channels,
            out_channels = 1,
            features     = (32, 64, 128, 256, 512, 32),
            act          = "LeakyReLU",
            norm         = "instance",
            dropout      = 0.1,
        )

        # Hook to capture bottleneck features
        self._bottleneck_feat: torch.Tensor | None = None
        self._hook_handle = None
        self._register_bottleneck_hook()

        # Load pretrained CoW segmentator weights
        if pretrained_path and Path(pretrained_path).exists():
            n = self._load_pretrained(str(pretrained_path))
            print(f"✓  Pretrained CoW weights: {n} layers from {Path(pretrained_path).name}")
        else:
            print("No pretrained weights — random initialisation")

        # Auxiliary sphere head
        self.aux_head = nn.Sequential(
            nn.Conv3d(self.BOTTLENECK_CH, 128, 3, padding=1, bias=False),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 1, 1),
        )

        feat_dim = self.BOTTLENECK_CH + 1   # + modality flag
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.head_binary = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 1),
        )
        self.head_multilabel = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, n_zones),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _register_bottleneck_hook(self) -> None:
        """Hook onto the deepest encoder layer of BasicUNet to get bottleneck."""
        # BasicUNet deepest encoder: down_4 → features[4] = 512 channels
        target = self.backbone.down_4
        self._hook_handle = target.register_forward_hook(
            lambda m, i, o: setattr(self, "_bottleneck_feat", o)
        )

    def _load_pretrained(self, path: str) -> int:
        """Load pretrained weights, skipping mismatched layers. Returns # loaded."""
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        m     = self.backbone
        model_dict = m.state_dict()
        matched = {k: v for k, v in state.items()
                   if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(matched)
        m.load_state_dict(model_dict)
        return len(matched)

    def get_param_groups(
        self,
        lr_encoder: float,
        lr_heads:   float,
    ) -> list[dict]:
        """Return parameter groups with differential learning rates."""
        m = self.module if hasattr(self, "module") else self
        encoder_p, head_p = [], []
        for name, param in m.named_parameters():
            if "backbone" in name:
                encoder_p.append(param)
            else:
                head_p.append(param)
        return [
            {"params": encoder_p, "lr": lr_encoder},
            {"params": head_p,    "lr": lr_heads},
        ]

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
        _ = self.backbone(x)                       # triggers hook
        feat = self._bottleneck_feat               # (B, 512, d, h, w)

        sphere_logits = self.aux_head(feat)

        pooled = self.gap(feat).flatten(1)
        pooled = torch.cat(
            [pooled, mod_flag.to(feat.device).view(pooled.shape[0], 1)], dim=1
        )

        return (
            self.head_binary(pooled).squeeze(1),
            self.head_multilabel(pooled),
            sphere_logits,
        )
