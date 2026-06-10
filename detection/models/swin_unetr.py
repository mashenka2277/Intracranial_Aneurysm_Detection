"""
Swin UNETR detector for intracranial aneurysm detection.

Architecture:
    - Swin Transformer encoder (via MONAI SwinUNETR backbone).
    - Bottleneck features (768 ch) extracted via forward hook on swinViT.
    - Auxiliary sphere head for CoW centroid localisation.
    - Binary head   : aneurysm presence.
    - Multilabel head: 13-zone anatomical localisation.

Pre-training:
    Supports MONAI SSL weights (self-supervised on 5 050 public CT scans).
    Download: https://github.com/Project-MONAI/MONAI-extra-test-data/releases/
              download/0.8.1/model_swinvit.pt
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETRDetector(nn.Module):
    """Swin UNETR backbone + classification heads for aneurysm detection.

    The Swin Transformer encoder is kept intact from MONAI SwinUNETR.
    A forward hook captures the bottleneck feature map (768 channels) which
    is used for both the auxiliary sphere prediction and the classification heads.

    Args:
        img_size    : input volume edge length (default 96 → 128 recommended)
        in_channels : number of input channels (default 2: vol + cow_mask)
        n_zones     : number of anatomical zones (default 13)
        dropout     : dropout rate for classification heads
    """

    BOTTLENECK_CH = 768

    def __init__(
        self,
        img_size:    int   = 96,
        in_channels: int   = 2,
        n_zones:     int   = 13,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()

        self.backbone = SwinUNETR(
            in_channels   = in_channels,
            out_channels  = 1,
            feature_size  = 48,
            use_checkpoint= True,
            spatial_dims  = 3,
        )

        # Hook to capture bottleneck encoder features
        self._bottleneck_feat: torch.Tensor | None = None
        self._register_hook()

        # Auxiliary sphere head
        self.aux_head = nn.Sequential(
            nn.Conv3d(self.BOTTLENECK_CH, 256, 3, padding=1, bias=False),
            nn.InstanceNorm3d(256),
            nn.GELU(),
            nn.Conv3d(256, 1, 1),
        )

        # Classification heads
        feat_dim = self.BOTTLENECK_CH + 1   # + modality flag
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.head_binary = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 1),
        )
        self.head_multilabel = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, n_zones),
        )

        self._init_heads()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _register_hook(self) -> None:
        """Register forward hook on swinViT to capture bottleneck features."""
        self._hook_handle = self.backbone.swinViT.register_forward_hook(
            lambda m, i, o: setattr(
                self,
                "_bottleneck_feat",
                o[-1] if isinstance(o, (list, tuple)) else o,
            )
        )
        print("Hook registered on: backbone.swinViT")

    def _init_heads(self) -> None:
        for m in list(self.head_binary.modules()) + list(self.head_multilabel.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    # ── SSL weight loading ────────────────────────────────────────────────────

    def load_ssl_weights(self, weights_path: str | Path) -> None:
        """Load MONAI SSL pretrained SwinViT weights.

        Args:
            weights_path : path to model_swinvit.pt

        Download:
            wget https://github.com/Project-MONAI/MONAI-extra-test-data/
                 releases/download/0.8.1/model_swinvit.pt
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"SSL weights not found at {weights_path}.\n"
                "Download from: https://github.com/Project-MONAI/MONAI-extra-test-data/"
                "releases/download/0.8.1/model_swinvit.pt"
            )
        m = self.module if hasattr(self, "module") else self
        m.backbone.load_from(
            weights=torch.load(str(weights_path), map_location="cpu", weights_only=False)
        )
        print(f"✓  MONAI SSL weights loaded from {weights_path.name}")

    # ── Param groups for differential LR ─────────────────────────────────────

    def get_param_groups(
        self,
        lr_encoder: float,
        lr_heads:   float,
    ) -> list[dict]:
        """Return parameter groups with different learning rates.

        Encoder (backbone) parameters get *lr_encoder* (lower).
        Head parameters get *lr_heads* (higher).
        """
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
        feat = self._bottleneck_feat               # (B, 768, d, h, w)

        sphere_logits = self.aux_head(feat)

        pooled = self.gap(feat).flatten(1)         # (B, 768)
        pooled = torch.cat(
            [pooled, mod_flag.to(feat.device).view(pooled.shape[0], 1)], dim=1
        )

        return (
            self.head_binary(pooled).squeeze(1),
            self.head_multilabel(pooled),
            sphere_logits,
        )
