"""
Loss functions for CoW segmentation and aneurysm detection.

Segmentation losses:
    FocalTverskyLoss  — recall-focused loss for thin vessel branches
    SkeletonRecallLoss — compound loss with explicit centerline coverage term

Detection loss:
    DetectorLoss — multi-task loss: binary detection + zone localisation + sphere aux
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss


# ── Segmentation ──────────────────────────────────────────────────────────────

class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for recall-focused segmentation of thin vessels.

    Setting alpha < beta penalises false negatives more than false positives,
    which increases recall at the cost of some precision — desirable for thin
    CoW branches where missing a vessel is worse than a small over-segmentation.

    At alpha=beta=0.5, gamma=1 this reduces to standard Dice loss.

    Args:
        alpha  : weight for false positives (default 0.3)
        beta   : weight for false negatives (default 0.7)
        gamma  : focal exponent — higher = more focus on hard examples
        smooth : numerical stability constant
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta:  float = 0.7,
        gamma: float = 1.5,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred_logits)
        tp   = (pred * target).sum()
        fp   = (pred * (1.0 - target)).sum()
        fn   = ((1.0 - pred) * target).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - tversky) ** (1.0 / self.gamma)


class SkeletonRecallLoss(nn.Module):
    """Compound segmentation loss with a Skeleton Recall term.

    Skeleton Recall measures what fraction of vessel centreline voxels are
    covered by the predicted mask. This encourages topological connectivity of
    thin branches that Dice alone tends to miss.

    Two modes:
        'DiceCE_Skel'  → DiceCE + BCE + SkeletonRecall  (w_skel=1)
            Balanced segmentation quality. Used for Fine-1 model.
        'Tversky_Skel' → FocalTversky + BCE + SkeletonRecall  (w_skel=3)
            Recall-focused. Used for Fine-2 model.

    Args:
        mode   : 'DiceCE_Skel' or 'Tversky_Skel'
        w_main : weight for the main (Dice/Tversky) component
        w_skel : weight for the skeleton recall component
        w_bce  : weight for the BCE component
    """

    def __init__(
        self,
        mode:   str   = "DiceCE_Skel",
        w_main: float = 1.0,
        w_skel: float = 1.0,
        w_bce:  float = 1.0,
    ) -> None:
        super().__init__()
        self.mode   = mode
        self.w_main = w_main
        self.w_skel = w_skel
        self.w_bce  = w_bce

        if mode == "DiceCE_Skel":
            self.main_loss = DiceCELoss(sigmoid=True)
        elif mode == "Tversky_Skel":
            self.main_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.5)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'DiceCE_Skel' or 'Tversky_Skel'.")

        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_logits: torch.Tensor,
        target:      torch.Tensor,
        skeleton:    torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute compound loss.

        Args:
            pred_logits : raw model output  (B, 1, D, H, W)
            target      : binary ground truth mask  (B, 1, D, H, W)
            skeleton    : binary centreline mask  (B, 1, D, H, W)

        Returns:
            total loss tensor, dict with per-component values
        """
        loss_main = self.main_loss(pred_logits, target)
        loss_bce  = self.bce(pred_logits, target)

        # Skeleton recall: fraction of centreline voxels predicted positive
        pred_prob = torch.sigmoid(pred_logits)
        loss_skel = 1.0 - (skeleton * pred_prob).sum() / (skeleton.sum() + 1e-8)

        total = (
            self.w_main * loss_main
            + self.w_bce  * loss_bce
            + self.w_skel * loss_skel
        )

        return total, {
            "main": loss_main.item(),
            "bce":  loss_bce.item(),
            "skel": loss_skel.item(),
        }


# ── Detection ─────────────────────────────────────────────────────────────────

class DetectorLoss(nn.Module):
    """Multi-task loss for aneurysm detection and zone localisation.

    Three components:
        binary  : Focal loss for binary aneurysm presence head
        multi   : BCE for 13-zone multilabel localisation head
        aux     : (BCE + Dice) for auxiliary sphere segmentation head

    Args:
        w_bin            : weight for binary component (default 1.0)
        w_multi          : weight for multilabel component (default 1.0)
        w_aux            : weight for auxiliary sphere component (default 0.5)
        pos_weight_bin   : positive class weight for binary BCE
        pos_weight_multi : per-zone positive weights tensor (13,)
    """

    def __init__(
        self,
        w_bin:            float               = 1.0,
        w_multi:          float               = 1.0,
        w_aux:            float               = 0.5,
        pos_weight_bin:   float               = 1.0,
        pos_weight_multi: torch.Tensor | None = None,
        device:           torch.device        = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.w_bin   = w_bin
        self.w_multi = w_multi
        self.w_aux   = w_aux

        self.bce_bin = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_bin]).to(device)
        )
        self.bce_multi = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_multi.to(device)
            if pos_weight_multi is not None else None
        )
        self.bce_aux = nn.BCEWithLogitsLoss()

    @staticmethod
    def _dice(
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-6,
    ) -> torch.Tensor:
        p  = torch.sigmoid(pred_logits)
        tp = (p * target).sum()
        return 1.0 - (2.0 * tp + smooth) / (p.sum() + target.sum() + smooth)

    def forward(
        self,
        out_bin:      torch.Tensor,
        out_multi:    torch.Tensor,
        sphere_logits: torch.Tensor,
        has_an:       torch.Tensor,
        labels13:     torch.Tensor,
        sphere_gt:    torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute multi-task loss.

        Args:
            out_bin       : binary head logits  (B, 1)
            out_multi     : zone head logits  (B, 13)
            sphere_logits : aux sphere seg logits  (B, 1, D, H, W)
            has_an        : binary labels  (B, 1)
            labels13      : zone labels  (B, 13)
            sphere_gt     : sphere GT mask  (B, 1, D, H, W)

        Returns:
            total loss tensor, dict with per-component float values
        """
        out_bin   = torch.clamp(out_bin,   -10, 10)
        out_multi = torch.clamp(out_multi, -10, 10)

        # Focal loss for binary head (γ=2)
        bce_raw  = F.binary_cross_entropy_with_logits(
            out_bin, has_an, reduction="none"
        )
        pt       = torch.exp(-bce_raw)
        loss_bin = ((1.0 - pt) ** 2.0 * bce_raw).mean()

        loss_multi = self.bce_multi(out_multi, labels13)

        # Auxiliary sphere head — downsample GT to match prediction resolution
        sphere_gt_ds = F.interpolate(
            sphere_gt,
            size=sphere_logits.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        loss_aux = (
            0.5 * self.bce_aux(sphere_logits, sphere_gt_ds)
            + 0.5 * self._dice(sphere_logits, sphere_gt_ds)
        )

        total = (
            self.w_bin   * loss_bin
            + self.w_multi * loss_multi
            + self.w_aux   * loss_aux
        )

        return total, {
            "bin":   loss_bin.item(),
            "multi": loss_multi.item(),
            "aux":   loss_aux.item(),
        }
