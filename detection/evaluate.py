"""
Evaluation utilities for aneurysm detection and zone localisation.

Functions:
    compute_sens_at_spec  — sensitivity at fixed specificity threshold
    evaluate_detector     — full evaluation loop with TTA (flip augmentation)
    print_metrics         — pretty-print binary + per-zone results
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LOCATION_COLS, N_ZONES

# Short zone names for display
ZONE_SHORT = [
    "L.ICA inf", "R.ICA inf",
    "L.ICA sup", "R.ICA sup",
    "L.MCA",     "R.MCA",
    "ACoA",      "L.ACA",
    "R.ACA",     "L.PCoA",
    "R.PCoA",    "Basilar",
    "Other post",
]


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_sens_at_spec(
    labels:   np.ndarray,
    preds:    np.ndarray,
    spec_thr: float = 0.95,
) -> float:
    """Compute sensitivity at the point where specificity >= *spec_thr*.

    Args:
        labels   : binary ground-truth array (0/1)
        preds    : predicted probability array in [0, 1]
        spec_thr : minimum specificity (default 0.95)

    Returns:
        sensitivity value, or 0.0 if no operating point satisfies the constraint
    """
    try:
        fpr, tpr, _ = roc_curve(labels, preds)
        mask = (1.0 - fpr) >= spec_thr
        return float(tpr[mask].max()) if mask.any() else 0.0
    except Exception:
        return 0.0


# ── Full evaluation loop ──────────────────────────────────────────────────────

def evaluate_detector(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    use_tta:    bool  = True,
    n_zones:    int   = N_ZONES,
) -> dict:
    """Run evaluation with optional test-time augmentation (TTA).

    TTA: original forward pass + horizontal flip (flip W axis).
    Binary and multilabel probabilities are averaged over augmentations.

    Args:
        model   : detector in eval mode
        loader  : validation DataLoader
        device  : torch device
        use_tta : apply TTA when True (default True)
        n_zones : number of anatomical zones

    Returns:
        dict with keys:
            probs_bin   : (N,) binary probabilities
            probs_multi : (N, n_zones) zone probabilities
            labs_bin    : (N,) binary labels
            labs_multi  : (N, n_zones) zone labels
            auc_bin     : AUC-ROC for binary detection
            sens95      : sensitivity at specificity 0.95
            macro_auc   : macro-averaged AUC across zones
            zone_aucs   : (n_zones,) per-zone AUC values
            cm          : 2×2 confusion matrix (threshold 0.5)
    """
    model.eval()
    all_pb, all_pm, all_lb, all_lm = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            img    = batch["image"].to(device)
            mod    = batch["mod_flag"].to(device)
            has_an = batch["has_aneurysm"].numpy()
            labels = batch["labels"].numpy()

            with torch.amp.autocast("cuda"):
                ob1, om1, _ = model(img, mod)
                if use_tta:
                    ob2, om2, _ = model(img.flip(-1), mod)
                    pb = ((torch.sigmoid(ob1) + torch.sigmoid(ob2)) / 2).cpu().numpy()
                    pm = ((torch.sigmoid(om1) + torch.sigmoid(om2)) / 2).cpu().numpy()
                else:
                    pb = torch.sigmoid(ob1).cpu().numpy()
                    pm = torch.sigmoid(om1).cpu().numpy()

            all_pb.extend(pb)
            all_pm.append(pm)
            all_lb.extend(has_an)
            all_lm.append(labels)

    probs_bin   = np.array(all_pb)
    probs_multi = np.concatenate(all_pm,  axis=0)
    labs_bin    = np.array(all_lb)
    labs_multi  = np.concatenate(all_lm, axis=0)

    # Binary metrics
    try:
        auc_bin = roc_auc_score(labs_bin, probs_bin)
    except Exception:
        auc_bin = 0.5
    sens95 = compute_sens_at_spec(labs_bin, probs_bin)
    cm     = confusion_matrix(labs_bin, (probs_bin >= 0.5).astype(int))

    # Per-zone metrics
    zone_aucs = []
    for i in range(n_zones):
        try:
            auc_i = roc_auc_score(labs_multi[:, i], probs_multi[:, i])
        except Exception:
            auc_i = 0.5
        zone_aucs.append(auc_i)
    macro_auc = float(np.mean(zone_aucs))

    return {
        "probs_bin":   probs_bin,
        "probs_multi": probs_multi,
        "labs_bin":    labs_bin,
        "labs_multi":  labs_multi,
        "auc_bin":     auc_bin,
        "sens95":      sens95,
        "macro_auc":   macro_auc,
        "zone_aucs":   zone_aucs,
        "cm":          cm,
    }


def print_metrics(results: dict, n_zones: int = N_ZONES) -> None:
    """Pretty-print evaluation results.

    Args:
        results : dict returned by :func:`evaluate_detector`
        n_zones : number of anatomical zones
    """
    cm = results["cm"]
    print(f"\n{'='*50}")
    print("BINARY DETECTION")
    print(f"{'='*50}")
    print(f"  AUC-ROC              : {results['auc_bin']:.4f}")
    print(f"  Sensitivity@Spec0.95 : {results['sens95']:.4f}")
    print(f"  Confusion (thr=0.5)  : "
          f"TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

    print(f"\n{'='*50}")
    print(f"ZONE LOCALISATION  (Macro AUC = {results['macro_auc']:.4f})")
    print(f"{'='*50}")
    print(f"  {'Zone':<28} {'AUC':>6}  {'N+':>5}")
    print(f"  {'-'*42}")
    n_pos = results["labs_multi"].sum(axis=0).astype(int)
    for name, auc_i, n in zip(LOCATION_COLS, results["zone_aucs"], n_pos):
        print(f"  {name[:28]:<28} {auc_i:>6.3f}  {n:>5}")
    print(f"{'='*50}\n")
