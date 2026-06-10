"""
Dataset for aneurysm detection and zone localisation.

Input tensor : 2-channel [vol, cow_mask], shape (2, D, H, W)
Targets      :
    has_aneurysm — scalar float  (binary presence)
    labels       — (13,) float   (per-zone multilabel)
    sphere       — (1, D, H, W)  auxiliary sphere GT around CoW centroid
    mod_flag     — scalar float  (1.0 = MRA, 0.0 = CTA)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)
from torch.utils.data import Dataset

# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_SIZE = 128   # centre-crop target (voxels per axis)
SPHERE_R   = 5     # radius of auxiliary sphere target (voxels)
N_ZONES    = 13    # number of anatomical CoW zones

LOCATION_COLS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop a 3-D array to (size, size, size), zero-padding if smaller."""
    Z, H, W = arr.shape
    out = np.zeros((size,) * 3, dtype=arr.dtype)
    z0 = max(0, Z // 2 - size // 2)
    y0 = max(0, H // 2 - size // 2)
    x0 = max(0, W // 2 - size // 2)
    d = min(Z - z0, size)
    h = min(H - y0, size)
    w = min(W - x0, size)
    out[:d, :h, :w] = arr[z0:z0 + d, y0:y0 + h, x0:x0 + w]
    return out


def make_sphere_mask(
    shape:  tuple[int, int, int],
    center: tuple[int, int, int],
    radius: int = SPHERE_R,
) -> np.ndarray:
    """Binary sphere mask around *center*. Used as auxiliary segmentation GT."""
    Z, H, W = shape
    cz, cy, cx = center
    z = np.arange(Z).reshape(-1, 1, 1)
    y = np.arange(H).reshape(1, -1, 1)
    x = np.arange(W).reshape(1, 1, -1)
    return ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2).astype(
        np.float32
    )


def scan_pkl_dir(*dirs: Path) -> list[dict]:
    """Scan one or more pkl cache directories and return records list.

    Each record contains:
        path     : str path to pkl file
        sid      : series UID
        modality : 'CTA' or 'MRA'
        labels   : (13,) float32 zone labels
        presence : float (0 or 1) aneurysm presence
    """
    records = []
    for d in dirs:
        for pkl in sorted(Path(d).glob("*.pkl")):
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            records.append({
                "path":     str(pkl),
                "sid":      data.get("sid", pkl.stem),
                "modality": data.get("modality", "CTA"),
                "labels":   data["labels"][:N_ZONES].astype(np.float32),
                "presence": float(data["labels"][-1]),
            })
    return records


# ── Dataset ───────────────────────────────────────────────────────────────────

class AneurysmDataset(Dataset):
    """Loads pre-processed pkl cache for aneurysm detection.

    Each pkl file must contain:
        vol      : (D, H, W) float16/32 — normalised volume
        cow_mask : (D, H, W) float16/32 — CoW binary mask
        labels   : (14,) float32 — 13 zone labels + presence flag

    The dataset stacks vol and cow_mask into a 2-channel input tensor and
    computes an auxiliary sphere target around the CoW centroid.

    Args:
        records : list of dicts from :func:`scan_pkl_dir`
        augment : apply spatial + intensity augmentations when True
        size    : centre-crop target edge length (default 128)
        sphere_r: sphere target radius in voxels (default 5)
    """

    def __init__(
        self,
        records:  list[dict],
        augment:  bool = False,
        size:     int  = INPUT_SIZE,
        sphere_r: int  = SPHERE_R,
    ) -> None:
        self.records  = records
        self.augment  = augment
        self.size     = size
        self.sphere_r = sphere_r

        self.tfm = Compose([
            # Spatial
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image"], prob=0.5, max_k=3),
            RandZoomd(keys=["image"], min_zoom=0.85, max_zoom=1.15,
                      prob=0.3, keep_size=True),
            # Intensity
            RandGaussianNoised(keys=["image"], prob=0.3, std=0.02),
            RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.4),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.4),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
            # Blur
            RandGaussianSmoothd(keys=["image"],
                                sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5),
                                sigma_z=(0.5, 1.5), prob=0.2),
        ]) if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        with open(rec["path"], "rb") as f:
            d = pickle.load(f)

        vol      = center_crop(d["vol"].astype(np.float32),      self.size)
        cow_mask = center_crop(d["cow_mask"].astype(np.float32), self.size)
        presence = rec["presence"]
        labels13 = rec["labels"]
        mod_flag = np.float32(1.0 if rec["modality"] == "MRA" else 0.0)

        # Auxiliary sphere GT: sphere centred on CoW centroid (only for positives)
        if presence > 0.5 and cow_mask.sum() > 0:
            coords = np.column_stack(np.where(cow_mask > 0))
            center = tuple(coords.mean(axis=0).astype(int))
        else:
            center = (self.size // 2,) * 3
        sphere = make_sphere_mask((self.size,) * 3, center, self.sphere_r) * presence

        # Stack into 2-channel input (C, D, H, W)
        image = np.stack([vol, cow_mask], axis=0)

        if self.augment and self.tfm:
            image = np.asarray(
                self.tfm({"image": image})["image"], dtype=np.float32
            )

        return {
            "image":        torch.from_numpy(image),
            "has_aneurysm": torch.tensor(presence,  dtype=torch.float32),
            "labels":       torch.from_numpy(labels13),
            "sphere":       torch.from_numpy(sphere[None]),  # (1, D, H, W)
            "mod_flag":     torch.tensor(mod_flag,  dtype=torch.float32),
        }


# ── Weighted sampler helper ───────────────────────────────────────────────────

def compute_sample_weights(records: list[dict]) -> list[float]:
    """Compute per-sample weights inversely proportional to class frequency.

    Upweights positive (aneurysm present) samples so they appear
    in every mini-batch proportionally to their clinical significance.
    """
    n = len(records)
    n_pos = sum(r["presence"] for r in records)
    n_neg = n - n_pos
    w_pos = n / (2.0 * max(n_pos, 1))
    w_neg = n / (2.0 * max(n_neg, 1))
    return [w_pos if r["presence"] > 0.5 else w_neg for r in records]
