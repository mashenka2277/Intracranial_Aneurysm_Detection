"""
Dataset classes for CoW segmentation training.

CoarseDataset : loads NPZ pairs, resamples to 64³, normalises.
FineDataset   : loads NPZ pairs at full resolution, crops on-the-fly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────
COARSE_SIZE = 64
CROP_SIZE   = 128


# ── Helpers ───────────────────────────────────────────────────────────────────

def resample_vol(vol: np.ndarray, size: int, order: int = 1) -> np.ndarray:
    """Resample volume to isotropic cube of edge *size* voxels."""
    Z, H, W = vol.shape
    return zoom(vol, (size / Z, size / H, size / W), order=order).astype(np.float32)


def normalize_vol(vol: np.ndarray) -> np.ndarray:
    """Clip to [p1, p99] and scale to [0, 1]."""
    p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99)
    vol = np.clip(vol, p1, p99)
    return ((vol - p1) / (p99 - p1 + 1e-8)).astype(np.float32)


def mask_crop(
    vol:    np.ndarray,
    cow:    np.ndarray,
    skel:   np.ndarray,
    size:   int,
    random: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a (size³) patch centred on the CoW mask.

    random=True  (train): random voxel inside mask + ±size/4 jitter.
    random=False (val):   deterministic centroid of mask.
    Falls back to volume centre when mask is empty.

    Returns:
        (vol_patch, cow_patch, skel_patch), each shape (size, size, size)
    """
    Z, H, W = vol.shape
    out_v = np.zeros((size,) * 3, dtype=np.float32)
    out_c = np.zeros((size,) * 3, dtype=np.float32)
    out_s = np.zeros((size,) * 3, dtype=np.float32)

    if cow.sum() > 0:
        coords = np.where(cow > 0)
        if random:
            i  = np.random.randint(len(coords[0]))
            cz = int(coords[0][i]) + np.random.randint(-size // 4, size // 4)
            cy = int(coords[1][i]) + np.random.randint(-size // 4, size // 4)
            cx = int(coords[2][i]) + np.random.randint(-size // 4, size // 4)
        else:
            cz = int(np.mean(coords[0]))
            cy = int(np.mean(coords[1]))
            cx = int(np.mean(coords[2]))
    else:
        cz, cy, cx = Z // 2, H // 2, W // 2

    z1 = max(0, cz - size // 2); z2 = min(Z, z1 + size)
    y1 = max(0, cy - size // 2); y2 = min(H, y1 + size)
    x1 = max(0, cx - size // 2); x2 = min(W, x1 + size)
    if z2 - z1 < size: z1 = max(0, z2 - size)
    if y2 - y1 < size: y1 = max(0, y2 - size)
    if x2 - x1 < size: x1 = max(0, x2 - size)

    d, h, w = z2 - z1, y2 - y1, x2 - x1
    out_v[:d, :h, :w] = vol[z1:z2, y1:y2, x1:x2]
    out_c[:d, :h, :w] = cow[z1:z2, y1:y2, x1:x2]
    out_s[:d, :h, :w] = skel[z1:z2, y1:y2, x1:x2]
    return out_v, out_c, out_s


# ── Datasets ──────────────────────────────────────────────────────────────────

class CoarseDataset(Dataset):
    """Dataset for Coarse CoW segmentator.

    Loads NPZ files (keys: 'vol', 'cow'), resamples to COARSE_SIZE³ and
    normalises the volume. Pre-caches all data in RAM on first call to
    :meth:`build_cache` to eliminate repeated zoom calls during training.

    Args:
        ids     : list of series UIDs (stems of NPZ files)
        npz_dir : directory containing ``{sid}.npz`` files
        augment : apply random flips / rotations / noise when True
    """

    def __init__(
        self,
        ids:     list[str],
        npz_dir: Path,
        augment: bool = False,
    ) -> None:
        self.ids     = ids
        self.npz_dir = Path(npz_dir)
        self.augment = augment
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        self.tfm = Compose([
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        ]) if augment else None

    def build_cache(self) -> None:
        """Pre-compute zoom + normalise for all series. Call once before training."""
        for sid in tqdm(self.ids, desc="Caching CoarseDataset"):
            d   = np.load(self.npz_dir / f"{sid}.npz")
            vol = normalize_vol(d["vol"].astype(np.float32))
            vol = resample_vol(vol, COARSE_SIZE, order=1)
            cow = resample_vol(d["cow"].astype(np.float32), COARSE_SIZE, order=0)
            cow = (cow > 0.5).astype(np.float32)
            self.cache[sid] = (vol, cow)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        sid      = self.ids[idx]
        vol, cow = self.cache[sid]
        vol, cow = vol.copy()[None], cow.copy()[None]  # (1, 64, 64, 64)

        if self.augment and self.tfm:
            t = self.tfm({"image": vol, "mask": cow})
            vol, cow = t["image"], t["mask"]

        return {
            "image": torch.from_numpy(np.asarray(vol, dtype=np.float32)),
            "mask":  torch.from_numpy(np.asarray(cow, dtype=np.float32)),
        }


class FineDataset(Dataset):
    """Dataset for Fine CoW segmentator.

    Pre-caches full-resolution (vol, cow, skel) volumes in RAM as float16.
    Patch crop is applied on-the-fly so each epoch sees different patches.

    Args:
        ids      : list of series UIDs
        npz_dir  : directory with ``{sid}.npz`` files (keys: vol, cow)
        skel_dir : directory with ``{sid}.npz`` files (key: skel)
        augment  : apply spatial + intensity augmentations when True
        crop_size: edge length of the extracted 3-D patch
    """

    def __init__(
        self,
        ids:       list[str],
        npz_dir:   Path,
        skel_dir:  Path,
        augment:   bool = False,
        crop_size: int  = CROP_SIZE,
    ) -> None:
        self.ids       = ids
        self.npz_dir   = Path(npz_dir)
        self.skel_dir  = Path(skel_dir)
        self.augment   = augment
        self.crop_size = crop_size
        self.cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        self.tfm = Compose([
            RandFlipd(keys=["image", "mask", "skel"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "mask", "skel"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "mask", "skel"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "mask", "skel"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.3, std=0.05),
            RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        ]) if augment else None

    def build_cache(self) -> None:
        """Pre-load all volumes to RAM as float16."""
        for sid in tqdm(self.ids, desc="Caching FineDataset"):
            d    = np.load(self.npz_dir / f"{sid}.npz")
            vol  = normalize_vol(d["vol"].astype(np.float32)).astype(np.float16)
            cow  = (d["cow"] > 0.5).astype(np.float16)
            s    = np.load(self.skel_dir / f"{sid}.npz")
            skel = (s["skel"] > 0.5).astype(np.float16)
            self.cache[sid] = (vol, cow, skel)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        sid           = self.ids[idx]
        vol, cow, skel = self.cache[sid]
        vol  = vol.astype(np.float32)
        cow  = cow.astype(np.float32)
        skel = skel.astype(np.float32)

        vol_p, cow_p, skel_p = mask_crop(
            vol, cow, skel, self.crop_size, random=self.augment
        )
        vol_p  = vol_p[None]   # (1, C, C, C)
        cow_p  = cow_p[None]
        skel_p = skel_p[None]

        if self.augment and self.tfm:
            t      = self.tfm({"image": vol_p, "mask": cow_p, "skel": skel_p})
            vol_p  = np.asarray(t["image"],  dtype=np.float32)
            cow_p  = np.asarray(t["mask"],   dtype=np.float32)
            skel_p = np.asarray(t["skel"],   dtype=np.float32)

        return {
            "image": torch.from_numpy(vol_p),
            "mask":  torch.from_numpy(cow_p),
            "skel":  torch.from_numpy(skel_p),
        }


# ── Split helper ──────────────────────────────────────────────────────────────

def make_split(
    npz_dir:   Path,
    val_size:  float = 0.15,
    seed:      int   = 42,
) -> tuple[list[str], list[str]]:
    """Scan *npz_dir* for .npz files and return (train_ids, val_ids)."""
    ids = sorted(p.stem for p in Path(npz_dir).glob("*.npz"))
    return train_test_split(ids, test_size=val_size, random_state=seed)
