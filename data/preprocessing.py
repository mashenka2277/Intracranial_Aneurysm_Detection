"""
Preprocessing utilities for RSNA Intracranial Aneurysm Detection.

Pipeline:
    DICOM / NIfTI  →  resample to 1×1×1 mm  →  normalize  →  CoW crop
"""

from __future__ import annotations

import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────

COARSE_SIZE = 64          # input size for Coarse model (voxels)
ROI_MM      = (180, 180, 180)  # fixed crop around CoW center (mm = voxels at 1mm)

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

TARGET_MODALITIES = ("CT", "CTA", "MR", "MRA", "MR ")


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load NIfTI volume.

    Returns:
        vol     : float32 array (Z, H, W)
        spacing : (sz, sy, sx) in mm
    """
    nii = nib.load(str(path))
    vol = np.transpose(nii.get_fdata().astype(np.float32), (2, 1, 0))
    z   = nii.header.get_zooms()
    return vol, (float(z[2]), float(z[1]), float(z[0]))


def load_dicom(series_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load DICOM series as float32 volume in Hounsfield Units.

    Returns:
        vol     : float32 array (Z, H, W) in HU
        spacing : (slice_thickness, row_spacing, col_spacing) in mm
    """
    dcm_files = sorted(series_path.glob("*.dcm"))
    if not dcm_files:
        raise ValueError(f"No DICOM files in {series_path}")

    slices = [pydicom.dcmread(str(f)) for f in dcm_files]
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        pass

    vol = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    ds0 = slices[0]

    # HU conversion
    vol = vol * float(getattr(ds0, "RescaleSlope", 1.0)) + \
               float(getattr(ds0, "RescaleIntercept", 0.0))

    # Spacing — fallback chain
    st = float(getattr(ds0, "SliceThickness", 1.0))
    if hasattr(ds0, "PixelSpacing"):
        row_sp, col_sp = float(ds0.PixelSpacing[0]), float(ds0.PixelSpacing[1])
    elif hasattr(ds0, "ImagerPixelSpacing"):
        row_sp, col_sp = float(ds0.ImagerPixelSpacing[0]), float(ds0.ImagerPixelSpacing[1])
    else:
        row_sp, col_sp = 1.0, 1.0

    return vol, (st, row_sp, col_sp)


# ── Spatial ───────────────────────────────────────────────────────────────────

def resample_to_iso(
    vol: np.ndarray,
    spacing: tuple[float, float, float],
    target: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Resample volume to isotropic voxel spacing using trilinear interpolation.

    Args:
        vol     : (Z, H, W) float32
        spacing : original voxel spacing (sz, sy, sx) in mm
        target  : desired spacing, default 1×1×1 mm

    Returns:
        resampled float32 array
    """
    factors = tuple(s / t for s, t in zip(spacing, target))
    return zoom(vol, factors, order=1).astype(np.float32)


def safe_crop(
    center: tuple[int, int, int],
    shape: tuple[int, int, int],
    roi: tuple[int, int, int] = ROI_MM,
) -> tuple[int, int, int, int, int, int]:
    """Compute crop indices for a fixed-size ROI centred at *center*.

    Indices are clamped so the crop stays within the volume.

    Returns:
        (z1, z2, y1, y2, x1, x2)
    """
    cz, cy, cx = center
    Z,  H,  W  = shape
    rz, ry, rx = roi

    z1 = max(0, cz - rz // 2); z2 = min(Z, z1 + rz)
    y1 = max(0, cy - ry // 2); y2 = min(H, y1 + ry)
    x1 = max(0, cx - rx // 2); x2 = min(W, x1 + rx)

    # shift back if crop was clamped at the far edge
    if z2 - z1 < rz: z1 = max(0, z2 - rz)
    if y2 - y1 < ry: y1 = max(0, y2 - ry)
    if x2 - x1 < rx: x1 = max(0, x2 - rx)

    return z1, z2, y1, y2, x1, x2


# ── Intensity ─────────────────────────────────────────────────────────────────

def normalize(vol: np.ndarray, modality: str) -> np.ndarray:
    """Modality-aware intensity normalisation to [0, 1].

    - CTA : clip to vascular HU window  [-200, 900]
    - MRA : clip to per-volume percentiles [p1, p99]

    Args:
        vol      : float32 array in HU (CTA) or raw signal (MRA)
        modality : DICOM modality string, e.g. 'CTA', 'MRA', 'MR'

    Returns:
        float32 array in [0, 1]
    """
    mod = modality.upper().strip()
    if "CT" in mod:
        lo, hi = -200.0, 900.0
    else:
        lo = float(np.percentile(vol, 1))
        hi = float(np.percentile(vol, 99))

    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo + 1e-8)).astype(np.float32)


# ── Model utilities ───────────────────────────────────────────────────────────

def load_cow_model(
    features: tuple,
    ckpt_path: Path,
    device: torch.device,
) -> nn.Module:
    """Build BasicUNet and load CoW segmentation checkpoint.

    Args:
        features  : feature channel sizes, e.g. (16, 32, 64, 128, 256, 16)
        ckpt_path : path to .pth checkpoint
        device    : torch device

    Returns:
        model in eval mode
    """
    from monai.networks.nets import BasicUNet  # local import to keep optional

    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=features,
        act="LeakyReLU",
        norm="instance",
        dropout=0.1,
    ).to(device)

    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def coarse_inference(
    model: nn.Module,
    vol: np.ndarray,
    device: torch.device,
    coarse_size: int = COARSE_SIZE,
) -> tuple[int, int, int]:
    """Run Coarse BasicUNet and find CoW centre via DBSCAN.

    The volume is downsampled to *coarse_size³*, the model predicts a binary
    foreground mask, and DBSCAN clusters the foreground voxels to find the
    dominant cluster centroid, which is then mapped back to original coordinates.

    Args:
        model       : Coarse BasicUNet in eval mode
        vol         : normalised float32 volume (Z, H, W)
        device      : torch device
        coarse_size : edge length for downsampling (default 64)

    Returns:
        (cz, cy, cx) — centroid in original voxel coordinates
    """
    Z, H, W = vol.shape
    vs  = zoom(vol, (coarse_size / Z, coarse_size / H, coarse_size / W), order=1)
    inp = torch.from_numpy(vs[None, None].astype(np.float32)).to(device)

    with torch.no_grad():
        pred = (torch.sigmoid(model(inp)) > 0.5)[0, 0].cpu().numpy()

    coords = np.column_stack(np.where(pred > 0))
    if len(coords) >= 10:
        db    = DBSCAN(eps=3, min_samples=5).fit(coords)
        lbl   = db.labels_
        valid = lbl[lbl >= 0]
        if len(valid) > 0:
            best     = np.bincount(valid).argmax()
            center_s = coords[lbl == best].mean(axis=0)
        else:
            center_s = coords.mean(axis=0)
    elif len(coords) > 0:
        center_s = coords.mean(axis=0)
    else:
        center_s = np.array([coarse_size // 2] * 3, dtype=float)

    return (
        int(center_s[0] * Z / coarse_size),
        int(center_s[1] * H / coarse_size),
        int(center_s[2] * W / coarse_size),
    )


def fine_ensemble_inference(
    m1: nn.Module,
    m2: nn.Module,
    crop: np.ndarray,
    device: torch.device,
    crop_size: int = 128,
) -> np.ndarray:
    """Fine-1 + Fine-2 sliding-window ensemble on a 180³ CoW crop.

    Both models run with 50 % overlap and Gaussian weighting; their probability
    maps are averaged and thresholded at 0.5 to produce a binary CoW mask.

    Args:
        m1, m2     : Fine BasicUNet models in eval mode
        crop       : float32 volume (≈180³) centred on CoW
        device     : torch device
        crop_size  : sliding window patch size (default 128)

    Returns:
        binary float16 mask, same spatial shape as *crop*
    """
    from monai.inferers import sliding_window_inference  # optional dependency

    inp = torch.from_numpy(crop[None, None].astype(np.float32)).to(device)

    with torch.no_grad():
        p1 = torch.sigmoid(sliding_window_inference(
            inp, (crop_size,) * 3, sw_batch_size=2,
            predictor=m1, overlap=0.5, mode="gaussian",
        ))
        p2 = torch.sigmoid(sliding_window_inference(
            inp, (crop_size,) * 3, sw_batch_size=2,
            predictor=m2, overlap=0.5, mode="gaussian",
        ))

    return ((p1 + p2) / 2 > 0.5)[0, 0].cpu().numpy().astype(np.float16)
