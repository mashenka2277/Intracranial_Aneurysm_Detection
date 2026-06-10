"""
Diagnostic report generator for intracranial aneurysm detection.

Produces a structured 2-panel matplotlib figure consistent with ESR SR:
    Left  : axial DICOM slice with CoW mask overlay + aneurysm marker
    Right : detection metadata + CoW anatomy visualisation (max projection)

Usage:
    report = DiagnosticReport(
        pkl_path   = "path/to/series.pkl",
        ckpt_swin  = "path/to/best_swin.pth",
        ckpt_fine1 = "path/to/best_fine_dicece_skel.pth",
        ckpt_fine2 = "path/to/best_fine_tversky_skel.pth",
    )
    fig = report.generate(dicom_dir="path/to/dicom/series")
    fig.savefig("report.png", dpi=200, bbox_inches="tight")
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet, SwinUNETR
from scipy.ndimage import zoom as nd_zoom

# ── Colour scheme ─────────────────────────────────────────────────────────────
DARK = "#0d1117"
TEXT = "#e6edf3"

# Per-class CoW colours (classes 1–13 match LOCATION_COLS order)
CLASS_COLORS: dict[int, str] = {
    1:  "#2196F3",  # L.ICA inf
    2:  "#2196F3",  # R.ICA inf
    3:  "#1976D2",  # L.ICA sup
    4:  "#1976D2",  # R.ICA sup
    5:  "#43A047",  # L.MCA
    6:  "#43A047",  # R.MCA
    7:  "#E53935",  # ACoA
    8:  "#FF8F00",  # L.ACA
    9:  "#FF8F00",  # R.ACA
    10: "#8E24AA",  # L.PCoA
    11: "#8E24AA",  # R.PCoA
    12: "#00897B",  # Basilar Tip
    13: "#6D4C41",  # Other Posterior
}

CLASS_NAMES: dict[int, str] = {
    1: "L.ICA inf",  2: "R.ICA inf",
    3: "L.ICA sup",  4: "R.ICA sup",
    5: "L.MCA",      6: "R.MCA",
    7: "ACoA",       8: "L.ACA",
    9: "R.ACA",      10: "L.PCoA",
    11: "R.PCoA",    12: "Basilar",
    13: "Other",
}

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


# ── Model loaders ─────────────────────────────────────────────────────────────

def _load_fine(ckpt_path: str | Path, device: torch.device) -> nn.Module:
    m = BasicUNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        features=(32, 64, 128, 256, 512, 32),
        act="LeakyReLU", norm="instance", dropout=0.1,
    ).to(device)
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    st = {k.replace("module.", ""): v for k, v in ck["model"].items()}
    m.load_state_dict(st)
    m.eval()
    print(f"✓ {Path(ckpt_path).name}: epoch={ck['epoch']}  dice={ck['dice']:.4f}")
    return m


def _load_swin_detector(ckpt_path: str | Path, device: torch.device) -> nn.Module:
    """Load SwinUNETRDetector from checkpoint. Minimal import to avoid circular deps."""
    from detection.models.swin_unetr import SwinUNETRDetector  # type: ignore

    model = SwinUNETRDetector(in_channels=2, n_zones=13).to(device)
    ck    = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"✓ Swin detector: epoch={ck['epoch']}  AUC={ck['auc']:.4f}")
    return model


# ── CoW mask helpers ──────────────────────────────────────────────────────────

def _make_rgba(cow_sl: np.ndarray) -> np.ndarray:
    """Convert integer class mask to RGBA image using CLASS_COLORS."""
    rgba = np.zeros((*cow_sl.shape, 4), dtype=np.float32)
    for cls_id, color in CLASS_COLORS.items():
        mask = cow_sl == cls_id
        if mask.any():
            r, g, b = mcolors.to_rgb(color)
            rgba[mask] = (r, g, b, 0.85)
    return rgba


def _fine_ensemble_cow(
    vol:    np.ndarray,
    fine1:  nn.Module,
    fine2:  nn.Module,
    device: torch.device,
    size:   int = 128,
) -> np.ndarray:
    """Run Fine-1 + Fine-2 ensemble to predict CoW mask on a crop.

    Returns binary float32 mask, same shape as *vol*.
    """
    inp = torch.from_numpy(vol[None, None].astype(np.float32)).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        p1 = torch.sigmoid(sliding_window_inference(
            inp, (size,) * 3, 2, fine1, overlap=0.5, mode="gaussian"))
        p2 = torch.sigmoid(sliding_window_inference(
            inp, (size,) * 3, 2, fine2, overlap=0.5, mode="gaussian"))
    return ((p1 + p2) / 2 > 0.5)[0, 0].cpu().numpy().astype(np.float32)


# ── Main report class ─────────────────────────────────────────────────────────

class DiagnosticReport:
    """Generate a structured diagnostic report figure.

    Args:
        pkl_path   : path to pre-processed cache .pkl file
        ckpt_swin  : path to Swin UNETR detector checkpoint
        ckpt_fine1 : path to Fine-1 CoW segmentator checkpoint
        ckpt_fine2 : path to Fine-2 CoW segmentator checkpoint
        device     : torch device (default: auto)
    """

    def __init__(
        self,
        pkl_path:   str | Path,
        ckpt_swin:  str | Path,
        ckpt_fine1: str | Path,
        ckpt_fine2: str | Path,
        device:     torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load cache
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.vol      = self.data["vol"].astype(np.float32)
        self.cow_mask = self.data["cow_mask"].astype(np.float32)
        self.labels   = self.data["labels"]
        self.modality = self.data.get("modality", "CTA")
        self.sid      = self.data.get("sid", "unknown")
        self.spacing  = self.data.get("spacing", (1.0, 1.0, 1.0))
        self.crop_box = self.data.get("crop_box", (0, self.vol.shape[0],
                                                    0, self.vol.shape[1],
                                                    0, self.vol.shape[2]))

        # Load models
        self.fine1 = _load_fine(ckpt_fine1, self.device)
        self.fine2 = _load_fine(ckpt_fine2, self.device)
        self.detector = _load_swin_detector(ckpt_swin, self.device)

    def _run_detector(self) -> tuple[float, int]:
        """Run Swin UNETR detector. Returns (probability, predicted_zone_idx)."""
        vol_t = torch.from_numpy(self.vol[None, None].astype(np.float32)).to(self.device)
        cow_t = torch.from_numpy(self.cow_mask[None, None].astype(np.float32)).to(self.device)
        x = torch.cat([vol_t, cow_t], dim=1)
        mod = torch.tensor([1.0 if self.modality == "MRA" else 0.0]).to(self.device)

        with torch.no_grad():
            ob, om, _ = self.detector(x, mod)
        prob     = float(torch.sigmoid(ob).cpu().item())
        zone_idx = int(torch.sigmoid(om).cpu().argmax().item())
        return prob, zone_idx

    def generate(
        self,
        dicom_dir:   str | Path | None = None,
        save_path:   str | Path | None = None,
        figsize:     tuple[int, int]   = (16, 8),
    ) -> plt.Figure:
        """Generate and optionally save the diagnostic report figure.

        Args:
            dicom_dir  : directory of original DICOM files (for axial slice).
                         If None, uses the normalised crop volume.
            save_path  : if provided, save PNG to this path.
            figsize    : figure size in inches.

        Returns:
            matplotlib Figure
        """
        prob, zone_idx = self._run_detector()
        an_present     = prob >= 0.5
        an_loc         = LOCATION_COLS[zone_idx] if an_present else "—"

        # ── Load axial DICOM slice ────────────────────────────────────────────
        if dicom_dir is not None:
            dcm_files = sorted(Path(dicom_dir).glob("*.dcm"))
            slices    = sorted(
                [pydicom.dcmread(str(f)) for f in dcm_files],
                key=lambda s: float(s.ImagePositionPatient[2]),
            )
            z_crop_mid   = self.vol.shape[0] // 2
            z0           = self.crop_box[0]
            z_dicom      = min(z0 + z_crop_mid, len(slices) - 1)
            ds           = slices[z_dicom]
            raw          = ds.pixel_array.astype(np.float32)
            slope        = float(getattr(ds, "RescaleSlope", 1))
            intercept    = float(getattr(ds, "RescaleIntercept", 0))
            hu_slice     = raw * slope + intercept
            dicom_slice  = np.clip((hu_slice - (400 - 300)) / 600, 0, 1)
            H_dcm, W_dcm = dicom_slice.shape
        else:
            z_dicom     = self.vol.shape[0] // 2
            dicom_slice = self.vol[z_dicom]
            H_dcm, W_dcm = dicom_slice.shape

        # ── CoW max-projection mask ───────────────────────────────────────────
        # Use Fine ensemble prediction as coloured CoW overlay
        pred_fine = _fine_ensemble_cow(self.vol, self.fine1, self.fine2, self.device)

        # For colour display we need class labels — use cow_mask directly if available
        # as integer-class NIfTI, otherwise use binary fine prediction
        cow_binary = (pred_fine > 0.5).astype(np.uint8)

        z_best   = int(cow_binary.sum(axis=(1, 2)).argmax())
        cow_one  = cow_binary[z_best]
        cow_wide = cow_binary[
            max(0, z_best - 15): min(cow_binary.shape[0], z_best + 15)
        ].max(axis=0)

        # Simple single-colour overlay (cyan) when no per-class labels
        rgba_one  = np.zeros((*cow_one.shape, 4),  dtype=np.float32)
        rgba_wide = np.zeros((*cow_wide.shape, 4), dtype=np.float32)
        rgba_one[cow_one > 0]   = [0.26, 0.76, 0.76, 0.85]
        rgba_wide[cow_wide > 0] = [0.26, 0.76, 0.76, 0.85]

        # CoW centroid in DICOM pixel space
        if cow_binary.sum() > 0:
            yz  = np.argwhere(cow_wide > 0).mean(axis=0)
            px_y, px_x = float(yz[0]), float(yz[1])
        else:
            px_y, px_x = H_dcm / 2, W_dcm / 2

        # ── Build figure ─────────────────────────────────────────────────────
        fig = plt.figure(figsize=figsize, facecolor=DARK)
        gs  = GridSpec(2, 2, figure=fig,
                       hspace=0.15, wspace=0.25,
                       left=0.03, right=0.97, top=0.92, bottom=0.03,
                       height_ratios=[1, 1.4])

        ax_slice = fig.add_subplot(gs[:, 0])
        ax_meta  = fig.add_subplot(gs[0, 1])
        ax_cow   = fig.add_subplot(gs[1, 1])

        for ax in [ax_slice, ax_meta, ax_cow]:
            ax.set_facecolor(DARK)
            for sp in ax.spines.values():
                sp.set_edgecolor("#21262d")

        # Left: axial slice + CoW overlay + aneurysm marker
        ax_slice.imshow(dicom_slice, cmap="gray", vmin=0, vmax=1)
        ax_slice.imshow(rgba_one, interpolation="nearest", alpha=0.85)
        ax_slice.plot(px_x, px_y, "+", color="#f85149",
                      markersize=18, markeredgewidth=2.5)
        c = Circle((px_x, px_y), radius=18,
                   fill=False, edgecolor="#f85149", lw=2.5)
        ax_slice.add_patch(c)
        if an_present:
            ax_slice.annotate(
                f"Аневризма\n{an_loc[:20]}",
                xy=(px_x, px_y), xytext=(px_x + 40, px_y - 40),
                color="#f85149", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#f85149", lw=1.5),
            )
        ax_slice.set_title(
            f"Axial  z={z_dicom}  |  {self.modality}",
            color=TEXT, fontsize=11, pad=4,
        )
        ax_slice.axis("off")

        # Right top: metadata
        ax_meta.axis("off")
        FS = 12.5
        status_color = "#f85149" if an_present else "#3fb950"
        status_text  = "ВИЯВЛЕНА ✓" if an_present else "НЕ ВИЯВЛЕНА"
        lines = [
            (f"Аневризма:  {status_text}", status_color),
            (f"Ймовірність:  p = {prob:.3f}", TEXT),
            (f"Судина:  {an_loc[:30]}", TEXT),
            (f"Впевненість:  {prob * 100:.1f}%", TEXT),
            (f"x={px_x:.0f}  y={px_y:.0f}  z={z_dicom}", TEXT),
            ("", TEXT),
            ("Модель:  Swin UNETR", "#8b949e"),
            (f"Модальність:  {self.modality}", "#8b949e"),
        ]
        y = 0.96
        for text, color in lines:
            ax_meta.text(0.05, y, text, color=color, fontsize=FS,
                         transform=ax_meta.transAxes, va="top")
            y -= 0.105 if text else 0.03

        # Right bottom: CoW max projection
        rows = np.where(cow_wide.any(axis=1))[0]
        cols = np.where(cow_wide.any(axis=0))[0]
        if len(rows) > 0 and len(cols) > 0:
            pad = 10
            r1 = max(0, rows[0] - pad); r2 = min(cow_wide.shape[0], rows[-1] + pad)
            c1 = max(0, cols[0] - pad); c2 = min(cow_wide.shape[1], cols[-1] + pad)
            cow_crop = rgba_wide[r1:r2, c1:c2]
        else:
            cow_crop = rgba_wide

        inset = ax_cow.inset_axes([0.1, 0.05, 0.8, 0.85])
        inset.set_facecolor(DARK)
        inset.imshow(cow_crop, interpolation="nearest")
        inset.axis("off")
        ax_cow.set_xlim(0, 1); ax_cow.set_ylim(0, 1)
        ax_cow.axis("off")
        ax_cow.set_title(
            "Вілізієве коло (CoW)  |  max proj ±15 зрізів",
            color=TEXT, fontsize=9.5, pad=4,
        )

        fig.suptitle(
            f"Діагностичний звіт  |  RSNA-2025  |  {self.modality}  |  {self.sid[:40]}",
            color=TEXT, fontsize=11, fontweight="bold",
        )

        if save_path:
            fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor=DARK)
            print(f"✓  Report saved to {save_path}")

        return fig
