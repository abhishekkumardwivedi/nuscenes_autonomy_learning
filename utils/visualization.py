from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch


def normalize_01(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(array, dtype=np.float32)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return (x - lo) / max(hi - lo, eps)


def tensor_magnitude(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.detach().float().cpu()
    if t.ndim == 3:  # C,H,W
        return torch.linalg.vector_norm(t, dim=0).numpy()
    if t.ndim == 2:
        return t.numpy()
    raise ValueError(f"Expected 2D or 3D tensor, got {tuple(t.shape)}")


def save_heatmap(array: np.ndarray, path: Path, title: str, xlabel: str = "BEV y / column", ylabel: str = "BEV x / row") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(array, origin="lower", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_camera_montage(images: Sequence[Image.Image], names: Sequence[str], path: Path) -> None:
    """Save six camera views in a 2x3 grid."""
    n = len(images)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, (img, name) in enumerate(zip(images, names)):
        axes[i].imshow(img)
        axes[i].set_title(name)
        axes[i].axis("off")
    fig.suptitle("nuScenes camera views used by the pipeline")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_feature_montage(features: torch.Tensor, names: Sequence[str], path: Path, title: str) -> None:
    """Visualize channel-mean feature maps for N cameras."""
    f = features.detach().float().cpu()
    n = f.shape[0]
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        vis = f[i].abs().mean(dim=0).numpy()
        axes[i].imshow(vis, origin="upper")
        axes[i].set_title(names[i])
        axes[i].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_xy_plot(
    series: Sequence[np.ndarray],
    labels: Sequence[str],
    path: Path,
    title: str,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    invert_axes_for_bev: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for pts, label in zip(series, labels):
        pts = np.asarray(pts)
        if len(pts):
            # Plot y horizontally and x vertically for a natural ego-centric BEV.
            if invert_axes_for_bev:
                ax.plot(pts[:, 1], pts[:, 0], marker="o", label=label)
            else:
                ax.plot(pts[:, 0], pts[:, 1], marker="o", label=label)
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    if invert_axes_for_bev:
        ax.set_xlabel("y left (m)")
        ax.set_ylabel("x forward (m)")
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    if len(labels) <= 12:
        ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_polygons_to_mask(polygons_rc: Iterable[np.ndarray], height: int, width: int) -> np.ndarray:
    """Rasterize row/column polygons into a uint8 BEV mask."""
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons_rc:
        p = np.asarray(poly, dtype=float)
        if len(p) >= 3:
            # PIL uses (x=column, y=row).
            xy = [(float(col), float(row)) for row, col in p]
            draw.polygon(xy, fill=1)
    return np.asarray(canvas, dtype=np.uint8)
