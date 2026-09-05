from __future__ import annotations

import numpy as np
import torch

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary
from utils.visualization import save_heatmap, tensor_magnitude

STAGE_NUMBER = 5
STAGE_NAME = "Camera features -> BEV (educational lift-splat)"
SHORT_NAME = "camera_bev"


def _depth_prior(depths: torch.Tensor) -> torch.Tensor:
    """Deterministic teaching prior over depth bins.

    A production Lift-Splat model learns per-pixel depth probabilities. Before a
    depth head has been trained, random depth logits make the geometry hard to
    understand. This smooth prior keeps the result deterministic while preserving
    the actual lift -> transform -> splat mechanics.
    """
    weights = torch.exp(-depths / 22.0)
    return weights / weights.sum()


def lift_splat_one_frame(
    features: torch.Tensor,          # [6,C,Hf,Wf]
    intrinsics: np.ndarray,          # [6,3,3] scaled to resized camera image
    T_ego_from_camera: np.ndarray,   # [6,4,4]
    cfg: PipelineConfig,
) -> torch.Tensor:
    """Lift 2D feature cells along depth rays and splat into an ego-frame BEV."""
    device = features.device
    N, C, Hf, Wf = features.shape
    Hbev, Wbev = cfg.bev_height, cfg.bev_width
    output = torch.zeros((C, Hbev * Wbev), device=device, dtype=features.dtype)
    weight_sum = torch.zeros((1, Hbev * Wbev), device=device, dtype=features.dtype)

    depths = torch.linspace(cfg.depth_min, cfg.depth_max, cfg.depth_bins, device=device)
    depth_weights = _depth_prior(depths).to(features.dtype)

    # Pixel centers in the resized camera image corresponding to feature cells.
    u = (torch.arange(Wf, device=device, dtype=features.dtype) + 0.5) * (cfg.image_width / Wf)
    v = (torch.arange(Hf, device=device, dtype=features.dtype) + 0.5) * (cfg.image_height / Hf)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    uu = uu.reshape(-1)
    vv = vv.reshape(-1)

    for cam in range(N):
        K = torch.as_tensor(intrinsics[cam], dtype=features.dtype, device=device)
        T = torch.as_tensor(T_ego_from_camera[cam], dtype=features.dtype, device=device)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        feat_flat = features[cam].reshape(C, -1)

        for d, d_weight in zip(depths, depth_weights):
            # Camera coordinate convention: x right, y down, z forward.
            x_cam = (uu - cx) / fx * d
            y_cam = (vv - cy) / fy * d
            z_cam = torch.full_like(x_cam, d)
            ones = torch.ones_like(x_cam)
            p_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=0)  # [4,P]
            p_ego = T @ p_cam
            x_ego, y_ego = p_ego[0], p_ego[1]

            row = torch.floor((x_ego - cfg.bev_x_min) / cfg.bev_resolution).long()
            col = torch.floor((y_ego - cfg.bev_y_min) / cfg.bev_resolution).long()
            valid = (row >= 0) & (row < Hbev) & (col >= 0) & (col < Wbev)
            if not valid.any():
                continue
            flat_idx = row[valid] * Wbev + col[valid]
            src = feat_flat[:, valid] * d_weight
            output.scatter_add_(1, flat_idx.unsqueeze(0).expand(C, -1), src)
            n_valid = int(valid.sum().item())
            weight_values = torch.ones((1, n_valid), device=device, dtype=features.dtype) * d_weight
            weight_sum.scatter_add_(1, flat_idx.unsqueeze(0), weight_values)

    output = output / weight_sum.clamp_min(1e-6)
    return output.view(C, Hbev, Wbev)


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 05: use real calibration to turn perspective features into a top-down grid."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("camera_features", "camera_intrinsics", "T_ego_from_camera")

    feats = ctx.get("camera_features")
    intr = ctx.get("camera_intrinsics")
    extr = ctx.get("T_ego_from_camera")

    log.substage(5, 1, "Create depth bins")
    log.info(
        f"Depth range = {cfg.depth_min:.1f}..{cfg.depth_max:.1f} m with {cfg.depth_bins} bins"
    )
    log.detail("Production model: learn depth probability per pixel. Teaching mode: deterministic smooth prior.")

    log.substage(5, 2, "Lift feature cells into 3D camera rays")
    log.info("For each feature cell: (u,v) + depth -> (x_cam,y_cam,z_cam)")

    log.substage(5, 3, "Transform camera 3D points into ego coordinates")
    log.info("Apply the Stage 03 camera->ego extrinsic matrix to every lifted point.")

    log.substage(5, 4, "Splat transformed features into the BEV grid")
    camera_bevs = []
    with torch.no_grad():
        for t in range(feats.shape[0]):
            bev_t = lift_splat_one_frame(feats[t], intr[t], extr[t], cfg)
            camera_bevs.append(bev_t)
            log.detail(f"Temporal frame {t}: camera BEV produced")
    camera_bev = torch.stack(camera_bevs, dim=0)
    log.tensor("camera_bev [T,C,Hbev,Wbev]", camera_bev)

    log.substage(5, 5, "Visualize top-down camera feature magnitude")
    bev_path = stage_dir / "current_camera_bev.png"
    if cfg.save_plots:
        save_heatmap(
            tensor_magnitude(camera_bev[-1]), bev_path,
            "Stage 05: camera BEV feature magnitude (untrained depth prior)",
        )
        log.info(f"Saved camera BEV -> {bev_path}")

    values = {
        "camera_bev": camera_bev,
        "camera_bev_depth_mode": "deterministic teaching prior",
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("Perspective camera features have been geometrically re-indexed into a shared ego-centric BEV.")
