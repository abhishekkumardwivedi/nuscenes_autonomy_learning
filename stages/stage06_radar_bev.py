from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nuscenes.utils.data_classes import RadarPointCloud

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.geometry import transform_matrix, metric_to_bev_indices
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary
from utils.nuscenes_utils import RADAR_CHANNELS
from utils.visualization import save_heatmap

STAGE_NUMBER = 6
STAGE_NAME = "Radar points -> radar BEV encoder"
SHORT_NAME = "radar_bev"


class RadarBEVEncoder(nn.Module):
    """Tiny CNN that converts interpretable radar grid channels into features."""

    def __init__(self, in_channels: int = 4, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _radar_grid_one_sample(nusc, sample, cfg: PipelineConfig):
    """Aggregate all five radar sensors into 4 human-readable BEV channels.

    channels:
      0 = point count
      1 = mean radar cross section (RCS)
      2 = mean compensated velocity in ego x
      3 = mean compensated velocity in ego y
    """
    H, W = cfg.bev_height, cfg.bev_width
    count = np.zeros((H, W), dtype=np.float32)
    rcs_sum = np.zeros((H, W), dtype=np.float32)
    vx_sum = np.zeros((H, W), dtype=np.float32)
    vy_sum = np.zeros((H, W), dtype=np.float32)
    all_points = []

    for channel in RADAR_CHANNELS:
        sd_token = sample["data"][channel]
        sd = nusc.get("sample_data", sd_token)
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pc = RadarPointCloud.from_file(nusc.get_sample_data_path(sd_token))
        pts = pc.points  # 18 x N
        if pts.shape[1] == 0:
            continue

        T_ego_from_radar = transform_matrix(calib["translation"], calib["rotation"])
        xyz_sensor = pts[:3, :]
        xyz_h = np.vstack([xyz_sensor, np.ones((1, xyz_sensor.shape[1]))])
        xyz_ego = (T_ego_from_radar @ xyz_h)[:3, :]

        # Compensated planar velocity indices in nuScenes radar format.
        v_sensor = np.vstack([pts[8, :], pts[9, :], np.zeros(pts.shape[1])])
        v_ego = T_ego_from_radar[:3, :3] @ v_sensor
        rcs = pts[5, :]

        rows, cols, valid = metric_to_bev_indices(
            xyz_ego[0], xyz_ego[1],
            cfg.bev_x_min, cfg.bev_y_min, cfg.bev_resolution, H, W,
        )
        valid_idx = np.flatnonzero(valid)
        rows, cols = rows[valid], cols[valid]
        for src_i, r, c in zip(valid_idx, rows, cols):
            count[r, c] += 1.0
            rcs_sum[r, c] += float(rcs[src_i])
            vx_sum[r, c] += float(v_ego[0, src_i])
            vy_sum[r, c] += float(v_ego[1, src_i])

        if valid.any():
            all_points.append(
                np.column_stack([
                    xyz_ego[0, valid], xyz_ego[1, valid], rcs[valid],
                    v_ego[0, valid], v_ego[1, valid]
                ])
            )

    denom = np.maximum(count, 1.0)
    rcs_mean = rcs_sum / denom
    vx_mean = vx_sum / denom
    vy_mean = vy_sum / denom

    # Stable scales make the four channels easier for a small CNN to consume.
    count_scaled = np.log1p(count) / np.log(8.0)
    rcs_scaled = np.clip(rcs_mean / 30.0, -1.0, 1.0)
    vx_scaled = np.clip(vx_mean / 20.0, -1.0, 1.0)
    vy_scaled = np.clip(vy_mean / 20.0, -1.0, 1.0)
    grid = np.stack([count_scaled, rcs_scaled, vx_scaled, vy_scaled], axis=0).astype(np.float32)
    points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 5), dtype=np.float32)
    return grid, points, int(count.sum())


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 06: preserve radar physics first, then encode the BEV channels."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("nusc", "history_samples")
    nusc = ctx.get("nusc")
    samples = ctx.get("history_samples")

    log.substage(6, 1, "Load five radar point clouds per time step")
    grids = []
    points_by_time = []
    radar_counts = []
    for t, sample in enumerate(samples):
        grid, points, count = _radar_grid_one_sample(nusc, sample, cfg)
        grids.append(torch.from_numpy(grid))
        points_by_time.append(points)
        radar_counts.append(count)
        log.detail(f"Frame {t}: {count} in-range radar returns after five-sensor aggregation")
    radar_raw = torch.stack(grids, dim=0).to(cfg.torch_device)
    log.tensor("radar_bev_raw [T,4,H,W]", radar_raw)

    log.substage(6, 2, "Interpret the four raw radar channels")
    log.info("ch0=count, ch1=mean RCS, ch2=compensated vx, ch3=compensated vy")
    if cfg.save_plots:
        save_heatmap(radar_raw[-1, 0].detach().cpu().numpy(), stage_dir / "current_radar_count.png", "Stage 06: radar return density")
        save_heatmap(radar_raw[-1, 2].detach().cpu().numpy(), stage_dir / "current_radar_vx.png", "Stage 06: compensated radar velocity x")

    log.substage(6, 3, "Encode radar grid into learned feature channels")
    encoder = RadarBEVEncoder(4, 32).to(cfg.torch_device).eval()
    with torch.no_grad():
        radar_features = encoder(radar_raw)
    log.tensor("radar_bev_features [T,32,H,W]", radar_features)
    log.detail("The CNN is untrained at this point; the raw 4-channel radar plots remain the meaningful visual baseline.")

    values = {
        "radar_bev_raw": radar_raw,
        "radar_bev_features": radar_features,
        "radar_points_by_time": points_by_time,
        "radar_point_counts": radar_counts,
        "radar_encoder_model": encoder,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("Five asynchronous-looking radar views are represented in the same ego-centric BEV grid as the cameras.")
