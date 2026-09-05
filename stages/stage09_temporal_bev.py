from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary
from utils.visualization import save_heatmap, tensor_magnitude

STAGE_NUMBER = 9
STAGE_NAME = "Temporal BEV: ego-motion alignment + memory fusion"
SHORT_NAME = "temporal_bev"


class ConvGRUCell(nn.Module):
    """A standard convolutional GRU cell for spatial memory."""

    def __init__(self, channels: int):
        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=1)
        z, r = torch.sigmoid(self.gates(combined)).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * candidate


class TemporalConvGRU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.cell = ConvGRUCell(channels)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: [T,C,H,W]
        h = torch.zeros_like(sequence[0:1])
        for t in range(sequence.shape[0]):
            h = self.cell(sequence[t:t+1], h)
        return h[0]


def warp_bev_to_current(
    source: torch.Tensor,
    T_current_from_source: np.ndarray,
    cfg: PipelineConfig,
) -> torch.Tensor:
    """Warp a source-frame BEV into current-ego coordinates using grid_sample.

    The output pixel asks: "which position in the source BEV corresponds to this
    current-frame metric coordinate?" Therefore we use the inverse transform
    T_source_from_current when building the sampling grid.
    """
    device, dtype = source.device, source.dtype
    C, H, W = source.shape
    T_source_from_current = np.linalg.inv(T_current_from_source)
    T = torch.as_tensor(T_source_from_current, device=device, dtype=dtype)

    rows = torch.arange(H, device=device, dtype=dtype)
    cols = torch.arange(W, device=device, dtype=dtype)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    x_cur = cfg.bev_x_min + (rr + 0.5) * cfg.bev_resolution
    y_cur = cfg.bev_y_min + (cc + 0.5) * cfg.bev_resolution
    zeros = torch.zeros_like(x_cur)
    ones = torch.ones_like(x_cur)
    p_cur = torch.stack([x_cur, y_cur, zeros, ones], dim=0).reshape(4, -1)
    p_src = T @ p_cur
    x_src = p_src[0].reshape(H, W)
    y_src = p_src[1].reshape(H, W)

    # grid_sample expects grid[...,0] = horizontal/W coordinate and grid[...,1] = vertical/H.
    grid_x = 2.0 * (y_src - cfg.bev_y_min) / (cfg.bev_y_max - cfg.bev_y_min) - 1.0
    grid_y = 2.0 * (x_src - cfg.bev_x_min) / (cfg.bev_x_max - cfg.bev_x_min) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    warped = F.grid_sample(
        source.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return warped[0]


def temporal_ema(aligned: torch.Tensor) -> torch.Tensor:
    """Interpretable deterministic baseline: newer frames get larger weights."""
    T = aligned.shape[0]
    weights = torch.linspace(1.0, 2.0, T, device=aligned.device, dtype=aligned.dtype)
    weights = weights / weights.sum()
    return (aligned * weights.view(T, 1, 1, 1)).sum(dim=0)


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("fused_bev", "T_current_from_history")

    fused = ctx.get("fused_bev")
    transforms = ctx.get("T_current_from_history")

    log.substage(9, 1, "Warp all historical BEVs into the current ego frame")
    aligned = []
    for t in range(fused.shape[0]):
        warped = warp_bev_to_current(fused[t], transforms[t], cfg)
        aligned.append(warped)
        log.detail(f"frame {t}: spatial BEV warped into current coordinates")
        if cfg.save_plots:
            save_heatmap(
                tensor_magnitude(warped),
                stage_dir / f"aligned_frame_{t:02d}.png",
                f"Stage 09: aligned BEV frame {t}",
            )
    aligned_bev = torch.stack(aligned, dim=0)
    log.tensor("aligned_bev [T,C,H,W]", aligned_bev)

    log.substage(9, 2, "Fuse temporal memory")
    ema_bev = temporal_ema(aligned_bev)
    log.tensor("temporal_ema_bev", ema_bev)

    convgru_model = TemporalConvGRU(aligned_bev.shape[1]).to(cfg.torch_device).eval()
    with torch.no_grad():
        convgru_bev = convgru_model(aligned_bev)
    log.tensor("temporal_convgru_bev", convgru_bev)
    log.detail("ConvGRU is architecturally real but untrained. EMA is the default visual baseline until training exists.")

    if cfg.temporal_model.lower() == "convgru":
        temporal_bev = convgru_bev
        selected = "convgru (untrained unless checkpoint added later)"
    else:
        temporal_bev = ema_bev
        selected = "recency-weighted aligned EMA"
    log.info(f"Selected temporal output = {selected}")

    log.substage(9, 3, "Compare current-only vs temporal representation")
    if cfg.save_plots:
        save_heatmap(tensor_magnitude(fused[-1]), stage_dir / "current_only_bev.png", "Stage 09: current spatial BEV only")
        save_heatmap(tensor_magnitude(temporal_bev), stage_dir / "temporal_bev.png", "Stage 09: selected temporal BEV")

    values = {
        "aligned_bev": aligned_bev,
        "temporal_ema_bev": ema_bev,
        "temporal_convgru_bev": convgru_bev,
        "temporal_bev": temporal_bev,
        "temporal_model_selected": selected,
        "temporal_convgru_model": convgru_model,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("Spatial information from several moments is now represented in one current-frame temporal BEV.")
