from __future__ import annotations

import torch
from torch import nn

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary
from utils.visualization import normalize_01, save_heatmap, tensor_magnitude

STAGE_NUMBER = 7
STAGE_NAME = "Spatial BEV fusion: camera + radar"
SHORT_NAME = "spatial_fusion"


class SpatialBEVFusion(nn.Module):
    """Concatenate modality features and learn local cross-modal mixing."""

    def __init__(self, camera_channels: int = 64, radar_channels: int = 32, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(camera_channels + radar_channels, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, camera_bev: torch.Tensor, radar_bev: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([camera_bev, radar_bev], dim=1))


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("camera_bev", "radar_bev_features", "radar_bev_raw")

    camera = ctx.get("camera_bev")
    radar = ctx.get("radar_bev_features")
    radar_raw = ctx.get("radar_bev_raw")

    log.substage(7, 1, "Verify both modalities share the same spatial grid")
    log.tensor("camera_bev", camera)
    log.tensor("radar_bev_features", radar)
    if camera.shape[-2:] != radar.shape[-2:]:
        raise RuntimeError("Camera BEV and radar BEV spatial shapes must match before fusion.")

    log.substage(7, 2, "Concatenate modality feature channels")
    log.info(f"Feature channels before fusion = {camera.shape[1]} camera + {radar.shape[1]} radar")

    log.substage(7, 3, "Run a small spatial fusion network")
    fusion_model = SpatialBEVFusion(camera.shape[1], radar.shape[1], 64).to(cfg.torch_device).eval()
    with torch.no_grad():
        fused_bev = fusion_model(camera, radar)
    log.tensor("fused_bev [T,64,H,W]", fused_bev)
    log.detail("This fusion CNN is not yet trained; it exists so the model boundary is explicit and trainable later.")

    log.substage(7, 4, "Create an interpretable fusion visualization")
    cam_mag = normalize_01(tensor_magnitude(camera[-1]))
    radar_density = normalize_01(radar_raw[-1, 0].detach().cpu().numpy())
    interpretable = 0.70 * cam_mag + 0.30 * radar_density
    if cfg.save_plots:
        save_heatmap(interpretable, stage_dir / "current_interpretable_fusion.png", "Stage 07: camera magnitude + radar density (visual teaching blend)")
        save_heatmap(tensor_magnitude(fused_bev[-1]), stage_dir / "current_learned_fusion_features.png", "Stage 07: untrained fusion-network feature magnitude")

    values = {
        "spatial_fusion_model": fusion_model,
        "fused_bev": fused_bev,
        "interpretable_fusion": interpretable,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("Camera and radar now occupy one spatial feature representation for every history frame.")
