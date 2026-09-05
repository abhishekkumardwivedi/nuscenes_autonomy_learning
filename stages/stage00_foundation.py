from __future__ import annotations

import random
import numpy as np
import torch

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary

STAGE_NUMBER = 0
STAGE_NAME = "Foundation / configuration"
SHORT_NAME = "foundation"


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 00: make execution deterministic and explain the coordinate system."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)

    log.substage(0, 1, "Fix random seeds")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    log.info(f"Random seed = {cfg.seed}")

    log.substage(0, 2, "Resolve compute device")
    device = cfg.torch_device
    log.info(f"Selected device = {device}")
    if device.type == "cuda":
        log.detail(f"GPU = {torch.cuda.get_device_name(device)}")

    log.substage(0, 3, "Define coordinate convention")
    log.info("nuScenes ego frame: +x forward, +y left, +z up")
    log.info(
        f"BEV area: x=[{cfg.bev_x_min},{cfg.bev_x_max}] m, "
        f"y=[{cfg.bev_y_min},{cfg.bev_y_max}] m"
    )
    log.info(
        f"BEV grid = {cfg.bev_height} x {cfg.bev_width} at "
        f"{cfg.bev_resolution:.2f} m/cell"
    )

    values = {
        "device": str(device),
        "seed": cfg.seed,
        "bev_shape": [cfg.bev_height, cfg.bev_width],
        "coordinate_system": "+x forward, +y left, +z up",
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("The pipeline now has a deterministic, explicit coordinate and BEV convention.")
