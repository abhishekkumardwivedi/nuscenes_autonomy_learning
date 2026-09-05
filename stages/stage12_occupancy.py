from __future__ import annotations

import numpy as np

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary
from utils.visualization import draw_polygons_to_mask, save_heatmap

STAGE_NUMBER = 12
STAGE_NAME = "Dynamic occupancy / occupied-space raster"
SHORT_NAME = "occupancy"


def _xy_polygon_to_rc(poly_xy: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    row = (x - cfg.bev_x_min) / cfg.bev_resolution
    col = (y - cfg.bev_y_min) / cfg.bev_resolution
    return np.column_stack([row, col])


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("gt_objects", "radar_bev_raw")

    log.substage(12, 1, "Rasterize dynamic 3D boxes into BEV occupancy")
    polygons_rc = [_xy_polygon_to_rc(o["bottom_corners_xy"], cfg) for o in ctx.get("gt_objects")]
    object_occupancy = draw_polygons_to_mask(polygons_rc, cfg.bev_height, cfg.bev_width)
    log.tensor("object_occupancy [H,W]", object_occupancy)

    log.substage(12, 2, "Add a radar-supported occupancy hint")
    radar_density = ctx.get("radar_bev_raw")[-1, 0].detach().cpu().numpy()
    radar_occupied = (radar_density > 0).astype(np.uint8)
    dynamic_occupancy = np.maximum(object_occupancy, radar_occupied)
    occupied_pct = 100.0 * float(dynamic_occupancy.mean())
    log.info(f"Occupied grid fraction = {occupied_pct:.2f}%")
    log.detail("This is dynamic-object occupancy, not yet semantic drivable/free-space. Map context is added in Stage 14.")

    if cfg.save_plots:
        save_heatmap(dynamic_occupancy, stage_dir / "dynamic_occupancy.png", "Stage 12: dynamic occupied-space proxy")

    values = {
        "object_occupancy": object_occupancy,
        "radar_occupied": radar_occupied,
        "dynamic_occupancy": dynamic_occupancy,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("Tracked objects and radar evidence are now represented as an explicit occupied-space grid.")
