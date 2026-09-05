from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nuscenes.map_expansion.map_api import NuScenesMap

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.geometry import yaw_from_quaternion
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary

STAGE_NUMBER = 14
STAGE_NAME = "HD-map / road context"
SHORT_NAME = "map_context"

MAP_LAYERS = ["drivable_area", "lane", "ped_crossing", "walkway"]


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 14: rasterize semantic map layers around the current ego pose."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("nusc", "scene", "ego_pose_records")
    nusc = ctx.get("nusc")
    scene = ctx.get("scene")
    pose = ctx.get("ego_pose_records")[-1]

    log.substage(14, 1, "Find which nuScenes map belongs to this scene")
    log_record = nusc.get("log", scene["log_token"])
    map_name = log_record["location"]
    log.info(f"Map location = {map_name}")

    x_global, y_global = pose["translation"][:2]
    yaw_deg = math.degrees(yaw_from_quaternion(pose["rotation"]))
    patch_box = (
        float(x_global), float(y_global),
        float(cfg.bev_x_max - cfg.bev_x_min),
        float(cfg.bev_y_max - cfg.bev_y_min),
    )

    log.substage(14, 2, "Rasterize semantic layers around the ego vehicle")
    map_masks = None
    map_error = None
    try:
        nusc_map = NuScenesMap(dataroot=cfg.dataroot, map_name=map_name)
        map_masks = nusc_map.get_map_mask(
            patch_box=patch_box,
            patch_angle=yaw_deg,
            layer_names=MAP_LAYERS,
            canvas_size=(cfg.bev_height, cfg.bev_width),
        ).astype(np.uint8)
        log.tensor("map_masks [layers,H,W]", map_masks)
        for i, layer in enumerate(MAP_LAYERS):
            log.detail(f"{layer}: {100.0 * map_masks[i].mean():.1f}% of patch")
    except Exception as exc:
        map_error = str(exc)
        log.info("Map expansion layers could not be loaded from this dataroot.")
        log.detail(map_error)
        log.detail("The rest of the pipeline can continue; install/extract the nuScenes map expansion to enable this stage fully.")

    log.substage(14, 3, "Visualize map channels")
    if cfg.save_plots and map_masks is not None:
        fig, axes = plt.subplots(1, len(MAP_LAYERS), figsize=(16, 4))
        for i, (ax, layer) in enumerate(zip(axes, MAP_LAYERS)):
            ax.imshow(map_masks[i], origin="lower")
            ax.set_title(layer)
            ax.axis("off")
        fig.suptitle("Stage 14: ego-centered nuScenes semantic map patch")
        fig.tight_layout()
        fig.savefig(stage_dir / "semantic_map_layers.png", dpi=150)
        plt.close(fig)

    log.substage(14, 4, "Route-context limitation")
    log.info("nuScenes provides semantic road/map context, but not a full mission route for our ego planner.")
    log.info("Mission-route planning is introduced properly when this pipeline is connected to nuPlan/CARLA.")

    values = {
        "map_name": map_name,
        "map_layers": MAP_LAYERS,
        "map_masks": map_masks,
        "map_error": map_error,
        "map_patch_box_global": patch_box,
        "map_patch_angle_deg": yaw_deg,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("The dynamic world now has static road semantics such as drivable area, lanes and crossings.")
