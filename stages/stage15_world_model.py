from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary
from utils.visualization import normalize_01, tensor_magnitude

STAGE_NUMBER = 15
STAGE_NAME = "World-model assembly"
SHORT_NAME = "world_model"


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 15: package perception, memory, tracks, occupancy, prediction and map context."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("temporal_bev", "tracks", "dynamic_occupancy", "agent_predictions")

    log.substage(15, 1, "Collect world-state components")
    world_model = {
        "temporal_bev": ctx.get("temporal_bev"),
        "tracks": ctx.get("tracks"),
        "dynamic_occupancy": ctx.get("dynamic_occupancy"),
        "agent_predictions": ctx.get("agent_predictions"),
        "map_masks": ctx.get("map_masks"),
        "ego_speed_mps": ctx.get("current_ego_speed_mps"),
    }
    log.info(f"tracks = {len(world_model['tracks'])}")
    log.info(f"ego speed = {world_model['ego_speed_mps']:.2f} m/s")
    log.info(f"map available = {world_model['map_masks'] is not None}")

    log.substage(15, 2, "Create one human-readable world-state view")
    if cfg.save_plots:
        bg = normalize_01(tensor_magnitude(world_model["temporal_bev"]))
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(
            bg, origin="lower",
            extent=[cfg.bev_y_min, cfg.bev_y_max, cfg.bev_x_min, cfg.bev_x_max],
        )

        # Occupancy outline proxy.
        occ = world_model["dynamic_occupancy"]
        ys = np.linspace(cfg.bev_y_min, cfg.bev_y_max, occ.shape[1])
        xs = np.linspace(cfg.bev_x_min, cfg.bev_x_max, occ.shape[0])
        if occ.max() > 0:
            ax.contour(ys, xs, occ, levels=[0.5], linewidths=0.8)

        for tr in world_model["tracks"].values():
            xy = tr["current_xy"]
            ax.scatter(xy[1], xy[0], s=18)
            pred = world_model["agent_predictions"].get(tr["instance_token"])
            if pred is not None and len(pred):
                ax.plot(pred[:, 1], pred[:, 0], "--", linewidth=0.8)

        ax.scatter(0, 0, marker="x", s=80, label="ego")
        ax.set_xlim(cfg.bev_y_min, cfg.bev_y_max)
        ax.set_ylim(cfg.bev_x_min, cfg.bev_x_max)
        ax.set_xlabel("y left (m)")
        ax.set_ylabel("x forward (m)")
        ax.set_title("Stage 15: temporal scene + tracks + predicted motion + occupancy")
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        fig.savefig(stage_dir / "world_model.png", dpi=160)
        plt.close(fig)

    values = {"world_model": world_model}
    ctx.update(values)
    save_stage_summary(stage_dir, {
        "track_count": len(world_model["tracks"]),
        "ego_speed_mps": world_model["ego_speed_mps"],
        "map_available": world_model["map_masks"] is not None,
        "temporal_bev": world_model["temporal_bev"],
        "dynamic_occupancy": world_model["dynamic_occupancy"],
    })
    log.outcome("The planner now receives one explicit world state instead of unrelated perception outputs.")
