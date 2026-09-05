from __future__ import annotations

import math
import numpy as np

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.geometry import yaw_from_matrix
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary
from utils.visualization import save_xy_plot

STAGE_NUMBER = 8
STAGE_NAME = "Localization / ego motion between temporal frames"
SHORT_NAME = "ego_motion"


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 08: compute source-ego -> current-ego rigid transforms from nuScenes poses."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("T_global_from_ego", "timestamps_sec")

    poses = ctx.get("T_global_from_ego")
    ts = ctx.get("timestamps_sec")
    T_global_from_current = poses[-1]
    T_current_from_global = np.linalg.inv(T_global_from_current)

    log.substage(8, 1, "Compute each history pose relative to the current ego frame")
    T_current_from_history = []
    relative_xyyaw = []
    history_positions_in_current = []
    for t, T_global_from_hist in enumerate(poses):
        T_cur_from_hist = T_current_from_global @ T_global_from_hist
        T_current_from_history.append(T_cur_from_hist)
        dx, dy = T_cur_from_hist[0, 3], T_cur_from_hist[1, 3]
        dyaw = yaw_from_matrix(T_cur_from_hist)
        relative_xyyaw.append([dx, dy, dyaw])
        history_positions_in_current.append([dx, dy])
        log.detail(
            f"frame {t}: history ego origin appears at current-frame "
            f"x={dx:+.2f} m, y={dy:+.2f} m, yaw={math.degrees(dyaw):+.2f} deg"
        )

    T_current_from_history = np.asarray(T_current_from_history, dtype=np.float64)
    relative_xyyaw = np.asarray(relative_xyyaw, dtype=np.float64)
    history_positions_in_current = np.asarray(history_positions_in_current, dtype=np.float64)

    log.substage(8, 2, "Estimate ego speed from successive recorded poses")
    speeds = np.zeros(len(poses), dtype=np.float64)
    for i in range(1, len(poses)):
        dt = max(ts[i] - ts[i - 1], 1e-6)
        dist = np.linalg.norm(poses[i][:2, 3] - poses[i - 1][:2, 3])
        speeds[i] = dist / dt
    if len(speeds) > 1:
        speeds[0] = speeds[1]
    current_speed = float(speeds[-1])
    log.info(f"Current ego speed estimate = {current_speed:.2f} m/s ({current_speed * 3.6:.1f} km/h)")

    log.substage(8, 3, "Visualize ego history in the current coordinate frame")
    if cfg.save_plots:
        save_xy_plot(
            [history_positions_in_current], ["ego history"],
            stage_dir / "ego_history_current_frame.png",
            "Stage 08: previous ego positions expressed in current ego frame",
            xlim=(cfg.bev_y_min, cfg.bev_y_max),
            ylim=(cfg.bev_x_min, cfg.bev_x_max),
            invert_axes_for_bev=True,
        )

    values = {
        "T_current_from_history": T_current_from_history,
        "relative_ego_xyyaw": relative_xyyaw,
        "ego_speeds_mps": speeds,
        "current_ego_speed_mps": current_speed,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("We can now compensate vehicle motion before combining BEVs from different times.")
