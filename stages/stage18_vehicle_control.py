from __future__ import annotations

import math
import numpy as np

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary

STAGE_NUMBER = 18
STAGE_NAME = "Vehicle control: trajectory -> steer / throttle / brake"
SHORT_NAME = "vehicle_control"


def _pure_pursuit(trajectory: np.ndarray, wheelbase: float = 2.7, lookahead: float = 6.0) -> float:
    """Return steering angle in radians using a simple pure-pursuit controller."""
    if len(trajectory) == 0:
        return 0.0
    distances = np.linalg.norm(trajectory, axis=1)
    candidates = np.flatnonzero(distances >= lookahead)
    idx = int(candidates[0]) if len(candidates) else len(trajectory) - 1
    x, y = trajectory[idx]
    ld2 = max(x * x + y * y, 1e-4)
    curvature = 2.0 * y / ld2
    return math.atan(wheelbase * curvature)


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("selected_ego_trajectory", "current_ego_speed_mps")

    trajectory = np.asarray(ctx.get("selected_ego_trajectory"), dtype=np.float64)
    current_speed = float(ctx.get("current_ego_speed_mps"))
    times = np.asarray(ctx.get("future_times_sec"), dtype=np.float64)
    if len(times) != len(trajectory):
        times = np.arange(1, len(trajectory) + 1, dtype=np.float64) * 0.5

    log.substage(18, 1, "Lateral control with Pure Pursuit")
    steering_rad = _pure_pursuit(trajectory)
    max_steer_rad = 0.60
    steer = float(np.clip(steering_rad / max_steer_rad, -1.0, 1.0))
    log.info(f"Steering angle request = {math.degrees(steering_rad):+.2f} deg")
    log.info(f"Normalized steering command = {steer:+.3f}")

    log.substage(18, 2, "Longitudinal speed control")
    if len(trajectory) >= 2:
        dt = max(float(times[-1] - times[0]), 1e-3)
        desired_speed = float(np.linalg.norm(trajectory[-1] - trajectory[0]) / dt)
    elif len(trajectory) == 1:
        desired_speed = float(np.linalg.norm(trajectory[0]) / max(times[0], 1e-3))
    else:
        desired_speed = 0.0
    speed_error = desired_speed - current_speed
    kp = 0.35
    accel_cmd = kp * speed_error
    throttle = float(np.clip(accel_cmd, 0.0, 1.0))
    brake = float(np.clip(-accel_cmd, 0.0, 1.0))
    log.info(f"Current speed = {current_speed:.2f} m/s")
    log.info(f"Desired speed = {desired_speed:.2f} m/s")
    log.info(f"Throttle = {throttle:.3f}, brake = {brake:.3f}")

    values = {
        "raw_control": {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
        },
        "desired_speed_mps": desired_speed,
        "steering_angle_rad": steering_rad,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("The selected trajectory is now converted into actuator-style commands.")
