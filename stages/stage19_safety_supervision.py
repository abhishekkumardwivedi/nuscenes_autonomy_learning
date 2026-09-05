from __future__ import annotations

import numpy as np
import torch

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary

STAGE_NUMBER = 19
STAGE_NAME = "Runtime safety / fallback supervision"
SHORT_NAME = "safety"


def _trajectory_collision(trajectory: np.ndarray, occupancy: np.ndarray, cfg: PipelineConfig) -> bool:
    H, W = occupancy.shape
    for x, y in trajectory:
        r = int((x - cfg.bev_x_min) / cfg.bev_resolution)
        c = int((y - cfg.bev_y_min) / cfg.bev_resolution)
        if 0 <= r < H and 0 <= c < W:
            r0, r1 = max(0, r-2), min(H, r+3)
            c0, c1 = max(0, c-2), min(W, c+3)
            if occupancy[r0:r1, c0:c1].max() > 0:
                return True
    return False


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("raw_control", "selected_ego_trajectory", "dynamic_occupancy", "temporal_bev")

    control = dict(ctx.get("raw_control"))
    trajectory = np.asarray(ctx.get("selected_ego_trajectory"), dtype=np.float64)
    occupancy = ctx.get("dynamic_occupancy")
    temporal_bev = ctx.get("temporal_bev")

    log.substage(19, 1, "Check numerical/model health")
    bev_finite = bool(torch.isfinite(temporal_bev).all().item())
    traj_finite = bool(np.isfinite(trajectory).all())
    controls_valid = all(np.isfinite(v) and -1.0 <= v <= 1.0 for v in control.values())
    log.info(f"temporal BEV finite = {bev_finite}")
    log.info(f"trajectory finite   = {traj_finite}")
    log.info(f"controls bounded    = {controls_valid}")

    log.substage(19, 2, "Check planned trajectory against occupied space")
    collision_risk = _trajectory_collision(trajectory, occupancy, cfg)
    log.info(f"collision along selected trajectory = {collision_risk}")

    unsafe_reasons = []
    if not bev_finite:
        unsafe_reasons.append("non-finite temporal BEV")
    if not traj_finite:
        unsafe_reasons.append("non-finite trajectory")
    if not controls_valid:
        unsafe_reasons.append("invalid control range")
    if collision_risk:
        unsafe_reasons.append("occupied cells intersect selected trajectory")

    log.substage(19, 3, "Apply a simple minimum-risk fallback")
    if unsafe_reasons:
        final_control = {"steer": 0.0, "throttle": 0.0, "brake": 1.0}
        safety_status = "FALLBACK_BRAKE"
        log.info("Safety supervisor overrides planner/controller -> full brake")
        for reason in unsafe_reasons:
            log.detail(f"reason: {reason}")
    else:
        final_control = control
        safety_status = "NOMINAL"
        log.info("No simple safety violation found; raw controller command is passed through.")

    log.detail("Production FuSa/SOTIF requires much more: sensor health, timing, uncertainty, ODD, redundancy, MRM state machine, diagnostics, etc.")

    values = {
        "safety_status": safety_status,
        "safety_reasons": unsafe_reasons,
        "final_control": final_control,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome(f"Runtime supervision status = {safety_status}.")
