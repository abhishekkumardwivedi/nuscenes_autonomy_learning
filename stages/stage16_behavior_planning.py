from __future__ import annotations

from enum import Enum
import numpy as np

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary

STAGE_NUMBER = 16
STAGE_NAME = "Behavior planning"
SHORT_NAME = "behavior_planning"


class Behavior(str, Enum):
    KEEP_LANE = "KEEP_LANE"
    FOLLOW = "FOLLOW"
    STOP = "STOP"


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 16: a transparent rule baseline for high-level behavior choice.

    nuScenes has no mission route here, so this is intentionally not presented as
    production behavior planning. The goal is to expose the interface between the
    world model and a high-level decision.
    """
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("world_model")
    wm = ctx.get("world_model")

    log.substage(16, 1, "Inspect objects in an ego-lane corridor")
    ahead = []
    for tr in wm["tracks"].values():
        x, y = tr["current_xy"]
        if 0.0 < x < 30.0 and abs(y) < 2.0:
            ahead.append((float(x), tr))
    ahead.sort(key=lambda z: z[0])

    log.substage(16, 2, "Choose a high-level behavior")
    if ahead and ahead[0][0] < 8.0:
        behavior = Behavior.STOP
        reason = f"nearest lane-corridor object is only {ahead[0][0]:.1f} m ahead"
    elif ahead and ahead[0][0] < 25.0:
        behavior = Behavior.FOLLOW
        reason = f"object detected {ahead[0][0]:.1f} m ahead in the lane corridor"
    else:
        behavior = Behavior.KEEP_LANE
        reason = "no close object found in the simple lane corridor"

    log.info(f"Selected behavior = {behavior.value}")
    log.info(f"Reason = {reason}")
    log.detail("Later with nuPlan/CARLA this stage grows to route following, lane change, yield, junction logic and overtaking.")

    values = {
        "behavior": behavior.value,
        "behavior_reason": reason,
        "objects_ahead_count": len(ahead),
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome(f"World state -> high-level decision: {behavior.value}.")
