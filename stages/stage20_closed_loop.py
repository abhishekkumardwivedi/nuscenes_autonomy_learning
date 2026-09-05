from __future__ import annotations

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary

STAGE_NUMBER = 20
STAGE_NAME = "Closed-loop integration boundary"
SHORT_NAME = "closed_loop"


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 20: explain the offline-vs-closed-loop boundary and optionally apply to CARLA."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("final_control")

    backend = getattr(cfg, "backend", "offline")
    log.substage(20, 1, "Identify execution environment")
    log.info(f"backend = {backend}")

    applied_actor_id = None
    if backend == "carla":
        log.substage(20, 2, "Apply one supervised command to an existing CARLA ego actor")
        from adapters.carla_adapter import CarlaVehicleAdapter
        adapter = CarlaVehicleAdapter(
            host=getattr(cfg, "carla_host", "127.0.0.1"),
            port=getattr(cfg, "carla_port", 2000),
        )
        applied_actor_id = adapter.apply(ctx.get("final_control"))
        log.info(f"Applied control to CARLA actor id={applied_actor_id}")
        status = "CARLA_ONE_STEP_APPLIED"
        log.detail("True closed loop requires replacing nuScenes input with live CARLA sensor callbacks and repeating Stages 02-19 every tick.")
    else:
        log.substage(20, 2, "Offline nuScenes replay limitation")
        log.info("nuScenes is recorded data: our steering/brake command cannot change the next recorded frame.")
        log.info("Therefore this run demonstrates sense->world->plan->control, but not causal closed-loop driving.")
        status = "OFFLINE_REPLAY_COMPLETE"

    values = {
        "closed_loop_status": status,
        "carla_actor_id": applied_actor_id,
        "final_control": ctx.get("final_control"),
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome(status)
