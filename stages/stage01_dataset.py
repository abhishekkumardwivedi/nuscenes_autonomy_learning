from __future__ import annotations

from pathlib import Path

from nuscenes.nuscenes import NuScenes

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary
from utils.nuscenes_utils import CAMERA_CHANNELS, RADAR_CHANNELS, scene_sample_tokens

STAGE_NUMBER = 1
STAGE_NAME = "nuScenes sequence loading"
SHORT_NAME = "dataset"


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 01: choose one short temporal sequence from nuScenes Mini."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)

    log.substage(1, 1, "Open nuScenes database")
    dataroot = Path(cfg.dataroot)
    if not dataroot.exists():
        raise FileNotFoundError(
            f"nuScenes dataroot does not exist: {dataroot}\n"
            "Download/extract v1.0-mini and pass --dataroot /path/to/nuscenes"
        )
    nusc = NuScenes(version=cfg.version, dataroot=str(dataroot), verbose=(cfg.verbose >= 3))
    log.info(f"Version = {cfg.version}")
    log.info(f"Scenes available = {len(nusc.scene)}")
    log.info(f"Annotated keyframe samples = {len(nusc.sample)}")

    log.substage(1, 2, "Choose one scene")
    if not (0 <= cfg.scene_index < len(nusc.scene)):
        raise IndexError(f"scene_index={cfg.scene_index} but only {len(nusc.scene)} scenes are available")
    scene = nusc.scene[cfg.scene_index]
    tokens = scene_sample_tokens(nusc, scene)
    log.info(f"Scene name = {scene['name']}")
    log.info(f"Samples in scene = {len(tokens)}")

    log.substage(1, 3, "Choose current frame with history and future")
    min_current = cfg.history_frames - 1
    max_current = len(tokens) - cfg.future_frames - 1
    if max_current < min_current:
        raise RuntimeError(
            "Selected scene is too short for requested history/future. "
            f"Need history={cfg.history_frames}, future={cfg.future_frames}."
        )
    if cfg.sample_index < 0:
        current_index = (min_current + max_current) // 2
    else:
        current_index = max(min_current, min(cfg.sample_index, max_current))

    history_tokens = tokens[current_index - cfg.history_frames + 1 : current_index + 1]
    future_tokens = tokens[current_index + 1 : current_index + 1 + cfg.future_frames]
    history_samples = [nusc.get("sample", t) for t in history_tokens]
    future_samples = [nusc.get("sample", t) for t in future_tokens]
    current_sample = history_samples[-1]

    log.info(f"History frames = {len(history_samples)} (includes current frame)")
    log.info(f"Future supervision frames = {len(future_samples)}")
    log.detail(f"Current timestamp = {current_sample['timestamp']} microseconds")

    log.substage(1, 4, "Inspect sensor channels")
    for name in CAMERA_CHANNELS:
        log.detail(f"Camera available: {name}")
    for name in RADAR_CHANNELS:
        log.detail(f"Radar available : {name}")
    log.detail("Reference sensor   : LIDAR_TOP (used only for sample ego-pose timing/reference)")

    values = {
        "nusc": nusc,
        "scene": scene,
        "scene_sample_tokens": tokens,
        "history_samples": history_samples,
        "future_samples": future_samples,
        "current_sample": current_sample,
        "current_scene_sample_index": current_index,
        "camera_channels": CAMERA_CHANNELS,
        "radar_channels": RADAR_CHANNELS,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, {
        "scene_name": scene["name"],
        "scene_description": scene["description"],
        "scene_samples": len(tokens),
        "current_index": current_index,
        "history_tokens": history_tokens,
        "future_tokens": future_tokens,
        "camera_channels": CAMERA_CHANNELS,
        "radar_channels": RADAR_CHANNELS,
    })
    log.outcome("A chronological mini-sequence is ready. Every later stage works on this same sequence.")
