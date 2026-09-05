from __future__ import annotations

from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.geometry import transform_points
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary

STAGE_NUMBER = 11
STAGE_NAME = "Object tracking and track history"
SHORT_NAME = "tracking"


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 11: use nuScenes instance IDs to expose what a tracker must recover."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("nusc", "history_samples", "T_global_from_ego", "gt_objects")
    nusc = ctx.get("nusc")
    samples = ctx.get("history_samples")
    T_current_from_global = np.linalg.inv(ctx.get("T_global_from_ego")[-1])
    current_instances = {o["instance_token"] for o in ctx.get("gt_objects")}

    log.substage(11, 1, "Collect the same instance across historical frames")
    histories = defaultdict(list)
    for sample in samples:
        t_sec = sample["timestamp"] / 1e6
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            inst = ann["instance_token"]
            if inst not in current_instances:
                continue
            center_global = np.asarray(ann["translation"], dtype=np.float64).reshape(1, 3)
            center_cur = transform_points(T_current_from_global, center_global)[0]
            histories[inst].append([t_sec, center_cur[0], center_cur[1]])

    tracks = {}
    for obj in ctx.get("gt_objects"):
        inst = obj["instance_token"]
        arr = np.asarray(histories.get(inst, []), dtype=np.float64)
        if len(arr) >= 2:
            dt = max(arr[-1, 0] - arr[-2, 0], 1e-6)
            vel = (arr[-1, 1:3] - arr[-2, 1:3]) / dt
        else:
            vel = np.asarray(obj["velocity_xy"], dtype=np.float64)
        tracks[inst] = {
            "instance_token": inst,
            "category": obj["category"],
            "history_txy": arr,
            "current_xy": np.asarray(obj["center"][:2], dtype=np.float64),
            "velocity_xy": vel,
        }

    log.info(f"Active tracks = {len(tracks)}")
    for i, tr in enumerate(tracks.values()):
        if i >= 8:
            break
        log.detail(
            f"{tr['category']:<30} history={len(tr['history_txy'])} "
            f"v=({tr['velocity_xy'][0]:+.2f},{tr['velocity_xy'][1]:+.2f}) m/s"
        )

    log.substage(11, 2, "Visualize track histories")
    if cfg.save_plots:
        fig, ax = plt.subplots(figsize=(8, 8))
        for tr in tracks.values():
            h = tr["history_txy"]
            if len(h):
                ax.plot(h[:, 2], h[:, 1], marker="o", linewidth=1)
        ax.scatter(0, 0, marker="x", s=70, label="ego now")
        ax.set_xlim(cfg.bev_y_min, cfg.bev_y_max)
        ax.set_ylim(cfg.bev_x_min, cfg.bev_x_max)
        ax.set_xlabel("y left (m)")
        ax.set_ylabel("x forward (m)")
        ax.set_title("Stage 11: GT identity track histories in current ego frame")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(stage_dir / "track_histories.png", dpi=150)
        plt.close(fig)

    log.substage(11, 3, "Learning note: GT IDs vs a real tracker")
    log.info("nuScenes instance_token gives perfect identity for teaching/evaluation.")
    log.info("Later, replace this oracle with detection -> association -> state filter (e.g. Kalman).")

    values = {"tracks": tracks}
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("Static detections have become time-consistent object tracks with velocity and history.")
