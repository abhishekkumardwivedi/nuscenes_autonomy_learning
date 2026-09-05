from __future__ import annotations

from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.geometry import transform_points
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary

STAGE_NUMBER = 13
STAGE_NAME = "Agent trajectory prediction"
SHORT_NAME = "prediction"


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 13: constant-velocity baseline + recorded future for comparison."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("nusc", "tracks", "future_samples", "current_sample", "T_global_from_ego")

    nusc = ctx.get("nusc")
    tracks = ctx.get("tracks")
    future_samples = ctx.get("future_samples")
    current_sample = ctx.get("current_sample")
    T_cur_from_global = np.linalg.inv(ctx.get("T_global_from_ego")[-1])
    t0 = current_sample["timestamp"] / 1e6
    future_times = np.asarray([s["timestamp"] / 1e6 - t0 for s in future_samples], dtype=np.float64)

    log.substage(13, 1, "Predict each agent with a constant-velocity baseline")
    predictions: Dict[str, np.ndarray] = {}
    for inst, tr in tracks.items():
        p0 = tr["current_xy"]
        v = tr["velocity_xy"]
        predictions[inst] = p0[None, :] + future_times[:, None] * v[None, :]
    log.info(f"Predicted agents = {len(predictions)}")
    log.detail(f"Prediction horizon timestamps = {np.round(future_times, 2).tolist()} s")

    log.substage(13, 2, "Extract recorded future agent positions for evaluation")
    future_gt: Dict[str, list] = {inst: [] for inst in tracks}
    for sample in future_samples:
        ann_by_inst = {}
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            ann_by_inst[ann["instance_token"]] = ann
        for inst in tracks:
            ann = ann_by_inst.get(inst)
            if ann is None:
                future_gt[inst].append([np.nan, np.nan])
            else:
                c = np.asarray(ann["translation"], dtype=np.float64).reshape(1, 3)
                c_cur = transform_points(T_cur_from_global, c)[0]
                future_gt[inst].append(c_cur[:2])
    future_gt = {k: np.asarray(v, dtype=np.float64) for k, v in future_gt.items()}

    ades, fdes = [], []
    for inst, pred in predictions.items():
        gt = future_gt[inst]
        valid = np.isfinite(gt).all(axis=1)
        if valid.any():
            errors = np.linalg.norm(pred[valid] - gt[valid], axis=1)
            ades.append(float(errors.mean()))
            fdes.append(float(errors[-1]))
    metrics = {
        "constant_velocity_ADE_m": float(np.mean(ades)) if ades else None,
        "constant_velocity_FDE_m": float(np.mean(fdes)) if fdes else None,
    }
    log.info(f"Constant-velocity ADE = {metrics['constant_velocity_ADE_m']}")
    log.info(f"Constant-velocity FDE = {metrics['constant_velocity_FDE_m']}")

    log.substage(13, 3, "Visualize predicted vs recorded future")
    if cfg.save_plots:
        fig, ax = plt.subplots(figsize=(8, 8))
        shown = 0
        for inst, pred in predictions.items():
            gt = future_gt[inst]
            ax.plot(pred[:, 1], pred[:, 0], "--", linewidth=1)
            valid = np.isfinite(gt).all(axis=1)
            if valid.any():
                ax.plot(gt[valid, 1], gt[valid, 0], linewidth=1)
            shown += 1
            if shown >= 20:
                break
        ax.scatter(0, 0, marker="x", s=70)
        ax.set_xlim(cfg.bev_y_min, cfg.bev_y_max)
        ax.set_ylim(cfg.bev_x_min, cfg.bev_x_max)
        ax.set_xlabel("y left (m)")
        ax.set_ylabel("x forward (m)")
        ax.set_title("Stage 13: dashed CV predictions vs solid recorded future")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(stage_dir / "agent_predictions.png", dpi=150)
        plt.close(fig)

    values = {
        "future_times_sec": future_times,
        "agent_predictions": predictions,
        "agent_future_gt": future_gt,
        "prediction_metrics": metrics,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("Tracks now have future hypotheses, and the baseline can be quantitatively compared with recorded motion.")
