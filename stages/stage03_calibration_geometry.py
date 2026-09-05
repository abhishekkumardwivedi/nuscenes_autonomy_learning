from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.geometry import transform_matrix
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary
from utils.nuscenes_utils import CAMERA_CHANNELS, sample_sensor_calibration

STAGE_NUMBER = 3
STAGE_NAME = "Calibration and coordinate geometry"
SHORT_NAME = "geometry"


def _scaled_intrinsic(K: np.ndarray, original_hw, target_hw) -> np.ndarray:
    original_h, original_w = original_hw
    target_h, target_w = target_hw
    sx = target_w / float(original_w)
    sy = target_h / float(original_h)
    out = K.copy().astype(np.float64)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 03: expose K matrices and camera->ego rigid transforms."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("nusc", "history_samples")
    nusc = ctx.get("nusc")
    samples = ctx.get("history_samples")

    log.substage(3, 1, "Read camera intrinsics K")
    intrinsics = []
    cam_to_ego = []
    original_hw = []

    for sample in samples:
        Ks_t = []
        Ts_t = []
        sizes_t = []
        for cam in CAMERA_CHANNELS:
            sd, calib = sample_sensor_calibration(nusc, sample, cam)
            K = np.asarray(calib["camera_intrinsic"], dtype=np.float64)
            size = (int(sd["height"]), int(sd["width"]))
            K_scaled = _scaled_intrinsic(
                K, size, (cfg.image_height, cfg.image_width)
            )
            T = transform_matrix(calib["translation"], calib["rotation"])
            Ks_t.append(K_scaled)
            Ts_t.append(T)
            sizes_t.append(size)
        intrinsics.append(Ks_t)
        cam_to_ego.append(Ts_t)
        original_hw.append(sizes_t)

    intrinsics = np.asarray(intrinsics, dtype=np.float64)  # [T,6,3,3]
    cam_to_ego = np.asarray(cam_to_ego, dtype=np.float64)  # [T,6,4,4]
    log.tensor("camera_intrinsics [T,6,3,3]", intrinsics)
    log.tensor("T_ego_from_camera [T,6,4,4]", cam_to_ego)

    log.substage(3, 2, "Interpret one calibration numerically")
    front_idx = CAMERA_CHANNELS.index("CAM_FRONT")
    K_front = intrinsics[-1, front_idx]
    T_front = cam_to_ego[-1, front_idx]
    log.info("CAM_FRONT scaled intrinsic matrix K:")
    for row in K_front:
        log.detail("[" + ", ".join(f"{v:8.3f}" for v in row) + "]")
    log.info("CAM_FRONT camera origin in ego coordinates:")
    log.detail(
        f"x={T_front[0,3]:.3f} m forward, y={T_front[1,3]:.3f} m left, z={T_front[2,3]:.3f} m up"
    )

    log.substage(3, 3, "Visualize camera positions and optical axes")
    plot_path = stage_dir / "camera_geometry_topdown.png"
    if cfg.save_plots:
        fig, ax = plt.subplots(figsize=(7, 7))
        for cam_i, cam in enumerate(CAMERA_CHANNELS):
            T = cam_to_ego[-1, cam_i]
            origin = T[:3, 3]
            # Camera optical axis is +z in camera coordinates.
            optical_ego = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
            ax.scatter(origin[1], origin[0], s=45)
            ax.arrow(
                origin[1], origin[0],
                optical_ego[1] * 2.0, optical_ego[0] * 2.0,
                head_width=0.15, length_includes_head=True,
            )
            ax.text(origin[1], origin[0], cam.replace("CAM_", ""), fontsize=8)
        ax.scatter(0, 0, marker="x", s=80, label="ego origin")
        ax.set_xlabel("y left (m)")
        ax.set_ylabel("x forward (m)")
        ax.set_title("Stage 03: camera origins + optical directions in ego frame")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        log.info(f"Saved geometry plot -> {plot_path}")

    values = {
        "camera_intrinsics": intrinsics,
        "T_ego_from_camera": cam_to_ego,
        "camera_original_hw": original_hw,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("Every camera pixel can now be related geometrically to the ego vehicle frame.")
