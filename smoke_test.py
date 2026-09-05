"""Dataset-free sanity checks for core geometry/temporal/planning code.

This is not a model-quality test. It only checks that important tensor paths run,
produce expected shapes, and avoid NaN/Inf before you download nuScenes.
"""
from __future__ import annotations

import tempfile
import numpy as np
import torch

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger
from stages.stage05_camera_bev import lift_splat_one_frame
from stages.stage09_temporal_bev import warp_bev_to_current
from stages import stage16_behavior_planning
from stages import stage17_motion_planning
from stages import stage18_vehicle_control
from stages import stage19_safety_supervision


def main():
    cfg = PipelineConfig(
        dataroot="/not-needed",
        output_dir=tempfile.mkdtemp(prefix="autonomy_smoke_"),
        save_plots=False,
        verbose=0,
        pretrained_backbone=False,
        bev_x_min=-10,
        bev_x_max=20,
        bev_y_min=-15,
        bev_y_max=15,
        bev_resolution=1.0,
        depth_min=2,
        depth_max=8,
        depth_bins=4,
    )

    # Synthetic lift-splat shape check.
    feature = torch.randn(6, 8, 4, 7)
    K = np.tile(np.array([[224, 0, 224], [0, 224, 128], [0, 0, 1]], dtype=float), (6, 1, 1))
    # Teaching rotation: camera +z -> ego +x.
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    Ts = []
    for _ in range(6):
        T = np.eye(4)
        T[:3, :3] = R
        Ts.append(T)
    bev = lift_splat_one_frame(feature, K, np.stack(Ts), cfg)
    assert bev.shape == (8, cfg.bev_height, cfg.bev_width)
    assert torch.isfinite(bev).all()

    # Temporal warp shape check.
    T_cur_from_src = np.eye(4)
    T_cur_from_src[0, 3] = 1.0
    warped = warp_bev_to_current(bev, T_cur_from_src, cfg)
    assert warped.shape == bev.shape
    assert torch.isfinite(warped).all()

    # Planner/control/safety interface check.
    ctx = PipelineContext()
    tracks = {
        "demo": {
            "instance_token": "demo",
            "category": "vehicle.car",
            "current_xy": np.array([18.0, 0.5]),
            "velocity_xy": np.array([2.0, 0.0]),
        }
    }
    ctx.update({
        "world_model": {"tracks": tracks},
        "current_ego_speed_mps": 8.0,
        "dynamic_occupancy": np.zeros((cfg.bev_height, cfg.bev_width), dtype=np.uint8),
        "temporal_bev": torch.zeros((64, cfg.bev_height, cfg.bev_width)),
        "future_times_sec": np.arange(1, 7, dtype=float) * 0.5,
    })
    log = LessonLogger(0)
    stage16_behavior_planning.run(ctx, cfg, log)
    stage17_motion_planning.run(ctx, cfg, log)
    stage18_vehicle_control.run(ctx, cfg, log)
    stage19_safety_supervision.run(ctx, cfg, log)
    assert ctx.get("selected_ego_trajectory").shape == (6, 2)
    assert set(ctx.get("final_control")) == {"steer", "throttle", "brake"}

    print("SMOKE TEST PASSED")
    print(f"BEV shape: {tuple(bev.shape)}")
    print(f"Behavior: {ctx.get('behavior')}")
    print(f"Safety: {ctx.get('safety_status')}")


if __name__ == "__main__":
    main()
