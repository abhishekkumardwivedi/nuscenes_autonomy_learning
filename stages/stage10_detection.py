from __future__ import annotations

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.geometry import metric_to_bev_indices
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary
from utils.nuscenes_utils import box_global_to_ego
from utils.visualization import normalize_01, tensor_magnitude

STAGE_NUMBER = 10
STAGE_NAME = "Object detection representation and targets"
SHORT_NAME = "detection"


class BEVDetectionHead(nn.Module):
    """Minimal CenterPoint-style teaching head.

    It predicts a center heatmap plus box regression channels. It is intentionally
    small; without training its predictions are not used as truth. Stage 10 uses
    nuScenes GT to show what this head should learn.
    """

    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(64, num_classes, 1)
        self.box = nn.Conv2d(64, 6, 1)  # dx,dy,w,l,sin(yaw),cos(yaw)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return {"heatmap": self.heatmap(h), "box": self.box(h)}


def _class_id(category: str) -> int:
    if category.startswith("vehicle.car"):
        return 0
    if category.startswith("vehicle"):
        return 1
    if category.startswith("human.pedestrian"):
        return 2
    return 3


def _gaussian_target(objects, cfg: PipelineConfig, num_classes: int = 4) -> np.ndarray:
    H, W = cfg.bev_height, cfg.bev_width
    target = np.zeros((num_classes, H, W), dtype=np.float32)
    rr_grid, cc_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    sigma = 2.0
    for obj in objects:
        x, y = obj["center"][:2]
        r, c, valid = metric_to_bev_indices(
            np.array([x]), np.array([y]),
            cfg.bev_x_min, cfg.bev_y_min, cfg.bev_resolution, H, W,
        )
        if not valid[0]:
            continue
        g = np.exp(-((rr_grid-r[0])**2 + (cc_grid-c[0])**2) / (2 * sigma**2))
        cls = _class_id(obj["category"])
        target[cls] = np.maximum(target[cls], g.astype(np.float32))
    return target


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("nusc", "current_sample", "T_global_from_ego", "temporal_bev")
    nusc = ctx.get("nusc")
    sample = ctx.get("current_sample")
    T_global_from_current = ctx.get("T_global_from_ego")[-1]
    temporal_bev = ctx.get("temporal_bev")

    log.substage(10, 1, "Transform nuScenes 3D GT boxes into current ego coordinates")
    objects = []
    for ann_token in sample["anns"]:
        obj = box_global_to_ego(nusc, ann_token, T_global_from_current)
        x, y = obj["center"][:2]
        if cfg.bev_x_min <= x < cfg.bev_x_max and cfg.bev_y_min <= y < cfg.bev_y_max:
            objects.append(obj)
    log.info(f"GT objects inside BEV = {len(objects)}")
    for obj in objects[:8]:
        log.detail(
            f"{obj['category']:<32} center=({obj['center'][0]:+.1f},{obj['center'][1]:+.1f}) m "
            f"v=({obj['velocity_xy'][0]:+.1f},{obj['velocity_xy'][1]:+.1f}) m/s"
        )

    log.substage(10, 2, "Build a center heatmap target")
    target_heatmap = _gaussian_target(objects, cfg)
    log.tensor("detection_target_heatmap [classes,H,W]", target_heatmap)
    log.detail("This heatmap is meaningful supervision: the network should learn peaks at object centers.")

    log.substage(10, 3, "Instantiate the learnable detection head")
    head = BEVDetectionHead(temporal_bev.shape[0], num_classes=4).to(cfg.torch_device).eval()
    with torch.no_grad():
        raw_pred = head(temporal_bev.unsqueeze(0))
    log.tensor("untrained heatmap logits", raw_pred["heatmap"])
    log.detail("We do NOT treat untrained logits as detections. GT remains the teaching reference until this head is trained.")

    log.substage(10, 4, "Visualize temporal BEV + ground-truth boxes")
    if cfg.save_plots:
        bg = normalize_01(tensor_magnitude(temporal_bev))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(bg, origin="lower", extent=[cfg.bev_y_min, cfg.bev_y_max, cfg.bev_x_min, cfg.bev_x_max])
        for obj in objects:
            p = obj["bottom_corners_xy"]
            closed = np.vstack([p, p[0]])
            ax.plot(closed[:, 1], closed[:, 0], linewidth=1.2)
            ax.scatter(obj["center"][1], obj["center"][0], s=8)
        ax.set_xlabel("y left (m)")
        ax.set_ylabel("x forward (m)")
        ax.set_title("Stage 10: temporal BEV with nuScenes GT 3D boxes")
        ax.set_xlim(cfg.bev_y_min, cfg.bev_y_max)
        ax.set_ylim(cfg.bev_x_min, cfg.bev_x_max)
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        fig.savefig(stage_dir / "gt_boxes_on_temporal_bev.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(target_heatmap.max(axis=0), origin="lower")
        ax.set_title("Stage 10: max-over-class detection center target")
        fig.tight_layout()
        fig.savefig(stage_dir / "detection_target_heatmap.png", dpi=150)
        plt.close(fig)

    values = {
        "gt_objects": objects,
        "detection_target_heatmap": target_heatmap,
        "detection_head_model": head,
        "untrained_detection_output": raw_pred,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, {
        "gt_object_count": len(objects),
        "categories": [o["category"] for o in objects],
        "detection_target_heatmap": target_heatmap,
        "untrained_detection_output": raw_pred,
    })
    log.outcome("You can now see both the detection supervision target and the neural head that will learn it.")
