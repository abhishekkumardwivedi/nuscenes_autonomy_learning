from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.geometry import transform_matrix
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary
from utils.nuscenes_utils import CAMERA_CHANNELS, sample_ego_pose
from utils.visualization import save_camera_montage

STAGE_NUMBER = 2
STAGE_NAME = "Sensor loading and camera preprocessing"
SHORT_NAME = "sensor_preprocess"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 02: decode six cameras for every temporal frame and normalize them."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("nusc", "history_samples")
    nusc = ctx.get("nusc")
    samples = ctx.get("history_samples")

    log.substage(2, 1, "Load and resize six camera images per time step")
    all_frames: List[torch.Tensor] = []
    current_raw_images: List[Image.Image] = []
    original_sizes = []

    for t, sample in enumerate(samples):
        frame_tensors = []
        for cam in CAMERA_CHANNELS:
            sd_token = sample["data"][cam]
            sd = nusc.get("sample_data", sd_token)
            path = Path(nusc.get_sample_data_path(sd_token))
            img = Image.open(path).convert("RGB")
            original_sizes.append((t, cam, img.height, img.width))
            resized = img.resize((cfg.image_width, cfg.image_height), Image.BILINEAR)
            if t == len(samples) - 1:
                current_raw_images.append(resized.copy())
            frame_tensors.append(_pil_to_tensor(resized))
        all_frames.append(torch.stack(frame_tensors, dim=0))

    images_01 = torch.stack(all_frames, dim=0)  # [T, 6, 3, H, W]
    mean = IMAGENET_MEAN.to(images_01.dtype)
    std = IMAGENET_STD.to(images_01.dtype)
    images_normalized = (images_01 - mean) / std

    log.tensor("images_01 [T,6,3,H,W]", images_01)
    log.tensor("images_normalized", images_normalized)
    log.detail("Normalization uses the standard ImageNet mean/std expected by ResNet backbones.")

    log.substage(2, 2, "Read ego pose for each temporal sample")
    ego_pose_records = []
    T_global_from_ego = []
    timestamps_sec = []
    for sample in samples:
        pose, T = sample_ego_pose(nusc, sample)
        ego_pose_records.append(pose)
        T_global_from_ego.append(T)
        timestamps_sec.append(sample["timestamp"] / 1e6)
    T_global_from_ego = np.stack(T_global_from_ego, axis=0)
    timestamps_sec = np.asarray(timestamps_sec, dtype=np.float64)
    log.tensor("T_global_from_ego [T,4,4]", T_global_from_ego)
    log.detail(f"Sequence duration = {timestamps_sec[-1] - timestamps_sec[0]:.2f} s")

    log.substage(2, 3, "Create a visual checkpoint")
    montage_path = stage_dir / "current_six_cameras.png"
    if cfg.save_plots:
        save_camera_montage(current_raw_images, CAMERA_CHANNELS, montage_path)
        log.info(f"Saved six-camera montage -> {montage_path}")

    values = {
        "images_01": images_01,
        "images_normalized": images_normalized,
        "current_raw_images": current_raw_images,
        "original_camera_sizes": original_sizes,
        "ego_pose_records": ego_pose_records,
        "T_global_from_ego": T_global_from_ego,
        "timestamps_sec": timestamps_sec,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, values)
    log.outcome("Raw camera files are now explicit tensors, and the ego pose timeline is available.")
