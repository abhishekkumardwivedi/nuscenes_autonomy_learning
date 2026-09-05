from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class PipelineConfig:
    """Central configuration for the learning pipeline.

    The project intentionally keeps the configuration explicit and small so that
    every important choice is visible while you are learning.
    """

    dataroot: str
    version: str = "v1.0-mini"
    output_dir: str = "outputs"
    scene_index: int = 0
    sample_index: int = -1
    history_frames: int = 4
    future_frames: int = 6

    # Camera network input size. nuScenes camera images are resized to this size.
    image_height: int = 256
    image_width: int = 448

    # Bird's-eye-view region in ego coordinates (meters).
    # Ego frame convention: +x forward, +y left, +z up.
    bev_x_min: float = -50.0
    bev_x_max: float = 50.0
    bev_y_min: float = -50.0
    bev_y_max: float = 50.0
    bev_resolution: float = 0.5

    # Camera lift-splat teaching configuration.
    depth_min: float = 4.0
    depth_max: float = 50.0
    depth_bins: int = 24

    # Model options.
    pretrained_backbone: bool = True
    temporal_model: str = "ema"  # "ema" (interpretable) or "convgru" (untrained model demo)
    planner_mode: str = "classical"  # "classical" or "learned"

    # Execution / display controls.
    verbose: int = 2
    save_plots: bool = True
    device: str = "auto"
    seed: int = 7

    # Stage 20 integration. "offline" is safe for nuScenes replay; "carla" applies one command.
    backend: str = "offline"
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000

    @property
    def bev_height(self) -> int:
        return int(round((self.bev_x_max - self.bev_x_min) / self.bev_resolution))

    @property
    def bev_width(self) -> int:
        return int(round((self.bev_y_max - self.bev_y_min) / self.bev_resolution))

    @property
    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
