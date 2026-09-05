from __future__ import annotations

from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger, make_stage_dir, save_stage_summary
from utils.nuscenes_utils import CAMERA_CHANNELS
from utils.visualization import save_feature_montage

STAGE_NUMBER = 4
STAGE_NAME = "Camera encoder: ResNet-50 + feature pyramid"
SHORT_NAME = "camera_encoder"


class ResNet50FPN(nn.Module):
    """Small teaching FPN built on top of torchvision ResNet-50.

    Output channels are intentionally reduced to 64 so later BEV stages remain
    understandable and lightweight. We expose C2..C5 and P3..P5 so you can see
    exactly where the feature pyramid comes from.
    """

    def __init__(self, pretrained: bool = True, out_channels: int = 64):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        try:
            net = resnet50(weights=weights)
            self.loaded_pretrained = weights is not None
        except Exception as exc:
            print(f"  [WARN] Could not load pretrained ResNet-50 weights: {exc}")
            print("  [WARN] Falling back to random initialization. Shapes are still valid, semantics are not trained.")
            net = resnet50(weights=None)
            self.loaded_pretrained = False

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # C2: 256 channels, 1/4
        self.layer2 = net.layer2  # C3: 512 channels, 1/8
        self.layer3 = net.layer3  # C4: 1024 channels, 1/16
        self.layer4 = net.layer4  # C5: 2048 channels, 1/32

        # Standard top-down FPN idea: lateral 1x1 projections + upsample + add.
        self.lat3 = nn.Conv2d(512, out_channels, 1)
        self.lat4 = nn.Conv2d(1024, out_channels, 1)
        self.lat5 = nn.Conv2d(2048, out_channels, 1)
        self.out3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5_lat = self.lat5(c5)
        p4_lat = self.lat4(c4) + F.interpolate(p5_lat, size=c4.shape[-2:], mode="nearest")
        p3_lat = self.lat3(c3) + F.interpolate(p4_lat, size=c3.shape[-2:], mode="nearest")

        p3 = self.out3(p3_lat)
        p4 = self.out4(p4_lat)
        p5 = self.out5(p5_lat)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5, "p3": p3, "p4": p4, "p5": p5}


def run(ctx: PipelineContext, cfg: PipelineConfig, log: LessonLogger) -> None:
    """Stage 04: convert normalized images into multi-scale semantic features."""
    log.stage(STAGE_NUMBER, STAGE_NAME)
    stage_dir = make_stage_dir(cfg.output_path, STAGE_NUMBER, SHORT_NAME)
    ctx.require("images_normalized")

    images = ctx.get("images_normalized")
    T, N, C, H, W = images.shape
    device = cfg.torch_device

    log.substage(4, 1, "Flatten time and camera dimensions for the CNN")
    batch = images.reshape(T * N, C, H, W).to(device)
    log.tensor("CNN input [T*6,3,H,W]", batch)

    log.substage(4, 2, "Run ResNet-50 backbone")
    model = ResNet50FPN(pretrained=cfg.pretrained_backbone, out_channels=64).to(device)
    model.eval()
    with torch.no_grad():
        feats = model(batch)
    log.info(f"Pretrained backbone loaded = {model.loaded_pretrained}")
    for name in ["c2", "c3", "c4", "c5"]:
        log.tensor(name.upper(), feats[name])

    log.substage(4, 3, "Build the feature pyramid")
    for name in ["p3", "p4", "p5"]:
        log.tensor(name.upper(), feats[name])
    log.detail("P3 has higher spatial resolution; P5 has stronger abstraction but lower resolution.")
    log.detail("For the teaching BEV stage we use P4: a useful middle ground between detail and cost.")

    # Restore [T, 6, C, Hf, Wf].
    features_by_level = {}
    for name, value in feats.items():
        features_by_level[name] = value.reshape(T, N, *value.shape[1:]).detach()
    camera_features = features_by_level["p4"]

    log.substage(4, 4, "Visual checkpoint: feature activation")
    feature_path = stage_dir / "current_p4_feature_montage.png"
    if cfg.save_plots:
        save_feature_montage(
            camera_features[-1], CAMERA_CHANNELS, feature_path,
            "Stage 04: P4 mean absolute activation for each camera",
        )
        log.info(f"Saved feature montage -> {feature_path}")

    values = {
        "camera_encoder_model": model,
        "camera_features_by_level": features_by_level,
        "camera_features": camera_features,
        "camera_feature_level": "p4",
        "camera_encoder_pretrained": model.loaded_pretrained,
    }
    ctx.update(values)
    save_stage_summary(stage_dir, {
        "camera_features": camera_features,
        "feature_level": "p4",
        "pretrained": model.loaded_pretrained,
        "levels": {k: v for k, v in features_by_level.items()},
    })
    log.outcome("Six RGB images per time step are now compact CNN/FPN feature tensors ready for 3D lifting.")
