from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class StageMeta:
    number: int
    alias: str
    title: str
    short_name: str
    module: str
    code_files: List[str]
    concept: str
    inputs: List[str]
    outputs: List[str]
    dataset: str
    substeps: int
    preferred_visual: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# This registry is intentionally explicit. It doubles as the learning curriculum,
# the dashboard navigation model and the execution registry.
STAGES: List[StageMeta] = [
    StageMeta(0, "foundation", "Foundation / configuration", "foundation", "stages.stage00_foundation",
              ["stages/stage00_foundation.py", "config.py"],
              "Fix reproducibility, compute device and the coordinate/BEV convention before processing data.",
              ["Pipeline configuration"], ["Device", "Seed", "BEV grid convention"], "nuScenes Mini", 3),
    StageMeta(1, "dataset", "nuScenes sequence loading", "dataset", "stages.stage01_dataset",
              ["stages/stage01_dataset.py", "utils/nuscenes_utils.py"],
              "Open nuScenes, select a scene and construct one temporal history plus future window.",
              ["nuScenes dataroot", "Scene index", "Sample index"], ["Selected scene", "History samples", "Future samples"], "nuScenes Mini", 4),
    StageMeta(2, "preprocess", "Sensor loading and camera preprocessing", "sensor_preprocess", "stages.stage02_sensor_preprocess",
              ["stages/stage02_sensor_preprocess.py"],
              "Load the six cameras for every temporal sample, resize/normalize them and read ego poses.",
              ["Temporal sample tokens"], ["Camera tensors", "Camera RGB previews", "Ego poses"], "nuScenes Mini", 3,
              "current_six_cameras.png"),
    StageMeta(3, "geometry", "Calibration and coordinate geometry", "geometry", "stages.stage03_calibration_geometry",
              ["stages/stage03_calibration_geometry.py", "utils/geometry.py"],
              "Understand camera intrinsics/extrinsics and how sensor coordinates relate to the ego frame.",
              ["Camera sample data", "Calibration records"], ["Intrinsics", "Extrinsics", "Camera-to-ego transforms"], "nuScenes Mini", 3,
              "camera_geometry_topdown.png"),
    StageMeta(4, "encoder", "Camera encoder: ResNet-50 + feature pyramid", "camera_encoder", "stages.stage04_camera_encoder",
              ["stages/stage04_camera_encoder.py"],
              "Convert pixels into multi-scale semantic feature maps using a CNN backbone and feature pyramid.",
              ["Preprocessed camera tensors"], ["C2/C3/C4/C5 backbone features", "FPN features"], "nuScenes Mini", 4,
              "current_p4_feature_montage.png"),
    StageMeta(5, "camera_bev", "Camera features → BEV", "camera_bev", "stages.stage05_camera_bev",
              ["stages/stage05_camera_bev.py", "utils/geometry.py"],
              "Lift image features along camera rays, transform them into ego coordinates and splat into bird's-eye view.",
              ["Camera features", "Intrinsics", "Extrinsics"], ["Camera BEV sequence"], "nuScenes Mini", 5,
              "current_camera_bev.png"),
    StageMeta(6, "radar_bev", "Radar points → radar BEV", "radar_bev", "stages.stage06_radar_bev",
              ["stages/stage06_radar_bev.py"],
              "Load five radars, transform returns into ego space and rasterize/encode them on the BEV grid.",
              ["Radar sweeps", "Radar calibration"], ["Raw radar BEV", "Encoded radar BEV"], "nuScenes Mini", 3,
              "current_radar_count.png"),
    StageMeta(7, "spatial_fusion", "Camera + radar spatial fusion", "spatial_fusion", "stages.stage07_spatial_fusion",
              ["stages/stage07_spatial_fusion.py"],
              "Bring camera and radar features onto the same BEV grid and fuse their complementary information.",
              ["Camera BEV", "Radar BEV"], ["Fused BEV sequence"], "nuScenes Mini", 4,
              "current_interpretable_fusion.png"),
    StageMeta(8, "ego_motion", "Localization / ego motion", "ego_motion", "stages.stage08_localization_ego_motion",
              ["stages/stage08_localization_ego_motion.py", "utils/geometry.py"],
              "Compute historical ego poses relative to the current frame so temporal BEVs can be aligned.",
              ["Ego poses over history"], ["Current-from-history transforms", "Estimated speed"], "nuScenes Mini", 3,
              "ego_history_current_frame.png"),
    StageMeta(9, "temporal", "Temporal BEV fusion", "temporal_bev", "stages.stage09_temporal_bev",
              ["stages/stage09_temporal_bev.py"],
              "Warp historical BEVs into current ego coordinates and fuse memory using an interpretable EMA or ConvGRU.",
              ["Fused BEV history", "Ego-motion transforms"], ["Aligned BEVs", "Temporal BEV"], "nuScenes Mini", 3,
              "temporal_bev.png"),
    StageMeta(10, "detection", "Object detection", "detection", "stages.stage10_detection",
              ["stages/stage10_detection.py"],
              "Represent 3D object detection targets in BEV and expose the learnable detection-head boundary.",
              ["Temporal BEV", "nuScenes 3D annotations"], ["Detection targets", "Detection-head output"], "nuScenes Mini", 4,
              "gt_boxes_on_temporal_bev.png"),
    StageMeta(11, "tracking", "Object tracking", "tracking", "stages.stage11_tracking",
              ["stages/stage11_tracking.py"],
              "Build temporal object histories and understand what data association must recover in a real tracker.",
              ["Historical annotations/detections"], ["Track histories", "Object state history"], "nuScenes Mini", 3,
              "track_histories.png"),
    StageMeta(12, "occupancy", "Occupancy / free-space representation", "occupancy", "stages.stage12_occupancy",
              ["stages/stage12_occupancy.py"],
              "Rasterize occupied space in BEV and combine dynamic-object and radar evidence.",
              ["3D boxes", "Radar BEV"], ["Dynamic occupancy grid"], "nuScenes Mini", 2,
              "dynamic_occupancy.png"),
    StageMeta(13, "prediction", "Agent trajectory prediction", "prediction", "stages.stage13_prediction",
              ["stages/stage13_prediction.py"],
              "Predict future motion of surrounding agents and compare a transparent baseline with recorded future trajectories.",
              ["Track histories", "Future annotations"], ["Predicted trajectories", "Prediction error references"], "nuScenes Mini", 3,
              "agent_predictions.png"),
    StageMeta(14, "map", "HD-map / road context", "map_context", "stages.stage14_map_context",
              ["stages/stage14_map_context.py"],
              "Rasterize semantic road/map layers around the ego vehicle and understand route-context limitations.",
              ["Scene location", "Ego pose", "nuScenes map"], ["Semantic map channels"], "nuScenes Mini", 4,
              "semantic_map_layers.png"),
    StageMeta(15, "world_model", "World-model assembly", "world_model", "stages.stage15_world_model",
              ["stages/stage15_world_model.py"],
              "Assemble temporal perception, occupancy, tracks, predictions, map and ego state into one planning-facing world state.",
              ["Temporal BEV", "Tracks", "Occupancy", "Predictions", "Map", "Ego state"], ["World-state bundle"], "nuScenes Mini", 2,
              "world_model.png"),
    StageMeta(16, "behavior", "Behavior planning", "behavior_planning", "stages.stage16_behavior_planning",
              ["stages/stage16_behavior_planning.py"],
              "Choose a high-level driving behavior from the current world state using a transparent baseline policy.",
              ["World model", "Route/context"], ["Behavior decision"], "nuPlan / CARLA later", 2),
    StageMeta(17, "motion_planning", "Motion planning / future ego trajectory", "motion_planning", "stages.stage17_motion_planning",
              ["stages/stage17_motion_planning.py"],
              "Generate candidate trajectories, score them and expose a learned trajectory-head alternative.",
              ["Behavior", "World model", "Occupancy"], ["Candidate trajectories", "Selected trajectory"], "nuPlan / CARLA later", 4,
              "motion_candidates.png"),
    StageMeta(18, "control", "Vehicle control", "vehicle_control", "stages.stage18_vehicle_control",
              ["stages/stage18_vehicle_control.py"],
              "Convert the selected trajectory into steering, throttle and brake commands using understandable controllers.",
              ["Selected trajectory", "Ego speed"], ["Steering", "Throttle", "Brake"], "CARLA later", 2),
    StageMeta(19, "safety", "Runtime safety / fallback supervision", "safety_supervision", "stages.stage19_safety_supervision",
              ["stages/stage19_safety_supervision.py"],
              "Check numerical health and collision risk, then override unsafe commands with a minimum-risk fallback.",
              ["World model", "Trajectory", "Control command"], ["Safety status", "Supervised command"], "CARLA later", 3),
    StageMeta(20, "closed_loop", "Closed-loop integration boundary", "closed_loop", "stages.stage20_closed_loop",
              ["stages/stage20_closed_loop.py", "adapters/carla_adapter.py"],
              "Show where offline replay ends and true action → new observation feedback begins in CARLA.",
              ["Supervised vehicle command"], ["Offline replay note or CARLA command application"], "CARLA", 2),
]

BY_NUMBER: Dict[int, StageMeta] = {s.number: s for s in STAGES}
BY_ALIAS: Dict[str, StageMeta] = {s.alias: s for s in STAGES}


def stage_dir(base: Path, stage: StageMeta) -> Path:
    return base / f"stage{stage.number:02d}_{stage.short_name}"
