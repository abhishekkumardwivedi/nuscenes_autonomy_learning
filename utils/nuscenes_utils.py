from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np
from pyquaternion import Quaternion

from .geometry import transform_matrix, yaw_from_matrix


CAMERA_CHANNELS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

RADAR_CHANNELS = [
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]


def scene_sample_tokens(nusc, scene_record: Dict) -> List[str]:
    tokens: List[str] = []
    token = scene_record["first_sample_token"]
    while token:
        tokens.append(token)
        sample = nusc.get("sample", token)
        token = sample["next"]
    return tokens


def sample_ego_pose(nusc, sample: Dict) -> Tuple[Dict, np.ndarray]:
    """Use LIDAR_TOP keyframe as the sample's reference ego pose."""
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return pose, transform_matrix(pose["translation"], pose["rotation"])


def sample_sensor_calibration(nusc, sample: Dict, channel: str) -> Tuple[Dict, Dict]:
    sd = nusc.get("sample_data", sample["data"][channel])
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    return sd, calib


def box_global_to_ego(nusc, ann_token: str, T_global_from_ego: np.ndarray) -> Dict:
    """Return a nuScenes GT box expressed in the chosen current ego frame."""
    ann = nusc.get("sample_annotation", ann_token)
    box = nusc.get_box(ann_token)
    T_ego_from_global = np.linalg.inv(T_global_from_ego)

    center_global = np.asarray(box.center, dtype=np.float64).reshape(1, 3)
    from .geometry import transform_points
    center_ego = transform_points(T_ego_from_global, center_global)[0]

    corners_global = box.bottom_corners().T  # [4,3]
    bottom_ego = transform_points(T_ego_from_global, corners_global)[:, :2]

    R_ego_from_global = T_ego_from_global[:3, :3]
    R_box_ego = R_ego_from_global @ box.orientation.rotation_matrix
    yaw = float(np.arctan2(R_box_ego[1, 0], R_box_ego[0, 0]))

    velocity_global = nusc.box_velocity(ann_token)
    if np.any(np.isnan(velocity_global)):
        velocity_global = np.zeros(3, dtype=np.float64)
    velocity_ego = R_ego_from_global @ np.asarray(velocity_global, dtype=np.float64)

    return {
        "ann_token": ann_token,
        "instance_token": ann["instance_token"],
        "category": ann["category_name"],
        "center": center_ego,
        "size_wlh": np.asarray(box.wlh, dtype=np.float64),
        "yaw": yaw,
        "bottom_corners_xy": bottom_ego,
        "velocity_xy": velocity_ego[:2],
    }
