from __future__ import annotations

import math
from typing import Iterable, Tuple
import numpy as np
from pyquaternion import Quaternion


def transform_matrix(translation: Iterable[float], rotation: Iterable[float]) -> np.ndarray:
    """Build T_parent_from_child from nuScenes translation + quaternion."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def yaw_from_quaternion(rotation: Iterable[float]) -> float:
    """Return yaw (radians) from a nuScenes quaternion."""
    R = Quaternion(rotation).rotation_matrix
    return math.atan2(R[1, 0], R[0, 0])


def yaw_from_matrix(T: np.ndarray) -> float:
    return math.atan2(float(T[1, 0]), float(T[0, 0]))


def transform_points(T_dst_from_src: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    """Transform Nx3 points from src coordinates into dst coordinates."""
    points = np.asarray(points_xyz, dtype=np.float64)
    ones = np.ones((len(points), 1), dtype=np.float64)
    p4 = np.concatenate([points, ones], axis=1)
    return (T_dst_from_src @ p4.T).T[:, :3]


def global_xy_to_ego_xy(global_xy: np.ndarray, T_global_from_ego: np.ndarray) -> np.ndarray:
    pts = np.asarray(global_xy, dtype=np.float64)
    xyz = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts))])
    return transform_points(np.linalg.inv(T_global_from_ego), xyz)[:, :2]


def metric_to_bev_indices(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map metric ego coordinates to BEV row/column.

    Array convention: row follows +x (forward); column follows +y (left).
    """
    row = np.floor((x - x_min) / resolution).astype(np.int64)
    col = np.floor((y - y_min) / resolution).astype(np.int64)
    valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    return row, col, valid


def bev_index_to_metric(
    row: np.ndarray,
    col: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = x_min + (np.asarray(row) + 0.5) * resolution
    y = y_min + (np.asarray(col) + 0.5) * resolution
    return x, y
