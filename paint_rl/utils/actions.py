from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

ACTION_VERSION = 2
ACTION_DIM = 10
DEFAULT_SIZE_MIN = 0.02
DEFAULT_SIZE_MAX = 1.0


class DecodedTriangleAction(TypedDict):
    action_version: int
    action: list[float]
    center: list[float]
    width: float
    height: float
    rotation: float
    skew: float
    vertices: list[list[float]]
    color: list[float]
    alpha: float


def decode_triangle_action(
    action: np.ndarray,
    *,
    alpha_min: float,
    alpha_max: float,
    size_min: float = DEFAULT_SIZE_MIN,
    size_max: float = DEFAULT_SIZE_MAX,
) -> DecodedTriangleAction:
    """Decode a 10-dim structured action into triangle parameters.

    Action layout:
        center_x, center_y, width_unit, height_unit,
        rotation_unit, skew_unit, r, g, b, alpha_unit
    """
    if not 0.0 < size_min <= size_max <= 1.0:
        raise ValueError("size range must satisfy 0 < size_min <= size_max <= 1")

    clipped = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
    center_x = float(clipped[0])
    center_y = float(clipped[1])
    width = float(size_min + float(clipped[2]) * (size_max - size_min))
    height = float(size_min + float(clipped[3]) * (size_max - size_min))
    rotation = float((2.0 * float(clipped[4]) - 1.0) * math.pi)
    skew = float(2.0 * float(clipped[5]) - 1.0)
    color = clipped[6:9].astype(np.float32)
    alpha_unit = float(clipped[9])
    alpha = float(alpha_min + alpha_unit * (alpha_max - alpha_min))

    local = np.array(
        [
            [-0.5 * width, 0.5 * height],
            [0.5 * width, 0.5 * height],
            [0.5 * skew * width, -0.5 * height],
        ],
        dtype=np.float32,
    )
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    rotation_matrix = np.array(
        [[cos_r, -sin_r], [sin_r, cos_r]],
        dtype=np.float32,
    )
    rotated = local @ rotation_matrix.T
    vertices = rotated + np.array([center_x, center_y], dtype=np.float32)

    return {
        "action_version": ACTION_VERSION,
        "action": [float(value) for value in clipped],
        "center": [center_x, center_y],
        "width": width,
        "height": height,
        "rotation": rotation,
        "skew": skew,
        "vertices": [[float(x), float(y)] for x, y in vertices],
        "color": [float(value) for value in color],
        "alpha": alpha,
    }
