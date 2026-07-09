from __future__ import annotations

import numpy as np


def decode_triangle_action(
    action: np.ndarray,
    *,
    alpha_min: float,
    alpha_max: float,
) -> dict[str, object]:
    """Decode a 10-dim action into triangle parameters."""
    clipped = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
    points = clipped[:6].reshape(3, 2)
    color = clipped[6:9]
    alpha_unit = float(clipped[9])
    alpha = float(alpha_min + alpha_unit * (alpha_max - alpha_min))

    return {
        "action": [float(value) for value in clipped],
        "vertices": [[float(x), float(y)] for x, y in points],
        "color": [float(value) for value in color],
        "alpha": alpha,
    }
