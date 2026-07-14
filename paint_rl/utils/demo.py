from __future__ import annotations

import numpy as np


def make_demo_target(image_width: int, image_height: int) -> np.ndarray:
    y, x = np.mgrid[0:image_height, 0:image_width].astype(np.float32)
    x = x / max(image_width - 1, 1)
    y = y / max(image_height - 1, 1)
    return np.stack([x, y, 1.0 - x], axis=-1).astype(np.float32)
