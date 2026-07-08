from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_target_image(path: str | Path, image_size: int = 64) -> np.ndarray:
    """Load an RGB target image as float32 values in [0, 1]."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def save_canvas(canvas: np.ndarray, path: str | Path) -> None:
    """Save a float canvas in [0, 1] as an RGB image."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_array = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_array, mode="RGB").save(output_path)
