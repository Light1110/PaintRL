import numpy as np
from PIL import Image

from paint_rl.utils.image import load_target_image, save_canvas


def test_load_target_image_resizes_and_normalizes_rgb(tmp_path):
    image_path = tmp_path / "target.png"
    Image.new("RGB", (8, 4), color=(255, 128, 0)).save(image_path)

    loaded = load_target_image(image_path, image_size=16)

    assert loaded.shape == (16, 16, 3)
    assert loaded.dtype == np.float32
    assert np.all((loaded >= 0.0) & (loaded <= 1.0))


def test_save_canvas_writes_uint8_rgb_image(tmp_path):
    output_path = tmp_path / "canvas.png"
    canvas = np.zeros((4, 4, 3), dtype=np.float32)
    canvas[..., 1] = 0.5

    save_canvas(canvas, output_path)

    saved = Image.open(output_path)
    assert saved.mode == "RGB"
    assert saved.size == (4, 4)
