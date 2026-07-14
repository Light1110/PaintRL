import numpy as np

from paint_rl.utils.demo import make_demo_target


def test_make_demo_target_shape_and_dtype():
    target = make_demo_target(8, 4)

    assert target.shape == (4, 8, 3)
    assert target.dtype == np.float32
    assert float(target.min()) >= 0.0
    assert float(target.max()) <= 1.0
