import numpy as np
import pytest

from paint_rl.utils.actions import decode_triangle_action


def test_decode_triangle_action_maps_alpha_into_range():
    action = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5],
        dtype=np.float32,
    )

    decoded = decode_triangle_action(action, alpha_min=0.05, alpha_max=0.8)

    assert np.allclose(decoded["vertices"], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    assert np.allclose(decoded["color"], [0.7, 0.8, 0.9])
    assert decoded["alpha"] == pytest.approx(0.425)
    assert np.allclose(decoded["action"], action.tolist())


def test_decode_triangle_action_clips_out_of_range_values():
    action = np.array(
        [1.5, -0.5, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2.0],
        dtype=np.float32,
    )

    decoded = decode_triangle_action(action, alpha_min=0.0, alpha_max=1.0)

    assert np.allclose(decoded["vertices"], [[1.0, 0.0], [0.3, 0.4], [0.5, 0.6]])
    assert np.allclose(decoded["color"], [0.7, 0.8, 0.9])
    assert decoded["alpha"] == 1.0
