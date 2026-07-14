import math

import numpy as np
import pytest

from paint_rl.utils.actions import ACTION_VERSION, decode_triangle_action


def _area(vertices: list[list[float]] | np.ndarray) -> float:
    points = np.asarray(vertices, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(
        x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])
    )


def test_decode_centered_symmetric_triangle():
    # center=(0.5,0.5), w=0.4, h=0.3, rotation=0, skew=0, RGB black, alpha mid
    size_min = 0.02
    size_max = 1.0
    width_unit = (0.4 - size_min) / (size_max - size_min)
    height_unit = (0.3 - size_min) / (size_max - size_min)
    action = np.array(
        [0.5, 0.5, width_unit, height_unit, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5],
        dtype=np.float32,
    )

    decoded = decode_triangle_action(
        action,
        alpha_min=0.05,
        alpha_max=0.8,
        size_min=size_min,
        size_max=size_max,
    )

    assert decoded["action_version"] == ACTION_VERSION
    assert decoded["center"] == pytest.approx([0.5, 0.5])
    assert decoded["width"] == pytest.approx(0.4)
    assert decoded["height"] == pytest.approx(0.3)
    assert decoded["rotation"] == pytest.approx(0.0)
    assert decoded["skew"] == pytest.approx(0.0)
    assert np.allclose(
        decoded["vertices"],
        [[0.3, 0.65], [0.7, 0.65], [0.5, 0.35]],
        atol=1e-6,
    )
    assert np.allclose(decoded["color"], [0.0, 0.0, 0.0])
    assert decoded["alpha"] == pytest.approx(0.425)
    assert _area(decoded["vertices"]) == pytest.approx(0.4 * 0.3 / 2.0)


def test_decode_applies_rotation_around_center():
    size_min = 0.02
    size_max = 1.0
    width_unit = (0.4 - size_min) / (size_max - size_min)
    height_unit = (0.2 - size_min) / (size_max - size_min)
    # target width=0.4, height=0.2, rotation=+π (unit=1.0), skew=0
    action = np.array(
        [0.5, 0.5, width_unit, height_unit, 1.0, 0.5, 0.1, 0.2, 0.3, 0.0],
        dtype=np.float32,
    )

    decoded = decode_triangle_action(
        action,
        alpha_min=0.0,
        alpha_max=1.0,
        size_min=size_min,
        size_max=size_max,
    )

    assert decoded["rotation"] == pytest.approx(math.pi)
    # 180° rotation: local (-0.2,0.1),(0.2,0.1),(0,-0.1) -> (0.2,-0.1),(-0.2,-0.1),(0,0.1)
    assert np.allclose(
        decoded["vertices"],
        [[0.7, 0.4], [0.3, 0.4], [0.5, 0.6]],
        atol=1e-5,
    )


def test_decode_applies_skew_to_apex():
    size_min = 0.02
    size_max = 1.0
    width_unit = (0.4 - size_min) / (size_max - size_min)
    height_unit = (0.2 - size_min) / (size_max - size_min)
    action = np.array(
        [0.5, 0.5, width_unit, height_unit, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    decoded = decode_triangle_action(
        action,
        alpha_min=0.0,
        alpha_max=1.0,
        size_min=size_min,
        size_max=size_max,
    )

    assert decoded["skew"] == pytest.approx(1.0)
    # apex x = center_x + skew * width/2 = 0.5 + 0.2
    assert np.allclose(
        decoded["vertices"],
        [[0.3, 0.6], [0.7, 0.6], [0.7, 0.4]],
        atol=1e-6,
    )


def test_decode_maps_size_bounds():
    action_min = np.array(
        [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    action_max = np.array(
        [0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    decoded_min = decode_triangle_action(
        action_min, alpha_min=0.05, alpha_max=0.8, size_min=0.02, size_max=1.0
    )
    decoded_max = decode_triangle_action(
        action_max, alpha_min=0.05, alpha_max=0.8, size_min=0.02, size_max=1.0
    )

    assert decoded_min["width"] == pytest.approx(0.02)
    assert decoded_min["height"] == pytest.approx(0.02)
    assert decoded_max["width"] == pytest.approx(1.0)
    assert decoded_max["height"] == pytest.approx(1.0)


def test_decode_clips_action_units_but_allows_vertices_outside_unit_square():
    # center near corner + large size; vertices may leave [0,1]
    action = np.array(
        [1.5, -0.5, 1.0, 1.0, 0.5, 0.5, 2.0, -1.0, 0.5, 2.0],
        dtype=np.float32,
    )

    decoded = decode_triangle_action(
        action, alpha_min=0.0, alpha_max=1.0, size_min=0.02, size_max=1.0
    )

    assert np.allclose(decoded["action"][:2], [1.0, 0.0])
    assert np.allclose(decoded["color"], [1.0, 0.0, 0.5])
    assert decoded["alpha"] == pytest.approx(1.0)
    vertices = np.asarray(decoded["vertices"])
    assert np.any(vertices < 0.0) or np.any(vertices > 1.0)


def test_decode_area_always_positive():
    rng = np.random.default_rng(0)
    for _ in range(50):
        action = rng.random(10).astype(np.float32)
        decoded = decode_triangle_action(
            action, alpha_min=0.05, alpha_max=0.8, size_min=0.02, size_max=1.0
        )
        assert _area(decoded["vertices"]) == pytest.approx(
            decoded["width"] * decoded["height"] / 2.0, rel=1e-5, abs=1e-6
        )
        assert _area(decoded["vertices"]) > 0.0
