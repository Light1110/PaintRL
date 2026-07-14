from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from paint_rl.training import FixedTargetReplayBuffer


HEIGHT = 4
WIDTH = 5


def make_observation_space(
    shape: tuple[int, int, int] = (11, HEIGHT, WIDTH),
) -> spaces.Box:
    return spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)


def make_action_space() -> spaces.Box:
    return spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)


def make_target() -> np.ndarray:
    return np.linspace(0.0, 1.0, HEIGHT * WIDTH * 3, dtype=np.float32).reshape(
        HEIGHT, WIDTH, 3
    )


def make_full_observation(
    canvas: np.ndarray,
    target: np.ndarray | None = None,
) -> np.ndarray:
    target = make_target() if target is None else target
    x = np.tile(np.linspace(0.0, 1.0, WIDTH, dtype=np.float32), (HEIGHT, 1))
    y = np.tile(
        np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None], (1, WIDTH)
    )
    return np.concatenate(
        [
            canvas.transpose(2, 0, 1),
            target.transpose(2, 0, 1),
            np.abs(canvas - target).transpose(2, 0, 1),
            np.stack([x, y]),
        ],
        axis=0,
    ).astype(np.float32)


def make_buffer(
    *,
    buffer_size: int = 4,
    n_envs: int = 1,
    target: np.ndarray | None = None,
    observation_space: spaces.Space | None = None,
    **kwargs: object,
) -> FixedTargetReplayBuffer:
    return FixedTargetReplayBuffer(
        buffer_size=buffer_size,
        observation_space=observation_space or make_observation_space(),
        action_space=make_action_space(),
        target_image=make_target() if target is None else target,
        device="cpu",
        n_envs=n_envs,
        **kwargs,
    )


def add_transition(
    buffer: FixedTargetReplayBuffer,
    canvas: np.ndarray,
    next_canvas: np.ndarray,
    *,
    done: float = 0.0,
    infos: list[dict[str, object]] | None = None,
) -> None:
    n_envs = buffer.n_envs
    obs = np.stack([make_full_observation(canvas)] * n_envs)
    next_obs = np.stack([make_full_observation(next_canvas)] * n_envs)
    buffer.add(
        obs,
        next_obs,
        np.zeros((n_envs, 2), dtype=np.float32),
        np.zeros(n_envs, dtype=np.float32),
        np.full(n_envs, done, dtype=np.float32),
        infos or [{} for _ in range(n_envs)],
    )


def test_stores_only_uint8_canvas_for_observations_and_next_observations():
    buffer = make_buffer(buffer_size=8)

    assert buffer.observations.shape == (8, 1, 3, HEIGHT, WIDTH)
    assert buffer.next_observations.shape == (8, 1, 3, HEIGHT, WIDTH)
    assert buffer.observations.dtype == np.uint8
    assert buffer.next_observations.dtype == np.uint8
    assert buffer.observations.nbytes == 8 * 3 * HEIGHT * WIDTH
    assert buffer.next_observations.nbytes == 8 * 3 * HEIGHT * WIDTH
    assert buffer.full_observation_space.shape == (11, HEIGHT, WIDTH)


def test_reconstructs_canvas_target_difference_and_coordinates():
    target = make_target()
    canvas = np.linspace(0.013, 0.987, HEIGHT * WIDTH * 3, dtype=np.float32).reshape(
        HEIGHT, WIDTH, 3
    )
    next_canvas = np.flip(canvas, axis=1).copy()
    buffer = make_buffer(target=target)
    add_transition(buffer, canvas, next_canvas)

    sample = buffer._get_samples(np.array([0]))
    obs = sample.observations.cpu().numpy()[0]
    next_obs = sample.next_observations.cpu().numpy()[0]
    expected_x = np.tile(
        np.linspace(0.0, 1.0, WIDTH, dtype=np.float32), (HEIGHT, 1)
    )
    expected_y = np.tile(
        np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None], (1, WIDTH)
    )

    assert obs.dtype == np.float32
    assert obs.shape == (11, HEIGHT, WIDTH)
    assert np.max(np.abs(obs[:3] - canvas.transpose(2, 0, 1))) <= 1 / 255
    assert np.max(np.abs(next_obs[:3] - next_canvas.transpose(2, 0, 1))) <= 1 / 255
    np.testing.assert_allclose(obs[3:6], target.transpose(2, 0, 1))
    np.testing.assert_allclose(obs[6:9], np.abs(obs[:3] - obs[3:6]))
    np.testing.assert_allclose(next_obs[6:9], np.abs(next_obs[:3] - next_obs[3:6]))
    np.testing.assert_allclose(obs[9], expected_x)
    np.testing.assert_allclose(obs[10], expected_y)


def test_time_limit_truncation_is_not_returned_as_terminal_done():
    buffer = make_buffer()
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    add_transition(
        buffer,
        canvas,
        canvas,
        done=1.0,
        infos=[{"TimeLimit.truncated": True}],
    )

    sample = buffer._get_samples(np.array([0]))

    assert sample.dones.item() == 0.0
    assert buffer.handle_timeout_termination is True


def test_ring_buffer_overwrites_oldest_compact_canvas():
    buffer = make_buffer(buffer_size=2)
    for value in (0.1, 0.5, 0.9):
        canvas = np.full((HEIGHT, WIDTH, 3), value, dtype=np.float32)
        add_transition(buffer, canvas, canvas)

    assert buffer.full
    assert buffer.pos == 1
    np.testing.assert_allclose(buffer.observations[0, 0], np.rint(0.9 * 255))
    np.testing.assert_allclose(buffer.observations[1, 0], np.rint(0.5 * 255))


@pytest.mark.parametrize(
    ("field", "wrong_shape"),
    [
        ("obs", (1, 10, HEIGHT, WIDTH)),
        ("next_obs", (11, HEIGHT, WIDTH)),
    ],
)
def test_add_rejects_observation_with_wrong_shape(
    field: str,
    wrong_shape: tuple[int, ...],
):
    buffer = make_buffer()
    valid = np.zeros((1, 11, HEIGHT, WIDTH), dtype=np.float32)
    values = {
        "obs": valid,
        "next_obs": valid,
    }
    values[field] = np.zeros(wrong_shape, dtype=np.float32)

    with pytest.raises(
        ValueError,
        match=rf"{field}.*\(1, 11, {HEIGHT}, {WIDTH}\)",
    ):
        buffer.add(
            values["obs"],
            values["next_obs"],
            np.zeros((1, 2), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            [{}],
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (np.zeros((HEIGHT, WIDTH), dtype=np.float32), "H x W x 3"),
        (np.zeros((HEIGHT + 1, WIDTH, 3), dtype=np.float32), "match"),
    ],
)
def test_rejects_invalid_target_shape(target: np.ndarray, message: str):
    with pytest.raises(ValueError, match=message):
        make_buffer(target=target)


@pytest.mark.parametrize(
    "observation_space",
    [
        spaces.Discrete(4),
        make_observation_space((10, HEIGHT, WIDTH)),
    ],
)
def test_rejects_invalid_full_observation_space(observation_space: spaces.Space):
    with pytest.raises(ValueError, match="observation_space"):
        make_buffer(observation_space=observation_space)


def test_rejects_optimize_memory_usage():
    with pytest.raises(ValueError, match="optimize_memory_usage"):
        make_buffer(optimize_memory_usage=True)


def test_rejects_disabling_timeout_termination_handling():
    with pytest.raises(ValueError, match="handle_timeout_termination"):
        FixedTargetReplayBuffer(
            4,
            make_observation_space(),
            make_action_space(),
            "cpu",
            1,
            False,
            False,
            target_image=make_target(),
        )


def test_supports_multiple_envs_with_shared_target(
    monkeypatch: pytest.MonkeyPatch,
):
    buffer = make_buffer(buffer_size=4, n_envs=2)
    canvases = np.stack(
        [
            np.full((HEIGHT, WIDTH, 3), 0.2, dtype=np.float32),
            np.full((HEIGHT, WIDTH, 3), 0.8, dtype=np.float32),
        ]
    )
    observations = np.stack([make_full_observation(canvas) for canvas in canvases])
    buffer.add(
        observations,
        observations,
        np.zeros((2, 2), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        [{}, {}],
    )
    monkeypatch.setattr(
        np.random,
        "randint",
        lambda low, high, size: np.array([0, 1]),
    )

    sample = buffer._get_samples(np.array([0, 0]))
    sampled = sample.observations.cpu().numpy()

    np.testing.assert_allclose(sampled[0, 3:6], sampled[1, 3:6])
    np.testing.assert_allclose(sampled[0, :3], np.rint(0.2 * 255) / 255)
    np.testing.assert_allclose(sampled[1, :3], np.rint(0.8 * 255) / 255)
