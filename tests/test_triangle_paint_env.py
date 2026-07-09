import numpy as np

from paint_rl.envs import TrianglePaintEnv


def test_reset_returns_canvas_target_and_diff_channels():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=16, max_steps=5)

    observation, info = env.reset(seed=123)

    assert observation.shape == (11, 16, 16)
    assert observation.dtype == np.float32
    assert np.all((observation >= 0.0) & (observation <= 1.0))
    assert info["mse"] == np.float32(1.0)


def test_observation_includes_absolute_coordinate_channels():
    target = np.zeros((4, 4, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=4, max_steps=5)

    observation, _ = env.reset(seed=123)

    expected_axis = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    np.testing.assert_allclose(observation[9], np.tile(expected_axis, (4, 1)))
    np.testing.assert_allclose(observation[10], np.tile(expected_axis[:, None], (1, 4)))


def test_step_draws_triangle_and_rewards_mse_improvement():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=16, max_steps=5, reward_scale=1.0)
    env.reset(seed=123)
    old_mse = env.current_mse

    action = np.array(
        [
            0.1,
            0.1,
            0.9,
            0.1,
            0.5,
            0.9,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float32,
    )
    observation, reward, terminated, truncated, info = env.step(action)

    assert observation.shape == (11, 16, 16)
    assert env.canvas.mean() > 0.0
    assert reward == np.float32(old_mse - env.current_mse)
    assert info["step"] == 1
    assert not terminated
    assert not truncated


def test_episode_truncates_at_max_steps():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=16, max_steps=1)
    env.reset(seed=123)

    _, _, terminated, truncated, info = env.step(env.action_space.sample())

    assert not terminated
    assert truncated
    assert info["step"] == 1


def test_terminal_info_includes_canvas_snapshot_copy():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=16, max_steps=1)
    env.reset(seed=123)

    action = np.array(
        [
            0.1,
            0.1,
            0.9,
            0.1,
            0.5,
            0.9,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float32,
    )
    _, _, terminated, truncated, info = env.step(action)

    assert not terminated
    assert truncated
    terminal_canvas = info["terminal_canvas"]
    np.testing.assert_allclose(terminal_canvas, env.canvas)
    assert not np.shares_memory(terminal_canvas, env.canvas)

    env.reset(seed=456)

    assert not np.allclose(terminal_canvas, env.canvas)
