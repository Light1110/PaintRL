import numpy as np

from paint_rl.envs import TrianglePaintEnv


def test_reset_returns_canvas_target_and_diff_channels():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=16, max_steps=5)

    observation, info = env.reset(seed=123)

    assert observation.shape == (9, 16, 16)
    assert observation.dtype == np.float32
    assert np.all((observation >= 0.0) & (observation <= 1.0))
    assert info["mse"] == np.float32(1.0)


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

    assert observation.shape == (9, 16, 16)
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
