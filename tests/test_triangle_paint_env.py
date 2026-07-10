import numpy as np

from paint_rl.envs import TrianglePaintEnv

BLACK_TRIANGLE_ACTION = np.array(
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

LARGE_BLACK_TRIANGLE_ACTION = np.array(
    [
        0.0,
        0.0,
        1.0,
        0.0,
        0.5,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)

DEGENERATE_TRIANGLE_ACTION = np.array(
    [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)


def test_reset_returns_canvas_target_and_diff_channels():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=16, max_steps=5)

    observation, info = env.reset(seed=123)

    assert observation.shape == (11, 16, 16)
    assert observation.dtype == np.float32
    assert np.all((observation >= 0.0) & (observation <= 1.0))
    assert info["mse"] == np.float32(1.0)
    assert env.initial_mse == np.float32(1.0)


def test_observation_includes_absolute_coordinate_channels():
    target = np.zeros((4, 4, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=4, max_steps=5)

    observation, _ = env.reset(seed=123)

    expected_axis = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    np.testing.assert_allclose(observation[9], np.tile(expected_axis, (4, 1)))
    np.testing.assert_allclose(observation[10], np.tile(expected_axis[:, None], (1, 4)))


def test_step_draws_triangle_and_rewards_area_normalized_improvement():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(
        target_image=target,
        image_size=16,
        max_steps=5,
        step_reward_scale=1.0,
        episode_reward_scale=0.0,
        final_mse_penalty_scale=0.0,
    )
    env.reset(seed=123)

    observation, reward, terminated, truncated, info = env.step(BLACK_TRIANGLE_ACTION)

    assert observation.shape == (11, 16, 16)
    assert env.canvas.mean() > 0.0
    assert reward == np.float32(info["step_reward"])
    assert info["terminal_reward"] == np.float32(0.0)
    assert info["triangle_area"] > 0
    assert info["mean_pixel_improvement"] > 0.0
    assert info["step"] == 1
    assert not terminated
    assert not truncated


def test_area_normalization_removes_linear_area_bias():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    small_env = TrianglePaintEnv(
        target_image=target,
        image_size=16,
        max_steps=5,
        step_reward_scale=1.0,
        episode_reward_scale=0.0,
        final_mse_penalty_scale=0.0,
    )
    large_env = TrianglePaintEnv(
        target_image=target,
        image_size=16,
        max_steps=5,
        step_reward_scale=1.0,
        episode_reward_scale=0.0,
        final_mse_penalty_scale=0.0,
    )
    small_env.reset(seed=123)
    large_env.reset(seed=123)

    _, small_reward, _, _, small_info = small_env.step(BLACK_TRIANGLE_ACTION)
    _, large_reward, _, _, large_info = large_env.step(LARGE_BLACK_TRIANGLE_ACTION)

    assert large_info["triangle_area"] > small_info["triangle_area"]
    np.testing.assert_allclose(
        small_reward,
        large_reward,
        rtol=0.05,
    )


def test_terminal_reward_applied_on_last_step():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(
        target_image=target,
        image_size=16,
        max_steps=1,
        step_reward_scale=1.0,
        episode_reward_scale=100.0,
        final_mse_penalty_scale=200.0,
    )
    env.reset(seed=123)
    initial_mse = env.initial_mse

    _, reward, terminated, truncated, info = env.step(BLACK_TRIANGLE_ACTION)

    expected_step_reward = info["step_reward"]
    expected_terminal_reward = np.float32(
        100.0 * (initial_mse - env.current_mse) - 200.0 * env.current_mse
    )

    assert not terminated
    assert truncated
    assert info["terminal_reward"] == expected_terminal_reward
    assert reward == np.float32(expected_step_reward + expected_terminal_reward)


COLINEAR_TRIANGLE_ACTION = np.array(
    [
        0.0,
        0.5,
        0.5,
        0.5,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)


def test_degenerate_triangle_does_not_crash():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=16, max_steps=5)
    env.reset(seed=123)

    _, reward, _, _, info = env.step(COLINEAR_TRIANGLE_ACTION)

    assert info["triangle_area"] >= 0
    assert np.isfinite(reward)
    assert "mean_pixel_improvement" in info
    assert "step_reward" in info


def test_coincident_triangle_covers_at_most_one_pixel():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=16, max_steps=5)
    env.reset(seed=123)

    _, reward, _, _, info = env.step(DEGENERATE_TRIANGLE_ACTION)

    assert info["triangle_area"] <= 1
    assert np.isfinite(reward)


def test_reward_scale_maps_to_step_reward_scale():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(
        target_image=target,
        image_size=16,
        max_steps=5,
        reward_scale=2.5,
        episode_reward_scale=0.0,
        final_mse_penalty_scale=0.0,
    )
    env.reset(seed=123)

    _, reward, _, _, info = env.step(BLACK_TRIANGLE_ACTION)

    assert env.step_reward_scale == np.float32(2.5)
    assert reward == np.float32(2.5 * info["mean_pixel_improvement"])


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

    _, _, terminated, truncated, info = env.step(BLACK_TRIANGLE_ACTION)

    assert not terminated
    assert truncated
    terminal_canvas = info["terminal_canvas"]
    np.testing.assert_allclose(terminal_canvas, env.canvas)
    assert not np.shares_memory(terminal_canvas, env.canvas)

    env.reset(seed=456)

    assert not np.allclose(terminal_canvas, env.canvas)
