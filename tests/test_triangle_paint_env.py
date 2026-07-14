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

WORSE_WHITE_TRIANGLE_ACTION = np.array(
    [
        0.1,
        0.1,
        0.9,
        0.1,
        0.5,
        0.9,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float32,
)

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


def test_step_rewards_global_mse_improvement():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(
        target_image=target,
        image_size=16,
        max_steps=5,
        reward_scale=1000.0,
    )
    env.reset(seed=123)
    old_mse = float(env.current_mse)

    observation, reward, terminated, truncated, info = env.step(BLACK_TRIANGLE_ACTION)

    expected_improvement = np.float32(old_mse - float(env.current_mse))
    assert observation.shape == (11, 16, 16)
    assert env.canvas.mean() > 0.0
    assert info["triangle_area"] > 0
    assert info["mse_improvement"] == expected_improvement
    assert info["dense_reward"] == np.float32(1000.0 * expected_improvement)
    assert reward == info["dense_reward"]
    assert "terminal_reward" not in info
    assert "step_reward" not in info
    assert info["step"] == 1
    assert not terminated
    assert not truncated


def test_larger_triangle_improves_global_mse_more():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    small_env = TrianglePaintEnv(
        target_image=target,
        image_size=16,
        max_steps=5,
        reward_scale=1000.0,
    )
    large_env = TrianglePaintEnv(
        target_image=target,
        image_size=16,
        max_steps=5,
        reward_scale=1000.0,
    )
    small_env.reset(seed=123)
    large_env.reset(seed=123)

    _, small_reward, _, _, small_info = small_env.step(BLACK_TRIANGLE_ACTION)
    _, large_reward, _, _, large_info = large_env.step(LARGE_BLACK_TRIANGLE_ACTION)

    assert large_info["triangle_area"] > small_info["triangle_area"]
    assert large_info["mse_improvement"] > small_info["mse_improvement"]
    assert large_reward > small_reward


def test_worsening_action_returns_negative_reward():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(
        target_image=target,
        image_size=16,
        max_steps=5,
        reward_scale=1000.0,
    )
    env.reset(seed=123)
    env.step(BLACK_TRIANGLE_ACTION)
    old_mse = float(env.current_mse)

    _, reward, _, _, info = env.step(WORSE_WHITE_TRIANGLE_ACTION)

    assert float(env.current_mse) > old_mse
    assert info["mse_improvement"] < 0.0
    assert reward < 0.0
    assert reward == info["dense_reward"]


def test_episode_reward_telescopes_to_initial_minus_final_mse():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(
        target_image=target,
        image_size=16,
        max_steps=3,
        reward_scale=1000.0,
    )
    env.reset(seed=123)
    initial_mse = float(env.initial_mse)

    total_reward = 0.0
    for action in [
        BLACK_TRIANGLE_ACTION,
        LARGE_BLACK_TRIANGLE_ACTION,
        WORSE_WHITE_TRIANGLE_ACTION,
    ]:
        _, reward, _, _, info = env.step(action)
        total_reward += float(reward)
        assert "terminal_reward" not in info

    expected_total = 1000.0 * (initial_mse - float(env.current_mse))
    np.testing.assert_allclose(total_reward, expected_total, rtol=1e-5, atol=1e-5)


def test_last_step_has_no_terminal_bonus():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(
        target_image=target,
        image_size=16,
        max_steps=1,
        reward_scale=1000.0,
    )
    env.reset(seed=123)
    old_mse = float(env.current_mse)

    _, reward, terminated, truncated, info = env.step(BLACK_TRIANGLE_ACTION)

    expected = np.float32(1000.0 * (old_mse - float(env.current_mse)))
    assert not terminated
    assert truncated
    assert reward == expected
    assert info["dense_reward"] == expected
    assert "terminal_reward" not in info


def test_degenerate_triangle_does_not_crash():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=16, max_steps=5)
    env.reset(seed=123)

    _, reward, _, _, info = env.step(COLINEAR_TRIANGLE_ACTION)

    assert info["triangle_area"] >= 0
    assert np.isfinite(reward)
    assert "mse_improvement" in info
    assert "dense_reward" in info


def test_coincident_triangle_covers_at_most_one_pixel():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=16, max_steps=5)
    env.reset(seed=123)

    _, reward, _, _, info = env.step(DEGENERATE_TRIANGLE_ACTION)

    assert info["triangle_area"] <= 1
    assert np.isfinite(reward)


def test_reward_scale_scales_global_mse_improvement():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(
        target_image=target,
        image_size=16,
        max_steps=5,
        reward_scale=2.5,
    )
    env.reset(seed=123)
    old_mse = float(env.current_mse)

    _, reward, _, _, info = env.step(BLACK_TRIANGLE_ACTION)

    assert env.reward_scale == np.float32(2.5)
    assert info["mse_improvement"] == np.float32(old_mse - float(env.current_mse))
    assert reward == np.float32(2.5 * float(info["mse_improvement"]))


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
