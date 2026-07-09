from pathlib import Path

import numpy as np
from stable_baselines3.common.monitor import Monitor

from paint_rl.envs import TrianglePaintEnv
from paint_rl.training import EpisodeCanvasSnapshotCallback
from scripts.train_sac import build_model, run_deterministic_rollout


def test_episode_canvas_snapshot_callback_saves_on_interval(tmp_path: Path):
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = Monitor(TrianglePaintEnv(target_image=target, image_size=16, max_steps=3))
    model = build_model(env, seed=0, total_timesteps=12, max_steps=3, device="cpu")
    callback = EpisodeCanvasSnapshotCallback(
        output_dir=tmp_path,
        snapshot_interval=2,
        verbose=0,
    )

    model.learn(total_timesteps=12, callback=callback)

    assert (tmp_path / "snapshots" / "episode_00002.png").exists()
    assert (tmp_path / "snapshots" / "episode_00004.png").exists()
    assert not (tmp_path / "snapshots" / "episode_00001.png").exists()


def test_episode_canvas_snapshot_callback_disabled_when_interval_zero(tmp_path: Path):
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = Monitor(TrianglePaintEnv(target_image=target, image_size=16, max_steps=3))
    model = build_model(env, seed=0, total_timesteps=12, max_steps=3, device="cpu")
    callback = EpisodeCanvasSnapshotCallback(
        output_dir=tmp_path,
        snapshot_interval=0,
        verbose=0,
    )

    model.learn(total_timesteps=12, callback=callback)

    assert not (tmp_path / "snapshots").exists()


def test_run_deterministic_rollout_records_triangle_parameters():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=16, max_steps=3)
    model = build_model(
        Monitor(env),
        seed=0,
        total_timesteps=10,
        max_steps=3,
        device="cpu",
    )

    rollout = run_deterministic_rollout(env, model, max_steps=3, seed=0)

    assert rollout["seed"] == 0
    assert rollout["max_steps"] == 3
    assert len(rollout["steps"]) == 3

    first_step = rollout["steps"][0]
    assert first_step["step"] == 1
    assert len(first_step["action"]) == 10
    assert len(first_step["vertices"]) == 3
    assert len(first_step["color"]) == 3
    assert isinstance(first_step["alpha"], float)
    assert isinstance(first_step["mse"], float)
