from pathlib import Path

import numpy as np
from PIL import Image
from stable_baselines3.common.monitor import Monitor

from paint_rl.envs import TrianglePaintEnv
from paint_rl.training import EpisodeCanvasSnapshotCallback, EpisodeTrainingLogCallback
from scripts.train_sac import build_model, run_deterministic_rollout


def test_episode_training_log_callback_writes_episode_metrics(tmp_path: Path):
    callback = EpisodeTrainingLogCallback(output_dir=tmp_path)

    callback.locals = {
        "dones": [False],
        "infos": [
            {
                "mse": np.float32(0.75),
                "step_reward": np.float32(0.5),
                "terminal_reward": np.float32(0.0),
                "mean_pixel_improvement": np.float32(0.25),
            }
        ],
        "rewards": [np.float32(0.5)],
    }
    assert callback._on_step()

    callback.locals = {
        "dones": [True],
        "infos": [
            {
                "mse": np.float32(0.25),
                "step_reward": np.float32(0.75),
                "terminal_reward": np.float32(-1.5),
                "mean_pixel_improvement": np.float32(0.125),
            }
        ],
        "rewards": [np.float32(-0.75)],
    }
    assert callback._on_step()

    log_text = (tmp_path / "episode_metrics.txt").read_text(encoding="utf-8")
    assert "episode=1" in log_text
    assert "steps=2" in log_text
    assert "episode_reward=-0.250000" in log_text
    assert "step_reward_total=1.250000" in log_text
    assert "terminal_reward=-1.500000" in log_text
    assert "final_mse=0.250000" in log_text
    assert "mean_pixel_improvement_total=0.375000" in log_text


def test_episode_training_log_callback_resets_accumulators_between_episodes(
    tmp_path: Path,
):
    callback = EpisodeTrainingLogCallback(output_dir=tmp_path)

    for mse, step_reward, terminal_reward, reward in [
        (0.5, 1.0, -0.25, 0.75),
        (0.25, 2.0, -0.5, 1.5),
    ]:
        callback.locals = {
            "dones": [True],
            "infos": [
                {
                    "mse": np.float32(mse),
                    "step_reward": np.float32(step_reward),
                    "terminal_reward": np.float32(terminal_reward),
                    "mean_pixel_improvement": np.float32(step_reward / 10.0),
                }
            ],
            "rewards": [np.float32(reward)],
        }
        assert callback._on_step()

    log_lines = (tmp_path / "episode_metrics.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(log_lines) == 2
    assert "episode=1" in log_lines[0]
    assert "step_reward_total=1.000000" in log_lines[0]
    assert "episode_reward=0.750000" in log_lines[0]
    assert "episode=2" in log_lines[1]
    assert "steps=1" in log_lines[1]
    assert "step_reward_total=2.000000" in log_lines[1]
    assert "episode_reward=1.500000" in log_lines[1]


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


def test_episode_canvas_snapshot_callback_prefers_terminal_canvas(tmp_path: Path):
    class DummyEnv:
        canvas = np.ones((4, 4, 3), dtype=np.float32)

    class DummyTrainingEnv:
        envs = [DummyEnv()]

    class DummyModel:
        def get_env(self) -> DummyTrainingEnv:
            return DummyTrainingEnv()

    terminal_canvas = np.zeros((4, 4, 3), dtype=np.float32)
    callback = EpisodeCanvasSnapshotCallback(
        output_dir=tmp_path,
        snapshot_interval=1,
        verbose=0,
    )
    callback.model = DummyModel()
    callback.locals = {
        "dones": [True],
        "infos": [{"terminal_canvas": terminal_canvas}],
    }

    assert callback._on_step()

    saved = np.asarray(
        Image.open(tmp_path / "snapshots" / "episode_00001.png"),
        dtype=np.uint8,
    )
    assert saved.mean() == 0.0


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
