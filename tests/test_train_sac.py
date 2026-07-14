from pathlib import Path

import numpy as np
import pytest
import yaml
from stable_baselines3.common.monitor import Monitor

from paint_rl.config import TrainConfig, load_train_config
from paint_rl.envs import TrianglePaintEnv
from paint_rl.models import PaintCNNFeaturesExtractor
from paint_rl.cli.train import build_model, parse_args, resolve_dimensions
from paint_rl.training import FixedTargetReplayBuffer


def test_build_model_uses_custom_cnn_features_extractor():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = Monitor(TrianglePaintEnv(target_image=target, image_size=16, max_steps=5))

    model = build_model(
        env,
        target_image=target,
        seed=0,
        total_timesteps=10,
        buffer_size=10,
    )

    assert isinstance(model.actor.features_extractor, PaintCNNFeaturesExtractor)


def test_build_model_uses_configured_buffer_size():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = Monitor(TrianglePaintEnv(target_image=target, image_size=16, max_steps=5))

    model = build_model(
        env,
        target_image=target,
        seed=0,
        total_timesteps=10,
        buffer_size=42,
    )

    assert model.buffer_size == 42


def test_build_model_respects_device():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = Monitor(TrianglePaintEnv(target_image=target, image_size=16, max_steps=5))

    model = build_model(
        env,
        target_image=target,
        seed=0,
        total_timesteps=10,
        buffer_size=10,
        device="cpu",
    )

    assert str(model.device) == "cpu"


def test_build_model_uses_compact_fixed_target_replay_buffer():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = Monitor(TrianglePaintEnv(target_image=target, image_size=16, max_steps=5))

    model = build_model(
        env,
        target_image=target,
        seed=0,
        total_timesteps=10,
        buffer_size=10,
        device="cpu",
    )

    assert isinstance(model.replay_buffer, FixedTargetReplayBuffer)
    assert model.replay_buffer.optimize_memory_usage is False
    np.testing.assert_array_equal(model.replay_buffer.target_image, target)


def test_compact_replay_buffer_supports_short_sac_training():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = Monitor(TrianglePaintEnv(target_image=target, image_size=16, max_steps=5))
    model = build_model(
        env,
        target_image=target,
        seed=0,
        total_timesteps=4,
        buffer_size=10,
        device="cpu",
    )

    model.learn(total_timesteps=4)
    sample = model.replay_buffer.sample(batch_size=2)

    assert model.num_timesteps == 4
    assert model.replay_buffer.size() == 4
    assert model._n_updates > 0
    assert sample.observations.shape == (2, 11, 16, 16)


def test_parse_args_requires_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "train.yaml"
    config_path.write_text("output_dir: outputs/sac\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["train_sac.py", "--config", str(config_path)],
    )

    args = parse_args()

    assert args.config == config_path


def test_parse_args_rejects_legacy_flags(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train_sac.py", "--total-timesteps", "1000", "--output-dir", "outputs/sac"],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_resolve_dimensions_from_train_config(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": "outputs/sac",
                "image_width": 96,
                "image_height": 64,
            }
        ),
        encoding="utf-8",
    )
    config = load_train_config(config_path)

    assert resolve_dimensions(config) == (96, 64)


def test_resolve_dimensions_falls_back_to_image_size(tmp_path: Path):
    config = TrainConfig(
        target=None,
        output_dir=tmp_path / "out",
        image_size=32,
        image_width=None,
        image_height=None,
        max_steps=10,
        total_timesteps=100,
        buffer_size=100,
        reward_scale=1.0,
        seed=0,
        snapshot_interval=0,
        check_env=False,
        device="cpu",
    )

    assert resolve_dimensions(config) == (32, 32)
