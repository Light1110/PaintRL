from pathlib import Path

import numpy as np
import pytest
import yaml
from stable_baselines3.common.monitor import Monitor

from paint_rl.config import TrainConfig, load_train_config
from paint_rl.envs import TrianglePaintEnv
from paint_rl.models import PaintCNNFeaturesExtractor
from paint_rl.cli.train import build_model, parse_args, resolve_dimensions


def test_build_model_uses_custom_cnn_features_extractor():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = Monitor(TrianglePaintEnv(target_image=target, image_size=16, max_steps=5))

    model = build_model(env, seed=0, total_timesteps=10, max_steps=5)

    assert isinstance(model.actor.features_extractor, PaintCNNFeaturesExtractor)


def test_build_model_scales_buffer_size_with_total_timesteps():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = Monitor(TrianglePaintEnv(target_image=target, image_size=16, max_steps=5))

    model = build_model(env, seed=0, total_timesteps=10, max_steps=5)

    assert model.buffer_size == 10


def test_build_model_caps_buffer_size_at_existing_max_steps_limit():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = Monitor(TrianglePaintEnv(target_image=target, image_size=16, max_steps=5))

    model = build_model(env, seed=0, total_timesteps=1_000, max_steps=5)

    assert model.buffer_size == 1000


def test_build_model_caps_buffer_size_with_long_horizon():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = Monitor(TrianglePaintEnv(target_image=target, image_size=16, max_steps=5))

    model = build_model(env, seed=0, total_timesteps=10_000, max_steps=5)

    assert model.buffer_size == 2500


def test_build_model_respects_device():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = Monitor(TrianglePaintEnv(target_image=target, image_size=16, max_steps=5))

    model = build_model(env, seed=0, total_timesteps=10, max_steps=5, device="cpu")

    assert str(model.device) == "cpu"


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
        reward_scale=1.0,
        seed=0,
        snapshot_interval=0,
        check_env=False,
        device="cpu",
    )

    assert resolve_dimensions(config) == (32, 32)
