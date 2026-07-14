import numpy as np
from stable_baselines3.common.monitor import Monitor

from paint_rl.envs import TrianglePaintEnv
from paint_rl.models import PaintCNNFeaturesExtractor
from scripts.train_sac import build_model


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
