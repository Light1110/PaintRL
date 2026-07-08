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
