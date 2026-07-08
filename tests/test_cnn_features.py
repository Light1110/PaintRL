import numpy as np
import torch

from paint_rl.envs import TrianglePaintEnv
from paint_rl.models import PaintCNNFeaturesExtractor


def test_paint_cnn_features_extractor_outputs_requested_feature_dim():
    target = np.zeros((16, 16, 3), dtype=np.float32)
    env = TrianglePaintEnv(target_image=target, image_size=16, max_steps=5)
    observation, _ = env.reset(seed=123)
    extractor = PaintCNNFeaturesExtractor(env.observation_space, features_dim=128)

    batch = torch.as_tensor(observation[None, ...])
    features = extractor(batch)

    assert features.shape == (1, 128)
