from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class FixedTargetReplayBuffer(ReplayBuffer):
    """Store quantized canvases and reconstruct fixed-target observations."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        *,
        target_image: np.ndarray,
    ) -> None:
        if optimize_memory_usage:
            raise ValueError(
                "FixedTargetReplayBuffer does not support optimize_memory_usage=True"
            )
        if not handle_timeout_termination:
            raise ValueError(
                "FixedTargetReplayBuffer requires handle_timeout_termination=True"
            )
        if not isinstance(observation_space, spaces.Box):
            raise ValueError("observation_space must be a Box with shape (11, H, W)")
        if len(observation_space.shape) != 3 or observation_space.shape[0] != 11:
            raise ValueError("observation_space must have shape (11, H, W)")

        _, height, width = observation_space.shape
        target = np.asarray(target_image, dtype=np.float32)
        if target.ndim != 3 or target.shape[2] != 3:
            raise ValueError("target_image must have shape H x W x 3")
        if target.shape[:2] != (height, width):
            raise ValueError("target_image dimensions must match observation_space")
        if target.max(initial=0.0) > 1.0:
            target = target / 255.0

        self.full_observation_space = observation_space
        self.target_image = np.clip(target, 0.0, 1.0).astype(np.float32)
        self._target_chw = self.target_image.transpose(2, 0, 1)
        x_axis = np.linspace(0.0, 1.0, width, dtype=np.float32)
        y_axis = np.linspace(0.0, 1.0, height, dtype=np.float32)
        self._coordinate_channels = np.stack(
            [
                np.tile(x_axis, (height, 1)),
                np.tile(y_axis[:, None], (1, width)),
            ]
        )

        compact_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, height, width),
            dtype=np.uint8,
        )
        super().__init__(
            buffer_size=buffer_size,
            observation_space=compact_observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=handle_timeout_termination,
        )

    @staticmethod
    def _quantize_canvas(observation: np.ndarray) -> np.ndarray:
        canvas = np.asarray(observation)[..., :3, :, :]
        return np.rint(np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _validate_full_observation(
        self,
        observation: np.ndarray,
        field: str,
    ) -> np.ndarray:
        observation = np.asarray(observation)
        expected_shape = (self.n_envs, *self.full_observation_space.shape)
        if observation.shape != expected_shape:
            raise ValueError(
                f"{field} must have shape {expected_shape}, got {observation.shape}"
            )
        return observation

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        obs = self._validate_full_observation(obs, "obs")
        next_obs = self._validate_full_observation(next_obs, "next_obs")
        super().add(
            self._quantize_canvas(obs),
            self._quantize_canvas(next_obs),
            action,
            reward,
            done,
            infos,
        )

    def _reconstruct_observation(self, canvases: np.ndarray) -> np.ndarray:
        canvas = canvases.astype(np.float32) / np.float32(255.0)
        batch_size = canvas.shape[0]
        target = np.broadcast_to(self._target_chw, (batch_size, *self._target_chw.shape))
        coordinates = np.broadcast_to(
            self._coordinate_channels,
            (batch_size, *self._coordinate_channels.shape),
        )
        return np.concatenate(
            [canvas, target, np.abs(canvas - target), coordinates],
            axis=1,
        ).astype(np.float32)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> ReplayBufferSamples:
        env_indices = np.random.randint(
            0,
            high=self.n_envs,
            size=(len(batch_inds),),
        )
        observations = self._normalize_obs(
            self._reconstruct_observation(
                self.observations[batch_inds, env_indices, :]
            ),
            env,
        )
        next_observations = self._normalize_obs(
            self._reconstruct_observation(
                self.next_observations[batch_inds, env_indices, :]
            ),
            env,
        )
        data = (
            observations,
            self.actions[batch_inds, env_indices, :],
            next_observations,
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1),
                env,
            ),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
