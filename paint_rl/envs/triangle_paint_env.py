from __future__ import annotations

from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from paint_rl.utils.actions import (
    ACTION_DIM,
    DEFAULT_SIZE_MAX,
    DEFAULT_SIZE_MIN,
    decode_triangle_action,
)


class TrianglePaintEnv(gym.Env):
    """Paint a target image by adding one transparent triangle per action."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        target_image: np.ndarray,
        image_size: int = 64,
        image_width: int | None = None,
        image_height: int | None = None,
        max_steps: int = 200,
        reward_scale: float = 1000.0,
        alpha_min: float = 0.05,
        alpha_max: float = 0.8,
        triangle_size_min: float = DEFAULT_SIZE_MIN,
        triangle_size_max: float = DEFAULT_SIZE_MAX,
        success_mse: float = 1e-4,
    ) -> None:
        super().__init__()
        if image_width is None and image_height is None:
            if image_size <= 0:
                raise ValueError("image_size must be positive")
            image_width = image_size
            image_height = image_size
        elif image_width is None:
            image_width = image_height
        elif image_height is None:
            image_height = image_width

        if image_width is None or image_height is None:
            raise ValueError("image_width and image_height must be resolved")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError("alpha range must satisfy 0 <= alpha_min <= alpha_max <= 1")
        if not 0.0 < triangle_size_min <= triangle_size_max <= 1.0:
            raise ValueError(
                "triangle size range must satisfy 0 < size_min <= size_max <= 1"
            )

        self.image_size = image_size
        self.image_width = image_width
        self.image_height = image_height
        self.max_steps = max_steps
        self.reward_scale = np.float32(reward_scale)
        self.alpha_min = np.float32(alpha_min)
        self.alpha_max = np.float32(alpha_max)
        self.triangle_size_min = float(triangle_size_min)
        self.triangle_size_max = float(triangle_size_max)
        self.success_mse = np.float32(success_mse)

        self.target = self._prepare_target(target_image)
        self.canvas = np.ones_like(self.target, dtype=np.float32)
        self.coordinate_channels = self._make_coordinate_channels()
        self.current_step = 0
        self.initial_mse = self._mse()
        self.current_mse = self.initial_mse

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(11, self.image_height, self.image_width),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        background = options.get("background", 1.0) if options else 1.0
        self.canvas = np.full_like(self.target, np.float32(background), dtype=np.float32)
        self.current_step = 0
        self.initial_mse = self._mse()
        self.current_mse = self.initial_mse
        return self._observation(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.float32, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        old_mse = self.current_mse

        mask_bool = self._draw_triangle(action)
        self.current_step += 1
        self.current_mse = self._mse()

        triangle_area = int(mask_bool.sum())
        mse_improvement = np.float32(old_mse - self.current_mse)
        dense_reward = np.float32(self.reward_scale * mse_improvement)

        terminated = bool(self.current_mse <= self.success_mse)
        truncated = bool(self.current_step >= self.max_steps and not terminated)

        info = self._info(
            triangle_area=triangle_area,
            mse_improvement=mse_improvement,
            dense_reward=dense_reward,
        )
        if terminated or truncated:
            info["terminal_canvas"] = np.copy(self.canvas)
        return self._observation(), dense_reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return np.clip(self.canvas * 255.0, 0, 255).astype(np.uint8)

    def _prepare_target(self, target_image: np.ndarray) -> np.ndarray:
        target = np.asarray(target_image, dtype=np.float32)
        if target.ndim != 3 or target.shape[2] != 3:
            raise ValueError("target_image must have shape H x W x 3")
        if target.max(initial=0.0) > 1.0:
            target = target / 255.0
        if target.shape[:2] != (self.image_height, self.image_width):
            target = cv2.resize(
                target,
                (self.image_width, self.image_height),
                interpolation=cv2.INTER_AREA,
            )
        return np.clip(target, 0.0, 1.0).astype(np.float32)

    def _observation(self) -> np.ndarray:
        diff = np.abs(self.canvas - self.target)
        return np.concatenate(
            [
                self.canvas.transpose(2, 0, 1),
                self.target.transpose(2, 0, 1),
                diff.transpose(2, 0, 1),
                self.coordinate_channels,
            ],
            axis=0,
        ).astype(np.float32)

    def _make_coordinate_channels(self) -> np.ndarray:
        x_axis = np.linspace(0.0, 1.0, self.image_width, dtype=np.float32)
        y_axis = np.linspace(0.0, 1.0, self.image_height, dtype=np.float32)
        x_grid = np.tile(x_axis, (self.image_height, 1))
        y_grid = np.tile(y_axis[:, None], (1, self.image_width))
        return np.stack([x_grid, y_grid], axis=0).astype(np.float32)

    def _draw_triangle(self, action: np.ndarray) -> np.ndarray:
        decoded = decode_triangle_action(
            action,
            alpha_min=float(self.alpha_min),
            alpha_max=float(self.alpha_max),
            size_min=self.triangle_size_min,
            size_max=self.triangle_size_max,
        )
        points = np.asarray(decoded["vertices"], dtype=np.float32)
        color = np.asarray(decoded["color"], dtype=np.float32)
        alpha = np.float32(decoded["alpha"])

        # Vertices may leave [0, 1]; OpenCV clips the filled polygon to the mask.
        x_points = np.rint(points[:, 0] * (self.image_width - 1))
        y_points = np.rint(points[:, 1] * (self.image_height - 1))
        pixel_points = np.stack([x_points, y_points], axis=1).astype(np.int32)
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        cv2.fillPoly(mask, [pixel_points], 1)
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            return mask_bool

        self.canvas[mask_bool] = (
            (1.0 - alpha) * self.canvas[mask_bool] + alpha * color
        ).astype(np.float32)
        return mask_bool

    def _mse(self) -> np.float32:
        return np.float32(np.mean(np.square(self.canvas - self.target)))

    def _info(
        self,
        *,
        triangle_area: int | None = None,
        mse_improvement: np.float32 | None = None,
        dense_reward: np.float32 | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "mse": self.current_mse,
            "step": self.current_step,
        }
        if triangle_area is not None:
            info["triangle_area"] = triangle_area
        if mse_improvement is not None:
            info["mse_improvement"] = mse_improvement
        if dense_reward is not None:
            info["dense_reward"] = dense_reward
        return info
