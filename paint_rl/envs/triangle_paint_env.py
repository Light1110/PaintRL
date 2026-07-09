from __future__ import annotations

from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


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

        self.image_size = image_size
        self.image_width = image_width
        self.image_height = image_height
        self.max_steps = max_steps
        self.reward_scale = np.float32(reward_scale)
        self.alpha_min = np.float32(alpha_min)
        self.alpha_max = np.float32(alpha_max)
        self.success_mse = np.float32(success_mse)

        self.target = self._prepare_target(target_image)
        self.canvas = np.ones_like(self.target, dtype=np.float32)
        self.coordinate_channels = self._make_coordinate_channels()
        self.current_step = 0
        self.current_mse = self._mse()

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
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
        self.current_mse = self._mse()
        return self._observation(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.float32, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        old_mse = self.current_mse

        self._draw_triangle(action)
        self.current_step += 1
        self.current_mse = self._mse()

        reward = np.float32((old_mse - self.current_mse) * self.reward_scale)
        terminated = bool(self.current_mse <= self.success_mse)
        truncated = bool(self.current_step >= self.max_steps and not terminated)
        info = self._info()
        if terminated or truncated:
            info["terminal_canvas"] = np.copy(self.canvas)
        return self._observation(), reward, terminated, truncated, info

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

    def _draw_triangle(self, action: np.ndarray) -> None:
        clipped = np.clip(action, 0.0, 1.0)
        points = clipped[:6].reshape(3, 2)
        color = clipped[6:9].astype(np.float32)
        alpha_unit = clipped[9]
        alpha = self.alpha_min + alpha_unit * (self.alpha_max - self.alpha_min)

        x_points = np.rint(points[:, 0] * (self.image_width - 1))
        y_points = np.rint(points[:, 1] * (self.image_height - 1))
        pixel_points = np.stack([x_points, y_points], axis=1).astype(np.int32)
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        cv2.fillPoly(mask, [pixel_points], 1)
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            return

        self.canvas[mask_bool] = (
            (1.0 - alpha) * self.canvas[mask_bool] + alpha * color
        ).astype(np.float32)

    def _mse(self) -> np.float32:
        return np.float32(np.mean(np.square(self.canvas - self.target)))

    def _info(self) -> dict[str, Any]:
        return {
            "mse": self.current_mse,
            "step": self.current_step,
        }
