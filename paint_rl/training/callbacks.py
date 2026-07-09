from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from paint_rl.utils.image import save_canvas


def unwrap_env(env: gym.Env) -> gym.Env:
    while hasattr(env, "env"):
        env = env.env
    return env


class EpisodeCanvasSnapshotCallback(BaseCallback):
    """Save the final canvas when selected training episodes finish."""

    def __init__(
        self,
        output_dir: Path,
        snapshot_interval: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if snapshot_interval < 0:
            raise ValueError("snapshot_interval must be non-negative")

        self.output_dir = Path(output_dir)
        self.snapshot_interval = snapshot_interval
        self.snapshot_dir = self.output_dir / "snapshots"
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.snapshot_interval == 0:
            return True

        dones = self.locals.get("dones")
        if dones is None or not dones[0]:
            return True

        self.episode_count += 1
        if self.episode_count % self.snapshot_interval != 0:
            return True

        env = unwrap_env(self.training_env.envs[0])
        canvas = np.copy(env.canvas)
        snapshot_path = self.snapshot_dir / f"episode_{self.episode_count:05d}.png"
        save_canvas(canvas, snapshot_path)

        if self.verbose > 0:
            print(f"Saved episode snapshot to {snapshot_path}")

        return True
