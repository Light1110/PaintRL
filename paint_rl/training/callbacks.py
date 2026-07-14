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

        infos = self.locals.get("infos")
        terminal_canvas = (
            infos[0].get("terminal_canvas")
            if infos is not None and len(infos) > 0
            else None
        )
        if terminal_canvas is None:
            env = unwrap_env(self.training_env.envs[0])
            canvas = np.copy(env.canvas)
        else:
            canvas = np.copy(terminal_canvas)
        snapshot_path = self.snapshot_dir / f"episode_{self.episode_count:05d}.png"
        save_canvas(canvas, snapshot_path)

        if self.verbose > 0:
            print(f"Saved episode snapshot to {snapshot_path}")

        return True


class EpisodeTrainingLogCallback(BaseCallback):
    """Append text metrics when training episodes finish."""

    def __init__(
        self,
        output_dir: Path,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.output_dir = Path(output_dir)
        self.log_path = self.output_dir / "episode_metrics.txt"
        self.episode_count = 0
        self._reset_episode_metrics()

    def _reset_episode_metrics(self) -> None:
        self.current_episode_steps = 0
        self.current_episode_reward = 0.0
        self.current_dense_reward_total = 0.0
        self.current_mse_improvement_total = 0.0
        self.current_final_mse = 0.0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        rewards = self.locals.get("rewards")

        if infos is None or len(infos) == 0:
            return True

        info = infos[0]
        reward = float(rewards[0]) if rewards is not None and len(rewards) > 0 else 0.0
        done = bool(dones[0]) if dones is not None and len(dones) > 0 else False

        self.current_episode_steps += 1
        self.current_episode_reward += reward
        self.current_dense_reward_total += float(info.get("dense_reward", 0.0))
        self.current_mse_improvement_total += float(info.get("mse_improvement", 0.0))
        self.current_final_mse = float(info.get("mse", self.current_final_mse))

        if not done:
            return True

        self.episode_count += 1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_line = (
            f"episode={self.episode_count} "
            f"steps={self.current_episode_steps} "
            f"episode_reward={self.current_episode_reward:.6f} "
            f"dense_reward_total={self.current_dense_reward_total:.6f} "
            f"mse_improvement_total={self.current_mse_improvement_total:.6f} "
            f"final_mse={self.current_final_mse:.6f}"
        )
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{log_line}\n")

        if self.verbose > 0:
            print(f"Saved episode metrics to {self.log_path}")

        self._reset_episode_metrics()
        return True
