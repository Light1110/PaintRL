from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paint_rl.config import TrainConfig, load_train_config, resolve_image_dimensions
from paint_rl.envs import TrianglePaintEnv
from paint_rl.models import PaintCNNFeaturesExtractor
from paint_rl.training import EpisodeCanvasSnapshotCallback, EpisodeTrainingLogCallback
from paint_rl.utils.actions import decode_triangle_action
from paint_rl.utils.image import load_target_image, save_canvas
from scripts.random_demo import make_demo_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC to paint with triangles.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML training config file.",
    )
    return parser.parse_args()


def build_model(
    env: Monitor,
    seed: int,
    total_timesteps: int,
    max_steps: int,
    device: str = "auto",
) -> SAC:
    buffer_size = min(total_timesteps, max_steps * 500)
    policy_kwargs = {
        "features_extractor_class": PaintCNNFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    return SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        learning_starts=min(100, total_timesteps // 10),
        buffer_size=buffer_size,
        batch_size=max(2, min(64, total_timesteps)),
        device=device,
    )


def resolve_dimensions(config: TrainConfig) -> tuple[int, int]:
    return resolve_image_dimensions(
        image_size=config.image_size,
        image_width=config.image_width,
        image_height=config.image_height,
    )


def run_deterministic_rollout(
    env: TrianglePaintEnv,
    model: SAC,
    *,
    max_steps: int,
    seed: int,
) -> dict[str, object]:
    observation, _ = env.reset(seed=seed)
    steps: list[dict[str, object]] = []

    for _ in range(max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        decoded = decode_triangle_action(
            action,
            alpha_min=float(env.alpha_min),
            alpha_max=float(env.alpha_max),
        )
        steps.append(
            {
                "step": int(info["step"]),
                **decoded,
                "mse": float(info["mse"]),
            }
        )
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "max_steps": max_steps,
        "steps": steps,
    }


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    image_width, image_height = resolve_dimensions(config)

    target = (
        load_target_image(
            config.target,
            image_size=config.image_size,
            image_width=image_width,
            image_height=image_height,
        )
        if config.target
        else make_demo_target(image_width, image_height)
    )
    env = TrianglePaintEnv(
        target_image=target,
        image_size=config.image_size,
        image_width=image_width,
        image_height=image_height,
        max_steps=config.max_steps,
        reward_scale=config.reward_scale,
    )
    if config.check_env:
        check_env(env, warn=True)

    target_path = config.output_dir / "target.png"
    save_canvas(target, target_path)

    monitored_env = Monitor(env)
    model = build_model(
        monitored_env,
        seed=config.seed,
        total_timesteps=config.total_timesteps,
        max_steps=config.max_steps,
        device=config.device,
    )
    snapshot_callback = EpisodeCanvasSnapshotCallback(
        output_dir=config.output_dir,
        snapshot_interval=config.snapshot_interval,
        verbose=1,
    )
    log_callback = EpisodeTrainingLogCallback(output_dir=config.output_dir)
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[snapshot_callback, log_callback],
    )

    model_path = config.output_dir / "triangle_sac_model"
    model.save(model_path)

    rollout = run_deterministic_rollout(
        env,
        model,
        max_steps=config.max_steps,
        seed=config.seed,
    )
    save_canvas(env.canvas, config.output_dir / "final_canvas.png")

    rollout_path = config.output_dir / "final_rollout.json"
    with rollout_path.open("w", encoding="utf-8") as rollout_file:
        json.dump(rollout, rollout_file, indent=2)

    print(f"Saved model to {model_path}.zip")
    print(f"Saved target to {target_path}")
    print(f"Saved final canvas to {config.output_dir / 'final_canvas.png'}")
    print(f"Saved rollout actions to {rollout_path}")
    print(f"Saved episode metrics to {log_callback.log_path}")


if __name__ == "__main__":
    main()
