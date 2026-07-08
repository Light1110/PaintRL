from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from paint_rl.envs import TrianglePaintEnv
from paint_rl.utils.image import load_target_image, save_canvas
from scripts.random_demo import make_demo_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC to paint with triangles.")
    parser.add_argument("--target", type=Path, default=None, help="Path to target image.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sac"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = (
        load_target_image(args.target, args.image_size)
        if args.target
        else make_demo_target(args.image_size)
    )
    env = TrianglePaintEnv(
        target_image=target,
        image_size=args.image_size,
        max_steps=args.max_steps,
    )
    if args.check_env:
        check_env(env, warn=True)

    monitored_env = Monitor(env)
    model = SAC(
        "MlpPolicy",
        monitored_env,
        verbose=1,
        seed=args.seed,
        learning_starts=min(100, args.total_timesteps // 10),
        buffer_size=max(1_000, args.total_timesteps),
        batch_size=max(2, min(64, args.total_timesteps)),
    )
    model.learn(total_timesteps=args.total_timesteps)

    model_path = args.output_dir / "triangle_sac_model"
    model.save(model_path)

    observation, _ = env.reset(seed=args.seed)
    for _ in range(args.max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    save_canvas(env.canvas, args.output_dir / "final_canvas.png")
    save_canvas(target, args.output_dir / "target.png")
    print(f"Saved model to {model_path}.zip")
    print(f"Saved final canvas to {args.output_dir / 'final_canvas.png'}")


if __name__ == "__main__":
    main()
