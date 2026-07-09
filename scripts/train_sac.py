from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paint_rl.envs import TrianglePaintEnv
from paint_rl.models import PaintCNNFeaturesExtractor
from paint_rl.utils.image import load_target_image, save_canvas
from scripts.random_demo import make_demo_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC to paint with triangles.")
    parser.add_argument("--target", type=Path, default=None, help="Path to target image.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sac"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device for SAC (e.g. auto, cpu, cuda, cuda:0, cuda:1).",
    )
    return parser.parse_args()


def build_model(
    env: Monitor,
    seed: int,
    total_timesteps: int,
    max_steps: int,
    device: str = "auto",
) -> SAC:
    buffer_size = min(total_timesteps, max_steps * 50)
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


def resolve_dimensions(args: argparse.Namespace) -> tuple[int, int]:
    image_width = args.image_width
    image_height = args.image_height
    if image_width is None and image_height is None:
        image_width = args.image_size
        image_height = args.image_size
    elif image_width is None:
        image_width = image_height
    elif image_height is None:
        image_height = image_width

    if image_width is None or image_height is None:
        raise ValueError("image dimensions must be resolved")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    return image_width, image_height


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_width, image_height = resolve_dimensions(args)

    target = (
        load_target_image(
            args.target,
            image_size=args.image_size,
            image_width=image_width,
            image_height=image_height,
        )
        if args.target
        else make_demo_target(image_width, image_height)
    )
    env = TrianglePaintEnv(
        target_image=target,
        image_size=args.image_size,
        image_width=image_width,
        image_height=image_height,
        max_steps=args.max_steps,
    )
    if args.check_env:
        check_env(env, warn=True)

    monitored_env = Monitor(env)
    model = build_model(
        monitored_env,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        max_steps=args.max_steps,
        device=args.device,
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