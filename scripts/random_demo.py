from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from paint_rl.envs import TrianglePaintEnv
from paint_rl.utils.image import load_target_image, save_canvas


def make_demo_target(image_size: int) -> np.ndarray:
    y, x = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    x = x / max(image_size - 1, 1)
    y = y / max(image_size - 1, 1)
    return np.stack([x, y, 1.0 - x], axis=-1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random triangle painting demo.")
    parser.add_argument("--target", type=Path, default=None, help="Path to target image.")
    parser.add_argument("--output", type=Path, default=Path("outputs/random_demo.png"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = (
        load_target_image(args.target, args.image_size)
        if args.target
        else make_demo_target(args.image_size)
    )
    env = TrianglePaintEnv(target_image=target, image_size=args.image_size, max_steps=args.steps)
    env.reset(seed=args.seed)

    for _ in range(args.steps):
        env.step(env.action_space.sample())

    save_canvas(env.canvas, args.output)
    print(f"Saved random demo to {args.output}")


if __name__ == "__main__":
    main()
