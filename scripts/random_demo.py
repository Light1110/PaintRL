from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from paint_rl.envs import TrianglePaintEnv
from paint_rl.utils.image import load_target_image, save_canvas


def make_demo_target(image_width: int, image_height: int) -> np.ndarray:
    y, x = np.mgrid[0:image_height, 0:image_width].astype(np.float32)
    x = x / max(image_width - 1, 1)
    y = y / max(image_height - 1, 1)
    return np.stack([x, y, 1.0 - x], axis=-1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random triangle painting demo.")
    parser.add_argument("--target", type=Path, default=None, help="Path to target image.")
    parser.add_argument("--output", type=Path, default=Path("outputs/random_demo.png"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


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
        max_steps=args.steps,
    )
    env.reset(seed=args.seed)

    for _ in range(args.steps):
        env.step(env.action_space.sample())

    save_canvas(env.canvas, args.output)
    print(f"Saved random demo to {args.output}")


if __name__ == "__main__":
    main()
