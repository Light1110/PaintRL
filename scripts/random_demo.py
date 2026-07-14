from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from paint_rl.config import RandomDemoConfig, load_random_demo_config, resolve_image_dimensions
from paint_rl.envs import TrianglePaintEnv
from paint_rl.utils.image import load_target_image, save_canvas


def make_demo_target(image_width: int, image_height: int) -> np.ndarray:
    y, x = np.mgrid[0:image_height, 0:image_width].astype(np.float32)
    x = x / max(image_width - 1, 1)
    y = y / max(image_height - 1, 1)
    return np.stack([x, y, 1.0 - x], axis=-1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random triangle painting demo.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML random-demo config file.",
    )
    return parser.parse_args()


def resolve_dimensions(config: RandomDemoConfig) -> tuple[int, int]:
    return resolve_image_dimensions(
        image_size=config.image_size,
        image_width=config.image_width,
        image_height=config.image_height,
    )


def main() -> None:
    args = parse_args()
    config = load_random_demo_config(args.config)
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
        max_steps=config.steps,
    )
    env.reset(seed=config.seed)

    for _ in range(config.steps):
        env.step(env.action_space.sample())

    config.output.parent.mkdir(parents=True, exist_ok=True)
    save_canvas(env.canvas, config.output)
    print(f"Saved random demo to {config.output}")


if __name__ == "__main__":
    main()
