from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a YAML config file is invalid."""


@dataclass(frozen=True)
class TrainConfig:
    target: Path | None
    output_dir: Path
    image_size: int
    image_width: int | None
    image_height: int | None
    max_steps: int
    total_timesteps: int
    reward_scale: float
    seed: int
    snapshot_interval: int
    check_env: bool
    device: str


@dataclass(frozen=True)
class RandomDemoConfig:
    target: Path | None
    output: Path
    image_size: int
    image_width: int | None
    image_height: int | None
    steps: int
    seed: int


_TRAIN_DEFAULTS: dict[str, Any] = {
    "target": None,
    "output_dir": Path("outputs/sac"),
    "image_size": 64,
    "image_width": None,
    "image_height": None,
    "max_steps": 200,
    "total_timesteps": 10_000,
    "reward_scale": 1000.0,
    "seed": 0,
    "snapshot_interval": 10,
    "check_env": False,
    "device": "auto",
}

_RANDOM_DEMO_DEFAULTS: dict[str, Any] = {
    "target": None,
    "output": Path("outputs/random_demo.png"),
    "image_size": 64,
    "image_width": None,
    "image_height": None,
    "steps": 200,
    "seed": 0,
}


def load_train_config(config_path: Path | str) -> TrainConfig:
    path = Path(config_path)
    raw = _load_mapping(path)
    values = _merge_known_fields(raw, set(_TRAIN_DEFAULTS), _TRAIN_DEFAULTS)
    base_dir = path.resolve().parent

    target = _optional_path(values["target"], base_dir, field_name="target")
    output_dir = _required_path(values["output_dir"], base_dir, field_name="output_dir")
    image_size = _positive_int(values["image_size"], field_name="image_size")
    image_width = _optional_positive_int(values["image_width"], field_name="image_width")
    image_height = _optional_positive_int(
        values["image_height"], field_name="image_height"
    )
    max_steps = _positive_int(values["max_steps"], field_name="max_steps")
    total_timesteps = _positive_int(
        values["total_timesteps"], field_name="total_timesteps"
    )
    reward_scale = _finite_float(values["reward_scale"], field_name="reward_scale")
    seed = _int(values["seed"], field_name="seed")
    snapshot_interval = _non_negative_int(
        values["snapshot_interval"], field_name="snapshot_interval"
    )
    check_env = _bool(values["check_env"], field_name="check_env")
    device = _str(values["device"], field_name="device")

    return TrainConfig(
        target=target,
        output_dir=output_dir,
        image_size=image_size,
        image_width=image_width,
        image_height=image_height,
        max_steps=max_steps,
        total_timesteps=total_timesteps,
        reward_scale=reward_scale,
        seed=seed,
        snapshot_interval=snapshot_interval,
        check_env=check_env,
        device=device,
    )


def load_random_demo_config(config_path: Path | str) -> RandomDemoConfig:
    path = Path(config_path)
    raw = _load_mapping(path)
    values = _merge_known_fields(raw, set(_RANDOM_DEMO_DEFAULTS), _RANDOM_DEMO_DEFAULTS)
    base_dir = path.resolve().parent

    target = _optional_path(values["target"], base_dir, field_name="target")
    output = _required_path(values["output"], base_dir, field_name="output")
    image_size = _positive_int(values["image_size"], field_name="image_size")
    image_width = _optional_positive_int(values["image_width"], field_name="image_width")
    image_height = _optional_positive_int(
        values["image_height"], field_name="image_height"
    )
    steps = _positive_int(values["steps"], field_name="steps")
    seed = _int(values["seed"], field_name="seed")

    return RandomDemoConfig(
        target=target,
        output=output,
        image_size=image_size,
        image_width=image_width,
        image_height=image_height,
        steps=steps,
        seed=seed,
    )


def resolve_image_dimensions(
    *,
    image_size: int,
    image_width: int | None,
    image_height: int | None,
) -> tuple[int, int]:
    width = image_width
    height = image_height
    if width is None and height is None:
        width = image_size
        height = image_size
    elif width is None:
        width = height
    elif height is None:
        height = width

    if width is None or height is None:
        raise ConfigError("image dimensions must be resolved")
    if width <= 0 or height <= 0:
        raise ConfigError("image_width and image_height must be positive")
    return width, height


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError("Config root must be a mapping")
    return payload


def _merge_known_fields(
    raw: dict[str, Any],
    allowed: set[str],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown config field: {', '.join(unknown)}")
    merged = dict(defaults)
    merged.update(raw)
    return merged


def _resolve_path(value: Path | str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _optional_path(value: Any, base_dir: Path, *, field_name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"{field_name} must be a path string or null")
    return _resolve_path(value, base_dir)


def _required_path(value: Any, base_dir: Path, *, field_name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"{field_name} must be a path string")
    return _resolve_path(value, base_dir)


def _int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field_name} must be an integer")
    return value


def _positive_int(value: Any, *, field_name: str) -> int:
    number = _int(value, field_name=field_name)
    if number <= 0:
        raise ConfigError(f"{field_name} must be positive")
    return number


def _non_negative_int(value: Any, *, field_name: str) -> int:
    number = _int(value, field_name=field_name)
    if number < 0:
        raise ConfigError(f"{field_name} must be non-negative")
    return number


def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name=field_name)


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} must be a number")
    number = float(value)
    if number != number:  # NaN
        raise ConfigError(f"{field_name} must be a finite number")
    return number


def _bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a boolean")
    return value


def _str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be a string")
    if not value:
        raise ConfigError(f"{field_name} must be a non-empty string")
    return value
