from pathlib import Path

import pytest
import yaml

from paint_rl.config import (
    ConfigError,
    RandomDemoConfig,
    TrainConfig,
    load_random_demo_config,
    load_train_config,
)


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_load_train_config_uses_defaults_and_resolves_relative_paths(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = _write_yaml(
        config_dir / "train.yaml",
        {
            "target": "targets/mona.png",
            "output_dir": "../outputs/sac",
            "image_size": 64,
            "max_steps": 200,
            "total_timesteps": 10_000,
            "reward_scale": 1000.0,
            "seed": 0,
            "snapshot_interval": 10,
            "check_env": False,
            "device": "auto",
        },
    )

    config = load_train_config(config_path)

    assert isinstance(config, TrainConfig)
    assert config.target == (config_dir / "targets" / "mona.png").resolve()
    assert config.output_dir == (tmp_path / "outputs" / "sac").resolve()
    assert config.image_size == 64
    assert config.image_width is None
    assert config.image_height is None
    assert config.max_steps == 200
    assert config.total_timesteps == 10_000
    assert config.buffer_size == 10_000
    assert config.reward_scale == 1000.0
    assert config.seed == 0
    assert config.snapshot_interval == 10
    assert config.check_env is False
    assert config.device == "auto"
    assert config.triangle_size_min == 0.02
    assert config.triangle_size_max == 1.0


def test_load_train_config_reads_triangle_size_range(tmp_path: Path):
    config_path = _write_yaml(
        tmp_path / "train.yaml",
        {
            "output_dir": "outputs/sac",
            "triangle_size_min": 0.05,
            "triangle_size_max": 0.5,
        },
    )

    config = load_train_config(config_path)

    assert config.triangle_size_min == 0.05
    assert config.triangle_size_max == 0.5


def test_load_train_config_rejects_invalid_triangle_size_range(tmp_path: Path):
    config_path = _write_yaml(
        tmp_path / "train.yaml",
        {
            "output_dir": "outputs/sac",
            "triangle_size_min": 0.8,
            "triangle_size_max": 0.2,
        },
    )

    with pytest.raises(ConfigError, match="triangle_size"):
        load_train_config(config_path)


def test_load_train_config_allows_missing_optional_target(tmp_path: Path):
    config_path = _write_yaml(
        tmp_path / "train.yaml",
        {
            "output_dir": "outputs/sac",
            "image_width": 96,
            "image_height": 64,
        },
    )

    config = load_train_config(config_path)

    assert config.target is None
    assert config.image_width == 96
    assert config.image_height == 64
    assert config.output_dir == (tmp_path / "outputs" / "sac").resolve()


def test_load_random_demo_config_resolves_paths_relative_to_config_dir(tmp_path: Path):
    config_dir = tmp_path / "nested" / "configs"
    config_dir.mkdir(parents=True)
    config_path = _write_yaml(
        config_dir / "demo.yaml",
        {
            "target": "../images/target.png",
            "output": "../../outputs/random_demo.png",
            "image_size": 32,
            "steps": 50,
            "seed": 7,
        },
    )

    config = load_random_demo_config(config_path)

    assert isinstance(config, RandomDemoConfig)
    assert config.target == (tmp_path / "nested" / "images" / "target.png").resolve()
    assert config.output == (tmp_path / "outputs" / "random_demo.png").resolve()
    assert config.image_size == 32
    assert config.steps == 50
    assert config.seed == 7
    assert config.triangle_size_min == 0.02
    assert config.triangle_size_max == 1.0


def test_load_random_demo_config_reads_triangle_size_range(tmp_path: Path):
    config_path = _write_yaml(
        tmp_path / "demo.yaml",
        {
            "output": "out.png",
            "triangle_size_min": 0.1,
            "triangle_size_max": 0.4,
        },
    )

    config = load_random_demo_config(config_path)

    assert config.triangle_size_min == 0.1
    assert config.triangle_size_max == 0.4


def test_load_train_config_rejects_unknown_fields(tmp_path: Path):
    config_path = _write_yaml(
        tmp_path / "train.yaml",
        {"output_dir": "outputs/sac", "learning_rate": 0.001},
    )

    with pytest.raises(ValueError, match="Unknown config field"):
        load_train_config(config_path)


def test_load_random_demo_config_rejects_non_mapping(tmp_path: Path):
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("- just a list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_random_demo_config(config_path)


def test_load_train_config_rejects_invalid_types(tmp_path: Path):
    config_path = _write_yaml(
        tmp_path / "train.yaml",
        {"output_dir": "outputs/sac", "max_steps": "many"},
    )

    with pytest.raises(ValueError, match="max_steps"):
        load_train_config(config_path)


def test_load_train_config_rejects_non_positive_dimensions(tmp_path: Path):
    config_path = _write_yaml(
        tmp_path / "train.yaml",
        {"output_dir": "outputs/sac", "image_size": 0},
    )

    with pytest.raises(ValueError, match="image_size"):
        load_train_config(config_path)


def test_load_train_config_rejects_non_positive_buffer_size(tmp_path: Path):
    config_path = _write_yaml(
        tmp_path / "train.yaml",
        {"output_dir": "outputs/sac", "buffer_size": 0},
    )

    with pytest.raises(ValueError, match="buffer_size"):
        load_train_config(config_path)


def test_load_train_config_rejects_negative_snapshot_interval(tmp_path: Path):
    config_path = _write_yaml(
        tmp_path / "train.yaml",
        {"output_dir": "outputs/sac", "snapshot_interval": -1},
    )

    with pytest.raises(ValueError, match="snapshot_interval"):
        load_train_config(config_path)


def test_load_random_demo_config_rejects_non_positive_steps(tmp_path: Path):
    config_path = _write_yaml(
        tmp_path / "demo.yaml",
        {"output": "out.png", "steps": 0},
    )

    with pytest.raises(ValueError, match="steps"):
        load_random_demo_config(config_path)


def test_load_train_config_keeps_absolute_paths(tmp_path: Path):
    absolute_target = (tmp_path / "abs_target.png").resolve()
    absolute_output = (tmp_path / "abs_out").resolve()
    config_path = _write_yaml(
        tmp_path / "train.yaml",
        {
            "target": str(absolute_target),
            "output_dir": str(absolute_output),
        },
    )

    config = load_train_config(config_path)

    assert config.target == absolute_target
    assert config.output_dir == absolute_output
