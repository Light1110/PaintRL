from pathlib import Path

import pytest
import yaml

from paint_rl.config import RandomDemoConfig, load_random_demo_config
from scripts.random_demo import parse_args, resolve_dimensions


def test_parse_args_requires_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("output: outputs/demo.png\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["random_demo.py", "--config", str(config_path)],
    )

    args = parse_args()

    assert args.config == config_path


def test_parse_args_rejects_legacy_flags(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "sys.argv",
        ["random_demo.py", "--steps", "10", "--output", "out.png"],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_resolve_dimensions_from_random_demo_config(tmp_path: Path):
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output": "out.png",
                "image_width": 48,
                "image_height": 24,
            }
        ),
        encoding="utf-8",
    )
    config = load_random_demo_config(config_path)

    assert resolve_dimensions(config) == (48, 24)


def test_resolve_dimensions_falls_back_to_image_size():
    config = RandomDemoConfig(
        target=None,
        output=Path("out.png"),
        image_size=16,
        image_width=None,
        image_height=None,
        steps=5,
        seed=0,
        triangle_size_min=0.02,
        triangle_size_max=1.0,
    )

    assert resolve_dimensions(config) == (16, 16)
