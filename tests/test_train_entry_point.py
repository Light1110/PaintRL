from pathlib import Path


def test_pyproject_declares_paint_train_entry_point():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert 'paint-train = "paint_rl.cli.train:main"' in text
