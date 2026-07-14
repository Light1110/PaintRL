# Installable `paint-train` CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After `pip install -e .`, users can run `paint-train --config <yaml>` via `paint_rl.cli.train:main`, with no `scripts/train_sac.py` compatibility shim.

**Architecture:** Extract shared `make_demo_target` into `paint_rl.utils.demo`. Move the full training CLI into `paint_rl.cli.train`. Register a setuptools console script in `pyproject.toml`. Delete `scripts/train_sac.py` and point tests/README at the new module.

**Tech Stack:** Python 3.10+, setuptools, pytest, existing Gymnasium / Stable-Baselines3 training code

**Spec:** `docs/superpowers/specs/2026-07-14-paint-train-cli-design.md`

---

## File structure

| Path | Responsibility |
|------|----------------|
| `paint_rl/utils/demo.py` | Shared `make_demo_target` helper |
| `paint_rl/cli/__init__.py` | CLI package marker |
| `paint_rl/cli/train.py` | Training argparse + `main` and helpers |
| `scripts/random_demo.py` | Import `make_demo_target` from package |
| `scripts/train_sac.py` | **Delete** |
| `pyproject.toml` | `[build-system]`, packages, `paint-train` entry point |
| `tests/test_demo_utils.py` | Unit tests for `make_demo_target` |
| `tests/test_train_entry_point.py` | Assert console-script mapping in `pyproject.toml` |
| `tests/test_train_sac.py` | Import helpers from `paint_rl.cli.train` |
| `tests/test_training_callbacks.py` | Import helpers from `paint_rl.cli.train` |
| `README.md` | Document `paint-train` |

---

### Task 1: Extract `make_demo_target` into `paint_rl.utils.demo`

**Files:**
- Create: `paint_rl/utils/demo.py`
- Create: `tests/test_demo_utils.py`
- Modify: `scripts/random_demo.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_demo_utils.py`:

```python
import numpy as np

from paint_rl.utils.demo import make_demo_target


def test_make_demo_target_shape_and_dtype():
    target = make_demo_target(8, 4)

    assert target.shape == (4, 8, 3)
    assert target.dtype == np.float32
    assert float(target.min()) >= 0.0
    assert float(target.max()) <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_demo_utils.py::test_make_demo_target_shape_and_dtype -v
```

Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `paint_rl.utils.demo`

- [ ] **Step 3: Implement `paint_rl/utils/demo.py`**

```python
from __future__ import annotations

import numpy as np


def make_demo_target(image_width: int, image_height: int) -> np.ndarray:
    y, x = np.mgrid[0:image_height, 0:image_width].astype(np.float32)
    x = x / max(image_width - 1, 1)
    y = y / max(image_height - 1, 1)
    return np.stack([x, y, 1.0 - x], axis=-1).astype(np.float32)
```

- [ ] **Step 4: Update `scripts/random_demo.py` to import the helper**

Remove the local `make_demo_target` function body. Near the other `paint_rl` imports, add:

```python
from paint_rl.utils.demo import make_demo_target
```

Also remove the now-unused `import numpy as np` from `scripts/random_demo.py` if nothing else in that file needs NumPy.

Keep `parse_args` / `resolve_dimensions` / `main` unchanged except for the import.

- [ ] **Step 5: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_demo_utils.py tests/test_random_demo.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```powershell
git add paint_rl/utils/demo.py tests/test_demo_utils.py scripts/random_demo.py
git commit -m "Extract make_demo_target into paint_rl.utils.demo."
```

---

### Task 2: Register `paint-train` console script in packaging

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_train_entry_point.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_entry_point.py` (string check stays 3.10-safe without `tomllib`):

```python
from pathlib import Path


def test_pyproject_declares_paint_train_entry_point():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert 'paint-train = "paint_rl.cli.train:main"' in text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_train_entry_point.py::test_pyproject_declares_paint_train_entry_point -v
```

Expected: FAIL (assertion / missing script line)

- [ ] **Step 3: Update `pyproject.toml`**

Replace / extend `pyproject.toml` so it includes build system, package discovery, and the console script. Full intended file content:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "paint-rl"
version = "0.1.0"
description = "A minimal reinforcement learning painter that draws a target image with transparent triangles."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "gymnasium>=1.0.0",
    "stable-baselines3>=2.4.0",
    "torch>=2.0.0",
    "numpy>=1.26.0",
    "pillow>=10.0.0",
    "opencv-python-headless>=4.9.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]

[project.scripts]
paint-train = "paint_rl.cli.train:main"

[tool.setuptools.packages.find]
include = ["paint_rl*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Do **not** include `scripts` as an installable package in `include` — only `paint_rl*`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m pytest tests/test_train_entry_point.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml tests/test_train_entry_point.py
git commit -m "Register paint-train console script in packaging."
```

---

### Task 3: Add `paint_rl.cli.train` and migrate existing train tests

**Files:**
- Create: `paint_rl/cli/__init__.py`
- Create: `paint_rl/cli/train.py`
- Modify: `tests/test_train_sac.py`
- Modify: `tests/test_training_callbacks.py`
- Delete: `scripts/train_sac.py`

- [ ] **Step 1: Point train tests at the new module (expect import failure)**

In `tests/test_train_sac.py`, change:

```python
from scripts.train_sac import build_model, parse_args, resolve_dimensions
```

to:

```python
from paint_rl.cli.train import build_model, parse_args, resolve_dimensions
```

In `tests/test_training_callbacks.py`, change:

```python
from scripts.train_sac import build_model, run_deterministic_rollout
```

to:

```python
from paint_rl.cli.train import build_model, run_deterministic_rollout
```

Leave all test bodies unchanged.

- [ ] **Step 2: Run one test to verify it fails for the right reason**

Run:

```powershell
python -m pytest tests/test_train_sac.py::test_parse_args_requires_config -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'paint_rl.cli'` (or similar import error)

- [ ] **Step 3: Create the CLI package and move training code**

Create empty `paint_rl/cli/__init__.py`:

```python
"""Installable command-line entry points for PaintRL."""
```

Create `paint_rl/cli/train.py` by moving the training logic from `scripts/train_sac.py` with these exact adjustments:

1. Drop `sys` import and the `PROJECT_ROOT` / `sys.path` bootstrap block.
2. Import demo target from the package helper.
3. Keep all public helpers and `main` behavior identical.

Full file:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from paint_rl.config import TrainConfig, load_train_config, resolve_image_dimensions
from paint_rl.envs import TrianglePaintEnv
from paint_rl.models import PaintCNNFeaturesExtractor
from paint_rl.training import EpisodeCanvasSnapshotCallback, EpisodeTrainingLogCallback
from paint_rl.utils.actions import decode_triangle_action
from paint_rl.utils.demo import make_demo_target
from paint_rl.utils.image import load_target_image, save_canvas


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
```

- [ ] **Step 4: Delete `scripts/train_sac.py`**

Delete the file entirely. Do not leave a re-export shim.

- [ ] **Step 5: Run train-related tests**

Run:

```powershell
python -m pytest tests/test_train_sac.py tests/test_training_callbacks.py tests/test_train_entry_point.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```powershell
git add paint_rl/cli/__init__.py paint_rl/cli/train.py tests/test_train_sac.py tests/test_training_callbacks.py
git add -u scripts/train_sac.py
git commit -m "Move SAC training CLI to paint_rl.cli.train."
```

---

### Task 4: Reinstall editable package and update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Reinstall editable package so `paint-train` is on PATH**

Run (with the project venv activated):

```powershell
python -m pip install -e ".[dev]"
```

Expected: install succeeds; no missing-module errors for `paint_rl.cli.train`.

- [ ] **Step 2: Smoke-check the console script resolves**

Run:

```powershell
paint-train --help
```

Expected: argparse help text mentioning `--config` and description about training SAC. Do **not** run a full training job in this task.

- [ ] **Step 3: Update README Train SAC section**

In `README.md`, replace the Train SAC command example:

Old:

```powershell
python -m scripts.train_sac --config configs/train_sac.yaml
```

New:

```powershell
paint-train --config configs/train_sac.yaml
```

Search the README for any remaining `scripts.train_sac` mentions and remove/replace them the same way. Leave the Random Demo section using `python -m scripts.random_demo` unchanged.

- [ ] **Step 4: Run the full test suite**

Run:

```powershell
python -m pytest tests -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```powershell
git add README.md
git commit -m "Document paint-train as the SAC training entrypoint."
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Console script `paint-train` → `paint_rl.cli.train:main` | Task 2, Task 3 |
| Move training logic into `paint_rl/cli/train.py` | Task 3 |
| Extract `make_demo_target` to `paint_rl.utils.demo` | Task 1 |
| Update `scripts/random_demo.py` import | Task 1 |
| Delete `scripts/train_sac.py` (no shim) | Task 3 |
| Packaging / build-system / package discovery | Task 2 |
| Retarget train tests | Task 3 |
| Entry-point assertion test | Task 2 |
| README uses `paint-train` | Task 4 |
| No demo installable CLI | intentionally omitted |
| No `scripts.train_sac` compatibility | Task 3 delete |

---

## Self-review notes

- No TBD / placeholder steps.
- TDD order preserved: fail → implement → pass per task.
- `make_demo_target` signature matches both CLI and random demo callers.
- 3.10-safe entry-point test avoids `tomllib`.
