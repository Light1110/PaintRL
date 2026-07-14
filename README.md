# PaintRL

PaintRL is a minimal reinforcement learning prototype that paints a target image
by adding one transparent triangle per environment step.

The current version uses:

- `Gymnasium` for the custom environment.
- `Stable-Baselines3` SAC for continuous control with a custom CNN feature extractor.
- `Pillow` and `OpenCV` for image loading and triangle rendering.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Random Demo

Run a quick rendering smoke test with random triangles:

```powershell
python -m scripts.random_demo --steps 200 --output outputs/random_demo.png
```

Use your own target image:

```powershell
python -m scripts.random_demo --target path\to\mona_lisa.png --output outputs/random_mona.png
```

Run with a non-square canvas:

```powershell
python -m scripts.random_demo --image-width 96 --image-height 64 --steps 200 --output outputs/random_demo_96x64.png
```

## Train SAC

Run a short training job with the built-in demo target:

```powershell
python -m scripts.train_sac --total-timesteps 10000 --output-dir outputs/sac
```

Train against a custom target image:

```powershell
python -m scripts.train_sac --target path\to\mona_lisa.png --total-timesteps 10000 --output-dir outputs/mona_sac
```

Train on a non-square canvas:

```powershell
python -m scripts.train_sac --image-width 96 --image-height 64 --total-timesteps 10000 --output-dir outputs/sac_96x64
```

Use a specific GPU on a multi-GPU machine:

```powershell
python -m scripts.train_sac --device cuda:1 --total-timesteps 10000 --output-dir outputs/sac_gpu1
```

Force CPU training:

```powershell
python -m scripts.train_sac --device cpu --total-timesteps 10000 --output-dir outputs/sac_cpu
```

Resolution arguments:

- `--image-width` and `--image-height` define the canvas resolution.
- `--image-size` remains as a backward-compatible square shortcut.
- If only one of `--image-width`/`--image-height` is provided, the other defaults to the same value.

Outputs include:

- `target.png`: the resized target used by the environment, saved at training start.
- `triangle_sac_model.zip`: the saved SAC model.
- `episode_metrics.txt`: per-episode training metrics with dense reward totals, MSE improvement, and final MSE.
- `snapshots/episode_XXXXX.png`: periodic snapshots of the final canvas from completed training episodes.
- `final_canvas.png`: the deterministic rollout after training.
- `final_rollout.json`: per-step triangle parameters from the deterministic rollout.

Snapshot interval:

```powershell
python -m scripts.train_sac --snapshot-interval 10 --output-dir outputs/sac
```

Set `--snapshot-interval 0` to disable training snapshots.

## Environment

`TrianglePaintEnv` observes an 11-channel image tensor:

- Current canvas RGB.
- Target image RGB.
- Absolute difference RGB.
- Absolute x/y coordinate channels.

Each action is a 10-value continuous vector:

```text
x1, y1, x2, y2, x3, y3, r, g, b, alpha
```

The first six values define the three triangle vertices. The next three values
define RGB color. The final value is mapped into the configured alpha range.

The reward is a global dense MSE improvement signal:

```text
mse_improvement = old_mse - new_mse
reward = reward_scale * mse_improvement
```

Positive rewards mean the canvas got closer to the target; negative rewards mean
it got farther away. Over an episode without discounting:

```text
sum(reward) / reward_scale ≈ initial_mse - final_mse
```

There is no separate terminal reward pulse. Tune the reward scale during training:

```powershell
python -m scripts.train_sac --reward-scale 1000.0 --output-dir outputs/sac
```

## Tests

```powershell
python -m pytest tests
```
