# PaintRL

PaintRL is a minimal reinforcement learning prototype that paints a target image
by adding one transparent triangle per environment step.

The current version uses:

- `Gymnasium` for the custom environment.
- `Stable-Baselines3` SAC for continuous control with a custom CNN feature extractor.
- `Pillow` and `OpenCV` for image loading and triangle rendering.
- YAML config files for demo and training inputs.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Configuration

Scripts accept a single required argument: `--config <path-to-yaml>`.

Relative paths inside a config file are resolved from that YAML file's directory.
The bundled examples under `configs/` therefore use `../outputs/...` so artifacts
still land in the project `outputs/` folder when run from the repo root.

## Random Demo

Run a quick rendering smoke test with the bundled config:

```powershell
python -m scripts.random_demo --config configs/random_demo.yaml
```

Use your own target image by editing the config (paths relative to the YAML file):

```yaml
target: ../path/to/mona_lisa.png
output: ../outputs/random_mona.png
```

Run with a non-square canvas by setting width/height in the config:

```yaml
image_width: 96
image_height: 64
steps: 200
output: ../outputs/random_demo_96x64.png
```

## Train SAC

Run a short training job with the bundled config:

```powershell
python -m scripts.train_sac --config configs/train_sac.yaml
```

Train against a custom target image:

```yaml
target: ../path/to/mona_lisa.png
total_timesteps: 10000
output_dir: ../outputs/mona_sac
```

Train on a non-square canvas:

```yaml
image_width: 96
image_height: 64
total_timesteps: 10000
output_dir: ../outputs/sac_96x64
```

Use a specific GPU on a multi-GPU machine:

```yaml
device: cuda:1
total_timesteps: 10000
output_dir: ../outputs/sac_gpu1
```

Force CPU training:

```yaml
device: cpu
total_timesteps: 10000
output_dir: ../outputs/sac_cpu
```

Resolution fields:

- `image_width` and `image_height` define the canvas resolution.
- `image_size` remains as a backward-compatible square shortcut.
- If only one of `image_width`/`image_height` is provided, the other defaults to the same value.

Outputs include:

- `target.png`: the resized target used by the environment, saved at training start.
- `triangle_sac_model.zip`: the saved SAC model.
- `episode_metrics.txt`: per-episode training metrics with dense reward totals, MSE improvement, and final MSE.
- `snapshots/episode_XXXXX.png`: periodic snapshots of the final canvas from completed training episodes.
- `final_canvas.png`: the deterministic rollout after training.
- `final_rollout.json`: per-step triangle parameters from the deterministic rollout.

Snapshot interval:

```yaml
snapshot_interval: 10
output_dir: ../outputs/sac
```

Set `snapshot_interval: 0` to disable training snapshots.

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

There is no separate terminal reward pulse. Tune the reward scale in the training config:

```yaml
reward_scale: 1000.0
output_dir: ../outputs/sac
```

## Tests

```powershell
python -m pytest tests
```
