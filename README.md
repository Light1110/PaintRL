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
paint-train --config configs/train_sac.yaml
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

### Replay buffer memory

SAC training always uses a compact replay buffer for fixed-target runs; there is
no YAML switch. The buffer stores only the current and next RGB canvases as
`uint8`. It reconstructs the fixed target, absolute difference, and coordinate
channels as `float32` when sampling. Canvas quantization introduces at most
`1/255` absolute error per channel.

For a 64x64 canvas and 10,000 transitions, current/next observation storage is
reduced from about 3.36 GiB (11-channel `float32`) to 234 MiB (3-channel
`uint8`). This buffer assumes every transition shares one fixed target and does
not support mixing experience from multiple targets.

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
- `final_rollout.json`: per-step triangle parameters from the deterministic rollout
  (`action_version: 2`, center/size/rotation/skew, decoded vertices, color, alpha).

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

Each action is a 10-value continuous vector (action schema version 2):

```text
center_x, center_y, width_unit, height_unit, rotation_unit, skew_unit, r, g, b, alpha_unit
```

The first two values are the triangle center in normalized image coordinates.
`width_unit` / `height_unit` map into `[triangle_size_min, triangle_size_max]`
(defaults `0.02` / `1.0`). `rotation_unit` maps to `[−π, π]`, and `skew_unit`
maps to `[−1, 1]`. Local vertices are:

```text
(-width/2,  height/2)
( width/2,  height/2)
(skew*width/2, -height/2)
```

These points are rotated around the center, then translated. Vertices may leave
the unit square; OpenCV crops the filled polygon to the canvas. The remaining
values are RGB color and alpha (mapped into the configured alpha range).

Because width and height are bounded away from zero, every continuous action
produces a non-degenerate triangle. Old checkpoints trained on the previous
absolute-vertex action layout are incompatible and must be retrained.

Tune size bounds in the YAML config:

```yaml
triangle_size_min: 0.02
triangle_size_max: 1.0
```

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
