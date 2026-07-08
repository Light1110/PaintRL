# PaintRL

PaintRL is a minimal reinforcement learning prototype that paints a target image
by adding one transparent triangle per environment step.

The first version uses:

- `Gymnasium` for the custom environment.
- `Stable-Baselines3` SAC for continuous control.
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

## Train SAC

Run a short training job with the built-in demo target:

```powershell
python -m scripts.train_sac --total-timesteps 10000 --output-dir outputs/sac
```

Train against a custom target image:

```powershell
python -m scripts.train_sac --target path\to\mona_lisa.png --total-timesteps 10000 --output-dir outputs/mona_sac
```

Outputs include:

- `triangle_sac_model.zip`: the saved SAC model.
- `final_canvas.png`: the deterministic rollout after training.
- `target.png`: the resized target used by the environment.

## Environment

`TrianglePaintEnv` observes a 9-channel image tensor:

- Current canvas RGB.
- Target image RGB.
- Absolute difference RGB.

Each action is a 10-value continuous vector:

```text
x1, y1, x2, y2, x3, y3, r, g, b, alpha
```

The first six values define the three triangle vertices. The next three values
define RGB color. The final value is mapped into the configured alpha range.

The reward is the improvement in mean squared error:

```text
reward = (old_mse - new_mse) * reward_scale
```

## Tests

```powershell
python -m pytest tests
```
