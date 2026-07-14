# Design: Installable `paint-train` CLI

**Date:** 2026-07-14  
**Status:** Approved for implementation planning

## Goal

After `pip install -e .` (or a normal install), users can train with:

```powershell
paint-train --config configs/train_sac.yaml
```

No `scripts` compatibility for the old training entrypoint.

## Non-goals

- Installable command for random demo
- Unified multi-subcommand CLI (e.g. `paint-rl train`)
- Keeping `python -m scripts.train_sac` working

## Command contract

| Item | Value |
|------|--------|
| Console script name | `paint-train` |
| Entry point | `paint_rl.cli.train:main` |
| Required argument | `--config <path-to-yaml>` |
| Behavior | Same training pipeline as today's `scripts/train_sac.py` |

## Layout

```text
paint_rl/
  cli/
    __init__.py
    train.py              # argparse + training main / helpers
  utils/
    demo.py               # make_demo_target (shared)
scripts/
  random_demo.py          # keeps working; imports make_demo_target from paint_rl
  train_sac.py            # DELETED
```

## Implementation details

### 1. Move training CLI into the package

Create `paint_rl/cli/train.py` with the current contents of `scripts/train_sac.py`, adjusted imports:

- Package imports stay on `paint_rl.*`
- Replace `from scripts.random_demo import make_demo_target` with `from paint_rl.utils.demo import make_demo_target`
- Remove `sys.path` / `PROJECT_ROOT` bootstrap (not needed when installed as a package module)

Public functions to keep callable for tests: `parse_args`, `build_model`, `resolve_dimensions`, `run_deterministic_rollout`, `main`.

### 2. Extract shared demo target helper

Move `make_demo_target` from `scripts/random_demo.py` into `paint_rl/utils/demo.py`.

Update `scripts/random_demo.py` to import it from `paint_rl.utils.demo`.

### 3. Delete old train script

Delete `scripts/train_sac.py`. Do not leave a re-export shim.

### 4. Packaging

Update `pyproject.toml`:

- Add `[build-system]` with setuptools (if missing)
- Ensure `paint_rl` packages are discoverable (including `paint_rl.cli`)
- Add:

```toml
[project.scripts]
paint-train = "paint_rl.cli.train:main"
```

### 5. Tests

- Change `tests/test_train_sac.py` and `tests/test_training_callbacks.py` to import from `paint_rl.cli.train`
- Update any references to `scripts.train_sac`
- Optional small test: assert `pyproject.toml` declares `paint-train = "paint_rl.cli.train:main"`

### 6. Docs

Update README Train SAC section:

- Primary example: `paint-train --config configs/train_sac.yaml`
- Remove or replace references to `python -m scripts.train_sac`

## Compatibility notes

| Old | New |
|-----|-----|
| `python -m scripts.train_sac --config ...` | Unsupported (file removed) |
| `paint-train --config ...` | Supported after install |
| `python -m scripts.random_demo ...` | Unchanged |

## Risks / mitigations

- **Installed package missing CLI module:** covered by package discovery config and import-based unit tests.
- **Broken random demo after moving helper:** update import and keep existing random demo tests green.
