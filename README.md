# autotrainer

[![CI](https://github.com/OriAlpha/autotrainer/actions/workflows/ci.yml/badge.svg)](https://github.com/OriAlpha/autotrainer/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/autotrainer.svg)](https://pypi.org/project/autotrainer/)
[![Python](https://img.shields.io/pypi/pyversions/autotrainer.svg)](https://pypi.org/project/autotrainer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/OriAlpha/autotrainer/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v0.json)](https://github.com/astral-sh/ruff)

**Automatic distributed training and optimization for ML models.**
Give it a model and data - it detects your hardware (local GPUs or a SLURM
cluster), picks the distribution strategy, and can infer the loss function,
optimizer, learning rate, and schedule, or search for the best training
hyperparameters.

Supports **PyTorch** (DDP, SLURM multi-node), **TensorFlow/Keras**
(Mirrored / MultiWorker strategies), **scikit-learn**, **XGBoost**, and
**LightGBM** - all through one API.

```python
import autotrainer

# Smart defaults: loss, optimizer, LR, and schedule inferred and printed
model, loader, opt, loss_fn, sched = autotrainer.auto(model, loader)

# Or search for the best training recipe for YOUR model
best_model, best_params, study = autotrainer.tune(model, train_loader, val_loader)
```

```bash
autotrainer run train.py         # local: 1 or many GPUs, auto-detected
srun autotrainer run train.py    # SLURM: multi-node, zero config
autotrainer doctor               # diagnose your environment first
```

## Install

From PyPI:

```bash
pip install autotrainer[torch]            # just PyTorch
pip install autotrainer[torch,boosting]   # PyTorch + XGBoost/LightGBM
```

Only `psutil` is required by default; each ML framework is an opt-in extra
(`torch`, `sklearn`, `tf`, `boosting`, `tune`). Install only what you use.

## Install (dev mode)

Using `uv` (recommended):

```bash
uv venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

uv pip install -e ".[dev,torch,sklearn,tf,boosting,tune]"
```

Or using standard pip:

```bash
pip install -e ".[dev,torch,sklearn,tf,boosting,tune]"
```

## Use

In your training script, add one line:

```python
import autotrainer
model, loader, optimizer = autotrainer.prepare(model, loader, optimizer)
# ... your normal training loop
```

Then launch:

```bash
autotrainer info                 # show what was detected
autotrainer run train.py         # local machine (1 or many GPUs)
```

On SLURM, inside your sbatch script:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
srun autotrainer run train.py
```

## Optional: auto batch size

```python
best = autotrainer.find_batch_size(model, my_one_step_fn)
```

## Documentation

- [CHANGELOG](CHANGELOG.md) - version history.
- [API reference](docs/README.md) - generated with pdoc; build locally with
  `pdoc -o docs/build src/autotrainer`.
- [Examples](examples/) - runnable scripts for each framework and SLURM
  `.sbatch` templates.
- [Environment variables](.env.example) - every knob autotrainer reads.
- [Contributing](CONTRIBUTING.md) | [Security policy](SECURITY.md) |
  [Code of Conduct](CODE_OF_CONDUCT.md).

## Release history

- **v0.7** Hyperparameter tuning: `autotrainer.tune()` searches lr, weight
  decay, optimizer, and batch size with Optuna (TPE + early pruning).
- **v0.6** Auto model optimization: `autotrainer.auto()` infers loss, optimizer,
  LR (range test), and a warmup+cosine schedule.
- **v0.5** Hardening: worker failure handling, rank-0 logging helpers, auto
  mixed precision, `autotrainer doctor`, tests, CI.
- **v0.4** XGBoost / LightGBM: single-node thread auto-tuning.
- **v0.3** TensorFlow/Keras: `autotrainer.scope()`, MirroredStrategy /
  MultiWorkerMirroredStrategy, auto `TF_CONFIG` on SLURM.
- **v0.2** PyTorch (DDP, SLURM, batch-size finder) + scikit-learn
  (SLURM-aware `n_jobs`).

## Roadmap

- **Multi-node boosting** (xgboost.dask / lightgbm.dask across a SLURM
  allocation) — currently single-node threads only.
- **Stabilization toward 1.0**: expand real distributed test coverage, tighten
  the type-check to strict mode, and freeze the public API.
- **More schedulers and search spaces** beyond warmup+cosine and the default
  Optuna recipe.

See [CHANGELOG.md](CHANGELOG.md) for the full version history, and open or
upvote [issues](https://github.com/OriAlpha/autotrainer/issues) to prioritize
these.
