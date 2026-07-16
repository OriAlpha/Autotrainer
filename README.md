# Autotrainer

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

# tune() also handles sklearn / XGBoost / LightGBM (curated default spaces)
best_est, params, study = autotrainer.tune(XGBClassifier(), (X, y), (X_val, y_val))

# Or fully hands-free: search the recipe, then train the winner to completion
model, params, study = autotrainer.fit(model, train_loader, val_loader)
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

In your training script, add one line (plus `set_epoch` at each epoch start,
so distributed shuffling gives every epoch a fresh order):

```python
import autotrainer
model, loader, optimizer = autotrainer.prepare(model, loader, optimizer)

for epoch in range(epochs):
    autotrainer.set_epoch(loader, epoch)  # no-op when not distributed
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

## One-call training: fit()

`fit()` is the whole pipeline in one call - give it a model and data, get
back the best model it can produce on your hardware:

```python
model, params, study = autotrainer.fit(model, train_loader, val_loader, trials=30)
```

1. **Tune**: Optuna searches lr / weight decay / optimizer / batch size on
   short trials. Launched distributed, the trials are split across all
   ranks via a shared journal-file study - one trial per process, every
   GPU busy during the search.
2. **Train**: the winning recipe is retrained from your model's original
   init through `prepare()` - so `autotrainer run` distributes it across
   every GPU/node - with warmup+cosine, mixed precision, and early stopping
   on the val loss. The best epoch's weights are returned.

Pass `checkpoint="fit.ckpt"` to make it preemption-safe: the full training
state is saved every epoch, and rerunning the same script resumes where it
died (skipping the search) - ideal for requeued SLURM jobs.

For long searches under multi-process launches, raise the collective
timeout with `AUTOTRAINER_TIMEOUT` (seconds) - see `.env.example`.

## Optional: auto batch size

```python
best = autotrainer.find_batch_size(model, my_one_step_fn)
```

## Documentation

- [CHANGELOG](CHANGELOG.md) - version history.
- [API reference](https://orialpha.github.io/Autotrainer/) - published from
  CI on every push to main; build locally with `pdoc -o docs/build src/autotrainer`.
- [Public API & deprecation policy](CONTRIBUTING.md#public-api-and-deprecation-policy) -
  what `autotrainer.__all__` exports is stable; the rest is internal.
- [Examples](examples/) - runnable scripts for each framework and SLURM
  `.sbatch` templates.
- [Environment variables](.env.example) - every knob autotrainer reads.
- [Contributing](CONTRIBUTING.md) | [Security policy](SECURITY.md) |
  [Code of Conduct](CODE_OF_CONDUCT.md).

## Roadmap

- **Multi-node boosting** (xgboost.dask / lightgbm.dask across a SLURM
  allocation) — currently single-node threads only.
- **Stabilization toward 1.0**: freeze the public API (mypy strict mode and
  real 2-rank distributed tests already gate CI).
- **More schedulers and search spaces** beyond warmup+cosine and the default
  Optuna recipe.

See [CHANGELOG.md](CHANGELOG.md) for the full version history, and open or
upvote [issues](https://github.com/OriAlpha/autotrainer/issues) to prioritize
these.
