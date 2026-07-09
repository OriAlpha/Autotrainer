# autotrainer

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

## Publish to PyPI

Using `uv`:

```bash
uv run python -m build
uv run twine upload --repository testpypi dist/*   # test first
uv run twine upload dist/*
```

Or using standard python:

```bash
python -m build
twine upload --repository testpypi dist/*
twine upload dist/*
```

## Roadmap

- v0.2 PyTorch (DDP, SLURM, batch-size finder) + scikit-learn (SLURM-aware n_jobs) -- done
- v0.3 TensorFlow/Keras: `autotrainer.scope()`, MirroredStrategy / MultiWorkerMirroredStrategy, auto TF_CONFIG on SLURM
- v0.4 XGBoost / LightGBM: single-node nthread auto-set, then multi-node via xgboost.dask
- v0.5 Hardening: worker failure handling, rank-0 logging helpers, auto mixed precision, `autotrainer doctor`, tests, CI
- v0.6 Auto model optimization: `autotrainer.auto(model, loader)` infers loss, optimizer, LR (range test), warmup+cosine schedule -- done
- v0.7 Hyperparameter tuning: `autotrainer.tune(model, train_loader, val_loader)` searches lr, weight decay, optimizer, and batch size with Optuna (TPE + early pruning) -- done. The model architecture stays the user's; only the training recipe is searched.
