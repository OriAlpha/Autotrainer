# Autotrainer

[![CI](https://github.com/OriAlpha/Autotrainer/actions/workflows/ci.yml/badge.svg)](https://github.com/OriAlpha/Autotrainer/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/autotrainer.svg)](https://pypi.org/project/autotrainer/)
[![Python](https://img.shields.io/pypi/pyversions/autotrainer.svg)](https://pypi.org/project/autotrainer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/OriAlpha/Autotrainer/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v0.json)](https://github.com/astral-sh/ruff)

**Automatic distributed training and optimization for ML models.**
Give it a model and data — it detects your hardware (local GPUs or a SLURM
cluster), picks the distribution strategy, and can infer the loss function,
optimizer, learning rate, and schedule, or search for the best training
hyperparameters.

Supports **PyTorch** (DDP, SLURM multi-node), **TensorFlow/Keras**
(Mirrored / MultiWorker strategies), **scikit-learn**, **XGBoost**, and
**LightGBM** — all through one API.

```python
import autotrainer

# 1-Line Full Train: infers recipe, hardware wins, saves model, & prints summary!
autotrainer.train(model, loader, epochs=5, save_path="model.pt")

# Or prepare your custom training loop (DDP + AMP + TF32 + DataLoader workers):
model, loader, opt = autotrainer.prepare(model, loader, opt)

# Or hands-free search & retraining: searches recipe, then trains the winner
model, params, study = autotrainer.fit(model, train_loader, val_loader, save_path="best_model.pt")
```

```bash
python train.py                  # local: prepare() auto-distributes across GPUs
srun autotrainer run train.py    # SLURM multi-node (srun starts the tasks)
autotrainer doctor               # diagnose your environment first
```

## Install

```bash
uv pip install "autotrainer[torch]"            # PyTorch (recommended)
uv pip install "autotrainer[torch,boosting]"   # PyTorch + XGBoost/LightGBM
uv pip install "autotrainer[all]"              # Everything (PyTorch, TensorFlow, Sklearn, XGBoost, Optuna)

# Or with standard pip:
pip install "autotrainer[all]"
```

Only `psutil` is required by default; each ML framework is an opt-in extra
(`torch`, `sklearn`, `tf`, `boosting`, `tune`, `all`). Install only what you use.


Setting up for development instead? See
[CONTRIBUTING.md](CONTRIBUTING.md#development-setup).

## Quickstart

### 1-Line Complete Training (Easiest)

Infers loss/optimizer/schedule, applies hardware acceleration, trains for $N$ epochs, saves the model checkpoint, and prints the performance summary:


```python
import autotrainer

# 1 Line: trains model, saves checkpoint, and prints performance summary!
model = autotrainer.train(model, loader, epochs=5, save_path="model.pt")
```

### Custom Training Loop with `prepare()`

Or wrap your existing PyTorch training loop to auto-enable hardware acceleration (DDP, AMP, TF32, DataLoader thread pools) without altering your recipe:

```python
import autotrainer

model, loader, optimizer = autotrainer.prepare(model, loader, optimizer)

for epoch in range(epochs):
    autotrainer.set_epoch(loader, epoch)  # no-op when not distributed
    # ... your normal training loop

# At the end: prints comprehensive training summary & cleans up process groups
autotrainer.finish()
```


On a GPU, `prepare()` also enables TF32, `cudnn.benchmark`, sensible
`num_workers` / `pin_memory` / `persistent_workers`, and AMP — **without
touching your lr, loss, schedule, or optimizer**. It is a no-op on CPU, so the
same script runs unchanged on a laptop and on an A100. Pass `optimize=False` to
opt out, and see
[Getting throughput out of your GPUs](docs/guide/gpu-optimization.md) for what
that replaces.

Then launch. On a multi-GPU box `prepare()` spawns one worker per GPU the first
time it's called, so a bare `python train.py` uses them all with no launcher:

```bash
python train.py                   # local: auto-spawns per-GPU workers
autotrainer run train.py          # equivalent, explicit (same spawn machinery)
autotrainer info                  # show what was detected
```

## Which entry point do I want?

| You want | Call | Guide |
|---|---|---|
| 1-line train, recipe infer, & model save across PyTorch/Sklearn/XGBoost/TF | `train(model, loader, save_path="model.pt")` | [Quickstart](#quickstart) |
| My loop, but using the hardware properly | `prepare(model, loader, opt)` | [GPU optimization](docs/guide/gpu-optimization.md) |
| The whole step written for me | `train_step(...)` | [GPU optimization](docs/guide/gpu-optimization.md#even-simpler-train_step-runs-the-whole-step) |
| Loss / optimizer / LR / schedule inferred | `auto(model, loader)` | — |
| The best recipe searched, then trained | `fit(model, train, val)` | [One-call training](docs/guide/fit.md) |
| Just the search, not the final train | `tune(model, train, val)` | [One-call training](docs/guide/fit.md) |
| A learning rate suggestion | `find_lr(model, loader, loss_fn)` | [Training loop](docs/guide/training-loop.md#finding-a-learning-rate) |
| The largest batch size that fits | `find_batch_size(model, step_fn)` | [Training loop](docs/guide/training-loop.md#batch-size) |
| Post-training summary & metrics | `finish()`, `log_epoch()`, `step()` | [Training loop](docs/guide/training-loop.md#post-training-summary-finish-log_epoch-step-summarytracker) |
| To know if the loader is the bottleneck | `BottleneckMonitor()` | [Monitors](docs/guide/monitors.md) |
| To know if training is going wrong | `TrainingMonitor()` | [Monitors](docs/guide/monitors.md) |
| To shard a model too big for one GPU | `prepare(..., fsdp=True)` | [Scaling up](docs/guide/scaling.md) |
| XGBoost/LightGBM params with sane threads | `boost_params(lib="xgboost")` | — |
| TensorFlow strategy scope | `scope()`, `scale_batch_size(n)` | — |


Everything in `autotrainer.__all__` is public and stable; the rest is internal.

## How this differs from the alternatives

Most tools in this space ask you to hand over your training loop, your launcher,
or both. Autotrainer's bet is that you shouldn't have to give up either to use
your hardware properly.

| Tool | What it gives you | What you change to adopt it |
|---|---|---|
| `torch.distributed` + `torchrun` | The primitives | Write the DDP wrap, the sampler, the launcher flags — every time |
| HF Accelerate | Device/precision/distribution abstraction | Restructure around `accelerator.*`; configure and use `accelerate launch` |
| PyTorch Lightning | A full training framework | Move your code into a `LightningModule` + `Trainer` |
| Optuna | Hyperparameter search | Write the objective and the loop it calls; distribution is yours |
| Ray Train / Tune | Distributed execution + search | Adopt the Ray runtime and wrap your function in it |
| **autotrainer** | **All of the above paths, on your existing loop** | **One line: `prepare(model, loader, opt)`** |

Concretely, four things are unusual here:

1. **Your loop stays yours.** `prepare()` returns the same three objects you
   passed in. There is no base class to inherit, no trainer object to configure,
   and no callback system to learn. Deleting the import leaves a working script.
2. **Your hyperparameters are never touched silently.** lr, loss, schedule, and
   optimizer choice are yours unless you explicitly call `auto()` or `fit()` to
   have them inferred or searched. Everything `optimize=True` does — TF32,
   `cudnn.benchmark`, workers, AMP — is throughput, not recipe. And it prints
   every change it makes, so nothing is a surprise.
3. **No launcher to configure.** `python train.py` spawns one worker per GPU on
   its own. There is no config file to generate and no separate launch command,
   and under SLURM the auto-spawn correctly stands down so `srun autotrainer
   run` behaves.
4. **Search and distribution are the same tool.** `fit()` runs an ASHA search
   *split across your ranks* and then trains the winner distributed. Getting
   that from Optuna + DDP means wiring a shared study and a launcher yourself.

It is also framework-plural: PyTorch, TensorFlow/Keras, scikit-learn, XGBoost,
and LightGBM go through one API, where most of the tools above are PyTorch-only.

### Reach for something else when

- **You're already happy on Lightning, Accelerate, or Ray.** Autotrainer doesn't
  integrate with them — it's an alternative to that layer, not an addition. If
  their abstractions already fit your work, switching buys you little.
- **You need multi-node beyond SLURM**, or a scheduler-agnostic cluster
  abstraction. Ray covers ground autotrainer doesn't.
- **You need experiment tracking, model registries, or a UI.** This library
  prints to stdout and returns objects; it is not a platform.
- **You need architecture search.** Width/depth are deliberately out of scope —
  the model is yours.
- **You can't take pre-1.0 churn.** The public API has been frozen since 0.10,
  but this is 0.x and multi-node SLURM validation is still the open item before
  1.0 (see [Roadmap](#roadmap)).

## Documentation

**Guide**

- [Getting throughput out of your GPUs](docs/guide/gpu-optimization.md) —
  what `optimize=True` replaces, and `train_step()`.
- [Training-loop helpers](docs/guide/training-loop.md) — `set_epoch`,
  `zero_grad`, `eval_mode`, `accumulate`, batch size, `find_lr`.
- [Monitors](docs/guide/monitors.md) — bottleneck, throughput, and training
  health.
- [Scaling up](docs/guide/scaling.md) — launching, `torch.compile`, FSDP, CPU
  offload, SLURM.
- [One-call training](docs/guide/fit.md) — `fit()`, `metric=`, data checks,
  surviving preemption.

**Reference**

- [API reference](https://orialpha.github.io/Autotrainer/) — published from CI
  on every push to main; build locally with `pdoc -o docs/build src/autotrainer`.
- [Examples](examples/) — runnable scripts for each framework and SLURM
  `.sbatch` templates.
- [Environment variables](.env.example) — every knob autotrainer reads.
- [CHANGELOG](CHANGELOG.md) — version history.
- [Public API & deprecation policy](CONTRIBUTING.md#public-api-and-deprecation-policy)
  | [Contributing](CONTRIBUTING.md) | [Security](SECURITY.md) |
  [Code of Conduct](CODE_OF_CONDUCT.md).

## Roadmap

Toward 1.0:

- **Stabilization**: the public API is frozen as of 0.10; 1.0 blocks on real multi-node SLURM validation.
- **Unified Execution & Explanations (Shipped in 0.14.0)**:
  - 1-line unified `train()` API across PyTorch, Scikit-Learn, XGBoost, LightGBM, CatBoost, and TensorFlow.
  - Comprehensive Executive Training Summaries (`SummaryTracker`) reporting cluster hardware, active GPU/CPU optimizations, throughput (samples/sec), loss reduction, and triage diagnostics.
  - Framework environment diagnostics in `autotrainer doctor`.

Understanding your training run (Post-1.0 features):

- **Preflight estimation** (`doctor --profile`): dry-run a few batches, then
  report projected training time, memory headroom, and cost per GPU count —
  answer "how many GPUs do I actually need?" before burning an allocation.
- **Training cards**: every `fit()` emits a reproducibility card (recipe,
  seeds, environment, val curve) and `replay` reruns it.

Deeper SLURM ergonomics:

- **`autotrainer sbatch train.py --nodes 2 --time 4h`**: generate and submit a
  correct sbatch script (no more `--ntasks-per-node` != GPUs footguns).
- **Automatic requeue**: `fit(checkpoint=...)` already stops cleanly on a
  written checkpoint when preempted, but resubmitting is still yours to do
  (`--requeue`, or `scontrol requeue`). Issuing it from inside the handler is
  the remaining step.

More breadth:

- **Multi-node boosting** (xgboost.dask / lightgbm.dask across a SLURM
  allocation) — currently single-node threads only.
- **Richer augmentation policies**: label-mixing policies (mixup/cutmix) and per-op RandAugment-style search.
- **Architecture-aware search**: width/depth remain out of scope (a bigger,
  opt-in commitment that would step beyond "the model is yours").


Open or upvote [issues](https://github.com/OriAlpha/Autotrainer/issues) to
prioritize these.
