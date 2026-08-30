# Autotrainer

[![CI](https://github.com/OriAlpha/Autotrainer/actions/workflows/ci.yml/badge.svg)](https://github.com/OriAlpha/Autotrainer/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/autotrainer.svg)](https://pypi.org/project/autotrainer/)
[![Python](https://img.shields.io/pypi/pyversions/autotrainer.svg)](https://pypi.org/project/autotrainer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/OriAlpha/Autotrainer/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v0.json)](https://github.com/astral-sh/ruff)

**Automatic distributed training and optimization for ML models.**
Give it a model and data — it detects your hardware (local GPUs or a SLURM
cluster), picks the distribution strategy, applies the hardware wins, and can
infer the loss function, optimizer, learning rate, and schedule.

Supports **PyTorch** (DDP, SLURM multi-node), **TensorFlow/Keras**
(Mirrored / MultiWorker strategies), **scikit-learn**, **XGBoost**, and
**LightGBM** — all through one API.

```python
import autotrainer

# 1-Line Full Train: infers recipe, hardware wins, saves model, & prints summary!
autotrainer.train(model, loader, epochs=5, save_path="model.pt")

# Or prepare your custom training loop (DDP + AMP + TF32 + DataLoader workers):
model, loader, opt = autotrainer.prepare(model, loader, opt)
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
uv pip install "autotrainer[all]"              # Everything (PyTorch, TensorFlow, Sklearn, XGBoost)

# Or with standard pip:
pip install "autotrainer[all]"
```

Only `psutil` is required by default; each ML framework is an opt-in extra
(`torch`, `sklearn`, `tf`, `boosting`, `all`). Install only what you use.

> **Tip for HPC / SLURM Clusters**: On shared network file systems (NFS/Lustre), speed up installation by routing the `uv` cache to local node storage (`export UV_CACHE_DIR=/tmp/$USER-uv-cache`) and passing `--link-mode=copy` (or creating `.venv` in `/tmp`).

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
    autotrainer.set_epoch(loader, epoch)  # optional: prepare()'s loader
    # ... your normal training loop      # already reshuffles each epoch

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
| 1-line train, recipe infer, & model save across PyTorch/Sklearn/XGBoost/TF | `train(model, loader, save_path="model.pt")` | [One-line training](docs/guide/one-line-training.md) |
| My loop, but using the hardware properly | `prepare(model, loader, opt)` | [GPU optimization](docs/guide/gpu-optimization.md) |
| The whole step written for me | `train_step(...)` | [GPU optimization](docs/guide/gpu-optimization.md#even-simpler-train_step-runs-the-whole-step) |
| Loss / optimizer / LR / schedule inferred | `auto(model, loader)` | [One-line training](docs/guide/one-line-training.md) |
| A learning rate suggestion | `find_lr(model, loader, loss_fn)` | [Training loop](docs/guide/training-loop.md#finding-a-learning-rate) |
| The largest batch size that fits | `find_batch_size(model, step_fn)` | [Training loop](docs/guide/training-loop.md#batch-size) |
| Zero-dependency Web UI dashboard | `autotrainer ui`, `run_ui_server()` | [Monitors](docs/guide/monitors.md#autotrainer-ui) |
| Native experiment tracking (CSV, JSONL, metadata) | `NativeTracker()`, `CSVTracker()`, `JSONLTracker()` | [Monitors](docs/guide/monitors.md#native-experiment-trackers) |
| Multi-framework callbacks | `AutotrainerCallback()`, `AutotrainerHuggingFaceCallback()`, `AutotrainerLightningCallback()`, `AutotrainerKerasCallback()`, `autotrainer_xgboost_callback()`, `autotrainer_lightgbm_callback()` | [Monitors](docs/guide/monitors.md#multi-framework-callbacks) |
| Multi-format exports (HTML, Markdown, CSV, JSON) | `autotrainer ui` &rarr; Export Dropdown | [Monitors](docs/guide/monitors.md#multi-format-exports) |
| Multi-user workspaces & directory aggregation | `autotrainer ui ./logs /cluster/runs` | [Monitors](docs/guide/monitors.md#multi-user-workspace) |
| To know if the loader is the bottleneck | `BottleneckMonitor()` | [Monitors](docs/guide/monitors.md) |
| Samples/sec and a rough MFU estimate | `ThroughputMonitor()` | [Monitors](docs/guide/monitors.md) |
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
| Ray Train | Distributed execution | Adopt the Ray runtime and wrap your function in it |
| **autotrainer** | **All of the above paths, on your existing loop** | **One line: `prepare(model, loader, opt)`** |

Concretely, four things are unusual here:

1. **Your loop stays yours.** `prepare()` returns the same three objects you
   passed in. There is no base class to inherit, no trainer object to configure,
   and no callback system to learn. Deleting the import leaves a working script.
2. **Your hyperparameters are never touched silently.** lr, loss, schedule, and
   optimizer choice are yours unless you explicitly call `auto()` or `train()`
   to have them inferred. Everything `optimize=True` does — TF32,
   `cudnn.benchmark`, workers, AMP — is throughput, not recipe. And it prints
   every change it makes, so nothing is a surprise.
3. **No launcher to configure.** `python train.py` spawns one worker per GPU on
   its own. There is no config file to generate and no separate launch command,
   and under SLURM the auto-spawn correctly stands down so `srun autotrainer
   run` behaves.
4. **It tells you why the run is bad.** `ThroughputMonitor` reports MFU,
   `BottleneckMonitor` says whether the loader is starving the GPU, and
   `TrainingMonitor` names the numerical failure (NaN loss, divergence, fp16
   overflow, vanishing grads) in plain language with the fix. Getting a
   straight answer to "why is this slow / broken" is usually the actual job.

It is also framework-plural: PyTorch, TensorFlow/Keras, scikit-learn, XGBoost,
and LightGBM go through one API, where most of the tools above are PyTorch-only.

## The Core Philosophy: Hardware vs. Math

Autotrainer maintains a strict separation between **hardware execution throughput** and **mathematical modeling recipe**:

| Dimension | What it Includes | How Autotrainer Treats It |
|---|---|---|
| **⚡ Hardware Throughput** | AMP (bf16/fp16), TF32, channels_last, DataLoader Workers, Pin Memory, cuDNN Benchmark, DDP Spawning | **100% Automatic & Safe** — `prepare()` enables these out of the box because they accelerate compute execution without altering your loss function or model convergence. |
| **🧠 Algorithmic / Math** | Gradient Clipping, Learning Rate, Optimizer Choice, Weight Decay, Loss Scaling | **Non-Invasive by Default** — Autotrainer never silently modifies your mathematical hyperparameters. Instead, the **AI Training Doctor** (`TrainingMonitor`) inspects live tensor dynamics and calculates concrete remediation steps. |

### Reach for something else when

- **You're already happy on Lightning, Accelerate, or Ray.** Autotrainer doesn't
  integrate with them — it's an alternative to that layer, not an addition. If
  their abstractions already fit your work, switching buys you little.
- **You need multi-node beyond SLURM**, or a scheduler-agnostic cluster
  abstraction. Ray covers ground autotrainer doesn't.
- **You need heavy cloud-hosted model registries.** Autotrainer provides a local
  zero-dependency Web UI (`autotrainer ui`) and local file trackers (CSV, JSONL, run metadata), but does not run external hosted cloud services.
- **You need hyperparameter or architecture search.** Deliberately out of
  scope — the recipe and the model are yours. Reach for Optuna or Ray Tune,
  and use `prepare()` inside the objective.
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
- [One-line training](docs/guide/one-line-training.md) — `train()`, `auto()`,
  and the data checks that run before the compute.

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
- **Training cards**: every `train()` emits a reproducibility card (recipe,
  seeds, environment, loss curve) and `replay` reruns it.

Deeper SLURM ergonomics:

- **`autotrainer sbatch train.py --nodes 2 --time 4h`**: generate and submit a
  correct sbatch script (no more `--ntasks-per-node` != GPUs footguns).
- **Automatic requeue**: `autotrainer.preempt` already stops your loop cleanly
  at an epoch boundary when preempted, but resubmitting is still yours to do
  (`--requeue`, or `scontrol requeue`). Issuing it from inside the handler is
  the remaining step.

More breadth:

- **Multi-node boosting** (xgboost.dask / lightgbm.dask across a SLURM
  allocation) — currently single-node threads only.
- **Architecture-aware search**: width/depth remain out of scope (a bigger,
  opt-in commitment that would step beyond "the model is yours").


Open or upvote [issues](https://github.com/OriAlpha/Autotrainer/issues) to
prioritize these.
