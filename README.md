# Autotrainer

[![CI](https://github.com/OriAlpha/Autotrainer/actions/workflows/ci.yml/badge.svg)](https://github.com/OriAlpha/Autotrainer/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/autotrainer.svg)](https://pypi.org/project/autotrainer/)
[![Python](https://img.shields.io/pypi/pyversions/autotrainer.svg)](https://pypi.org/project/autotrainer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/OriAlpha/Autotrainer/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v0.json)](https://github.com/astral-sh/ruff)

**Run your training loop fast on the hardware you have.**

Autotrainer detects your hardware (local GPUs or a SLURM cluster), picks the
distribution strategy, applies the throughput wins people forget, and tells you
when a run is slow or broken — **without touching your lr, loss, schedule, or
optimizer**.

Works with **PyTorch** (DDP, SLURM multi-node), **TensorFlow/Keras**,
**scikit-learn**, **XGBoost**, and **LightGBM** through one API.

## Install

```bash
pip install "autotrainer[torch]"   # PyTorch (recommended)
pip install "autotrainer[all]"     # + TensorFlow, sklearn, XGBoost, LightGBM
```

Only `psutil` is required by default; each framework is an opt-in extra
(`torch`, `sklearn`, `tf`, `boosting`, `all`). Install only what you use.

## Use it

**Two lines, full training run** — infers loss/optimizer/schedule, applies the
hardware wins, saves the checkpoint, prints a summary:

```python
import autotrainer

model = autotrainer.train(model, loader, epochs=5, save_path="model.pt")
```

**Or keep your own loop** — `prepare()` returns the same three objects you
passed in, now distributed and optimized:

```python
import autotrainer

model, loader, optimizer = autotrainer.prepare(model, loader, optimizer)

for epoch in range(epochs):
    autotrainer.set_epoch(loader, epoch)   # reshuffle each epoch
    ...                                    # your normal training loop

autotrainer.finish()                       # summary + process-group cleanup
```

Then run it. No launcher to configure — on a multi-GPU box `prepare()` spawns
one worker per GPU by itself:

```bash
python train.py                  # local: auto-distributes across GPUs
srun autotrainer run train.py    # SLURM multi-node
autotrainer doctor               # check the environment before you burn an allocation
autotrainer ui                   # zero-dependency dashboard for your runs
```

## Why use it

- **Your loop stays yours.** No base class, no trainer object, no callback
  system. Deleting the import leaves a working script.
- **Your hyperparameters are never touched silently.** lr, loss, schedule, and
  optimizer are yours unless you call `auto()` or `train()`. `optimize=True`
  only does throughput — TF32, `cudnn.benchmark`, `channels_last`, workers,
  AMP, fused optimizers — and prints every change it makes.
- **It's faster.** +22% end to end vs torch defaults on a 4-conv net, +123% on
  optimizer-bound MLPs. [Measurements](docs/performance.md).
- **No launcher to configure.** `python train.py` uses every GPU; under SLURM
  the auto-spawn stands down so `srun autotrainer run` behaves.
- **It tells you why the run is bad.** MFU and samples/sec, whether the loader
  is starving the GPU, and NaN / divergence / fp16-overflow named in plain
  language with the fix.
- **One API across frameworks.** PyTorch, TensorFlow/Keras, scikit-learn,
  XGBoost, LightGBM — most alternatives are PyTorch-only.

Worker counts come from the CPUs you were actually *allocated*
(`SLURM_CPUS_PER_TASK`, then the affinity mask), so a job granted 8 CPUs of a
128-core node does not spawn 128 workers.

## Documentation

**Start here**

- [Entry points](docs/api-map.md) — which function do I call for what.
- [Performance](docs/performance.md) — what the optimizations measurably buy.
- [How it compares](docs/comparison.md) — vs Lightning, Accelerate, Ray,
  `torchrun`, and when to reach for something else.

**Guide**

- [Getting throughput out of your GPUs](docs/guide/gpu-optimization.md) — what
  `optimize=True` replaces, and `train_step()`.
- [Training-loop helpers](docs/guide/training-loop.md) — `set_epoch`,
  `zero_grad`, `eval_mode`, `accumulate`, batch size, `find_lr`.
- [Monitors](docs/guide/monitors.md) — bottleneck, throughput, training health,
  the web UI, trackers, and framework callbacks.
- [Scaling up](docs/guide/scaling.md) — launching, `torch.compile`, FSDP, CPU
  offload, SLURM.
- [One-line training](docs/guide/one-line-training.md) — `train()`, `auto()`,
  and the data checks that run before the compute.

**Reference**

- [API reference](https://orialpha.github.io/Autotrainer/) — generated from the
  source on every push to main.
- [Examples](examples/) — runnable scripts per framework, SLURM `.sbatch`
  templates.
- [Environment variables](.env.example) — every knob autotrainer reads.
- [Roadmap](docs/roadmap.md) · [CHANGELOG](CHANGELOG.md) ·
  [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) ·
  [Code of Conduct](CODE_OF_CONDUCT.md)

> **HPC tip**: on shared network filesystems (NFS/Lustre), route the `uv` cache
> to local node storage (`export UV_CACHE_DIR=/tmp/$USER-uv-cache`) and pass
> `--link-mode=copy`, or create `.venv` in `/tmp`.
