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

### Want throughput, not magic? `prepare(..., optimize=True)`

`prepare()` by itself makes your model/loader distribution-ready and leaves
everything else alone. Pass `optimize=True` to also flip on the GPU wins
users forget — TF32, `cudnn.benchmark` for CNNs, sane `num_workers` /
`pin_memory` / `persistent_workers` defaults on bare loaders, and AMP —
**without touching your lr, loss, schedule, or optimizer choice**. No-op on
CPU and when the flag is off, so existing scripts see no change.

#### Before: the boilerplate you write today to "use your GPUs well"

None of this touches your recipe (lr, loss, schedule), yet you have to
remember all of it every time — and forgetting any one of them silently
leaves 2–3× on the table.

```python
import torch

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # your hyperparameter
loss_fn = nn.CrossEntropyLoss()                             # your hyperparameter

# Manual GPU optimization — ~10 lines of boilerplate:
torch.backends.cuda.matmul.allow_tf32 = True               # ships off by default
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True                       # free win for CNNs
loader = DataLoader(ds, batch_size=64, shuffle=True,
                    num_workers=8, pin_memory=True,         # avoid GPU starvation
                    persistent_workers=True)
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)

for epoch in range(epochs):
    for xb, yb in loader:
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            loss = loss_fn(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
```

#### After: one line

```python
import autotrainer

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # your hyperparameter
loss_fn = nn.CrossEntropyLoss()                             # your hyperparameter

# ONE line detects hardware and sets TF32 / cudnn.benchmark / num_workers /
# pin_memory / persistent_workers / AMP. lr, loss, schedule, optimizer untouched.
model, loader, optimizer = autotrainer.prepare(model, loader, optimizer, optimize=True)

scaler = autotrainer.GradScaler()   # no-op when bf16 is available
for epoch in range(epochs):
    autotrainer.set_epoch(loader, epoch)                    # reshuffles in DDP
    for xb, yb in loader:
        with autotrainer.autocast_context():                # bf16 if supported, else fp16
            loss = loss_fn(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
```

What changed: **one `prepare(..., optimize=True)` call + two no-op-on-CPU
helpers.** What didn't: your lr, your loss, your schedule, your optimizer.
Same script runs unchanged on a laptop (everything degrades to no-ops) and
on an A100.

#### What it prints when it runs

Nothing is silent — every speedup is named, and the user is explicitly told
their hyperparameters weren't touched:

```
[autotrainer] mode=local_multi_gpu nodes=1 procs/node=4 world_size=4
[autotrainer] DistributedSampler installed (shuffle=True) - call autotrainer.set_epoch(loader, epoch) ...
[autotrainer] optimize: TF32, cudnn.benchmark, num_workers=8, pin_memory, persistent_workers, AMP (hyperparameters untouched)
```

| | Manual | `prepare(optimize=True)` |
|---|---|---|
| Lines of "optimize my GPUs" boilerplate | ~10, hand-written, easy to forget | **1** |
| Hyperparameters touched | none (correct) | none (correct) |
| Works on CPU | guard every line yourself | automatic (all no-ops) |
| Works on SLURM | you'd never write this path | same script, `srun autotrainer run` |
| Knows what it did | silent | prints it |

### Training-loop helpers (`zero_grad`, `eval_mode`, `accumulate`)

The small things users forget *inside* the loop. None touch lr / loss /
schedule / optimizer choice.

```python
import autotrainer

for epoch in range(epochs):
    autotrainer.set_epoch(loader, epoch)
    model.train()
    for xb, yb in loader:
        with autotrainer.autocast_context():
            loss = loss_fn(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        autotrainer.zero_grad(opt)            # set_to_none=True, saves memory

    # eval_mode restores the prior train/eval state - kills the classic
    # "forgot to flip back to train() after validation" silent bug.
    with autotrainer.eval_mode(model):
        val_loss = evaluate(model, val_loader)
```

**Gradient accumulation** when the effective batch is larger than the
physical one — scales the step count, **not** the lr:

```python
# Effective batch = 4 micro-batches; opt steps once per 4 backwards.
with autotrainer.accumulate(opt, steps=4, scaler=scaler) as acc:
    for micro_xb, micro_yb in micro_batches:
        with autotrainer.autocast_context():
            loss = loss_fn(model(micro_xb), micro_yb) / 4
        acc.backward(loss)
```

### Auto batch size

```python
# Grow batch size until OOM, back off one step. Pass loss_fn for an
# accurate fwd+bwd measurement; without it the sweep is forward-only
# (conservative). lr and schedule are NOT changed - pair with accumulate()
# to scale the step to the new effective batch.
model, loader, opt = autotrainer.prepare(
    model, loader, opt, optimize=True, auto_bs=True, loss_fn=loss_fn
)
```

### Dataloader bottleneck monitor

The cheapest piece of the roadmap's training-triage theme. Sample per-step
data-load vs compute time and get a plain-language warning when the loader
is starving the GPU:

```python
mon = autotrainer.BottleneckMonitor(warmup=10)
for xb, yb in loader:
    with mon.data_time():
        pass  # the wait for the next batch
    with mon.step_time():
        loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
    mon.tick()
    if mon.should_report():
        mon.report()   # -> "[autotrainer] bottleneck: dataloader is 78% of
                       #     step time ... - raise num_workers / pin_memory / prefetch"
```

Opt-in; zero overhead when not constructed.

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

Toward 1.0:

- **Stabilization**: the public API is frozen as of 0.10; 1.0 removes the
  deprecated `train_loader=`/`val_loader=` aliases after a soak period and
  real multi-node SLURM validation.

Understanding your training run (the theme after 1.0 - autotrainer should
explain runs, not just launch them):

- **Preflight estimation** (`doctor --profile`): dry-run a few batches,
  then report projected training time, memory headroom, and cost per GPU
  count - answer "how many GPUs do I actually need?" before burning an
  allocation.
- **Training triage**: watch the loop and diagnose failures in plain
  language - NaN loss traced to a too-high LR, GPU idle time traced to a
  dataloader bottleneck, fp16 overflow with a bf16 suggestion.
- **Data sanity checks in `auto()`**: the same one-batch peek that infers
  the loss can flag class imbalance (suggest weighted loss), unnormalized
  inputs, and train/val overlap.
- **Training cards**: every `fit()` emits a reproducibility card
  (recipe, seeds, environment, val curve) and `replay` reruns it.

Deeper SLURM ergonomics:

- **`autotrainer sbatch train.py --nodes 2 --time 4h`**: generate and
  submit a correct sbatch script (no more `--ntasks-per-node` != GPUs
  footguns).
- **Preemption handling**: catch SLURM's requeue signal, checkpoint via
  `fit()`'s resume support, and continue after requeue automatically.

More breadth:

- **Multi-node boosting** (xgboost.dask / lightgbm.dask across a SLURM
  allocation) — currently single-node threads only.
- **More schedulers and search spaces** beyond warmup+cosine and the default
  Optuna recipe.

See [CHANGELOG.md](CHANGELOG.md) for the full version history, and open or
upvote [issues](https://github.com/OriAlpha/autotrainer/issues) to prioritize
these.
