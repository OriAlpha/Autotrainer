# Getting throughput out of your GPUs

`prepare()` makes your model and loader distribution-ready and — by default on
a GPU — also flips on the wins people forget: TF32, `cudnn.benchmark` for CNNs,
sane `num_workers` / `pin_memory` / `persistent_workers` defaults on bare
loaders, and AMP — **without touching your lr, loss, schedule, or optimizer
choice**.

It is a no-op on CPU (every flag gates on a visible CUDA device), so CPU-only
scripts see no change. Pass `optimize=False` to opt out.

## Before: the boilerplate you write today

None of this touches your recipe (lr, loss, schedule), yet you have to remember
all of it every time — and forgetting any one of them silently leaves 2–3× on
the table.

```python
import torch

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # your hyperparameter
loss_fn = nn.CrossEntropyLoss()                             # your hyperparameter

# Manual GPU optimization — ~10 lines of boilerplate:
torch.backends.cuda.matmul.allow_tf32 = True                # ships off by default
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True                       # free win for CNNs
loader = DataLoader(ds, batch_size=64, shuffle=True,
                    num_workers=8, pin_memory=True,         # avoid GPU starvation
                    persistent_workers=True)
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

for epoch in range(epochs):
    for xb, yb in loader:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            loss = loss_fn(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
```

## After: one line

```python
import autotrainer

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # your hyperparameter
loss_fn = nn.CrossEntropyLoss()                             # your hyperparameter

# ONE line detects hardware and sets TF32 / cudnn.benchmark / num_workers /
# pin_memory / persistent_workers / AMP. lr, loss, schedule, optimizer untouched.
model, loader, optimizer = autotrainer.prepare(model, loader, optimizer)

scaler = autotrainer.GradScaler()   # no-op when bf16 is available
for epoch in range(epochs):
    autotrainer.set_epoch(loader, epoch)                    # optional: prepare()'s
                                                            # loader self-advances
    for xb, yb in loader:
        with autotrainer.autocast_context():                # bf16 if supported, else fp16
            loss = loss_fn(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
```

What changed: **one `prepare(...)` call plus two no-op-on-CPU helpers.** What
didn't: your lr, your loss, your schedule, your optimizer. The same script runs
unchanged on a laptop (everything degrades to no-ops) and on an A100.

| | Manual | `prepare()` (optimize default) |
|---|---|---|
| Lines of "optimize my GPUs" boilerplate | ~10, hand-written, easy to forget | **1** |
| Hyperparameters touched | none (correct) | none (correct) |
| Works on CPU | guard every line yourself | automatic (all no-ops) |
| Works on SLURM | you'd never write this path | same script, `srun autotrainer run` |
| Knows what it did | silent | prints it |

## Even simpler: `train_step()` runs the whole step

`prepare()` can't wrap *your* loop, so the AMP block above is the one thing you
still hand-write. `train_step()` does it for you — forward under autocast, loss,
scale/backward/step/update, then zero the grads — in one call, returning the
(detached) loss for logging:

```python
model, loader, optimizer = autotrainer.prepare(model, loader, optimizer)
scaler = autotrainer.GradScaler()          # no-op on CPU / bf16; omit to skip

for epoch in range(epochs):
    autotrainer.set_epoch(loader, epoch)
    model.train()
    for xb, yb in loader:
        loss = autotrainer.train_step(model, loss_fn, xb, yb, optimizer, scaler=scaler)
```

Same contract: lr / loss / schedule / optimizer are yours; `train_step` only
runs the forgettable, order-sensitive bookkeeping (backward stays outside
autocast). Pass `autocast=False` to keep full precision, or omit `scaler` on
CPU / bf16 GPUs.

## What it prints when it runs

Nothing is silent — every speedup is named, you are explicitly told your
hyperparameters weren't touched, and when AMP is on `prepare()` points at
`train_step()` (and still shows the manual form) so the loop is one call either
way. The helpers are no-ops on CPU, so the snippet it prints is safe verbatim:

```
[autotrainer] mode=local_multi_gpu nodes=1 procs/node=4 world_size=4
[autotrainer] DistributedSampler installed (shuffle=True) - epoch shuffling auto-managed
[autotrainer] optimize: TF32, cudnn.benchmark, num_workers=8, pin_memory, persistent_workers, AMP (hyperparameters untouched)
[autotrainer] optimize: AMP is on. Simplest - one call per step:
    scaler = autotrainer.GradScaler()   # once, before the loop
    loss = autotrainer.train_step(model, loss_fn, xb, yb, opt, scaler=scaler)
  or wrap the step yourself:
    with autotrainer.autocast_context():
        out = model(x); loss = loss_fn(out, y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
```

## Next

- [Training-loop helpers](training-loop.md) — the small things people forget
  *inside* the loop.
- [Scaling up](scaling.md) — `torch.compile`, FSDP, CPU offload, SLURM.
