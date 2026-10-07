# What autotrainer's optimizations actually buy


Measured on one RTX 5070 Laptop GPU. Your mileage varies with model, GPU and
data — these are the shape of the win, not a promise:

| Change | Effect | Measured on |
|---|---|---|
| `channels_last`, **AMP on** | **+28%** per step, **+34%** when `train_step()` also converts the input | 4-conv net, 64x3x128x128 |
| `channels_last`, **AMP off** | **−68%** — which is why it is gated on AMP and never applied without it | same |
| Fused optimizer kernels in `auto()` | **+11–12%** per step | MLP stacks, 40x64 and 20x512 |
| Fused optimizer kernels in `auto()` | **+123%** when the optimizer step dominates | MLP stack, 4x2048 |
| `optimize=True` vs torch defaults, end to end | **+22%** | 4-conv net, 64x3x128x128 |

Nothing here changes your maths. `channels_last` is a memory layout and fused
optimizers are the same update in one kernel instead of one launch per tensor —
identical results, fewer cycles. Every one of them is printed when it fires:

```
[autotrainer] optimize: TF32, cudnn.benchmark, channels_last, num_workers=8, pin_memory, persistent_workers, AMP (hyperparameters untouched)
```
