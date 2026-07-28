# Monitors

Three opt-in monitors, answering three different questions. All of them only
*observe* — none touches your lr, loss, schedule, or optimizer. Each has zero
overhead when you don't construct it.

| Monitor | Answers | Costs |
|---|---|---|
| `BottleneckMonitor` | Is the dataloader starving the GPU? | two `with` blocks per step |
| `ThroughputMonitor` | How fast is this, in samples/sec and MFU? | one `with` block per step |
| `TrainingMonitor` | Is the training numerically healthy? | one call per step |

## `BottleneckMonitor` — is the loader the problem?

The cheapest piece of the family. Sample per-step data-load vs compute time and
get a plain-language warning when the loader dominates:

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

## `ThroughputMonitor` — how fast is this?

Tracks samples/sec and peak GPU memory over a rolling window (last 50 steps by
default):

```python
mon = autotrainer.ThroughputMonitor(batch_size=64)
for xb, yb in loader:
    with mon.step_time():
        loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
    mon.tick()
    if mon.should_report():
        mon.report()
```

Pass `model_flops=` (per-step forward+backward FLOPs for one batch) to also get
an MFU estimate. It is deliberately not auto-counted — inferring FLOPs for an
arbitrary model is unreliable enough that a wrong number is worse than none.

`stats()` returns the same figures programmatically if you'd rather log them
yourself; it is safe to call at any time and returns zeros or `None` until
enough samples have accumulated.

## `TrainingMonitor` — is it healthy?

`BottleneckMonitor` and `ThroughputMonitor` answer *"is it fast?"*.
`TrainingMonitor` answers *"is it healthy?"* — the silent numerical failures.

Call `step()` once per optimizer step (after `backward()`, before `zero_grad()`)
and it flags each problem **once**, in plain language, with the fix:

```python
mon = autotrainer.TrainingMonitor()
for xb, yb in loader:
    loss = loss_fn(model(xb), yb)
    loss.backward()
    mon.step(loss, model=model, optimizer=opt, scaler=scaler)
    opt.step(); opt.zero_grad()
mon.report()   # one-line all-clear, or a recap of what fired
```

It catches NaN/Inf loss (lr too high, or bad inputs), loss divergence, fp16
gradient overflow (→ "switch to bf16"), and non-finite / spiking / vanishing
gradients — e.g.:

```
[autotrainer] triage: gradient norm spiked to 4.1e+03 (128x the recent median)
    - add gradient clipping (torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
```

Everything but `loss` is optional: pass `model` for gradient checks, `optimizer`
for a concrete lr hint, `scaler` for fp16 overflow detection. The thresholds
(`divergence_factor`, `spike_factor`, `vanish_threshold`, `plateau_patience`,
`overflow_patience`) are constructor arguments if the defaults are too noisy or
too quiet for your model.

## Next

- [One-call training](fit.md) — `fit()` runs its own checks before spending the
  compute.
