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

## autotrainer ui

Launch the zero-dependency dark-mode Web UI dashboard directly from CLI or Python:

```bash
autotrainer ui                          # Launches server at http://localhost:8501
autotrainer ui ./logs /cluster/runs     # Watch multiple directories simultaneously
```

Or programmatically via `run_ui_server(logs_dirs=["logs", "/shared/cluster/runs"], port=8501)`.

### Access and exposure

The dashboard is not read-only — it renames, archives and **deletes** run
directories — so it is locked down by default:

- **Binds `127.0.0.1`.** Only this machine can reach it. `--host 0.0.0.0`
  opens it to the network and prints a warning saying what that exposes; on a
  shared login node that is every other user on the cluster. Prefer an SSH
  tunnel: `ssh -L 8501:localhost:8501 you@cluster`, then browse locally.
- **Requires a session token.** One is generated per launch and included in
  the URL printed at startup — open that link rather than bare
  `localhost:8501`. The browser keeps it in a `SameSite=Strict` cookie
  afterwards. Scripts can send `X-Autotrainer-Token` instead. `--no-token`
  disables the check, which is only reasonable if you trust every local user.

```bash
autotrainer ui --host 0.0.0.0           # network-visible; warns, still needs the token
autotrainer ui --no-token               # no auth; trusted single-user machines only
```

A `run_id` is always resolved inside the directories you passed, so a crafted
id cannot read or delete anything elsewhere on disk.

### Multi-User Workspace
The UI automatically detects and aggregates runs by user identity (`AUTOTRAINER_USER`, `SLURM_JOB_USER`, or OS username). Use the **Workspace User** dropdown to filter runs by specific team members or view the cluster as a whole (`👥 All Users (X users, Y runs)`).

### Multi-Format Exports
Export any training run with 1 click directly from the dashboard:
- **🌐 Standalone HTML Report (`.html`)**: Complete interactive report with embedded Chart.js curves, AI triage doctor diagnoses, and print/PDF optimization.
- **📝 Markdown Summary (`.md`)**: Formatted table ready to paste into GitHub PRs, Slack, or Notion.
- **📊 Metrics CSV (`.csv`)**: Raw numerical step & epoch logs.
- **📦 Raw JSON (`.json`)**: Complete telemetry payload.

## Native Experiment Trackers

`NativeTracker()`, `CSVTracker()`, and `JSONLTracker()` log structured metrics locally under `./logs/<run_id>/` without requiring external 3rd-party tracking packages:

```python
tracker = autotrainer.NativeTracker(run_name="my_experiment", user="suhas")
tracker.log_params({"lr": 0.001, "batch_size": 32})
tracker.log_epoch(1, {"train_loss": 0.45, "val_loss": 0.41, "val_acc": 88.5})
tracker.close()
```

## Multi-Framework Callbacks

Use built-in callbacks to stream telemetry directly from your favorite ML framework:
- `AutotrainerCallback()` for standard PyTorch or custom loops
- `AutotrainerHuggingFaceCallback()` for Hugging Face `Trainer(callbacks=[...])`
- `AutotrainerLightningCallback()` for PyTorch Lightning `Trainer(callbacks=[...])`
- `AutotrainerKerasCallback()` for Keras `model.fit(callbacks=[...])`
- `autotrainer_xgboost_callback()` for XGBoost `xgb.train(callbacks=[...])`
- `autotrainer_lightgbm_callback()` for LightGBM `lgb.train(callbacks=[...])`

## Next

- [One-call training](fit.md) — `fit()` runs its own checks before spending the compute.
