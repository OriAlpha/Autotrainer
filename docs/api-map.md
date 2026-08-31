# Entry points

Everything below is exported from `autotrainer`. Everything in
`autotrainer.__all__` is public and stable; the rest is internal.


| You want | Call | Guide |
|---|---|---|
| 1-line train, recipe infer, & model save across PyTorch/Sklearn/XGBoost/TF | `train(model, loader, save_path="model.pt")` | [One-line training](guide/one-line-training.md) |
| My loop, but using the hardware properly | `prepare(model, loader, opt)` | [GPU optimization](guide/gpu-optimization.md) |
| The whole step written for me | `train_step(...)` | [GPU optimization](guide/gpu-optimization.md#even-simpler-train_step-runs-the-whole-step) |
| Loss / optimizer / LR / schedule inferred | `auto(model, loader)` | [One-line training](guide/one-line-training.md) |
| A learning rate suggestion | `find_lr(model, loader, loss_fn)` | [Training loop](guide/training-loop.md#finding-a-learning-rate) |
| The largest batch size that fits | `find_batch_size(model, step_fn)` | [Training loop](guide/training-loop.md#batch-size) |
| Zero-dependency Web UI dashboard | `autotrainer ui`, `run_ui_server()` | [Monitors](guide/monitors.md#autotrainer-ui) |
| Native experiment tracking (CSV, JSONL, metadata) | `NativeTracker()`, `CSVTracker()`, `JSONLTracker()` | [Monitors](guide/monitors.md#native-experiment-trackers) |
| Multi-framework callbacks | `AutotrainerCallback()`, `AutotrainerHuggingFaceCallback()`, `AutotrainerLightningCallback()`, `AutotrainerKerasCallback()`, `autotrainer_xgboost_callback()`, `autotrainer_lightgbm_callback()` | [Monitors](guide/monitors.md#multi-framework-callbacks) |
| Multi-format exports (HTML, Markdown, CSV, JSON) | `autotrainer ui` &rarr; Export Dropdown | [Monitors](guide/monitors.md#multi-format-exports) |
| Multi-user workspaces & directory aggregation | `autotrainer ui ./logs /cluster/runs` | [Monitors](guide/monitors.md#multi-user-workspace) |
| To know if the loader is the bottleneck | `BottleneckMonitor()` | [Monitors](guide/monitors.md) |
| Samples/sec and a rough MFU estimate | `ThroughputMonitor()` | [Monitors](guide/monitors.md) |
| To know if training is going wrong | `TrainingMonitor()` | [Monitors](guide/monitors.md) |
| To check the environment before it costs an allocation | `autotrainer doctor` | [Scaling up](guide/scaling.md) |
| To shard a model too big for one GPU | `prepare(..., fsdp=True)` | [Scaling up](guide/scaling.md) |
| XGBoost/LightGBM params with sane threads | `boost_params(lib="xgboost")` | — |
| TensorFlow strategy scope | `scope()`, `scale_batch_size(n)` | — |
