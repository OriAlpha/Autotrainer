# autotrainer documentation

## Start here

- [Entry points](api-map.md) - which function to call for what.
- [Performance](performance.md) - what the optimizations measurably buy.
- [How it compares](comparison.md) - vs Lightning, Accelerate, Ray, `torchrun`.
- [Roadmap](roadmap.md) - what is planned toward 1.0 and after.

## Guide

Task-oriented walkthroughs. Start at whichever question you actually have:

- [Getting throughput out of your GPUs](guide/gpu-optimization.md) — what
  `prepare(optimize=True)` replaces, and `train_step()`.
- [Training-loop helpers](guide/training-loop.md) — `set_epoch`, `zero_grad`,
  `eval_mode`, `accumulate`, batch size, `find_lr`.
- [Monitors](guide/monitors.md) — bottleneck, throughput, and training health.
- [Scaling up](guide/scaling.md) — launching, `torch.compile`, FSDP, CPU
  offload, SLURM.
- [One-line training](guide/one-line-training.md) — `train()`, `auto()`, and
  the data checks that run before the compute.

## API reference

A generated API reference for the `autotrainer` package is built with
[pdoc](https://pdoc.dev). The published copy lives at
<https://orialpha.github.io/Autotrainer/>. To build it locally:

```bash
pip install -e ".[dev]"
pdoc -o docs/build src/autotrainer
```

Then open `docs/build/index.html` in a browser. The CI `docs` job builds this
on every push/PR and uploads it as an artifact (`api-docs`).

The reference covers everything exported in `autotrainer.__all__`:

- **Entry points:** `prepare`, `auto`, `train`, `find_lr`,
  `find_batch_size`, `scope`, `scale_batch_size`, `boost_params`, `finish`.

- **Training-loop helpers:** `train_step`, `accumulate`, `zero_grad`,
  `eval_mode`, `train_mode`, `set_epoch`, `GradScaler`, `autocast_context`.
- **Rank-aware utilities:** `rank`, `is_main`, `print0`, `save0`, `barrier`.
- **Monitors:** `ThroughputMonitor`, `BottleneckMonitor`, `TrainingMonitor`,
  `SummaryTracker`, plus `log_epoch` and `step` for recording into the active
  summary.
- **Trackers and UI:** `NativeTracker`, `CSVTracker`, `JSONLTracker`,
  `run_ui_server`.
- **Framework callbacks:** `AutotrainerCallback`,
  `AutotrainerHuggingFaceCallback`, `AutotrainerLightningCallback`,
  `AutotrainerKerasCallback`, `autotrainer_xgboost_callback`,
  `autotrainer_lightgbm_callback`.
- **SLURM helpers:** `configure_scratch`, `configure_nccl`, `node_scratch`.


plus the per-framework backends. Everything else (submodules, `_`-prefixed
helpers) is internal - see the [public API policy](../CONTRIBUTING.md#public-api-and-deprecation-policy).

## Other docs

- [../README.md](../README.md) - quickstart, install, and how autotrainer
  compares to the alternatives.
- [../CHANGELOG.md](../CHANGELOG.md) - version history.
- [../CONTRIBUTING.md](../CONTRIBUTING.md) - dev setup and PR guidelines.
- [runner-setup.md](runner-setup.md) - self-hosted GPU CI runner setup.

- [../.env.example](../.env.example) - every environment variable autotrainer
  reads, with comments.
- [../examples/](../examples/) - runnable example scripts and SLURM `.sbatch`
  templates.
