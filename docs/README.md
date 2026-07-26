# autotrainer documentation

## API reference

A generated API reference for the `autotrainer` package is built with
[pdoc](https://pdoc.dev). To build it locally:

```bash
pip install -e ".[dev]"
pdoc -o docs/build src/autotrainer
```

Then open `docs/build/index.html` in a browser. The CI `docs` job builds this
on every push/PR and uploads it as an artifact (`api-docs`).

The reference covers everything exported in `autotrainer.__all__`:

- **Entry points:** `prepare`, `auto`, `tune`, `fit`, `find_lr`,
  `find_batch_size`, `scope`, `scale_batch_size`, `boost_params`.
- **Training-loop helpers:** `train_step`, `accumulate`, `zero_grad`,
  `eval_mode`, `train_mode`, `set_epoch`, `GradScaler`, `autocast_context`.
- **Rank-aware utilities:** `rank`, `is_main`, `print0`, `save0`, `barrier`.
- **Monitors:** `ThroughputMonitor`, `BottleneckMonitor`.
- **SLURM helpers:** `configure_scratch`, `configure_nccl`, `node_scratch`.

plus the per-framework backends. Everything else (submodules, `_`-prefixed
helpers) is internal - see the [public API policy](../CONTRIBUTING.md#public-api-and-deprecation-policy).

## Other docs

- [../README.md](../README.md) - quickstart, install, and SLURM usage.
- [../CHANGELOG.md](../CHANGELOG.md) - version history.
- [../CONTRIBUTING.md](../CONTRIBUTING.md) - dev setup and PR guidelines.
- [../.env.example](../.env.example) - every environment variable autotrainer
  reads, with comments.
- [../examples/](../examples/) - runnable example scripts and SLURM `.sbatch`
  templates.
